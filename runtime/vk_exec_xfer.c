// Transfers of the Vulkan op-stream replay: buffer and texture copies, texture
// clears, event signals, counter reports and conditional rendering. vk_exec.c calls vk_exec_op_xfer for these ops.
//
// Buffers are guest memory: vk_mem_resolve maps an address to the VkBuffer behind its pool, textures are the vk_texture.c images
// (linear texel layout, GENERAL layout; block-linear storage is only converted when a mirrored texture is uploaded). So a
// copy is one Vulkan transfer command and the pitch of a buffer side comes from the payload (OP_COPY_STRIDE or the
// tight row). Every copy ends the open pass and starts with a full barrier. Destinations: an image is marked
// GPU-written at record time; a CPU-visible buffer range is reported to vk_gpu_written once the GPU is done (write
// watch does not see GPU writes, textures stored there re-upload).
//
// Events and counters are CPU-visible side effects. Their values go into guest memory on the completion thread through
// vk_completion_add, in submit order, as gfx_null.c does: event WRITE / ADD, counter {0, timestamp} for every type but
// samples passed. A timestamp report puts 0 in the counter word: NX2NAVI gates its clears on two such words being unequal.
//
// Samples passed (NVN_COUNTER_TYPE_SAMPLES_PASSED) is a running count on the queue: RESET_COUNTER zeroes it, every
// draw after it adds the samples that passed the depth and stencil tests, REPORT_COUNTER writes the running value.
// NX2NAVI resets and reports it 3 to 8 times a frame in the field and its CPU reads the results back (f_70010c3298); a 0
// there culls what the query tested. The replay counts with Vulkan occlusion queries ("occlusion counts" below).
//
// Conditional rendering is a CPU-side skip, not VK_EXT_conditional_rendering: the extension compares one 32-bit word
// with zero, NVN compares the counter words at addr and addr + 16. The condition is evaluated when the op is recorded
// from guest memory; a counter reported by an earlier submission that is not done yet counts as 0 if it is a timestamp,
// as unknown if it is samples passed. An unknown count renders: the report usually sits in the command buffer being
// recorded, so its value cannot be waited for, and the word in memory still holds an older query's count. Rendering
// costs at most the work the condition would have saved; in the field
// NX2NAVI conditions only on timestamp reports. The condition sets VkExec.render_disabled, which vk_exec.c honours for
// clears and vk_exec_draw.c for draws and dispatches. A condition that cannot be read (address outside every pool)
// renders.
// Copies and CLEAR_TEX are not conditional.
//
// Contains code Copyright (c) 2026 jhaizhou, Copyright (c) 2026 Peiikai (MIT; see THIRD_PARTY_NOTICES.md).
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "platform/platform.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_dlaa.h"
#include "resource_core_registry.h"
#include "resource_core_vma.h"
#include "resource_image_write.h"
#include "resource_host_write.h"
#include "resource_command_effects.h"

enum { W_FORMAT, W_REGION, W_BUFFER, W_OVERLAP, W_COPY, W_CLEAR, W_QUERY, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

static uint32_t max1(uint32_t v) { return v ? v : 1; }

// ---- completion callbacks (completion thread)

typedef struct {
    uint64_t addr, size;
    int back, gpu_only;
    void *provider;
    VkBuffer buffer;
} Range;   // back: written into a VRAM mirror (vk_mem_mirrored)

static void guest_written_cb(void *arg) {
    Range *r = arg;
    if (!r->gpu_only) vk_mem_buffer_write_completed(r->provider, r->buffer, r->addr, r->size, r->back);
    vk_mem_recorded_provider_release(r->provider);
    free(r);
}

static void guest_written_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;
    Range *r = arg;
    vk_mem_recorded_provider_release(r->provider);
    free(r);
}

typedef struct {
    MemBackingCapture *mapping;
    ResourceHostWrite canonical;
    RcRegistryWrite raw;
    uint64_t address, size, fence, compatibility_mapping;
} CpuResultTarget;

static CpuResultTarget cpu_result_prepare(uint64_t address, uint64_t size, uint64_t fence) {
    CpuResultTarget target = {.address = address, .size = size, .fence = fence};
    uint64_t compatibility_mapping = mem_mapping_serial();
    mem_backing_capture(address, size, &target.mapping);
    if (!target.mapping && mem_mapping_range_unchanged(address, size, compatibility_mapping))
        target.compatibility_mapping = compatibility_mapping;
    uint32_t count;
    const RcInterval *source = mem_backing_intervals(target.mapping, &count);
    RcInterval *intervals;
    uint32_t copied = resource_host_interval_slice(source, count, address, size, &intervals);
    RcRegistryWrite *mapped = copied ? calloc(copied, sizeof *mapped) : NULL;
    if (mapped && resource_effects_buffer_result_take(address, size, intervals, copied, &target.raw, mapped)) {
        target.canonical = (ResourceHostWrite){.intervals = intervals, .tickets = mapped, .count = copied};
    } else {
        free(mapped);
        target.canonical = resource_host_write_prepare(intervals, copied, RC_REGISTRY_BUFFER, 0);
        rc_registry_write_begin(rc_registry_runtime(), (RcSourceRange){1, address, size}, RC_REGISTRY_BUFFER, &target.raw);
    }
    return target;
}

static int cpu_result_read_hold(const CpuResultTarget *target) {
    if (target->mapping) return mem_backing_read_hold(target->mapping);
    if (!target->compatibility_mapping || !mem_read_hold(target->address, target->size)) return 0;
    if (mem_mapping_range_unchanged(target->address, target->size, target->compatibility_mapping)) return 1;
    mem_read_release();
    return 0;
}

static void cpu_result_dispose(CpuResultTarget *target, int written) {
    if (written) vk_mem_cpu_result_written(target->address, target->size);
    RcRegistry *registry = rc_registry_runtime();
    resource_host_write_dispose(&target->canonical, written ? target->fence : 0, written, written);
    if (target->raw.id) {
        if (written) {
            if (rc_registry_write_submit(registry, target->raw, target->fence) == RC_REGISTRY_OK)
                rc_registry_write_complete(registry, target->raw, 1);
            else rc_registry_write_abandon(registry, target->raw, 1);
        } else {
            rc_registry_write_cancel(registry, target->raw);
        }
    }
    if (written) mem_read_release();
    mem_backing_release(&target->mapping);
}

typedef struct {
    CpuResultTarget target;
    volatile uint32_t *val;
    uint32_t mode, value;
} EventWrite;

// The side effects below that ran after a waiter on the executed timeline had gone on (vk_sync.c vk_sync_effect_late):
// the game was already past the wait after its submit when these reached its memory.
static volatile uint64_t late_events, late_counters, late_counts;

void vk_exec_late_stats(uint64_t *events, uint64_t *counters, uint64_t *counts) {
    *events = __atomic_exchange_n(&late_events, 0, __ATOMIC_RELAXED);
    *counters = __atomic_exchange_n(&late_counters, 0, __ATOMIC_RELAXED);
    *counts = __atomic_exchange_n(&late_counts, 0, __ATOMIC_RELAXED);
}

static void event_cb(void *arg) {
    EventWrite *e = arg;
    if (vk_sync_effect_late()) __atomic_fetch_add(&late_events, 1, __ATOMIC_RELAXED);
    int held = cpu_result_read_hold(&e->target);
    if (held) {
        if (e->mode == NVN_EVENT_SIGNAL_MODE_ADD) __atomic_fetch_add(e->val, e->value, __ATOMIC_RELEASE);
        else __atomic_store_n(e->val, e->value, __ATOMIC_RELEASE);
    }
    cpu_result_dispose(&e->target, held);
    free(e);
}

static void event_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;
    EventWrite *e = arg;
    cpu_result_dispose(&e->target, 0);
    free(e);
}

// NVNcounterData = {counter, timestamp}; timestamps are Tegra 19.2 MHz ticks (the base of GetSystemTick).
static void write_counter(uint64_t addr, uint64_t counter) {
    uint64_t rec[2] = {counter, (uint64_t)((unsigned __int128)kern_now_ns() * 12 / 625)};
    memcpy((void *)(uintptr_t)addr, rec, sizeof rec);
}

// Reports other than samples passed (timestamps; the statistics counters are not modelled) write a counter of 0. The
// counter word matters: Ocarina writes {1, 0} into a record from the CPU, reports TIMESTAMP_TOP into it and gates
// clears on it differing from a zero record (mode 1), clears that must not run.
static void pending_finished(uint64_t addr, uint64_t seq, int wrote);

typedef struct { CpuResultTarget target; uint64_t seq; } CounterWrite;

static void counter_cb(void *arg) {
    CounterWrite *w = arg;
    if (vk_sync_effect_late()) __atomic_fetch_add(&late_counters, 1, __ATOMIC_RELAXED);
    int held = cpu_result_read_hold(&w->target);
    if (held) write_counter(w->target.address, 0);
    pending_finished(w->target.address, w->seq, held);
    cpu_result_dispose(&w->target, held);
    free(w);
}

static void counter_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;
    CounterWrite *w = arg;
    pending_finished(w->target.address, w->seq, 0);
    cpu_result_dispose(&w->target, 0);
    free(w);
}

// Reports queued by the replay that the GPU has not finished. The condition of RENDER_COND reads a timestamp one as 0
// and a samples-passed one as unknown.
// The newest report per address, in a table probed over PENDING_PROBE slots from the address's hash: Ocarina issues
// hundreds of occlusion reports a frame and conditions draws on them in the same frame, so a ring of the last 256
// reports had dropped the report a condition referred to, which then read an older query's count (often 0) and culled
// visible ground and trees. It also rotates through tens of thousands of record addresses, so an entry whose value is
// in memory (written >= seq: a condition reads memory anyway) gives its slot to a new address; a table that only grew
// filled up after a minute and conditions on the newer records read the CPU's 1 again.
#define PENDING (1u << 16)
#define PENDING_PROBE 32
// seq: the newest report to addr; written: the newest finished result; canceled: the newest discarded result. A
// report counts as pending until then, not until its fence passes: the callback runs after the fence, and in between
// the record still held what the CPU wrote (Ocarina re-arms its condition records with 1 for the next frame), so a
// condition read 1 where the GPU had written 0, a depth clear ran after the depth copy and frames showed only sky.
// frame: the present count (vk_exec_cond_frame_end, in replay order) when the report was replayed. The game arms its
// condition records from the CPU between frames, so within a frame a timestamp report decides the record's value even
// after it reached memory: Ocarina's CPU, running a frame ahead, had re-armed a record with 1 by the time a later depth
// clear of the frame read it, the clear ran after the depth copy and the frame showed only sky (white flashes).
typedef struct {
    uint64_t addr, seq, frame, mapping;
    volatile uint64_t written, canceled;
    int unknown, type;
} Pending;
static Pending pending[PENDING];
static uint64_t report_seq;
static uint64_t cond_frame;

// Per-frame counts of conditional work (NX2NAVI_LOG_RCOND=1, vk_exec_cond_frame_end logs them): clears and
// draws issued under a render condition, run or skipped.
unsigned vk_cond_clears_run, vk_cond_clears_skip, vk_cond_draws_skip, vk_cond_depth_run;

void vk_exec_cond_frame_end(uint64_t frame) {
    static int on = -1;
    if (on < 0) on = rt_knob("LOG_RCOND", "0");
    if (on && (vk_cond_clears_run || vk_cond_clears_skip || vk_cond_draws_skip))
        rt_log("vk: frame %llu conditions: clears run %u skipped %u, draws skipped %u, depth clears run %u",
               (unsigned long long)frame, vk_cond_clears_run, vk_cond_clears_skip, vk_cond_draws_skip, vk_cond_depth_run);
    vk_cond_clears_run = vk_cond_clears_skip = vk_cond_draws_skip = vk_cond_depth_run = 0;
    if (on == 2) rt_log("SEQ present %llu", (unsigned long long)frame);
    cond_frame++;
}

static Pending *pending_slot(uint64_t addr, int insert);

static void pending_advance(volatile uint64_t *watermark, uint64_t seq) {
    uint64_t old = __atomic_load_n(watermark, __ATOMIC_RELAXED);
    while (old < seq && !__atomic_compare_exchange_n(watermark, &old, seq, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {}
}

// Completion thread: only an applied report can decide a conditional read.
static void pending_finished(uint64_t addr, uint64_t seq, int wrote) {
    Pending *e = pending_slot(addr, 0);
    if (!e) return;
    if (!wrote) pending_advance(&e->canceled, seq);
    pending_advance(&e->written, seq);
}

// Lookups run on the replay thread and the completion thread; only the replay thread inserts.
static Pending *pending_slot(uint64_t addr, int insert) {
    if (!addr) return NULL;
    uint32_t i = (uint32_t)((addr * 0x9E3779B97F4A7C15ull) >> 48) & (PENDING - 1);
    Pending *free_ = NULL, *victim = NULL;
    for (unsigned n = 0; n < PENDING_PROBE; n++, i = (i + 1) & (PENDING - 1)) {
        Pending *e = &pending[i];
        uint64_t a = __atomic_load_n(&e->addr, __ATOMIC_ACQUIRE);
        if (a == addr) return e;
        if (!insert) continue;
        if (!a) { if (!free_) free_ = e; continue; }
        if (__atomic_load_n(&e->written, __ATOMIC_ACQUIRE) >= e->seq && e->frame != cond_frame && (!victim || e->seq < victim->seq))
            victim = e;   // done, and not from this frame (whose reports still decide their records)
    }
    Pending *e = free_ ? free_ : victim;
    if (!e) return NULL;   // every probed entry still has a report in flight
    e->seq = 0, e->frame = 0, e->unknown = 0, e->type = -1;
    __atomic_store_n(&e->canceled, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&e->written, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&e->addr, addr, __ATOMIC_RELEASE);
    return e;
}

// ---- occlusion counts (samples passed)
//
// After a SAMPLES_PASSED reset, the first draw of a pass begins an occlusion query (precise when the device has
// occlusionQueryPrecise); the query ends with the pass, or at the next report or reset of the counter, and the next
// draw begins another. A Vulkan query begun inside a render pass must end inside it. NX2NAVI always resets, draws the
// tested proxies, reports, so counting stops at a report and restarts at the next reset: queries cover only the draws
// between the two, not every pass of the frame. Once the game reports twice without a reset in between, counting stays
// on for good (cumulative NVN semantics; that one report misses the draws since the previous one). The queries closed since the last
// report or reset form a segment, handed to the completion thread with that report or reset: in submit order it sums
// the segment's results into the running count (a reset drops the sum and zeroes the count) and writes the count of a
// report into guest memory. Slots come from one ring; the completion thread reads and host-resets them
// (hostQueryReset, Vulkan 1.2), so the replay records no reset commands. A full ring (no slot read back yet) leaves the
// draw uncounted and the segment's count at least 1, as does a query still unavailable after its fence. Without hostQueryReset the counts stay 0.
// The ring holds about four frames of Hyrule Field's queries (about 3,500 a frame): the replay workers' chunks take
// blocks of slots sized to their draws (vk_exec_xfer_slots), more than they use, and with 4096 the ring came round to
// slots the completion thread had not reset yet, leaving draws uncounted.
#define OQ_N 16384
#define LOG_ONCE(...) do { static volatile char once_; if (!__atomic_exchange_n(&once_, 1, __ATOMIC_RELAXED)) rt_log(__VA_ARGS__); } while (0)
static VkQueryPool oq_pool;
static int oq_state;                     // 0 not set up, 1 counting available, -1 unavailable
static VkQueryControlFlags oq_flags;
static volatile uint8_t oq_busy[OQ_N];   // from begin until the completion thread read and reset the slot
static uint32_t oq_next;                 // replay: next ring slot
static int oq_counting;                  // replay: draws are counted (a reset since the last report, or oq_keep)
static int oq_reported;                  // replay: a samples-passed report since the last reset
static int oq_keep;                      // replay: the game reported twice without a reset: count every draw
static int oq_open = -1;                 // replay: the slot whose query is active in the open pass
static uint32_t *oq_seg;                 // replay: slots closed since the last report or reset
static uint32_t oq_nseg, oq_capseg;
static int oq_lost;                      // replay: the segment missed a draw (ring full)
static uint64_t oq_count;                // completion thread: samples since the last reset

typedef struct OqBlock OqBlock;
typedef struct {
    CpuResultTarget target;
    uint64_t addr, seq;
    int reset;
    volatile int32_t lost;
    uint32_t n, cap, first;
    uint32_t *q;
    OqBlock *block;
} OqSegment;

// One producer reference while filling a block and one reference per result. Shutdown can cancel later unsent
// results while earlier handed slots are still live, so the last slot's callback cannot own the whole block.
enum { OQ_SEGMENTS_PER_BLOCK = 128 };
struct OqBlock { uint32_t refs; OqSegment segments[OQ_SEGMENTS_PER_BLOCK]; };
static OqBlock *oq_block;
static unsigned oq_used;

static void oq_block_release(OqBlock *block) {
    if (block && !__atomic_sub_fetch(&block->refs, 1, __ATOMIC_ACQ_REL)) free(block);
}

static OqSegment *oq_new_segment(void) {
    if (!oq_block) {
        oq_block = calloc(1, sizeof *oq_block);
        if (!oq_block) rt_fatal("vk: out of memory (occlusion reports)");
        oq_block->refs = 1;
    }
    OqSegment *s = &oq_block->segments[oq_used++];
    s->block = oq_block;
    __atomic_add_fetch(&oq_block->refs, 1, __ATOMIC_RELAXED);
    s->q = &s->first;
    s->cap = 1;
    if (oq_used == OQ_SEGMENTS_PER_BLOCK) {
        oq_block_release(oq_block);
        oq_block = NULL;
        oq_used = 0;
    }
    return s;
}

static void oq_reserve_slots(OqSegment *s, uint32_t cap) {
    if (cap <= s->cap) return;
    if (s->q == &s->first) {
        s->q = malloc(sizeof *s->q * cap);
        if (s->q && s->n) s->q[0] = s->first;
    } else {
        s->q = realloc(s->q, sizeof *s->q * cap);
    }
    if (!s->q) rt_fatal("vk: out of memory (occlusion slots)");
    s->cap = cap;
}

// The replay workers (vk_exec.c "parallel groups") record the draws of one group at once. The front pass replays a
// group's samples-passed resets and reports in the order of the stream before the workers record (vk_exec_xfer_front):
// each makes its segment, queued for the completion thread in that order, and they are listed in GS. A worker follows
// the counting as the stream does from its chunk's first op (vk_exec_xfer_chunk_state): XW.counting and the segment its
// closed queries belong to, the next one at or after its position in the stream (XW.seg; past the group's last, the
// pending list oq_seg, which a later reset or report takes). It keeps the query open in its own pass (XW.open), takes
// ring slots from its chunk's block and keeps the slots it closes with their segment in its chunk's list (CP), which
// vk_exec_xfer_group_merge puts into the segments once the group is done, chunk after chunk: under a lock per slot, the
// workers waited for each other at every query. XW.cond: the decisions of its chunk's render conditions, made by the
// front pass in the order of the stream.
// The blocks (vk_exec_xfer_slots): the front pass gives each chunk as many slots as it has draws that count samples and
// render, one after the other, so the slots follow the stream as the replay's own do and the completion thread reads
// and resets them in long runs (oq_take). Taken from the ring at once, the slots of a group's chunks interleaved: four
// times the result calls and a reset per slot, 3.5 ms more of the wait after a frame's first submission (RX 7900 XT,
// Hyrule Field, 3 workers). A chunk that needs more than its block takes them from the ring.
static _Thread_local struct {
    int on, open, counting, reported, keep, chunk, g;
    uint32_t seg, qnext, qend;
    const uint8_t *cond;
    uint32_t ncond, i;
} XW = {.open = -1};
// GS: per group slot (vk_exec.c LIVE_GROUPS), the front pass's segments of the group. tail: with NX2NAVI_REPLAY_PIPE
// the submitting thread replays on past the group before its chunks are done: the first segment made after the
// group's front pass (tail_wait), by the next group's front pass or the replay, takes the queries the chunks close
// after the group's last report or reset, which in the stream come before that one's own.
#define GS_GROUPS 4
static struct { int tail_wait; OqSegment **s, *tail; uint32_t n, cap; } GS[GS_GROUPS];
static int gs_on = -1;                   // the group slot whose front pass runs
#define CP_CHUNKS 32
static struct { uint32_t *seg, *q, n, cap; } CP[GS_GROUPS][CP_CHUNKS];   // per chunk: closed slots and their segment

void vk_exec_xfer_group(int group) {   // a group's front pass begins (its slot), or it ended (-1): the list stays
    if (group >= GS_GROUPS) rt_fatal("vk: parallel group slot %d of %d", group, GS_GROUPS);
    if (group >= 0) GS[group].n = 0, GS[group].tail_wait = 0, GS[group].tail = NULL;
    else if (gs_on >= 0) GS[gs_on].tail_wait = 1;   // until vk_exec_xfer_group_merge
    gs_on = group;
}

int vk_exec_xfer_counting(void) { return oq_counting; }

uint32_t vk_exec_xfer_slots(uint32_t n) {   // the workers of a group the submitting thread went on past take some too
    return __atomic_fetch_add(&oq_next, n, __ATOMIC_RELAXED);
}

uint64_t vk_exec_xfer_chunk_state(void) {
    return (uint64_t)GS[gs_on].n << 8 | (uint64_t)(oq_counting & 1) | (uint64_t)(oq_reported & 1) << 1 | (uint64_t)(oq_keep & 1) << 2;
}

void vk_exec_xfer_worker(int on, const uint8_t *cond, uint32_t ncond, uint64_t state, uint32_t slot0, uint32_t nslots,
                         int chunk, int group) {
    if (chunk < 0 || chunk >= CP_CHUNKS) rt_fatal("vk: replay worker chunk %d of at most %d", chunk, CP_CHUNKS);
    if (group < 0 || group >= GS_GROUPS) rt_fatal("vk: parallel group slot %d of %d", group, GS_GROUPS);
    XW.on = on, XW.open = -1, XW.cond = cond, XW.ncond = ncond, XW.i = 0, XW.chunk = chunk, XW.g = group;
    XW.qnext = slot0, XW.qend = slot0 + nslots;
    XW.counting = (int)(state & 1), XW.reported = (int)(state >> 1 & 1), XW.keep = (int)(state >> 2 & 1);
    XW.seg = (uint32_t)(state >> 8);
}

// Adds closed slot q to the segment a worker's queries belong to now, or the pending list.
static void seg_add(uint32_t q) {
    __typeof__(CP[0][0]) *l = &CP[XW.g][XW.chunk];
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 256;
        if (!(l->seg = realloc(l->seg, sizeof *l->seg * l->cap)) || !(l->q = realloc(l->q, sizeof *l->q * l->cap)))
            rt_fatal("vk: out of memory");
    }
    l->seg[l->n] = XW.seg, l->q[l->n++] = q;
}

void vk_exec_xfer_group_merge(int group, int nchunks) {
    __typeof__(GS[0]) *gs = &GS[group];
    for (int c = 0; c < nchunks && c < CP_CHUNKS; c++) {
        __typeof__(CP[0][0]) *l = &CP[group][c];
        for (uint32_t i = 0; i < l->n; i++) {
            uint32_t q = l->q[i];
            OqSegment *s = l->seg[i] < gs->n ? gs->s[l->seg[i]] : gs->tail;
            if (s) {
                if (s->n == s->cap) oq_reserve_slots(s, s->cap * 2);
                s->q[s->n++] = q;
            } else {
                if (oq_nseg == oq_capseg) {
                    oq_capseg = oq_capseg ? oq_capseg * 2 : 64;
                    if (!(oq_seg = realloc(oq_seg, sizeof *oq_seg * oq_capseg))) rt_fatal("vk: out of memory");
                }
                oq_seg[oq_nseg++] = q;
            }
        }
        l->n = 0;
    }
    gs->tail_wait = 0, gs->tail = NULL;
}

static int oq_ready(void) {
    if (oq_state) return oq_state > 0;
    VkCtx *v = vk_ctx();
    // vk_device.c enables every feature the device reports, so supported means enabled.
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(v->phys, &f2);
    oq_state = -1;
    if (!f12.hostQueryReset) {
        rt_log("vk: no hostQueryReset: samples-passed counters report 0");
        return 0;
    }
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION,
                                .queryCount = OQ_N};
    VkResult r = vkCreateQueryPool(v->device, &qi, NULL, &oq_pool);
    if (r != VK_SUCCESS) {
        rt_log("vk: occlusion query pool creation failed (%d): samples-passed counters report 0", (int)r);
        return 0;
    }
    vkResetQueryPool(v->device, oq_pool, 0, OQ_N);
    oq_flags = f2.features.occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0;
    oq_state = 1;
    return 1;
}

static void oq_close(VkExec *ex) {
    int *open = XW.on ? &XW.open : &oq_open;
    if (*open < 0) return;
    vkCmdEndQuery(ex->cb, oq_pool, (uint32_t)*open);
    ex->recorded++;
    if (XW.on) seg_add((uint32_t)*open);
    else {
        if (oq_nseg == oq_capseg) {
            oq_capseg = oq_capseg ? oq_capseg * 2 : 64;
            oq_seg = realloc(oq_seg, sizeof *oq_seg * oq_capseg);
            if (!oq_seg) rt_fatal("vk: out of memory");
        }
        oq_seg[oq_nseg++] = (uint32_t)*open;
    }
    *open = -1;
}

void vk_exec_oq_draw(VkExec *ex) {
    int *open = XW.on ? &XW.open : &oq_open;
    if (!(XW.on ? XW.counting : oq_counting) || *open >= 0) return;
    // a worker takes the next slot of its chunk's block, past it one from the ring at once (as the replay does, while
    // a group's workers may still take some); a busy one is lost, not tried again
    uint32_t q = (XW.on && XW.qnext != XW.qend ? XW.qnext++ : __atomic_fetch_add(&oq_next, 1, __ATOMIC_RELAXED)) % OQ_N;
    if (__atomic_load_n(&oq_busy[q], __ATOMIC_ACQUIRE)) {
        warn_once(W_QUERY, "occlusion query ring full: draws left uncounted");
        if (XW.on && XW.seg < GS[XW.g].n) __atomic_store_n(&GS[XW.g].s[XW.seg]->lost, 1, __ATOMIC_RELAXED);
        else __atomic_store_n(&oq_lost, 1, __ATOMIC_RELAXED);
        return;
    }
    __atomic_store_n(&oq_busy[q], 1, __ATOMIC_RELAXED);
    vkCmdBeginQuery(ex->cb, oq_pool, q, oq_flags);
    ex->recorded++;
    *open = (int)q;
}

void vk_exec_oq_pass_end(VkExec *ex) { oq_close(ex); }

static void oq_batch_drop(void);   // the results kept by the completion thread (oq_take)

void vk_exec_xfer_shutdown(void) {
    oq_block_release(oq_block);
    oq_block = NULL;
    oq_used = 0;
    if (oq_state > 0) vkDestroyQueryPool(vk_ctx()->device, oq_pool, NULL);
    oq_pool = VK_NULL_HANDLE, oq_state = 0, oq_next = 0, oq_counting = oq_reported = oq_keep = 0, oq_open = -1;
    oq_nseg = 0, oq_lost = 0, oq_count = 0;
    memset((void *)oq_busy, 0, sizeof oq_busy);
    oq_batch_drop();
}

// The completion thread reads the results in batches (NX2NAVI_OQ_BATCH, default 1; 0: every segment by itself, as
// before). The game tests one proxy per report, so a segment holds one query, and a frame of Hyrule Field has about a
// hundred of them: one vkGetQueryPoolResults and one vkResetQueryPool each. On an RX 7900 XT (26.10.1) each of those
// calls goes into the kernel, and the two hundred took 3.9 ms of every frame on the completion thread, which runs a
// submission's callbacks before it lets the submission's waiters go: the game's main thread waits for the previous
// frame's fence before it lets the next frame's GPU work start, so the frame after one replayed inside its submit
// call began 4 to 6 ms late with every thread idle. Slots come from the ring in order, so the slots of the segments
// that follow lie behind the one asked for: one call reads OQ_BATCH slots from there on with their availability, the
// available ones are kept (the results stay in the pool until the reset), and the segments after it take theirs from
// what is kept. A slot not available in the batch (its part had not run yet) is read again when its segment comes.
// The slots taken are reset in runs, one vkResetQueryPool for each: a run ends when a slot that does not continue it
// is taken or a new batch is read. Until then its slots stay busy for the replay, at most OQ_BATCH of the ring's
// OQ_N, also while the game reports nothing.
#define OQ_BATCH 256
static struct {
    uint32_t first, n;               // the slots kept: [first, first + n)
    uint64_t v[OQ_BATCH * 2];        // per slot: samples, available
    uint8_t taken[OQ_BATCH];
    uint32_t r0, rn;                 // the run of taken slots still to be reset
    uint64_t calls, slots, segs;     // since the last perf line: result calls, slots they read, segments served
} OQB;
static int oq_batch_mode = -1;

static void oq_batch_drop(void) { OQB.n = OQB.rn = 0; }

static void oq_reset_flush(VkDevice dev) {
    if (!OQB.rn) return;
    vkResetQueryPool(dev, oq_pool, OQB.r0, OQB.rn);
    for (uint32_t k = 0; k < OQB.rn; k++) __atomic_store_n(&oq_busy[OQB.r0 + k], 0, __ATOMIC_RELEASE);
    OQB.rn = 0;
}

static void oq_read(VkDevice dev, uint32_t first, uint32_t n) {
    VkResult r = vkGetQueryPoolResults(dev, oq_pool, first, n, sizeof OQB.v[0] * 2 * n, OQB.v, 2 * sizeof OQB.v[0],
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (r != VK_SUCCESS && r != VK_NOT_READY) {
        vk_device_lost_check(r, "the read of occlusion query results");
        rt_fatal("vk: occlusion query results failed (%d)", (int)r);
    }
    OQB.first = first, OQB.n = n;
    memset(OQB.taken, 0, n);
    OQB.calls++, OQB.slots += n;
}

// The samples of slot q, whose submission has completed; *avail 0 when the query has no result.
static uint64_t oq_take(VkDevice dev, uint32_t q, int *avail) {
    uint32_t k = q - OQB.first;
    if (!(q >= OQB.first && k < OQB.n && !OQB.taken[k] && OQB.v[2 * k + 1])) {
        // not kept, or kept as unavailable (read before its part ran): from this slot on, up to the replay's next slot
        oq_reset_flush(dev);   // before the kept results go: the run's slots are among them
        uint32_t next = __atomic_load_n(&oq_next, __ATOMIC_RELAXED) % OQ_N, n = OQ_N - q;
        if (next > q && next - q < n) n = next - q;
        if (n > OQ_BATCH) n = OQ_BATCH;
        if (oq_batch_mode == 0) n = 1;
        oq_read(dev, q, n);
        k = 0;
    }
    OQB.taken[k] = 1;
    if (OQB.rn && q != OQB.r0 + OQB.rn) oq_reset_flush(dev);
    if (!OQB.rn) OQB.r0 = q;
    OQB.rn++;
    *avail = OQB.v[2 * k + 1] != 0;
    return OQB.v[2 * k];
}

static void oq_cb(void *arg) {
    OqSegment *s = arg;
    VkDevice dev = vk_ctx()->device;
    if (oq_batch_mode < 0) oq_batch_mode = cfg_bool("NX2NAVI_OQ_BATCH");
    // a segment the replay workers filled lists its slots in the order the chunks closed them: in ring order, they are
    // taken in runs (oq_take)
    for (uint32_t i = 1; i < s->n; i++) {
        uint32_t v = s->q[i], j = i;
        for (; j && s->q[j - 1] > v; j--) s->q[j] = s->q[j - 1];
        s->q[j] = v;
    }
    uint64_t sum = 0;
    for (uint32_t i = 0; i < s->n; i++) {
        // The fence has passed, so a query that is still unavailable will not become available: waiting would block the
        // completion thread, and with it every later callback, for good. It counts as visible (one sample).
        int avail;
        uint64_t v = oq_take(dev, s->q[i], &avail);
        if (avail) sum += v;
        else {
            sum += 1;
            vk_device_lost_poll();   // a query its submission never wrote: on a lost device the run ends here
            LOG_ONCE("vk: occlusion query slot %u not available after its fence, counted as visible", s->q[i]);
        }
    }
    if (s->n && ++OQB.segs >= 12000) {
        if (rt_trace & RT_TRACE_PERF)
            rt_log("perf: occlusion queries: %llu segments read with %llu result calls of %.1f slots on average",
                   (unsigned long long)OQB.segs, (unsigned long long)OQB.calls,
                   OQB.calls ? (double)OQB.slots / (double)OQB.calls : 0.0);
        OQB.calls = OQB.slots = OQB.segs = 0;
    }
    if (s->reset) oq_count = 0;
    else {
        oq_count += sum;
        if (s->lost && !oq_count) oq_count = 1;
        if (s->addr) {
            if (vk_sync_effect_late()) __atomic_fetch_add(&late_counts, 1, __ATOMIC_RELAXED);
            int held = cpu_result_read_hold(&s->target);
            if (held) write_counter(s->addr, oq_count);
            pending_finished(s->addr, s->seq, held);
            cpu_result_dispose(&s->target, held);
        }
    }
    if (s->q != &s->first) free(s->q);
    oq_block_release(s->block);
}

static void oq_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;
    OqSegment *s = arg;
    if (s->addr) {
        pending_finished(s->addr, s->seq, 0);
        cpu_result_dispose(&s->target, 0);
    }
    if (s->q != &s->first) free(s->q);
    oq_block_release(s->block);
}

// Hands the segment to the completion thread with a report (addr; 0: no CPU-visible memory) or a reset.
static void oq_segment(VkExec *ex, uint64_t addr, int reset) {
    oq_close(ex);
    OqSegment *s = oq_new_segment();
    oq_reserve_slots(s, oq_nseg);
    s->addr = addr, s->seq = report_seq, s->reset = reset, s->lost = oq_lost, s->n = oq_nseg;
    if (addr) s->target = cpu_result_prepare(addr, sizeof(NVNcounterData), ex->fence);
    if (oq_nseg) memcpy(s->q, oq_seg, sizeof *s->q * oq_nseg);
    if (gs_on >= 0) {   // the front pass: the workers add the queries of the group's ops before it
        __typeof__(GS[0]) *gs = &GS[gs_on];
        if (gs->n == gs->cap) {
            gs->cap = gs->cap ? gs->cap * 2 : 64;
            if (!(gs->s = realloc(gs->s, sizeof *gs->s * gs->cap))) rt_fatal("vk: out of memory");
        }
        gs->s[gs->n++] = s;
    }
    for (int g = 0; g < GS_GROUPS; g++)   // the first made after a group's front pass
        if (GS[g].tail_wait && !GS[g].tail) GS[g].tail = s;
    oq_nseg = 0, oq_lost = 0;
    vk_completion_add_result_owned(ex->fence, oq_cb, oq_cancel, s);
}

static void reset_counter(VkExec *ex, const p_reset_counter *p) {
    if (p->type != NVN_COUNTER_TYPE_SAMPLES_PASSED || !oq_ready()) return;
    oq_segment(ex, 0, 1);
    oq_counting = 1, oq_reported = 0;
}

// A samples-passed report: the segment goes with it (addr 0: no CPU-visible memory) and counting stops until the next
// reset, unless reports came without resets before.
static void oq_report(VkExec *ex, uint64_t addr) {
    oq_segment(ex, addr, 0);
    if (oq_reported && !oq_keep) {
        oq_keep = 1;
        rt_log("vk: samples passed reported twice without a reset: counting every draw from now on");
    }
    oq_reported = 1;
    if (!oq_keep) oq_counting = 0;
}

// ---- event order check
// The game orders work between its queues with events: a command buffer waits (WaitEvent) until another's SignalEvent
// has set the value. Every submission replays here on one queue in the order it came, and a wait records nothing, so a
// wait whose signal comes in a later submission does not hold its work back: that work runs before what it waits for.
// The replay keeps the value each event reaches in replay order (ev_seen, the signals recorded so far, on top of the
// value in memory) and counts the waits it reaches unmet; NX2NAVI_TRACE=perf logs the count, and the first few are
// logged as they happen. Replay thread only.
static VkExecEventOrder ev_seen;
static uint64_t ev_waits, ev_unmet, ev_late, ev_late_subs;
// Waits reached unmet, until a replayed signal meets them (ev_late: the signal came that many submissions later, so
// the work after the wait ran before the work it waited for) or they are pushed out by newer ones.
#define EV_OPEN 64
static struct { volatile uint32_t *val; uint32_t mode, value; uint64_t sub; } ev_open[EV_OPEN];
static int ev_open_n;

static uint32_t *ev_slot(VkExecEventOrder *order, volatile uint32_t *val, int create, uint32_t memory) {
    for (int i = 0; i < order->count; i++)
        if (order->slots[i].val == val) return &order->slots[i].value;
    if (!create) return NULL;
    int i = order->count < VK_EXEC_EVENTS_SEEN ? order->count++ : order->next++ % VK_EXEC_EVENTS_SEEN;
    order->slots[i].val = val;
    order->slots[i].value = memory;
    return &order->slots[i].value;
}

void vk_exec_event_order_copy(VkExecEventOrder *order) { *order = ev_seen; }

static int ev_holds(uint32_t have, uint32_t mode, uint32_t want) {
    return mode == NVN_EVENT_WAIT_MODE_GEQUAL_WRAP ? (int32_t)(have - want) >= 0 : have == want;
}

static int ev_holds_either(uint32_t memory, const uint32_t *seen, const p_wait_event *wait) {
    return ev_holds(memory, wait->mode, wait->value) ||
           (seen && ev_holds(*seen, wait->mode, wait->value));
}

int vk_exec_event_order_met(VkExecEventOrder *order, const p_wait_event *wait, uint32_t memory) {
    return ev_holds_either(memory, ev_slot(order, wait->val, 0, 0), wait);
}

uint32_t vk_exec_event_order_signal(VkExecEventOrder *order, const p_signal_event *signal, uint32_t memory) {
    uint32_t *seen = ev_slot(order, signal->val, 1, memory);
    *seen = signal->mode == NVN_EVENT_SIGNAL_MODE_ADD ? *seen + signal->value : signal->value;
    return *seen;
}

// ---- events the CPU signals
// Ocarina gates its main queue on the CPU: each frame the game sets one of two events to 0 (they take turns frame by
// frame), submits the worker queues' command buffers and the main queue's, whose first command waits for the event
// to be 1, and signals 1 from the CPU (nvnEventSignal) once it has finished writing what the frame's commands read.
// The signal comes before the main queue's submit call, after it, or while that call is replaying. On the console
// the GPU starts the queue at the signal and reads memory from then on.
// Here a submission's recording begins with the copies of what it reads: the VRAM mirrors' sync and the uniform
// snapshot (vk_exec_record), and the wait is tested when the walk reaches it, some hundred microseconds later. A
// signal that came in between passed the wait, and the frame was drawn from copies taken before the game had
// finished writing: single frames with the scene's shadowing missing or the whole scene black, the interface as it
// was, most of them when other work keeps the CPU busy, since the game's threads then finish later and the signal
// falls into the replay more often.
// So a wait is tested against the value the event had when the submission took its copies: every nvnEventSignal is
// numbered and keeps the value it replaced (vk_exec_event_cpu, before the write), vk_exec_record notes the number at
// its start (vk_exec_event_sync), and a signal with a later number does not count for that submission. The wait then
// stops the submission like any wait reached before its signal, and vk_backend.c submit_sync lets it go on at once
// in a submission of its own, whose copies are taken after the signal.
// NX2NAVI_EVENT_AT_SYNC=0: the value in memory when the walk reaches the wait, as before.
#define CPU_EV 16
#define CPU_EV_HIST 8
static struct { const void *val; uint64_t seq[CPU_EV_HIST]; uint32_t before[CPU_EV_HIST]; unsigned n; } cpu_ev[CPU_EV];
static int cpu_ev_n;                             // cpu_ev_lock
static uint64_t cpu_ev_seq;                      // cpu_ev_lock: nvnEventSignal calls so far
static uint64_t ev_sync_seq = UINT64_MAX;        // cpu_ev_lock: cpu_ev_seq at the recording's start; UINT64_MAX outside one
static PlatMutex cpu_ev_lock = PLAT_MUTEX_INIT;
static uint64_t ev_after_sync;                   // replay: waits only a signal after the copies would have passed

int vk_exec_event_at_sync(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("EVENT_AT_SYNC", "1");
    return on;
}

void vk_exec_event_cpu(const void *event) {
    plat_mutex_lock(&cpu_ev_lock);
    int i = 0;
    while (i < cpu_ev_n && cpu_ev[i].val != event) i++;
    if (i == cpu_ev_n && cpu_ev_n < CPU_EV) cpu_ev[cpu_ev_n++].val = event, cpu_ev[i].n = 0;
    cpu_ev_seq++;
    if (i < cpu_ev_n) {
        unsigned k = cpu_ev[i].n++ % CPU_EV_HIST;
        cpu_ev[i].seq[k] = cpu_ev_seq, cpu_ev[i].before[k] = __atomic_load_n((const volatile uint32_t *)event, __ATOMIC_ACQUIRE);
    }
    plat_mutex_unlock(&cpu_ev_lock);
}

// 1 when the CPU has signalled this event before (nvnEventSignal): its waits are gates the game opens from the CPU.
int vk_exec_event_cpu_known(const void *event) {
    plat_mutex_lock(&cpu_ev_lock);
    int i = 0;
    while (i < cpu_ev_n && cpu_ev[i].val != event) i++;
    int known = i < cpu_ev_n;
    plat_mutex_unlock(&cpu_ev_lock);
    return known;
}

// The wait's condition on the value in memory now, whatever the replay has seen. Any thread.
int vk_exec_event_holds_now(const void *payload) {
    const p_wait_event *p = payload;
    return ev_holds(__atomic_load_n(p->val, __ATOMIC_ACQUIRE), p->mode, p->value);
}

void vk_exec_event_sync(int begin) {
    plat_mutex_lock(&cpu_ev_lock);
    ev_sync_seq = begin ? cpu_ev_seq : UINT64_MAX;
    plat_mutex_unlock(&cpu_ev_lock);
}

uint64_t vk_exec_event_late_take(void) {
    uint64_t n = ev_after_sync;
    ev_after_sync = 0;
    return n;
}

// The event's value in memory, and in *at_sync the value it had when the submission being recorded took its copies
// (the same outside a recording, and for an event the CPU has not signalled since).
static uint32_t ev_mem(volatile uint32_t *val, uint32_t *at_sync) {
    uint32_t mem = __atomic_load_n(val, __ATOMIC_ACQUIRE);
    if (vk_exec_frozen_event(val, at_sync)) return mem;
    *at_sync = mem;
    plat_mutex_lock(&cpu_ev_lock);
    if (ev_sync_seq != UINT64_MAX)
        for (int i = 0; i < cpu_ev_n; i++) {
            if (cpu_ev[i].val != (const void *)val) continue;
            uint64_t first = UINT64_MAX;
            unsigned n = cpu_ev[i].n < CPU_EV_HIST ? cpu_ev[i].n : CPU_EV_HIST;
            for (unsigned k = 0; k < n; k++)   // the earliest signal after the copies replaced the value they saw
                if (cpu_ev[i].seq[k] > ev_sync_seq && cpu_ev[i].seq[k] < first) first = cpu_ev[i].seq[k], *at_sync = cpu_ev[i].before[k];
            break;
        }
    plat_mutex_unlock(&cpu_ev_lock);
    return mem;
}

// Whether the wait holds. *late: it holds only by a CPU signal that came after the submission's copies.
static int ev_met(const p_wait_event *p, uint32_t *mem, const uint32_t **seen, int *late) {
    uint32_t at_sync;
    *mem = ev_mem(p->val, &at_sync);
    *seen = ev_slot(&ev_seen, p->val, 0, 0);
    int now = ev_holds_either(*mem, *seen, p), then = ev_holds_either(at_sync, *seen, p);
    *late = now && !then;
    return vk_exec_event_at_sync() ? then : now;
}

int vk_exec_event_met(const void *payload) {
    uint32_t mem;
    const uint32_t *seen;
    int late;
    return ev_met(payload, &mem, &seen, &late);
}

int vk_exec_signalled;

int vk_exec_wait_event(const void *payload) {
    const p_wait_event *p = payload;
    ev_waits++;
    uint32_t mem;
    const uint32_t *seen;
    int late;
    int met = ev_met(p, &mem, &seen, &late);
    if (late) {
        ev_after_sync++;
        vk_exec_evt('L', (const void *)p->val, 0);
    }
    if (met) return 1;
    ev_unmet++;
    if (ev_open_n == EV_OPEN) memmove(ev_open, ev_open + 1, sizeof ev_open - sizeof *ev_open), ev_open_n--;
    ev_open[ev_open_n].val = p->val, ev_open[ev_open_n].mode = p->mode, ev_open[ev_open_n].value = p->value;
    ev_open[ev_open_n++].sub = vk_exec_submit_seq;
    static int logged;
    if (logged++ < 8)
        rt_log("vk: event wait reached before its signal: event %p, mode %u, wants %#x, memory %#x, replayed signals %s%#x",
               (void *)p->val, p->mode, p->value, mem, seen ? "" : "none ", seen ? *seen : 0);
    return 0;
}

void vk_exec_event_stats(uint64_t *waits, uint64_t *unmet, uint64_t *late, uint64_t *late_subs) {
    *waits = ev_waits, *unmet = ev_unmet, *late = ev_late, *late_subs = ev_late_subs;
    ev_waits = ev_unmet = ev_late = ev_late_subs = 0;
}

static void signal_event(VkExec *ex, const p_signal_event *p) {
    uint32_t memory;
    if (!vk_exec_frozen_event(p->val, &memory)) memory = __atomic_load_n(p->val, __ATOMIC_ACQUIRE);
    uint32_t seen = vk_exec_event_order_signal(&ev_seen, p, memory);
    vk_exec_evt('s', (const void *)p->val, seen);
    vk_exec_signalled = 1;
    for (int i = 0; i < ev_open_n;)
        if (ev_open[i].val == p->val && ev_holds(seen, ev_open[i].mode, ev_open[i].value)) {
            ev_late++, ev_late_subs += vk_exec_submit_seq - ev_open[i].sub;
            ev_open[i] = ev_open[--ev_open_n];
        } else i++;
    EventWrite *e = malloc(sizeof *e);
    if (!e) rt_fatal("vk: out of memory");
    *e = (EventWrite){.target = cpu_result_prepare((uintptr_t)p->val, sizeof *p->val, ex->fence),
                      .val = p->val, .mode = p->mode, .value = p->value};
    vk_completion_add_result_owned(ex->fence, event_cb, event_cancel, e);
    resource_effects_buffers_commit();
}

static void report_counter(VkExec *ex, const p_report_counter *p) {
    uint64_t mapping = mem_mapping_serial();
    int samples = p->type == NVN_COUNTER_TYPE_SAMPLES_PASSED && oq_ready();
    report_seq++;   // before oq_report / counter_cb take it
    if (nvn_gpu_only(p->addr)) {   // CPU-invisible pool: no host memory behind it
        if (samples) oq_report(ex, 0);
        return;
    }
    VkMemSpan m;
    if (!vk_mem_resolve(p->addr, sizeof(NVNcounterData), &m))
        rt_fatal("vk: counter report at %#llx is in no memory pool", (unsigned long long)p->addr);
    vk_mem_count(m.cls, VK_MUSE_GPU_WRITE, sizeof(NVNcounterData));
    if (samples) oq_report(ex, p->addr);
    else {
        CounterWrite *a = malloc(sizeof *a);
        if (!a) rt_fatal("vk: out of memory");
        *a = (CounterWrite){cpu_result_prepare(p->addr, sizeof(NVNcounterData), ex->fence), report_seq};
        vk_completion_add_result_owned(ex->fence, counter_cb, counter_cancel, a);
    }
    resource_effects_buffers_commit();
    Pending *e = pending_slot(p->addr, 1);
    if (e) e->seq = report_seq, e->frame = cond_frame, e->mapping = mapping, e->unknown = samples, e->type = (int)p->type;
    else LOG_ONCE("vk: a counter report's table neighbourhood is all in flight: conditions on %#llx read memory",
                  (unsigned long long)p->addr);
}

// ---- conditional rendering

static const Pending *pending_current(uint64_t addr) {
    const Pending *e = pending_slot(addr, 0);
    if (!e || __atomic_load_n(&e->canceled, __ATOMIC_ACQUIRE) >= e->seq ||
        !mem_mapping_range_unchanged(addr, sizeof(NVNcounterData), e->mapping)) return NULL;
    return e;
}

// 1: *v read, 0: not readable, 2: a samples-passed report not in memory yet (unknown)
static int counter_word(uint64_t addr, uint64_t *v) {
    const Pending *e = pending_current(addr);   // the last report to this mapping decides
    if (e && !e->unknown && e->frame == cond_frame) return *v = 0, 1;   // a timestamp/statistics report of this frame
    if (e && __atomic_load_n(&e->written, __ATOMIC_ACQUIRE) < e->seq) return *v = 0, e->unknown ? 2 : 1;
    VkMemSpan m;
    if (nvn_gpu_only(addr) || !vk_mem_resolve(addr, sizeof *v, &m) || !mem_read_hold(addr, sizeof *v)) return 0;
    memcpy(v, (const void *)(uintptr_t)addr, sizeof *v);
    mem_read_release();
    vk_mem_count(m.cls, VK_MUSE_CPU_READ, sizeof *v);
    return 1;
}

// The counter type of the newest report to addr the ring still holds, -1 when there is none (NX2NAVI_LOG_RCOND).
static int last_type(uint64_t addr) {
    const Pending *e = pending_current(addr);
    return e ? e->type : -1;
}

// NX2NAVI_LOG_RCOND=ab (diagnosis): the words a condition compares and where each came from: "f" a report replayed
// this frame (0), "p" a pending report (0 or unknown), "m" memory, with the newest report's type and frame.
void vk_exec_cond_why(const p_render_cond *c, char *buf, size_t cap) {
    int o = snprintf(buf, cap, "%#llx mode %d:", (unsigned long long)c->addr, c->mode);
    for (int k = 0; k < 2 && o < (int)cap; k++) {
        uint64_t a = c->addr + 16 * (uint64_t)k;
        const Pending *e = pending_current(a);
        const char *src = !e ? "m" : !e->unknown && e->frame == cond_frame ? "f"
                        : __atomic_load_n(&e->written, __ATOMIC_ACQUIRE) < e->seq ? "p" : "m";
        o += snprintf(buf + o, cap - (size_t)o, " %llu %s(type %d, frame %lld of %llu)",
                      (unsigned long long)*(const uint64_t *)(uintptr_t)a, src, e ? e->type : -1,
                      e ? (long long)e->frame : -1LL, (unsigned long long)cond_frame);
    }
}

static int cond_enabled(const p_render_cond *c) {
    uint64_t x, y;
    int rx = counter_word(c->addr, &x), ry = counter_word(c->addr + 16, &y);
    if (!rx || !ry) {
        warn_once(W_BUFFER, "conditional render counters at %#llx are not readable: rendering", (unsigned long long)c->addr);
        return 1;
    }
    if (rx == 2 || ry == 2) return 1;   // a samples-passed count not done yet
    int on = c->mode == 0 ? x == y : x != y;
    // NX2NAVI_LOG_RCOND=1: one condition in 499 (up to 300 lines), on or off, with the counter types last reported there
    static int log_on = -1, nlog;
    static unsigned ncond;
    if (log_on < 0) log_on = rt_knob("LOG_RCOND", "0");
    if (log_on && ncond++ % 499 == 0 && nlog < 300) {
        nlog++;
        // (pending: the report value has not reached guest memory yet)
        int px = 0, py = 0;   // the word came from a report not done yet (forced to 0) rather than memory
        const Pending *ex_ = pending_current(c->addr), *ey_ = pending_current(c->addr + 16);
        px = ex_ && ex_->written < ex_->seq, py = ey_ && ey_->written < ey_->seq;
        const uint64_t *m = (const uint64_t *)(uintptr_t)c->addr;
        rt_log("vk: render condition %s at %#llx mode %d: %llu (type %d%s) vs %llu (type %d%s); memory %llx %llx %llx %llx",
               on ? "on" : "off", (unsigned long long)c->addr, c->mode, (unsigned long long)x, last_type(c->addr), px ? ", pending" : "",
               (unsigned long long)y, last_type(c->addr + 16), py ? ", pending" : "", (unsigned long long)m[0],
               (unsigned long long)m[1], (unsigned long long)m[2], (unsigned long long)m[3]);
    }
    return on;
}

// ---- textures

typedef struct {
    const VkTexImage *img;
    VkImageAspectFlags aspect;
    uint32_t level, layer0;
    int bpb, bw, bh;
    int32_t x, y, z;           // region origin
    uint32_t w, h, nz;         // region extent; nz: slices (3D) or layers
} Sub;

// Resolves tex, its view and a copy region to image coordinates. 0 (warned) when the copy cannot be replayed.
static int sub_region(const tex_t *t, const view_t *v, int has_v, const NVNcopyRegion *r, int defaults_to_level, Sub *s) {
    const VkTexImage *img = s->img;
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) s->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    else if (img->aspect == VK_IMAGE_ASPECT_DEPTH_BIT) s->aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    else {
        warn_once(W_FORMAT, "copy of depth-stencil format %#x texture: skipped", t->d.format);
        return 0;
    }
    s->level = has_v && v->set_levels ? v->base_level : 0;
    s->layer0 = has_v && v->set_layers ? v->min_layer : 0;
    if (s->level >= img->levels || s->level >= 32) {
        warn_once(W_REGION, "copy region outside texture level %u: skipped", s->level);
        return 0;
    }
    uint32_t lw = max1(img->width >> s->level), lh = max1(img->height >> s->level);
    int k = img->scale > 1 ? img->scale : 1;   // a larger shadow map: the game's region in its texels
    int64_t x = (int64_t)r->xoffset * k, y = (int64_t)r->yoffset * k;
    uint64_t w = r->width > 0 ? (uint64_t)r->width * k : defaults_to_level && x >= 0 && x < lw ? lw - x : 0;
    uint64_t h = r->height > 0 ? (uint64_t)r->height * k : defaults_to_level && y >= 0 && y < lh ? lh - y : 0;
    s->nz = r->depth > 0 ? (uint32_t)r->depth : 1;
    s->z = r->zoffset;
    int is3d = img->type == VK_IMAGE_TYPE_3D;
    if (x < 0 || y < 0 || x > INT32_MAX || y > INT32_MAX || s->z < 0 || !w || !h ||
        (uint64_t)x + w > lw || (uint64_t)y + h > lh ||
        (uint64_t)(uint32_t)s->z + s->nz + (is3d ? 0 : s->layer0) > (is3d ? max1(img->depth >> s->level) : img->layers)) {
        warn_once(W_REGION, "copy region %lld,%lld,%d %llux%llux%u outside level %u of a %ux%u texture: skipped",
                  (long long)x, (long long)y, s->z, (unsigned long long)w, (unsigned long long)h, s->nz,
                  s->level, img->width, img->height);
        return 0;
    }
    s->x = (int32_t)x; s->y = (int32_t)y; s->w = (uint32_t)w; s->h = (uint32_t)h;
    return 1;
}

static int sub_format(const tex_t *t, Sub *s) {
    if (!t) return 0;
    TexVkFormat f;
    if (!tex_vk_format(t->d.format, &f) || f.vk != f.ideal || f.expand_rgb || f.astc_bw) {
        warn_once(W_FORMAT, "copy of format %#x texture (decoded or expanded on upload): skipped", t->d.format);
        return 0;
    }
    s->bpb = nvn_fmt_bpb(t->d.format, &s->bw, &s->bh);
    return s->bpb != 0;
}

static int sub_resolve(const tex_t *t, const view_t *v, int has_v, const NVNcopyRegion *r, int defaults_to_level,
                       int write, Sub *s) {
    memset(s, 0, sizeof *s);
    if (!sub_format(t, s)) return 0;
    s->img = write == 2 ? vk_texture_get_alias_transfer_write(t) :
             write ? vk_texture_get_transfer_write(t) : vk_texture_get(t);
    if (!s->img && vk_texture_deferred()) return 0;
    if (!s->img) {
        warn_once(W_FORMAT, "copy of format %#x texture without an image: skipped", t->d.format);
        return 0;
    }
    return sub_region(t, v, has_v, r, defaults_to_level, s);
}

static VkImageSubresourceLayers sub_layers(const Sub *s) {
    int is3d = s->img->type == VK_IMAGE_TYPE_3D;
    return (VkImageSubresourceLayers){s->aspect, s->level, is3d ? 0 : s->layer0 + (uint32_t)s->z, is3d ? 1 : s->nz};
}

static VkOffset3D sub_offset(const Sub *s) {
    return (VkOffset3D){s->x, s->y, s->img->type == VK_IMAGE_TYPE_3D ? s->z : 0};
}

static VkExtent3D sub_extent(const Sub *s) {
    return (VkExtent3D){s->w, s->h, s->img->type == VK_IMAGE_TYPE_3D ? s->nz : 1};
}

typedef struct {
    uint64_t row_bytes, stride, rows, size;
    uint32_t row_length, image_height;
} BufferImagePitch;

enum { COPY_PITCH_OK, COPY_PITCH_INVALID, COPY_PITCH_OVERFLOW };

static int buffer_image_pitch(uint32_t w, uint32_t h, uint32_t layers, int bw, int bh, int bpb,
                              int64_t stride, BufferImagePitch *pitch) {
    if (!w || !h || !layers || bw <= 0 || bh <= 0 || bpb <= 0) return COPY_PITCH_INVALID;
    uint64_t blocks = ((uint64_t)w + (unsigned)bw - 1) / (unsigned)bw;
    uint64_t block_rows = ((uint64_t)h + (unsigned)bh - 1) / (unsigned)bh;
    uint64_t tight = blocks * (unsigned)bpb;
    uint64_t row = stride > 0 ? (uint64_t)stride : tight;
    if (row < tight || row % (unsigned)bpb) return COPY_PITCH_INVALID;
    uint64_t rows = block_rows * layers;
    if (row / (unsigned)bpb > UINT32_MAX / (unsigned)bw ||
        block_rows > UINT32_MAX / (unsigned)bh ||
        (rows > 1 && row > (UINT64_MAX - tight) / (rows - 1))) return COPY_PITCH_OVERFLOW;
    *pitch = (BufferImagePitch){tight, row, rows, (rows - 1) * row + tight,
        (uint32_t)(row / (unsigned)bpb) * (unsigned)bw, (uint32_t)block_rows * (unsigned)bh};
    return COPY_PITCH_OK;
}

static uint64_t buffer_image_runs(const BufferImagePitch *pitch) {
    return pitch->row_bytes == pitch->stride ? 1 : pitch->rows;
}

static uint64_t buffer_image_run_bytes(const BufferImagePitch *pitch) {
    return buffer_image_runs(pitch) == 1 ? pitch->size : pitch->row_bytes;
}

static int buffer_layout(const VkExec *ex, const Sub *s, BufferImagePitch *pitch, VkBufferImageCopy *copy) {
    int result = buffer_image_pitch(s->w, s->h, s->nz, s->bw, s->bh, s->bpb, ex->copy_stride, pitch);
    if (result != COPY_PITCH_OK) {
        if (result == COPY_PITCH_INVALID)
            warn_once(W_REGION, "texture copy row stride does not fit the selected block row: skipped");
        else
            warn_once(W_REGION, "texture copy pitches overflow the buffer extent: skipped");
        return 0;
    }
    *copy = (VkBufferImageCopy){.bufferRowLength = pitch->row_length, .bufferImageHeight = pitch->image_height,
        .imageSubresource = sub_layers(s), .imageOffset = sub_offset(s), .imageExtent = sub_extent(s)};
    return 1;
}

// A VRAM mirror supplies the buffer side when the pool has one.
static int buffer_side(const VkExec *ex, const Sub *s, uint64_t addr, BufferImagePitch *pitch, VkBufferImageCopy *c,
                       VkMemSpan *span, int preflight) {
    if (!buffer_layout(ex, s, pitch, c)) return 0;
    uint64_t size = pitch->size;
    VkMemSpan m;
    if (!addr || size > UINT64_MAX - addr ||
        !(preflight ? vk_mem_lookup(addr, size, &m) : vk_mem_resolve(addr, size, &m)) ||
        m.avail < size || size > UINT64_MAX - m.offset) {
        warn_once(W_BUFFER, "texture copy buffer %#llx (%llu bytes) is not one backed span: skipped",
                  (unsigned long long)addr, (unsigned long long)size);
        return 0;
    }
    if (!preflight) VK_MEM_PICK(m);
    *span = m;
    c->bufferOffset = m.offset;
    return 1;
}

static int buffer_to_texture_ready(VkExec *ex, VkExecBufferToTexture *copy) {
    return copy->read.scene == ex->fence && copy->buffer &&
        !rc_registry_position_compare(copy->read, vk_exec_resource_position()) &&
        vk_exec_buffer_to_texture_source_current(copy) &&
        (copy->source_alias || vk_mem_span_record_input(copy->source_hold, &copy->source));
}

static void buffer_to_texture_issue(VkExec *ex, VkExecBufferToTexture *copy, VkImage image) {
    vk_mem_count(copy->source.cls, VK_MUSE_COPY_SRC, copy->size);
    vk_dlaa_hud_write(image);
    if (copy->source_alias) vk_texture_alias_read_use(copy->source_alias, ex->fence);
    vk_exec_record_native_input_provider(ex, &copy->source_hold);
    vkCmdCopyBufferToImage(ex->cb, copy->buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy->region);
    ex->recorded++;
}

// Positive means a captured source or a verified no-op. No raw provider is selected again at issue.
static int buffer_to_texture_capture_source(VkExecBufferToTexture *copy, uint64_t address,
    const BufferImagePitch *pitch, const Sub *dst) {
    VkMemSpan raw;
    if (!address || pitch->size > UINT64_MAX - address || !vk_mem_lookup(address, pitch->size, &raw) ||
        raw.avail < pitch->size) {
        warn_once(W_BUFFER, "texture copy buffer %#llx (%llu bytes) is not one backed span: skipped",
                  (unsigned long long)address, (unsigned long long)pitch->size);
        return 1;
    }
    const uint32_t alignment = dst->aspect == VK_IMAGE_ASPECT_DEPTH_BIT && dst->bpb < 4 ? 4 : (unsigned)dst->bpb;
    if (raw.offset % alignment) {
        warn_once(W_REGION, "buffer copy offset is not aligned to its texture format: skipped");
        return 1;
    }
    RcRegistryResult status = vk_mem_span_prepare_native_input(address, pitch->size, 0, VK_MEM_MIRROR_IMPORT,
                                                               &copy->source, &copy->source_hold);
    if (status != RC_REGISTRY_OK || !copy->source_hold || copy->source.offset % alignment ||
        copy->source.avail < pitch->size || pitch->size > UINT64_MAX - copy->source.offset) return -1;
    copy->read = vk_exec_resource_position();
    copy->size = pitch->size;
    copy->offset_alignment = alignment;
    copy->format = dst->img->format;
    copy->region.bufferOffset = copy->source.offset;
    VkTextureAliasFailure failure = {0};
    status = vk_texture_alias_buffer_capture_rows(copy->source_hold, &copy->source, address, copy->size,
        pitch->row_bytes, pitch->stride, &copy->source_alias, &failure);
    if (status != RC_REGISTRY_OK)
        rt_log("vk: buffer-to-texture source refused read=%llu/%llu address=%#llx bytes=%#llx "
               "row=%#llx stride=%#llx registry=%d gate=%d cause=%d mapping=%llu backing=%llu offset=%#llx",
            (unsigned long long)copy->read.scene, (unsigned long long)copy->read.ordinal, (unsigned long long)address,
            (unsigned long long)copy->size, (unsigned long long)pitch->row_bytes, (unsigned long long)pitch->stride, status,
            failure.coverage.gate, failure.coverage.cause, (unsigned long long)failure.coverage.interval.mapping,
            (unsigned long long)failure.coverage.interval.range.backing,
            (unsigned long long)failure.coverage.interval.range.offset);
    return status == RC_REGISTRY_OK ? 1 : -1;
}

int vk_exec_buffer_to_texture_capture_compat(VkExec *ex, const p_copy_b2t *op, VkExecBufferToTexture *copy) {
    VkTexImage shape;
    Sub dst = {.img = &shape};
    if (!sub_format(op->tex, &dst) || !vk_texture_transfer_write_geometry(op->tex, &shape)) return 1;
    if (shape.scale > 1 || shape.samples != 1) {
        warn_once(W_REGION, "buffer copy to a scaled or multisample texture: skipped");
        return 1;
    }
    BufferImagePitch pitch;
    if (!sub_region(op->tex, &op->view, op->has_view, &op->r, 1, &dst) ||
        !buffer_layout(ex, &dst, &pitch, &copy->region)) return 1;
    return buffer_to_texture_capture_source(copy, op->src, &pitch, &dst);
}

static int buffer_to_texture_geometry_equal(const VkBufferImageCopy *a, const VkBufferImageCopy *b) {
    return a->bufferRowLength == b->bufferRowLength && a->bufferImageHeight == b->bufferImageHeight &&
        a->imageSubresource.aspectMask == b->imageSubresource.aspectMask &&
        a->imageSubresource.mipLevel == b->imageSubresource.mipLevel &&
        a->imageSubresource.baseArrayLayer == b->imageSubresource.baseArrayLayer &&
        a->imageSubresource.layerCount == b->imageSubresource.layerCount &&
        a->imageOffset.x == b->imageOffset.x && a->imageOffset.y == b->imageOffset.y && a->imageOffset.z == b->imageOffset.z &&
        a->imageExtent.width == b->imageExtent.width && a->imageExtent.height == b->imageExtent.height &&
        a->imageExtent.depth == b->imageExtent.depth;
}

void vk_exec_buffer_to_texture_commit_compat(VkExec *ex, const p_copy_b2t *p, VkExecBufferToTexture *copy) {
    if (!copy->source_hold || vk_exec_recording_failed() || vk_sync_stopping() || vk_device_is_lost()) return;
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    vk_contact_flush(ex);
    const int output = vk_texture_alias_region_transfer(p->tex, p->has_view ? &p->view : NULL, &p->r) ? 2 : 1;
    Sub dst;
    BufferImagePitch pitch;
    VkBufferImageCopy region;
    if (!sub_resolve(p->tex, &p->view, p->has_view, &p->r, 1, output, &dst)) {
        if (!vk_texture_deferred() && !vk_sync_stopping() && !vk_device_is_lost())
            vk_exec_required_graphics_unavailable(ex, OP_COPY_B2T, "retained buffer-to-texture destination unavailable");
        return;
    }
    if (dst.img->scale > 1 || dst.img->samples != 1 || dst.img->format != copy->format ||
        !(dst.img->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || !buffer_layout(ex, &dst, &pitch, &region) ||
        pitch.size != copy->size || !buffer_to_texture_geometry_equal(&copy->region, &region)) {
        vk_exec_required_graphics_unavailable(ex, OP_COPY_B2T, "buffer-to-texture destination geometry changed");
        return;
    }
    vk_gpu_gap_note(VK_GAP_COPY_B2T, (uint64_t)dst.w * dst.h);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (vk_exec_recording_failed() || vk_sync_stopping() || vk_device_is_lost()) return;
    if (!buffer_to_texture_ready(ex, copy)) {
        vk_exec_required_graphics_unavailable(ex, OP_COPY_B2T, "retained buffer-to-texture source changed before issue");
        return;
    }
    buffer_to_texture_issue(ex, copy, dst.img->image);
    resource_image_written((uint64_t)(uintptr_t)dst.img->image, dst.img->addr);
    vk_texture_gpu_written(p->tex);
    const NVNcopyRegion actual = {dst.x, dst.y, dst.z, (int)dst.w, (int)dst.h, (int)dst.nz};
    vk_texture_alias_transfer_written(p->tex, dst.img->image, p->has_view ? &p->view : NULL, &actual);
}

int vk_exec_buffer_to_texture_alias_candidate(const VkExec *ex, const p_copy_b2t *op) {
    if (!vk_texture_alias_region_transfer(op->tex, op->has_view ? &op->view : NULL, &op->r)) return 0;
    const VkTexImage image = {.type = VK_IMAGE_TYPE_2D};
    Sub box = {.img = &image, .aspect = VK_IMAGE_ASPECT_COLOR_BIT, .x = op->r.xoffset, .y = op->r.yoffset,
               .w = (uint32_t)op->r.width, .h = (uint32_t)op->r.height, .nz = 1};
    VkBufferImageCopy layout;
    BufferImagePitch pitch;
    VkMemSpan source;
    return sub_format(op->tex, &box) && buffer_side(ex, &box, op->src, &pitch, &layout, &source, 1);
}

int vk_exec_buffer_to_texture_capture(VkExec *ex, const p_copy_b2t *op, VkExecBufferToTexture *copy) {
    // These prior native hooks can still change an image when the transfer closes rendering.
    if (ex->tonemap_extent.width || ex->bc_written_mask || vk_contact_pending()) return 0;
    VkTexImage shape = {0};
    Sub dst = {.img = &shape};
    if (!sub_format(op->tex, &dst)) return 1;
    if (!vk_texture_write_target_shape(op->tex, &shape)) return 0;
    if (shape.scale > 1 || shape.type != VK_IMAGE_TYPE_2D || shape.aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
        shape.samples != 1 || dst.bw != 1 || dst.bh != 1 || !(shape.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) ||
        (op->has_view && op->view.set_format && op->view.format != op->tex->d.format)) return 0;
    BufferImagePitch pitch;
    if (!sub_region(op->tex, &op->view, op->has_view, &op->r, 1, &dst) ||
        !buffer_layout(ex, &dst, &pitch, &copy->region)) return 1;
    int source = buffer_to_texture_capture_source(copy, op->src, &pitch, &dst);
    if (source < 0 || !copy->source_hold) return source;
    int captured = vk_texture_write_target_capture(op->tex, &shape, &copy->target);
    if (captured <= 0) vk_exec_buffer_to_texture_cancel(copy);
    return captured;
}

int vk_exec_buffer_to_texture_source_current(const VkExecBufferToTexture *copy) {
    return copy->source_hold && vk_mem_span_capture_current(copy->source_hold, &copy->source) &&
        (!copy->source_alias || vk_texture_alias_read_current(copy->source_alias));
}

static int buffer_to_texture_source_current(void *opaque) {
    return vk_exec_buffer_to_texture_source_current(opaque);
}

RcRegistryResult vk_exec_buffer_to_texture_prepare_source(VkExec *ex, VkExecBufferToTexture *copy) {
    if (rc_registry_position_compare(copy->read, vk_exec_resource_position()) ||
        !vk_exec_buffer_to_texture_source_current(copy) || !vk_mem_span_record_input(copy->source_hold, &copy->source))
        return RC_REGISTRY_STALE;
    copy->buffer = copy->source.buffer;
    if (copy->source_alias) {
        VkExecRestoreResult prepared = vk_exec_prepare_alias_readback(ex, &copy->source_alias, 1,
                                                                     buffer_to_texture_source_current, copy);
        if (prepared == VK_EXEC_RESTORE_STOPPED) return RC_REGISTRY_BUSY;
        if (prepared == VK_EXEC_RESTORE_STALE) return RC_REGISTRY_STALE;
        if (prepared != VK_EXEC_RESTORE_READY) return RC_REGISTRY_CAPACITY;
        if (!vk_texture_alias_buffer_source(copy->source_alias, &copy->buffer, &copy->region.bufferOffset) ||
            copy->region.bufferOffset % copy->offset_alignment) return RC_REGISTRY_STALE;
    }
    return RC_REGISTRY_OK;
}

static int b2t_probe_on;
static uint64_t b2t_probe_copies;
uint64_t vk_exec_buffer_to_texture_probe_count(void) {
    __atomic_store_n(&b2t_probe_on, 1, __ATOMIC_RELAXED);
    return __atomic_load_n(&b2t_probe_copies, __ATOMIC_RELAXED);
}

void vk_exec_buffer_to_texture_commit(VkExec *ex, VkExecBufferToTexture *copy) {
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    vk_gpu_gap_note(VK_GAP_COPY_B2T, (uint64_t)copy->region.imageExtent.width * copy->region.imageExtent.height);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (vk_exec_recording_failed() || vk_sync_stopping() || vk_device_is_lost()) return;
    if (!buffer_to_texture_ready(ex, copy) || !vk_texture_write_target_current(copy->target)) {
        vk_exec_required_graphics_unavailable(ex, OP_COPY_B2T, "retained buffer-to-texture source or target changed");
        return;
    }
    ex->rt_resolved = 0;
    const VkTexImage *image = vk_texture_write_target_begin(copy->target, ex->cb);
    buffer_to_texture_issue(ex, copy, image->image);
    vk_texture_write_target_written(copy->target);
    copy->target = NULL;
    if (__atomic_load_n(&b2t_probe_on, __ATOMIC_RELAXED))
        __atomic_fetch_add(&b2t_probe_copies, 1, __ATOMIC_RELAXED);
}

void vk_exec_buffer_to_texture_cancel(VkExecBufferToTexture *copy) {
    vk_texture_write_target_cancel(copy->target);
    vk_mem_span_capture_release(&copy->source_hold);
    vk_texture_alias_read_cancel(copy->source_alias);
    *copy = (VkExecBufferToTexture){0};
}

int vk_exec_transfer_image_input_capture(uint32_t opcode, const void *payload, VkTextureTransferInput *input,
                                         VkTextureAliasFailure *failure) {
    const p_copy_t2t *t2t = opcode == OP_COPY_T2T ? payload : NULL;
    const p_copy_b2t *t2b = t2t ? NULL : payload;
    const tex_t *texture = t2t ? t2t->src : t2b->tex;
    const view_t *view = t2t ? &t2t->sv : &t2b->view;
    const NVNcopyRegion *region = t2t ? &t2t->sr : &t2b->r;
    const int has_view = t2t ? t2t->has_sv : t2b->has_view;
    VkTexImage shape;
    Sub source = {.img = &shape};
    if (!texture || !nvn_gpu_only((uintptr_t)texture->mem) || !sub_format(texture, &source)) return 0;
    if (!vk_texture_transfer_source_geometry(texture, &shape)) return 0;
    if (!sub_region(texture, view, has_view, region, 1, &source)) return 0;
    NVNcopyRegion box = {source.x, source.y, source.z, (int)source.w, (int)source.h, (int)source.nz};
    if (t2t && (t2t->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) && source.aspect == VK_IMAGE_ASPECT_COLOR_BIT) {
        VkTexImage destination;
        Sub target = {.img = &destination};
        const int equal = vk_texture_transfer_write_geometry(t2t->dst, &destination) &&
            sub_region(t2t->dst, &t2t->dv, t2t->has_dv, &t2t->dr, 1, &target) &&
            source.w == target.w && source.h == target.h && shape.format == destination.format;
        if (!equal) {
            box.xoffset = box.yoffset = 0;
            box.width = (int)max1(shape.width >> source.level);
            box.height = (int)max1(shape.height >> source.level);
        }
    }
    return vk_texture_transfer_input_capture(texture, &shape, sub_layers(&source), &box, input, failure);
}


static int restore_copy_capture_current(const p_copy_b2t *op, const VkExecRestoreCopy *copy) {
    uint64_t upload;
    if (copy->input.alias && vk_texture_alias_read_raw_progress(copy->input.alias, vk_exec_resource_position(), &upload) &&
        upload)
        return copy->destination &&
            vk_mem_canonical_source_qualify(copy->destination, op->src, copy->size) == RC_REGISTRY_OK;
    return vk_exec_restore_copy_current(op, copy);
}

int vk_exec_restore_copy_capture(VkExec *ex, const p_copy_b2t *op, VkExecRestoreCopy *copy,
                                 VkTextureAliasFailure *failure) {
    VkTexImage shape;
    Sub source = {.img = &shape};
    if (!sub_format(op->tex, &source)) return 0;
    const int retained = vk_exec_transfer_image_input_capture(OP_COPY_T2B, op, &copy->input, failure);
    if (retained < 0) return retained;
    if (retained) {
        shape = copy->input.image;
        if (!copy->input.alias) copy->image = &copy->input.image;
    } else copy->texture = vk_texture_restore_capture(op->tex, &shape);
    if (!retained && !copy->texture) {
        int captured = vk_texture_alias_read_capture(op->tex, &copy->alias, &shape, failure);
        if (captured <= 0) return captured;
    }
    VkBufferImageCopy layout;
    BufferImagePitch pitch;
    VkMemSpan destination;
    if (shape.scale <= 1 && sub_region(op->tex, &op->view, op->has_view, &op->r, 1, &source) &&
        buffer_side(ex, &source, op->src, &pitch, &layout, &destination, 1) && destination.pool) {
        copy->size = pitch.size;
        if (mem_backing_capture(op->src, copy->size, &copy->destination) == RC_BACKING_OK &&
            restore_copy_capture_current(op, copy)) return 1;
    }
    int required_alias = copy->alias != NULL || retained;
    if (required_alias && failure) {
        uint32_t count;
        const RcInterval *mapped = vk_texture_alias_read_intervals(copy->alias ? copy->alias : copy->input.alias, &count);
        *failure = (VkTextureAliasFailure){.phase = VK_TEXTURE_ALIAS_MAPPING, .status = RC_REGISTRY_STALE, .receiver =
                                          count ? mapped[0] : (RcInterval){0}};
    }
    vk_exec_restore_copy_cancel(copy);
    return required_alias ? -1 : 0;
}

struct VkExecTextureBufferWrite {
    BufferImagePitch pitch;
    VkBufferImageCopy region;
    VkMemSpan span;
    VkMemSpanCapture *hold;
    VkTexturePresentSource *resident;
    VkImage image;
    RcBudgetTicket memory, written_memory;
    RcRegistryBufferLocation location;
    RcInterval *mapped, *written;
    uint32_t mapped_count, written_count, raw_count;
    uint64_t address, fence;
    int immediate, mirror, gpu_only;
    RcRegistryWrite raw[];
};

// The selected mapping covers the entire footprint. Partition each written run at real
// mapping boundaries. Packed rows already have exactly the same coverage as the mapping.
static uint32_t texture_buffer_written_intervals(const BufferImagePitch *pitch, uint64_t address,
    RcInterval *mapped, uint32_t mapped_count, RcBudgetTicket *memory, RcInterval **out) {
    *out = NULL;
    if (!mapped_count) return 0;
    uint64_t runs = buffer_image_runs(pitch);
    if (runs == 1) {
        *out = mapped;
        return mapped_count;
    }
    uint64_t capacity = runs + mapped_count - 1;
    if (capacity > UINT32_MAX || capacity > SIZE_MAX / sizeof(RcInterval)) return 0;
    void *allocation;
    if (vk_replay_metadata_allocate((size_t)capacity * sizeof(RcInterval), 0, RC_BUDGET_BUFFER, memory,
                                    &allocation) != VK_SUCCESS) return 0;
    RcInterval *written = allocation;
    uint32_t i = 0, count = 0;
    for (uint64_t run = 0; run < runs; ++run) {
        uint64_t at = address + run * pitch->stride, end = at + pitch->row_bytes;
        while (at < end) {
            while (i < mapped_count && mapped[i].address + mapped[i].range.size <= at) ++i;
            if (i == mapped_count || mapped[i].address > at || count == capacity) {
                vk_replay_metadata_free(written, memory);
                return 0;
            }
            const RcInterval *m = &mapped[i];
            uint64_t next = m->address + m->range.size;
            if (next > end) next = end;
            written[count++] = (RcInterval){at, m->mapping,
                {m->range.backing, m->range.offset + (at - m->address), next - at}};
            at = next;
        }
    }
    *out = written;
    return count;
}

static void texture_buffer_write_release(VkExecTextureBufferWrite *write) {
    if (!write) return;
    vk_texture_present_release(write->resident);
    vk_mem_span_capture_release(&write->hold);
    if (write->written_memory.token.ledger)
        vk_replay_metadata_free(write->written, &write->written_memory);
    free(write->mapped);
    RcBudgetTicket memory = write->memory;
    vk_replay_metadata_free(write, &memory);
}

static void texture_buffer_writeback(const VkExecTextureBufferWrite *write) {
    uint64_t runs = buffer_image_runs(&write->pitch), bytes = buffer_image_run_bytes(&write->pitch);
    VkMemSpan row = write->span;
    for (uint64_t i = 0; i < runs; ++i) {
        row.offset = write->span.offset + i * write->pitch.stride;
        vk_mem_gpu_wrote_exact(&row, bytes);
    }
}

static void texture_buffer_write_done(void *arg) {
    VkExecTextureBufferWrite *write = arg;
    RcRegistry *registry = rc_registry_runtime();
    if (write->raw_count) {
        if (rc_registry_write_submit_extents(registry, write->raw, write->raw_count, write->fence) == RC_REGISTRY_OK) {
            for (uint32_t i = 0; i < write->raw_count; ++i)
                if (rc_registry_write_complete(registry, write->raw[i], write->immediate) != RC_REGISTRY_OK)
                    rc_registry_fail_closed(registry);
        } else if (rc_registry_write_abandon_extents(registry, write->raw, write->raw_count) != RC_REGISTRY_OK)
            rc_registry_fail_closed(registry);
    }
    if (!write->gpu_only) {
        uint64_t runs = buffer_image_runs(&write->pitch), bytes = buffer_image_run_bytes(&write->pitch);
        for (uint64_t i = 0; i < runs; ++i)
            vk_mem_buffer_write_completed(write->span.pool, write->span.buffer,
                write->address + i * write->pitch.stride, bytes, write->mirror);
    }
    texture_buffer_write_release(write);
}

static void texture_buffer_write_cancel(void *arg, int may_have_executed) {
    VkExecTextureBufferWrite *write = arg;
    if (!write) return;
    RcRegistry *registry = rc_registry_runtime();
    if (write->raw_count) {
        RcRegistryResult result = may_have_executed
            ? rc_registry_write_abandon_extents(registry, write->raw, write->raw_count)
            : rc_registry_write_cancel_extents(registry, write->raw, write->raw_count);
        if (result != RC_REGISTRY_OK) rc_registry_fail_closed(registry);
    }
    texture_buffer_write_release(write);
}

int vk_exec_restore_copy_current(const p_copy_b2t *op, const VkExecRestoreCopy *copy) {
    return (!copy->destination ||
            vk_mem_canonical_source_qualify(copy->destination, op->src, copy->size) == RC_REGISTRY_OK) &&
        vk_texture_transfer_input_current(&copy->input) &&
        (!copy->alias || vk_texture_alias_read_current(copy->alias)) &&
        (!copy->output || vk_mem_span_capture_current(copy->output->hold, &copy->output->span));
}

static int texture_buffer_hold_output(VkExecTextureBufferWrite *write) {
    const VkMemSpan *span = &write->span;
    RcRegistryResult result = vk_mem_span_capture_bound_input(write->address, write->pitch.size, span, &write->hold);
    if (result == RC_REGISTRY_UNSUPPORTED && span->pool && vk_mirror_on) {
        // A cold pool can keep writing to the existing import. Resolve this scene's import policy before
        // declaring the output, and qualify the lease against the exact buffer and offset already selected.
        VkMemSpan selected;
        result = vk_mem_span_prepare_native_input(write->address, write->pitch.size, 0, VK_MEM_MIRROR_IMPORT,
                                                  &selected, &write->hold);
        if (result == RC_REGISTRY_OK &&
            (selected.buffer != span->buffer || selected.offset != span->offset || selected.pool != span->pool ||
             selected.alloc != span->alloc || selected.alloc_offset != span->alloc_offset)) return 0;
    }
    return result == RC_REGISTRY_OK && vk_mem_span_record_input(write->hold, span) &&
        vk_mem_span_buffer_location(write->hold, span, &write->location);
}

int vk_exec_restore_copy_prepare(VkExec *ex, const p_copy_b2t *op, VkExecRestoreCopy *copy) {
    Sub source = {.img = copy->image};
    VkTexturePresentSource *resident = NULL;
    if (copy->image) {
        if (!sub_format(op->tex, &source) || !sub_region(op->tex, &op->view, op->has_view, &op->r, 1, &source))
            return -1;
    } else {
        if (!sub_resolve(op->tex, &op->view, op->has_view, &op->r, 1, 0, &source))
            return vk_texture_deferred_take() ? -1 : 0;
        if (source.img->scale > 1) {
            warn_once(W_REGION, "buffer copy of a shadow map enlarged by NX2NAVI_SHADOW_SCALE: skipped");
            return 0;
        }
        resident = vk_texture_present_retain_image(source.img, source.img->image);
        if (!resident) return -1;
    }

    VkImage image = source.img->image;
    BufferImagePitch pitch;
    VkBufferImageCopy region;
    VkMemSpan span;
    if (!buffer_side(ex, &source, op->src, &pitch, &region, &span, 0)) {
        vk_texture_present_release(resident);
        return copy->image ? -1 : 0;
    }
    if (copy->image && (copy->size != pitch.size || !vk_exec_restore_copy_current(op, copy))) return -1;
    int gpu_only = nvn_gpu_only(op->src);
    uint64_t runs = gpu_only ? 0 : buffer_image_runs(&pitch);
    if (runs > UINT32_MAX || runs > (SIZE_MAX - sizeof(VkExecTextureBufferWrite)) / sizeof(RcRegistryWrite)) {
        vk_texture_present_release(resident);
        return -1;
    }
    RcBudgetTicket memory = {0};
    void *allocation;
    if (vk_replay_metadata_allocate(sizeof(VkExecTextureBufferWrite) + (size_t)runs * sizeof(RcRegistryWrite),
                                    0, RC_BUDGET_BUFFER, &memory, &allocation) != VK_SUCCESS) {
        vk_texture_present_release(resident);
        return -1;
    }
    VkExecTextureBufferWrite *out = allocation;
    *out = (VkExecTextureBufferWrite){.pitch = pitch, .region = region, .span = span, .resident = resident,
        .image = image, .memory = memory, .address = op->src, .gpu_only = gpu_only,
        .mirror = vk_mem_recorded_write_mirrored(&span)};
    copy->output = out;
    if (!texture_buffer_hold_output(out)) return -1;
    out->immediate = vk_mem_span_input_cpu_visible(out->hold);
    out->mapped_count = vk_mem_span_input_intervals(out->hold, op->src, pitch.size, &out->mapped);
    out->written_count = texture_buffer_written_intervals(&pitch, op->src, out->mapped, out->mapped_count,
                                                          &out->written_memory, &out->written);
    return out->written_count ? 1 : -1;
}

RcRegistryResult vk_exec_restore_copy_declare(const p_copy_b2t *op, VkExecRestoreCopy *copy,
                                              ResourceCommandEffects *effects, RcRegistryPosition position) {
    VkExecTextureBufferWrite *out = copy->output;
    if (!out || !out->written_count || out->raw_count || !vk_exec_restore_copy_current(op, copy))
        return RC_REGISTRY_STALE;
    RcRegistryResult result = resource_effects_add_intervals(effects, out->written, out->written_count,
        out->mapped, out->mapped_count, RC_REGISTRY_BUFFER);
    if (result != RC_REGISTRY_OK || out->gpu_only) return result;

    uint32_t runs = (uint32_t)buffer_image_runs(&out->pitch);
    uint64_t bytes = buffer_image_run_bytes(&out->pitch);
    RcRegistryWriteExtent local[8], *extents = local;
    RcBudgetTicket memory = {0};
    if (runs > sizeof local / sizeof *local) {
        void *allocation;
        if ((uint64_t)runs * sizeof *extents > SIZE_MAX ||
            vk_replay_metadata_allocate((size_t)runs * sizeof *extents, 0, RC_BUDGET_BUFFER, &memory,
                                        &allocation) != VK_SUCCESS) return RC_REGISTRY_CAPACITY;
        extents = allocation;
    }
    for (uint32_t i = 0; i < runs; ++i)
        extents[i] = (RcRegistryWriteExtent){{RC_GUEST_COMPATIBILITY_BACKING,
            out->address + i * out->pitch.stride, bytes}, RC_REGISTRY_BUFFER};
    result = rc_registry_write_begin_extents(rc_registry_runtime(), extents, runs, position, out->raw);
    if (extents != local) vk_replay_metadata_free(extents, &memory);
    if (result == RC_REGISTRY_OK) out->raw_count = runs;
    return result;
}

void vk_exec_restore_copy_commit(VkExec *ex, const p_copy_b2t *op, VkExecRestoreCopy *copy) {
    VkExecTextureBufferWrite *write = copy->output;
    if (!write) return;
    if (vk_exec_recording_failed()) return;
    vk_contact_flush(ex);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (vk_exec_recording_failed()) return;
    RcRegistryBufferLocation location;
    if (!vk_exec_restore_copy_current(op, copy) ||
        !vk_mem_span_buffer_location(write->hold, &write->span, &location) ||
        location.allocation != write->location.allocation || location.incarnation != write->location.incarnation ||
        location.offset != write->location.offset ||
        !resource_effects_buffer_prepare_intervals(write->mapped, write->mapped_count, write->immediate)) {
        vk_exec_required_graphics_unavailable(ex, OP_COPY_T2B, "retained texture copy destination changed");
        return;
    }
    resource_effects_buffer_prepare_native(write->mapped, write->mapped_count, write->address, location);
    vk_gpu_gap_note(VK_GAP_COPY_T2B, (uint64_t)write->region.imageExtent.width * write->region.imageExtent.height);
    if (!vk_texture_transfer_input_use(&copy->input)) {
        vk_exec_required_graphics_unavailable(ex, OP_COPY_T2B, "retained texture copy source lost its written pixels");
        return;
    }
    if (copy->texture) vk_texture_restore_use(copy->texture);
    if (write->resident) vk_texture_transfer_read_use(write->resident, write->image);
    vkCmdCopyImageToBuffer(ex->cb, write->image, VK_IMAGE_LAYOUT_GENERAL, write->span.buffer, 1, &write->region);
    ex->recorded++;
    resource_effects_buffers_commit();

    // Bounding coverage is only a page/shadow hazard. Bytes and writeback follow the emitted runs.
    vk_usnap_drop(write->address, write->pitch.size);
    if (!write->gpu_only) vk_mem_gpu_pending_span(&write->span, write->address, write->pitch.size);
    texture_buffer_writeback(write);
    vk_mem_count(write->span.cls, VK_MUSE_GPU_WRITE, write->pitch.rows * write->pitch.row_bytes);
    write->fence = ex->fence;
    copy->output = NULL;
    vk_completion_add_owned(ex->fence, texture_buffer_write_done, texture_buffer_write_cancel, write);
}

void vk_exec_restore_copy_cancel(VkExecRestoreCopy *copy) {
    texture_buffer_write_cancel(copy->output, 0);
    vk_texture_restore_cancel(copy->texture);
    vk_texture_transfer_input_cancel(&copy->input);
    vk_texture_alias_read_cancel(copy->alias);
    mem_backing_release(&copy->destination);
    *copy = (VkExecRestoreCopy){0};
}

static int image_copy_same_layers(VkImageSubresourceLayers src, VkImageSubresourceLayers dst) {
    return src.aspectMask == dst.aspectMask && src.mipLevel == dst.mipLevel &&
        (uint64_t)src.baseArrayLayer + src.layerCount > dst.baseArrayLayer &&
        (uint64_t)dst.baseArrayLayer + dst.layerCount > src.baseArrayLayer;
}

static int image_copy_same_box(const Sub *src, const Sub *dst, int halo) {
    return (int64_t)src->x + src->w + halo > dst->x && (int64_t)dst->x + dst->w + halo > src->x &&
        (int64_t)src->y + src->h + halo > dst->y && (int64_t)dst->y + dst->h + halo > src->y &&
        (src->img->type != VK_IMAGE_TYPE_3D ||
        ((int64_t)src->z + src->nz + halo > dst->z && (int64_t)dst->z + dst->nz + halo > src->z));
}

static int image_copy_overlap(const Sub *src, const Sub *dst, int linear) {
    return src->img->image && src->img->image == dst->img->image &&
        image_copy_same_layers(sub_layers(src), sub_layers(dst)) && image_copy_same_box(src, dst, linear);
}

static int image_copy_shape(const VkTexImage *shape, const Sub *sub, const tex_t *tex, const view_t *view, int has_view,
                            VkImageAspectFlags aspect) {
    return shape->type == VK_IMAGE_TYPE_2D && shape->aspect == aspect && shape->scale <= 1 &&
        sub->bw == 1 && sub->bh == 1 && (!has_view || !view->set_format || view->format == tex->d.format);
}

static int image_copy_blit_supported(VkFormat format, VkFilter filter) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, format, &props);
    VkFormatFeatureFlags features = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    if (filter == VK_FILTER_LINEAR) features |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (props.optimalTilingFeatures & features) == features;
}

static int image_copy_geometry(const p_copy_t2t *op, Sub *src, Sub *dst, const VkTexImage *shape,
                               VkExecTextureToTexture *copy, int *status) {
    *status = 1;
    if (!sub_region(op->src, &op->sv, op->has_sv, &op->sr, 1, src)) return 0;
    if (!sub_region(op->dst, &op->dv, op->has_dv, &op->dr, 1, dst)) return 0;
    if (dst->nz != src->nz) {
        NVNcopyRegion destination = op->dr;
        destination.depth = (int)src->nz;
        if (!sub_region(op->dst, &op->dv, op->has_dv, &destination, 1, dst)) return 0;
    }
    if (src->aspect != dst->aspect) return 0;
    const int depth = src->aspect == VK_IMAGE_ASPECT_DEPTH_BIT;
    if (depth && (copy->source.format != VK_FORMAT_D32_SFLOAT || shape->format != VK_FORMAT_D32_SFLOAT ||
                  copy->source.samples != 1 || shape->samples != 1)) return 0;
    int same_size = src->w == dst->w && src->h == dst->h;
    copy->op = VK_EXEC_IMAGE_COPY;
    if (copy->source.samples != shape->samples) {
        if (copy->source.samples <= 1 || shape->samples != 1 || !same_size) return 0;
        copy->op = VK_EXEC_IMAGE_RESOLVE;
    } else if (!same_size) {
        copy->op = VK_EXEC_IMAGE_BLIT;
        copy->filter = !depth && (op->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        if (shape->samples != 1 || !image_copy_blit_supported(shape->format, copy->filter)) {
            *status = 0;
            return 0;
        }
    }
    copy->source_layers = sub_layers(src);
    copy->destination_layers = sub_layers(dst);
    copy->source_offset = sub_offset(src);
    copy->destination_offset = sub_offset(dst);
    copy->source_extent = sub_extent(src);
    copy->destination_extent = sub_extent(dst);
    return 1;
}

static void image_copy_prepare_overlap(VkExecTextureToTexture *copy, const Sub *src, const Sub *dst,
                                       const VkTexImage *shape) {
    // A private target writes the same native image. An immutable target instead receives a distinct COW image.
    int blit = copy->op == VK_EXEC_IMAGE_BLIT;
    if (image_copy_overlap(src, dst, blit && copy->filter == VK_FILTER_LINEAR)) {
        VkExtent3D extent = copy->source_extent;
        copy->scratch_offset = copy->source_offset;
        if (blit) {
            // Preserve the texels a filter may sample outside its rectangle at the selected mip's edges.
            extent = (VkExtent3D){max1(src->img->width >> src->level), max1(src->img->height >> src->level), 1};
            copy->scratch_offset = (VkOffset3D){0};
        }
        copy->scratch_recipe = (VkImageCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D, .format = shape->format, .extent = extent, .mipLevels = 1,
            .arrayLayers = copy->source_layers.layerCount, .samples = shape->samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    }
}

int vk_exec_texture_to_texture_capture(VkExec *ex, const p_copy_t2t *op, VkExecTextureToTexture *copy) {
    if (ex->tonemap_extent.width || ex->bc_written_mask || vk_contact_pending()) return 0;
    VkTexImage shape = {0};
    Sub src = {.img = &copy->source}, dst = {.img = &shape};
    if (!sub_format(op->src, &src) || !sub_format(op->dst, &dst)) return 1;
    int target_resident = vk_texture_write_target_shape(op->dst, &shape);
    if ((!target_resident && !vk_texture_write_target_restore_shape(op->dst, &shape)) ||
        !image_copy_shape(&shape, &dst, op->dst, &op->dv, op->has_dv, VK_IMAGE_ASPECT_COLOR_BIT) ||
        !(shape.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return 0;
    if (copy->input) copy->source = copy->input->image;
    else {
        copy->source_hold = vk_texture_transfer_read_capture(op->src, &copy->source);
        if (!copy->source_hold) copy->restore = vk_texture_restore_capture(op->src, &copy->source);
        if (!copy->source_hold && !copy->restore) return 0;
    }
    int status = 0;
    if (!target_resident && !copy->restore) goto discard;
    if (!image_copy_shape(&copy->source, &src, op->src, &op->sv, op->has_sv, VK_IMAGE_ASPECT_COLOR_BIT) ||
        !(copy->source.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || copy->source.format != shape.format) goto discard;
    if (!image_copy_geometry(op, &src, &dst, &shape, copy, &status)) goto discard;
    if (shape.gpu_owned) image_copy_prepare_overlap(copy, &src, &dst, &shape);
    status = vk_texture_write_target_capture(op->dst, &shape, &copy->target);
    if (status > 0) {
        if (vk_texture_write_target_needs_allocation(copy->target)) {
            if (copy->scratch_recipe.sType) rt_fatal("vk: exact image transfer changed its private destination placement");
            if (copy->restore) vk_texture_restore_pair_write_target(copy->restore, copy->target);
        }
        return status;
    }
discard:
    vk_exec_texture_to_texture_cancel(copy);
    return status;
}

VkResult vk_exec_texture_to_texture_allocate(VkExecTextureToTexture *copy, RcVmaDependency *dependency) {
    RcVmaRequest request = {.producer = RC_BUDGET_TEXTURE, .required = 1, .dependency = dependency};
    VmaAllocationCreateInfo settings = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    return resource_core_vma_create_image(&request, vk_ctx()->vma, &copy->scratch_recipe, &settings,
                                          &copy->scratch, &copy->scratch_memory, NULL, NULL);
}

static void texture_copy_read_done(void *opaque) {
    vk_texture_present_release(opaque);
}

static void texture_copy_read_cancel(void *opaque, int may_have_executed) {
    (void)may_have_executed;
    texture_copy_read_done(opaque);
}

static VkImage texture_copy_stage_source(VkExec *ex, VkExecTextureToTexture *copy) {
    if (!copy->scratch) return copy->source.image;
    const VkImageAspectFlags aspect = copy->source_layers.aspectMask;
    VkImageMemoryBarrier2 transition = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = copy->scratch, .subresourceRange = {aspect, 0, 1, 0, copy->source_layers.layerCount}};
    VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                                     .pImageMemoryBarriers = &transition};
    vkCmdPipelineBarrier2(ex->cb, &dependency);
    VkImageSubresourceLayers scratch_layers = {aspect, 0, 0, copy->source_layers.layerCount};
    VkImageCopy command = {copy->source_layers, copy->scratch_offset, scratch_layers, {0}, copy->scratch_recipe.extent};
    vkCmdCopyImage(ex->cb, copy->source.image, VK_IMAGE_LAYOUT_GENERAL, copy->scratch, VK_IMAGE_LAYOUT_GENERAL, 1, &command);
    ex->recorded++;
    vk_exec_barrier(ex);
    copy->source_layers = scratch_layers;
    copy->source_offset.x -= copy->scratch_offset.x;
    copy->source_offset.y -= copy->scratch_offset.y;
    return copy->scratch;
}

static void image_copy_record(VkExec *ex, const VkExecTextureToTexture *copy, VkImage source, VkImage destination) {
    VkImageSubresourceLayers sl = copy->source_layers, dl = copy->destination_layers;
    VkOffset3D so = copy->source_offset, d = copy->destination_offset;
    VkExtent3D se = copy->source_extent, de = copy->destination_extent;
    switch (copy->op) {
    case VK_EXEC_IMAGE_COPY: {
        VkImageCopy command = {sl, so, dl, d, se};
        vkCmdCopyImage(ex->cb, source, VK_IMAGE_LAYOUT_GENERAL, destination, VK_IMAGE_LAYOUT_GENERAL, 1, &command);
        break;
    }
    case VK_EXEC_IMAGE_RESOLVE: {
        VkImageResolve command = {sl, so, dl, d, se};
        vkCmdResolveImage(ex->cb, source, VK_IMAGE_LAYOUT_GENERAL, destination, VK_IMAGE_LAYOUT_GENERAL, 1, &command);
        break;
    }
    case VK_EXEC_IMAGE_BLIT: {
        VkImageBlit command = {sl, {so, {so.x + (int32_t)se.width, so.y + (int32_t)se.height, so.z + (int32_t)se.depth}},
                               dl, {d, {d.x + (int32_t)de.width, d.y + (int32_t)de.height, d.z + (int32_t)de.depth}}};
        vkCmdBlitImage(ex->cb, source, VK_IMAGE_LAYOUT_GENERAL, destination, VK_IMAGE_LAYOUT_GENERAL, 1, &command, copy->filter);
        break;
    }
    }
    ex->recorded++;
}

static int t2t_probe_on;
static uint64_t t2t_probe_copies, t2t_probe_staged;
uint64_t vk_exec_texture_to_texture_probe_count(uint64_t *staged) {
    __atomic_store_n(&t2t_probe_on, 1, __ATOMIC_RELAXED);
    *staged = __atomic_load_n(&t2t_probe_staged, __ATOMIC_RELAXED);
    return __atomic_load_n(&t2t_probe_copies, __ATOMIC_RELAXED);
}

void vk_exec_texture_to_texture_commit(VkExec *ex, VkExecTextureToTexture *copy) {
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    vk_gpu_gap_note(VK_GAP_COPY_T2T, (uint64_t)copy->source_extent.width * copy->source_extent.height);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    ex->rt_resolved = 0;
    if (copy->input) {
        if (!vk_texture_transfer_input_use(copy->input)) {
            vk_exec_required_graphics_unavailable(ex, OP_COPY_T2T, "retained texture copy source changed before issue");
            vk_exec_texture_to_texture_cancel(copy);
            return;
        }
    } else if (copy->source_hold) {
        if (vk_texture_transfer_read_use(copy->source_hold, copy->source.image))
            vk_texture_present_release(copy->source_hold);
        else vk_completion_add_owned(ex->fence, texture_copy_read_done, texture_copy_read_cancel, copy->source_hold);
        copy->source_hold = NULL;
    } else vk_texture_restore_use(copy->restore);
    VkImage source = texture_copy_stage_source(ex, copy);
    const VkTexImage *image = vk_texture_write_target_begin(copy->target, ex->cb);
    vk_dlaa_hud_write(image->image);
    image_copy_record(ex, copy, source, image->image);
    vk_texture_write_target_written(copy->target);
    copy->target = NULL;
    if (__atomic_load_n(&t2t_probe_on, __ATOMIC_RELAXED)) {
        __atomic_fetch_add(&t2t_probe_copies, 1, __ATOMIC_RELAXED);
        if (copy->scratch) __atomic_fetch_add(&t2t_probe_staged, 1, __ATOMIC_RELAXED);
    }
    if (copy->scratch) {
        vk_defer_image(copy->scratch, copy->scratch_memory);
        copy->scratch = VK_NULL_HANDLE;
    }
    vk_exec_texture_to_texture_cancel(copy);
}

void vk_exec_texture_to_texture_cancel(VkExecTextureToTexture *copy) {
    vk_texture_present_release(copy->source_hold);
    vk_texture_restore_cancel(copy->restore);
    vk_texture_write_target_cancel(copy->target);
    if (copy->scratch) resource_core_vma_destroy_image(vk_ctx()->vma, copy->scratch, copy->scratch_memory);
    *copy = (VkExecTextureToTexture){0};
}

int vk_exec_texture_to_texture_alias_candidate(const p_copy_t2t *p) {
    Sub source;
    return vk_texture_alias_region_transfer(p->dst, p->has_dv ? &p->dv : NULL, &p->dr) &&
        sub_format(p->src, &source) && p->sr.depth == 1;
}

void vk_exec_texture_to_texture_compat(VkExec *ex, const p_copy_t2t *p, const VkTextureTransferInput *input) {
    Sub s = {0}, d;
    if (input && input->texture) {
        s.img = &input->image;
        if (!sub_format(p->src, &s) || !sub_region(p->src, &p->sv, p->has_sv, &p->sr, 1, &s)) {
            vk_exec_required_graphics_unavailable(ex, OP_COPY_T2T, "retained texture copy source geometry changed");
            return;
        }
    } else if (!sub_resolve(p->src, &p->sv, p->has_sv, &p->sr, 1, 0, &s)) return;
    const int projected = vk_exec_texture_to_texture_alias_candidate(p);
    if (!sub_resolve(p->dst, &p->dv, p->has_dv, &p->dr, 1, projected ? 2 : 1, &d)) return;
    if (d.nz != s.nz) {
        NVNcopyRegion destination = p->dr;
        destination.depth = (int)s.nz;
        if (!sub_region(p->dst, &p->dv, p->has_dv, &destination, 1, &d)) return;
    }
    vk_gpu_gap_note(VK_GAP_COPY_T2T, (uint64_t)s.w * s.h);
    int same_size = s.w == d.w && s.h == d.h;
    int same_format = s.img->format == d.img->format;
    if (s.img->samples != d.img->samples && !(s.img->samples > 1 && d.img->samples == 1 && same_size && same_format)) {
        warn_once(W_COPY, "texture copy between %u and %u samples: skipped", s.img->samples, d.img->samples);
        return;
    }
    if (s.aspect != d.aspect || (s.aspect != VK_IMAGE_ASPECT_COLOR_BIT && !same_format)) {
        warn_once(W_COPY, "texture copy between formats %#x and %#x: skipped", p->src->d.format, p->dst->d.format);
        return;
    }
    VkExecImageTransfer op = s.img->samples > 1 && d.img->samples == 1 ? VK_EXEC_IMAGE_RESOLVE :
        same_size && same_format ? VK_EXEC_IMAGE_COPY : VK_EXEC_IMAGE_BLIT;
    int linear = (p->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) && s.aspect == VK_IMAGE_ASPECT_COLOR_BIT;
    if (image_copy_overlap(&s, &d, op == VK_EXEC_IMAGE_BLIT && linear))
        rt_fatal("vk: overlapping texture copy needs an exact scratch image for its selected representation");
    VkExecTextureToTexture copy = {.op = op, .source_layers = sub_layers(&s), .destination_layers = sub_layers(&d),
        .source_offset = sub_offset(&s), .destination_offset = sub_offset(&d),
        .source_extent = sub_extent(&s), .destination_extent = sub_extent(&d),
        .filter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST};
    vk_dlaa_hud_write(d.img->image);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    if (input && input->texture) {
        if (!vk_texture_transfer_input_use(input)) {
            vk_exec_required_graphics_unavailable(ex, OP_COPY_T2T, "compatible texture copy source changed");
            return;
        }
    } else vk_texture_explicit_read_use(p->src, s.img->image);
    image_copy_record(ex, &copy, s.img->image, d.img->image);
    resource_image_written((uint64_t)(uintptr_t)d.img->image, d.img->addr);
    vk_texture_gpu_written(p->dst);
    if (projected) {
        const NVNcopyRegion actual = {d.x, d.y, d.z, (int)d.w, (int)d.h, (int)d.nz};
        vk_texture_alias_transfer_written(p->dst, d.img->image, p->has_dv ? &p->dv : NULL, &actual);
    }
    static int lg = -1;
    if (lg < 0) lg = rt_knob("LOG_RCOND", "0");
    if (lg == 2 && (d.img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && d.w >= 1600)
        rt_log("SEQ depth copy %#llx -> %#llx", (unsigned long long)s.img->addr, (unsigned long long)d.img->addr);
}

// Projected pool transfers keep the exact Vulkan endpoints and the declared image placement from
// ordered selection through issue. The fallback still accepts shapes outside this projected lane.
static int projected_b2t_capture(VkExec *ex, const p_copy_b2t *op, VkExecProjectedTransfer *transfer) {
    const view_t *view = op->has_view ? &op->view : NULL;
    const int volume = vk_texture_alias_volume_transfer(op->tex, view) ||
        vk_texture_host_volume_transfer(op->tex, view);
    if (!volume && !vk_texture_alias_region_transfer(op->tex, view, &op->r)) return 0;
    VkExecBufferToTexture *copy = &transfer->buffer;
    VkTexImage shape = {.type = VK_IMAGE_TYPE_2D, .aspect = VK_IMAGE_ASPECT_COLOR_BIT};
    Sub dst = {.img = &shape, .aspect = shape.aspect, .x = op->r.xoffset, .y = op->r.yoffset,
               .w = (uint32_t)op->r.width, .h = (uint32_t)op->r.height, .nz = 1};
    BufferImagePitch pitch;
    if (!sub_format(op->tex, &dst) || (volume && (!vk_texture_transfer_write_geometry(op->tex, &shape) ||
        !sub_region(op->tex, &op->view, op->has_view, &op->r, 1, &dst))) ||
        !buffer_layout(ex, &dst, &pitch, &copy->region)) return 1;
    int source = buffer_to_texture_capture_source(copy, op->src, &pitch, &dst);
    if (source <= 0 || !copy->source_hold) return source;
    const NVNcopyRegion actual = {dst.x, dst.y, dst.z, (int32_t)dst.w, (int32_t)dst.h, (int32_t)dst.nz};
    return vk_texture_projected_transfer_capture(op->tex, view, &actual, &shape, &transfer->target);
}

static VkImageAspectFlags projected_t2t_aspect(const p_copy_t2t *op) {
    if (op->src && vk_texture_host_color_transfer(op->dst, op->has_dv ? &op->dv : NULL) &&
        !op->dr.xoffset && !op->dr.yoffset && !op->dr.zoffset && op->dr.depth <= 1 &&
        (op->dr.width <= 0 || (uint32_t)op->dr.width == op->dst->d.width) &&
        (op->dr.height <= 0 || (uint32_t)op->dr.height == op->dst->d.height))
        return VK_IMAGE_ASPECT_COLOR_BIT;
    if (vk_exec_texture_to_texture_alias_candidate(op)) return VK_IMAGE_ASPECT_COLOR_BIT;
    if (!op->src || !op->dst || op->src->d.format != NVN_FORMAT_DEPTH32F || op->src->d.samples > 1 ||
        op->sr.depth > 1 || (op->has_sv && !vk_texture_attachment_base_view(op->src, &op->sv))) return 0;
    NVNcopyRegion box = op->dr;
    // Match sub_region's default to the remaining mip extent.
    if (box.xoffset < 0 || box.yoffset < 0 || op->dst->d.width > INT32_MAX || op->dst->d.height > INT32_MAX) return 0;
    if (box.width <= 0) box.width = (int32_t)op->dst->d.width - box.xoffset;
    if (box.height <= 0) box.height = (int32_t)op->dst->d.height - box.yoffset;
    if (box.depth <= 0) box.depth = 1;
    return vk_texture_alias_depth_transfer(op->dst, op->has_dv ? &op->dv : NULL, &box) ? VK_IMAGE_ASPECT_DEPTH_BIT : 0;
}

static int projected_t2t_capture(VkExec *ex, const p_copy_t2t *op, VkExecProjectedTransfer *transfer) {
    (void)ex;
    const VkImageAspectFlags aspect = projected_t2t_aspect(op);
    if (!aspect) return 0;
    VkExecTextureToTexture *copy = &transfer->image;
    VkTexImage shape = {0};
    Sub src = {.img = &copy->source}, dst = {.img = &shape};
    if (!sub_format(op->src, &src) || !sub_format(op->dst, &dst)) return 1;
    if (copy->input) copy->source = copy->input->image;
    else {
        // A depth box can be current even when other parts of its image still need reconstruction.
        if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT)
            copy->source_hold = vk_texture_transfer_read_capture(op->src, &copy->source);
        if (!copy->source_hold) copy->source_hold = vk_texture_transfer_read_select(op->src, &copy->source);
        if (!copy->source_hold) return vk_texture_deferred_take() ? -1 : 0;
    }
    if (!image_copy_shape(&copy->source, &src, op->src, &op->sv, op->has_sv, aspect)) return 0;
    // Geometry and filtering are validated before taking a writable destination placement.
    TexVkFormat format;
    if (!tex_vk_format(op->dst->d.format, &format) || copy->source.format != format.vk) return 0;
    shape = (VkTexImage){.type = VK_IMAGE_TYPE_2D, .aspect = aspect, .levels = 1, .layers = 1,
                         .width = op->dst->d.width, .height = op->dst->d.height, .depth = 1,
                         .samples = 1, .format = format.vk};
    int handled;
    if (!image_copy_geometry(op, &src, &dst, &shape, copy, &handled)) return handled;
    NVNcopyRegion source_box = {src.x, src.y, src.z, (int)src.w, (int)src.h, (int)src.nz};
    if (copy->op == VK_EXEC_IMAGE_BLIT && copy->filter == VK_FILTER_LINEAR) {
        source_box.xoffset = source_box.yoffset = 0;
        source_box.width = (int)max1(copy->source.width >> src.level);
        source_box.height = (int)max1(copy->source.height >> src.level);
    }
    int proof = copy->input ? vk_texture_transfer_input_current(copy->input) :
        vk_texture_transfer_read_prove(op->src, &copy->source, copy->source_layers, &source_box);
    if (proof <= 0) return proof;
    NVNcopyRegion destination_box = {dst.x, dst.y, dst.z, (int)dst.w, (int)dst.h, (int)dst.nz};
    handled = vk_texture_projected_transfer_capture(op->dst, op->has_dv ? &op->dv : NULL, &destination_box,
                                                    &shape, &transfer->target);
    if (handled > 0) image_copy_prepare_overlap(copy, &src, &dst, &shape);
    return handled;
}

int vk_exec_projected_transfer_capture(VkExec *ex, uint32_t opcode, const void *payload,
                                      VkExecProjectedTransfer *transfer) {
    if (ex->tonemap_extent.width || ex->bc_written_mask || vk_contact_pending()) return 0;
    transfer->opcode = opcode;
    int handled = opcode == OP_COPY_B2T ? projected_b2t_capture(ex, payload, transfer) :
                                          projected_t2t_capture(ex, payload, transfer);
    if (handled <= 0 || !transfer->target) vk_exec_projected_transfer_cancel(transfer);
    return handled;
}

int vk_exec_projected_transfer_current(const VkExecProjectedTransfer *transfer) {
    return transfer->target && vk_texture_projected_transfer_current(transfer->target) &&
        (transfer->opcode == OP_COPY_B2T ? vk_exec_buffer_to_texture_source_current(&transfer->buffer) :
         transfer->image.input ? vk_texture_transfer_input_current(transfer->image.input) :
         vk_texture_transfer_read_current(transfer->image.source_hold, transfer->image.source.image));
}

void vk_exec_projected_transfer_cancel(VkExecProjectedTransfer *transfer) {
    vk_texture_projected_transfer_cancel(&transfer->target);
    if (transfer->opcode == OP_COPY_B2T) vk_exec_buffer_to_texture_cancel(&transfer->buffer);
    else if (transfer->opcode == OP_COPY_T2T) vk_exec_texture_to_texture_cancel(&transfer->image);
    *transfer = (VkExecProjectedTransfer){0};
}

static void projected_transfer_done(void *opaque) {
    VkTextureProjectedTransfer *transfer = opaque;
    vk_texture_projected_transfer_cancel(&transfer);
}

static void projected_transfer_canceled(void *opaque, int may_have_executed) {
    (void)may_have_executed;
    projected_transfer_done(opaque);
}

static void projected_b2t_record(VkExec *ex, VkExecProjectedTransfer *transfer, VkImage target) {
    VkExecBufferToTexture *copy = &transfer->buffer;
    if (!buffer_to_texture_ready(ex, copy) || !vk_texture_projected_transfer_begin(transfer->target)) {
        vk_exec_required_graphics_unavailable(ex, OP_COPY_B2T, "selected image transfer changed before issue");
        return;
    }
    vk_gpu_gap_note(VK_GAP_COPY_B2T, (uint64_t)copy->region.imageExtent.width * copy->region.imageExtent.height);
    buffer_to_texture_issue(ex, copy, target);
    vk_texture_projected_transfer_written(transfer->target);
    if (__atomic_load_n(&b2t_probe_on, __ATOMIC_RELAXED))
        __atomic_fetch_add(&b2t_probe_copies, 1, __ATOMIC_RELAXED);
    vk_completion_add_owned(ex->fence, projected_transfer_done, projected_transfer_canceled, transfer->target);
    transfer->target = NULL;
}

static void projected_t2t_record(VkExec *ex, VkExecProjectedTransfer *transfer, VkImage target) {
    VkExecTextureToTexture *copy = &transfer->image;
    if ((copy->input ? !vk_texture_transfer_input_use(copy->input) :
         !vk_texture_transfer_read_current(copy->source_hold, copy->source.image)) ||
        !vk_texture_projected_transfer_begin(transfer->target)) {
        vk_exec_required_graphics_unavailable(ex, OP_COPY_T2T, "selected image transfer changed before issue");
        return;
    }
    if (!copy->input) {
        if (vk_texture_transfer_read_use(copy->source_hold, copy->source.image))
            vk_texture_present_release(copy->source_hold);
        else vk_completion_add_owned(ex->fence, texture_copy_read_done, texture_copy_read_cancel, copy->source_hold);
        copy->source_hold = NULL;
    }
    vk_gpu_gap_note(VK_GAP_COPY_T2T, (uint64_t)copy->source_extent.width * copy->source_extent.height);
    VkImage source = texture_copy_stage_source(ex, copy);
    vk_dlaa_hud_write(target);
    image_copy_record(ex, copy, source, target);
    vk_texture_projected_transfer_written(transfer->target);
    if (__atomic_load_n(&t2t_probe_on, __ATOMIC_RELAXED)) {
        __atomic_fetch_add(&t2t_probe_copies, 1, __ATOMIC_RELAXED);
        if (copy->scratch) __atomic_fetch_add(&t2t_probe_staged, 1, __ATOMIC_RELAXED);
    }
    if (copy->scratch) {
        vk_defer_image(copy->scratch, copy->scratch_memory);
        copy->scratch = VK_NULL_HANDLE;
    }
    vk_completion_add_owned(ex->fence, projected_transfer_done, projected_transfer_canceled, transfer->target);
    transfer->target = NULL;
}

void vk_exec_projected_transfer_record(VkExec *ex, VkExecProjectedTransfer *transfer) {
    if (!transfer->target || vk_exec_recording_failed() || vk_sync_stopping() || vk_device_is_lost()) return;
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    vk_exec_pass_end(ex);
    if (vk_exec_recording_failed() || vk_sync_stopping() || vk_device_is_lost()) return;
    vk_exec_barrier(ex);
    if (vk_exec_recording_failed() || vk_sync_stopping() || vk_device_is_lost()) return;
    ex->rt_resolved = 0;
    VkImage target = vk_texture_projected_transfer_image(transfer->target);
    if (transfer->opcode == OP_COPY_B2T) projected_b2t_record(ex, transfer, target);
    else projected_t2t_record(ex, transfer, target);
}

// ---- buffer to buffer and buffer fills


typedef struct {
    VkMemSpan source, destination;
    VkMemSpanCapture *source_hold, *destination_hold;
    VkTextureAliasRead *source_image;
    VkBuffer alias_buffer;
    VkDeviceSize alias_offset;
    RcInterval *writes;
    uint32_t write_count;
    uint64_t src, dst, size;
    Range *completion;
} BufferTransferLeg;

enum { BUFFER_TRANSFER_COMPLETE, BUFFER_TRANSFER_HOLE, BUFFER_TRANSFER_OVERLAP, BUFFER_TRANSFER_UNALIGNED };

struct VkExecBufferTransfer {
    BufferTransferLeg *legs, terminal;
    size_t count, capacity;
    uint64_t src, dst, size, missing_address;
    RcRegistryPosition read;
    uint32_t opcode, value;
    int stop, missing, recorded;
};

static int buffer_transfer_probe;
static VkBufferTransferProbeStats buffer_transfer_stats;

void vk_exec_buffer_transfer_probe_arm(int arm) {
    // Only the native probe enables these counters, at a fence after all preceding probe copies.
    if (arm) buffer_transfer_stats = (VkBufferTransferProbeStats){0};
    __atomic_store_n(&buffer_transfer_probe, arm, __ATOMIC_RELEASE);
}

VkBufferTransferProbeStats vk_exec_buffer_transfer_probe_stats(void) {
    return (VkBufferTransferProbeStats){
        .copies=__atomic_load_n(&buffer_transfer_stats.copies,__ATOMIC_RELAXED),
        .clears=__atomic_load_n(&buffer_transfer_stats.clears,__ATOMIC_RELAXED),
        .legs=__atomic_load_n(&buffer_transfer_stats.legs,__ATOMIC_RELAXED),
        .bytes=__atomic_load_n(&buffer_transfer_stats.bytes,__ATOMIC_RELAXED),
        .partial=__atomic_load_n(&buffer_transfer_stats.partial,__ATOMIC_RELAXED),
        .mirror_twin=__atomic_load_n(&buffer_transfer_stats.mirror_twin,__ATOMIC_RELAXED),
    };
}

static void buffer_transfer_leg_release(BufferTransferLeg *leg) {
    vk_mem_span_capture_release(&leg->source_hold);
    vk_mem_span_capture_release(&leg->destination_hold);
    if (leg->source_image) vk_texture_alias_read_cancel(leg->source_image);
    free(leg->writes);
    free(leg->completion);
}

void vk_exec_buffer_transfer_release(VkExecBufferTransfer *transfer) {
    if (!transfer) return;
    for (size_t i = 0; i < transfer->count; i++) buffer_transfer_leg_release(&transfer->legs[i]);
    buffer_transfer_leg_release(&transfer->terminal);
    free(transfer->legs);
    free(transfer);
}

static RcRegistryResult capture_transfer_leg(VkExecBufferTransfer *transfer, BufferTransferLeg *leg, uint64_t left) {
    int copy = transfer->opcode == OP_COPY_B2B;
    RcRegistryResult result;
    if (copy) {
        // Copies and fills use an installed mirror or retain the current import through this scene.
        // This path never initiates optional mirror creation.
        result = leg->src ? vk_mem_span_prepare_native_input(leg->src, left, 1, VK_MEM_MIRROR_IMPORT,
            &leg->source, &leg->source_hold) : RC_REGISTRY_UNSUPPORTED;
        if (result != RC_REGISTRY_OK) {
            VkMemSpan check;
            if (result == RC_REGISTRY_UNSUPPORTED && !vk_mem_lookup(leg->src, 1, &check)) {
                transfer->stop = BUFFER_TRANSFER_HOLE;
                transfer->missing = 1;
                transfer->missing_address = leg->src;
                return RC_REGISTRY_OK;
            }
            return result;
        }
    }
    result = leg->dst ? vk_mem_span_prepare_native_input(leg->dst, left, 1, VK_MEM_MIRROR_IMPORT,
        &leg->destination, &leg->destination_hold) : RC_REGISTRY_UNSUPPORTED;
    if (result != RC_REGISTRY_OK) {
        VkMemSpan check;
        if (result == RC_REGISTRY_UNSUPPORTED && !vk_mem_lookup(leg->dst, 1, &check)) {
            transfer->stop = BUFFER_TRANSFER_HOLE;
            transfer->missing = 1;
            transfer->missing_address = leg->dst;
            return RC_REGISTRY_OK;
        }
        return result;
    }
    uint64_t n = leg->destination.avail;
    if (copy) {
        if (n > leg->source.avail) n = leg->source.avail;
        if (leg->source.buffer == leg->destination.buffer &&
            leg->source.offset < leg->destination.offset + n && leg->destination.offset < leg->source.offset + n) {
            transfer->stop = BUFFER_TRANSFER_OVERLAP;
            return RC_REGISTRY_OK;
        }
    } else {
        if (leg->destination.offset % 4) {
            transfer->stop = BUFFER_TRANSFER_HOLE;
            return RC_REGISTRY_OK;
        }
        n &= ~3ull;
    }
    if (!n) return RC_REGISTRY_OK;
    leg->write_count = vk_mem_span_input_intervals(leg->destination_hold, leg->dst, n, &leg->writes);
    if (!leg->write_count) return RC_REGISTRY_CAPACITY;
    leg->completion = malloc(sizeof *leg->completion);
    if (!leg->completion) return RC_REGISTRY_CAPACITY;
    leg->size = n;
    return RC_REGISTRY_OK;
}

RcRegistryResult vk_exec_buffer_transfer_prepare(uint32_t opcode, const void *payload, RcRegistryPosition read,
                                                 VkExecBufferTransfer **out) {
    if (!out || *out || !payload || !read.scene || read.ordinal < 2 || (read.ordinal & 1))
        return RC_REGISTRY_INVALID;
    uint64_t src = 0, dst, size;
    uint32_t value = 0;
    if (opcode == OP_COPY_B2B) {
        const p_copy_b2b *op = payload;
        src = op->src; dst = op->dst; size = op->size;
    } else if (opcode == OP_CLEAR_BUF) {
        const p_clear_buf *op = payload;
        dst = op->addr; size = op->size; value = op->value;
    } else return RC_REGISTRY_INVALID;
    int unaligned = opcode == OP_CLEAR_BUF && ((dst | size) & 3);
    if (!unaligned && (size > UINT64_MAX - src || size > UINT64_MAX - dst)) return RC_REGISTRY_UNSUPPORTED;
    VkExecBufferTransfer *transfer = calloc(1, sizeof *transfer);
    if (!transfer) return RC_REGISTRY_CAPACITY;
    *transfer = (VkExecBufferTransfer){.opcode = opcode, .src = src, .dst = dst, .size = size, .value = value,
        .read = read, .stop = unaligned ? BUFFER_TRANSFER_UNALIGNED : BUFFER_TRANSFER_COMPLETE};
    RcRegistryResult result = RC_REGISTRY_OK;
    uint64_t left = unaligned ? 0 : size;
    while (left) {
        BufferTransferLeg leg = {.src = src, .dst = dst};
        result = capture_transfer_leg(transfer, &leg, left);
        if (result == RC_REGISTRY_OK && leg.size && transfer->count == transfer->capacity) {
            size_t capacity = transfer->capacity ? transfer->capacity * 2 : 4;
            BufferTransferLeg *copy = capacity >= transfer->capacity && capacity <= SIZE_MAX / sizeof *copy ?
                realloc(transfer->legs, capacity * sizeof *copy) : NULL;
            if (!copy) result = RC_REGISTRY_CAPACITY;
            else { transfer->legs = copy; transfer->capacity = capacity; }
        }
        if (result != RC_REGISTRY_OK) {
            buffer_transfer_leg_release(&leg);
            break;
        }
        if (!leg.size) {
            transfer->terminal = leg;
            break;
        }
        transfer->legs[transfer->count++] = leg;
        if (opcode == OP_COPY_B2B) src += leg.size;
        dst += leg.size;
        left -= leg.size;
    }
    if (result == RC_REGISTRY_OK) *out = transfer;
    else vk_exec_buffer_transfer_release(transfer);
    return result;
}

static int buffer_transfer_sources_current(void *opaque) {
    const VkExecBufferTransfer *transfer = opaque;
    for (size_t i = 0; i < transfer->count; i++) {
        const BufferTransferLeg *leg = &transfer->legs[i];
        if (!vk_mem_span_capture_current(leg->source_hold, &leg->source) ||
            !vk_mem_span_capture_current(leg->destination_hold, &leg->destination) ||
            (leg->source_image && !vk_texture_alias_read_current(leg->source_image))) return 0;
    }
    return 1;
}

static int buffer_transfer_source_follows_overlapping_write(const VkExecBufferTransfer *transfer, size_t index) {
    uint32_t count;
    const RcInterval *ranges = vk_texture_alias_read_intervals(transfer->legs[index].source_image, &count);
    for (size_t earlier = 0; earlier < index; earlier++) {
        const BufferTransferLeg *write = &transfer->legs[earlier];
        for (uint32_t j = 0; j < count; j++)
            for (uint32_t k = 0; k < write->write_count; k++)
                if (rc_range_overlap(ranges[j].range, write->writes[k].range)) return 1;
    }
    return 0;
}

RcRegistryResult vk_exec_buffer_transfer_sources_prepare(VkExec *ex, VkExecBufferTransfer *transfer) {
    if (transfer->opcode != OP_COPY_B2B || transfer->read.scene != ex->fence ||
        transfer->read.ordinal != ex->resource_op * 2) return RC_REGISTRY_STALE;
    VkTextureAliasRead *gathers[VK_EXEC_ALIAS_READ_LIMIT];
    uint32_t count = 0;
    for (size_t i = 0; i < transfer->count; i++) {
        BufferTransferLeg *leg = &transfer->legs[i];
        VkTextureAliasFailure failure = {0};
        RcRegistryResult result = vk_texture_alias_buffer_capture(leg->source_hold, &leg->source,
            leg->src, leg->size, &leg->source_image, &failure);
        if (result != RC_REGISTRY_OK) {
            rt_log("vk: buffer copy retained image source refused scene %llu operation %llu source %#llx+%#llx "
                   "status %d gate %d cause %d backing %llu mapping %llu offset %#llx",
                (unsigned long long)ex->fence, (unsigned long long)ex->resource_op,
                (unsigned long long)leg->src, (unsigned long long)leg->size, result, failure.coverage.gate,
                failure.coverage.cause, (unsigned long long)failure.coverage.interval.range.backing,
                (unsigned long long)failure.coverage.interval.mapping,
                (unsigned long long)failure.coverage.interval.range.offset);
            return result;
        }
        if (!leg->source_image) continue;
        if (buffer_transfer_source_follows_overlapping_write(transfer, i)) return RC_REGISTRY_BLOCKED;
        if (count == VK_EXEC_ALIAS_READ_LIMIT) return RC_REGISTRY_CAPACITY;
        gathers[count++] = leg->source_image;
    }
    if (!count) return RC_REGISTRY_OK;
    VkExecRestoreResult prepared = vk_exec_prepare_alias_readback(ex, gathers, count,
        buffer_transfer_sources_current, transfer);
    if (prepared == VK_EXEC_RESTORE_STOPPED) return RC_REGISTRY_BUSY;
    if (prepared == VK_EXEC_RESTORE_STALE) return RC_REGISTRY_STALE;
    if (prepared != VK_EXEC_RESTORE_READY) return RC_REGISTRY_CAPACITY;

    for (size_t i = 0; i < transfer->count; i++) {
        BufferTransferLeg *leg = &transfer->legs[i];
        if (leg->source_image && !vk_texture_alias_buffer_source(leg->source_image, &leg->alias_buffer, &leg->alias_offset))
            return RC_REGISTRY_STALE;
    }
    return RC_REGISTRY_OK;
}

RcRegistryResult vk_exec_buffer_transfer_declare(const VkExecBufferTransfer *transfer, ResourceCommandEffects *effects,
                                                 RcRegistryPosition output) {
    if (!transfer || !effects) return RC_REGISTRY_INVALID;
    if (output.scene != transfer->read.scene || output.ordinal != transfer->read.ordinal + 1) return RC_REGISTRY_STALE;
    if (!transfer->count) return RC_REGISTRY_OK;
    uint32_t count = 0;
    for (size_t i = 0; i < transfer->count; i++) {
        if (transfer->legs[i].write_count > UINT32_MAX - count) return RC_REGISTRY_CAPACITY;
        count += transfer->legs[i].write_count;
    }
    RcInterval *parts = malloc((size_t)count * sizeof *parts);
    if (!parts) return RC_REGISTRY_CAPACITY;
    uint32_t at = 0;
    for (size_t i = 0; i < transfer->count; i++) {
        const BufferTransferLeg *leg = &transfer->legs[i];
        memcpy(parts + at, leg->writes, (size_t)leg->write_count * sizeof *parts);
        at += leg->write_count;
    }
    RcRegistryResult result = resource_effects_add_intervals(effects, parts, count, parts, count, RC_REGISTRY_BUFFER);
    free(parts);
    if (result == RC_REGISTRY_OK) for (size_t i = 0; i < transfer->count; i++) {
        const BufferTransferLeg *leg = &transfer->legs[i];
        RcSourceRange raw = {RC_GUEST_COMPATIBILITY_BACKING, leg->dst, leg->size};
        resource_effects_add(effects, raw, raw, RC_REGISTRY_BUFFER);
    }
    return result;
}

static void buffer_transfer_record(VkExec *ex, VkExecBufferTransfer *transfer) {
    int copy = transfer->opcode == OP_COPY_B2B;
    if (transfer->recorded) {
        vk_exec_required_graphics_unavailable(ex, transfer->opcode, "buffer transfer already recorded");
        return;
    }
    const BufferTransferLeg *terminal = &transfer->terminal;
    VkMemSpan check;
    if ((transfer->missing && vk_mem_lookup(transfer->missing_address, 1, &check)) ||
        (terminal->source_hold && !vk_mem_span_capture_current(terminal->source_hold, &terminal->source)) ||
        (terminal->destination_hold && !vk_mem_span_capture_current(terminal->destination_hold, &terminal->destination))) {
        vk_exec_required_graphics_unavailable(ex, transfer->opcode, "buffer transfer terminal span changed");
        return;
    }
    // Confirm every segment before writing any. The typed effect owns both projections and the actual batch
    // inherits each pinned provider; no destination or mirror is looked up again during emission.
    for (size_t i = 0; i < transfer->count; i++) {
        BufferTransferLeg *leg = &transfer->legs[i];
        int visible = vk_mem_span_input_cpu_visible(leg->destination_hold);
        if ((copy && !vk_mem_span_record_input(leg->source_hold, &leg->source)) ||
            (leg->source_image && (!leg->alias_buffer || !vk_texture_alias_read_current(leg->source_image))) ||
            !vk_mem_span_record_input(leg->destination_hold, &leg->destination) ||
            !resource_effects_buffer_prepare_intervals(leg->writes, leg->write_count, visible) ||
            !resource_effects_buffer_prepare(leg->dst, leg->size, visible)) {
            vk_exec_required_graphics_unavailable(ex, transfer->opcode, "retained buffer transfer span or write changed");
            return;
        }
        RcRegistryBufferLocation native;
        if (vk_mem_span_buffer_location(leg->destination_hold, &leg->destination, &native))
            resource_effects_buffer_prepare_native(leg->writes, leg->write_count, leg->dst, native);
    }
    transfer->recorded = 1;
    if (transfer->stop == BUFFER_TRANSFER_UNALIGNED) {
        warn_once(W_BUFFER, "buffer fill at %#llx of %llu bytes is not 4-byte aligned: skipped",
                  (unsigned long long)transfer->dst, (unsigned long long)transfer->size);
        return;
    }
    int probe = __atomic_load_n(&buffer_transfer_probe, __ATOMIC_ACQUIRE);
    uint64_t bytes = 0, twins = 0;
    vk_gpu_gap_note(copy ? VK_GAP_COPY_B2B : VK_GAP_CLEAR_BUF, transfer->size);
    if (transfer->count) {
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
    }
    for (size_t i = 0; i < transfer->count; i++) {
        BufferTransferLeg *leg = &transfer->legs[i];
        Range *write = leg->completion;
        *write = (Range){.addr = leg->dst, .size = leg->size,
            .back = vk_mem_recorded_write_mirrored(&leg->destination), .gpu_only = nvn_gpu_only(leg->dst),
            .provider = vk_mem_recorded_provider_acquire(&leg->destination), .buffer = leg->destination.buffer};
        vk_exec_record_native_input_provider(ex, &leg->destination_hold);
        if (copy) {
            vk_exec_record_native_input_provider(ex, &leg->source_hold);
            if (leg->source_image) vk_texture_alias_read_use(leg->source_image, ex->fence);
            VkBuffer buffer = leg->source_image ? leg->alias_buffer : leg->source.buffer;
            VkDeviceSize offset = leg->source_image ? leg->alias_offset : leg->source.offset;
            VkBufferCopy region = {offset, leg->destination.offset, leg->size};
            vkCmdCopyBuffer(ex->cb, buffer, leg->destination.buffer, 1, &region);
            vk_mem_count(leg->source.cls, VK_MUSE_COPY_SRC, leg->size);
        } else {
            vkCmdFillBuffer(ex->cb, leg->destination.buffer, leg->destination.offset, leg->size, transfer->value);
            VkBuffer twin = vk_mem_guest_twin(&leg->destination);
            if (twin) {
                vkCmdFillBuffer(ex->cb, twin, leg->destination.offset, leg->size, transfer->value);
                if (probe) twins++;
                ex->recorded++;
            }
        }
        leg->completion = NULL;
        vk_usnap_drop(leg->dst, leg->size);
        if (!write->gpu_only) vk_mem_gpu_pending_span(&leg->destination, leg->dst, leg->size);
        vk_completion_add_owned(ex->fence, guest_written_cb, guest_written_cancel, write);
        vk_mem_gpu_wrote(&leg->destination, leg->size);
        vk_mem_count(leg->destination.cls, VK_MUSE_GPU_WRITE, leg->size);
        if (probe) bytes += leg->size;
        ex->recorded++;
    }
    if (probe) {
        uint64_t *commands = copy ? &buffer_transfer_stats.copies : &buffer_transfer_stats.clears;
        __atomic_fetch_add(commands, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&buffer_transfer_stats.legs, transfer->count, __ATOMIC_RELAXED);
        __atomic_fetch_add(&buffer_transfer_stats.bytes, bytes, __ATOMIC_RELAXED);
        __atomic_fetch_add(&buffer_transfer_stats.partial, bytes < transfer->size, __ATOMIC_RELAXED);
        __atomic_fetch_add(&buffer_transfer_stats.mirror_twin, twins, __ATOMIC_RELAXED);
    }
    resource_effects_buffers_commit();
    if (transfer->stop == BUFFER_TRANSFER_OVERLAP)
        warn_once(W_OVERLAP, "buffer copy %#llx -> %#llx overlaps within one buffer: skipped",
                  (unsigned long long)transfer->src, (unsigned long long)transfer->dst);
    else if (transfer->stop == BUFFER_TRANSFER_HOLE) {
        if (copy) warn_once(W_BUFFER, "buffer copy %#llx -> %#llx (%llu bytes) leaves backed memory: rest skipped",
                            (unsigned long long)transfer->src, (unsigned long long)transfer->dst,
                            (unsigned long long)transfer->size);
        else warn_once(W_BUFFER, "buffer fill at %#llx (%llu bytes) leaves backed memory: rest skipped",
                       (unsigned long long)transfer->dst, (unsigned long long)transfer->size);
    }
}

// ---- texture clears

static int int_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
        return 1;
    default: return 0;
    }
}

static int clear_region(const p_clear_tex *p, const Sub *s, VkExecClearTexture *clear) {
    const VkTexImage *img = s->img;
    int is3d = img->type == VK_IMAGE_TYPE_3D;
    uint32_t lw = max1(img->width >> s->level), lh = max1(img->height >> s->level);
    int reformat = p->has_view && p->view.set_format && p->view.format != p->tex->d.format;
    if (s->x || s->y || s->w != lw || s->h != lh || (p->mask & 0xf) != 0xf || reformat ||
        (is3d && (s->z || s->nz != max1(img->depth >> s->level)))) {
        warn_once(W_CLEAR, "texture clear of region %d,%d %ux%u mask %#x (level %ux%u, view format change %d): skipped",
                  s->x, s->y, s->w, s->h, p->mask, lw, lh, reformat);
        return 0;
    }
    clear->range = (VkImageSubresourceRange){img->aspect, s->level, 1,
        is3d ? 0 : s->layer0 + (uint32_t)s->z, is3d ? 1 : s->nz};
    clear->texels = (uint64_t)lw * lh;
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
        clear->range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        memcpy(&clear->color, &p->c, sizeof clear->color);
        if (int_format(img->format) != (p->is_uint != 0)) {
            // A value of the other kind that is all zero bits (0.0f is) is the same clear either way: Ocarina clears its
            // rg16ui edge map (480x270) with a float 0 each frame, and skipped, the post chain's edge classification
            // read the previous frames' edges. Other values stay skipped.
            const uint32_t *bits = clear->color.uint32;
            if (bits[0] | bits[1] | bits[2] | bits[3]) {
                warn_once(W_CLEAR, "texture clear value kind (uint %d) does not match format %d: skipped", p->is_uint,
                          (int)img->format);
                return 0;
            }
        }
    }
    return 1;
}

int vk_exec_clear_alias_candidate(const p_clear_tex *p) {
    if (!p->tex) return 0;
    view_t view = p->view;
    if (view.set_format && view.format == p->tex->d.format) view.set_format = 0;
    if (!vk_texture_alias_region_transfer(p->tex, p->has_view ? &view : NULL, &p->r) ||
        (p->mask & 0xf) != 0xf || p->r.xoffset || p->r.yoffset ||
        (uint32_t)p->r.width != p->tex->d.width || (uint32_t)p->r.height != p->tex->d.height) return 0;
    TexVkFormat format;
    if (!tex_vk_format(p->tex->d.format, &format)) return 0;
    if (int_format(format.vk) == (p->is_uint != 0)) return 1;
    uint32_t bits[4];
    memcpy(bits, &p->c, sizeof bits);
    return !(bits[0] | bits[1] | bits[2] | bits[3]);
}

int vk_exec_clear_capture(const p_clear_tex *p, VkExecClearTexture *clear) {
    if (!p->tex) return 1;
    if ((p->mask & 0xf) != 0xf || (p->has_view && p->view.set_format && p->view.format != p->tex->d.format)) {
        warn_once(W_CLEAR, "texture clear changes a channel mask or view format: skipped");
        return 1;
    }
    VkTexImage shape;
    if (!vk_texture_write_target_shape(p->tex, &shape)) return 0;
    Sub s = {.img = &shape};
    if (!sub_region(p->tex, &p->view, p->has_view, &p->r, 1, &s) || !clear_region(p, &s, clear)) return 1;
    return vk_texture_write_target_capture(p->tex, &shape, &clear->target);
}

void vk_exec_clear_commit(VkExec *ex, VkExecClearTexture *clear) {
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    vk_gpu_gap_note(VK_GAP_CLEAR_TEX, clear->texels);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    ex->rt_resolved = 0;
    const VkTexImage *image = vk_texture_write_target_begin(clear->target, ex->cb);
    vk_dlaa_hud_write(image->image);
    vkCmdClearColorImage(ex->cb, image->image, VK_IMAGE_LAYOUT_GENERAL, &clear->color, 1, &clear->range);
    ex->recorded++;
    vk_texture_write_target_clear_written(clear->target, &clear->range);
    clear->target = NULL;
}

// Whole levels only (vkCmdClear*Image has no rectangle). Other previously supported producers resolve here.
static void clear_tex(VkExec *ex, const p_clear_tex *p) {
    const tex_t *t = p->tex;
    if (!t) return;
    Sub s;
    int projected = vk_exec_clear_alias_candidate(p);
    if (!sub_resolve(t, &p->view, p->has_view, &p->r, 1, projected ? 2 : 1, &s)) return;
    VkExecClearTexture clear = {0};
    if (!clear_region(p, &s, &clear)) return;
    const VkTexImage *img = s.img;
    vk_gpu_gap_note(VK_GAP_CLEAR_TEX, clear.texels);
    if (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vk_dlaa_hud_write(img->image);
        vkCmdClearColorImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &clear.color, 1, &clear.range);
    } else {
        int both = (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);
        VkClearDepthStencilValue dv = {0, 0};
        if (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) dv.depth = p->c.f[0];
        if (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) dv.stencil = both ? p->c.u[1] : p->c.u[0];
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);
        vkCmdClearDepthStencilImage(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, &dv, 1, &clear.range);
    }
    ex->recorded++;
    resource_image_written((uint64_t)(uintptr_t)img->image, img->addr);
    vk_texture_gpu_written(t);
    vk_texture_alias_clear_written(t, img->image, &clear.range);
}

// ---- parallel groups (vk_exec.c): the CPU side in the front pass, the rest in the workers

// The front pass, in the order of the stream, before the workers record (vk_exec_xfer_group on): a counter report or
// reset (completion callbacks, pending entries, the occlusion segments and counting), a render enable and a render
// condition (decided here, on the reports replayed so far). Returns the condition's decision: 1 renders.
int vk_exec_xfer_front(VkExec *fx, uint32_t op, const void *payload) {
    switch (op) {
    case OP_REPORT_COUNTER: report_counter(fx, payload); return 1;
    case OP_RESET_COUNTER: reset_counter(fx, payload); return 1;
    case OP_RENDER_ENABLE: fx->render_disabled = !((const p_flag *)payload)->value; return 1;
    case OP_RENDER_COND:
        fx->render_cond = *(const p_render_cond *)payload;
        fx->has_render_cond = 1;
        fx->render_disabled = !cond_enabled(payload);
        return !fx->render_disabled;
    default: rt_fatal("vk: op %s has no front pass", nvn_op_name(op));
    }
    return 1;
}

// A worker: what the front pass left, in its chunk. The contact shadows' pass (vk_contact_flush) is recorded where
// the replay would, once every earlier chunk of the group is done.
static void xfer_worker(VkExec *ex, uint32_t op, const void *payload) {
    if (op != OP_RENDER_ENABLE && vk_contact_pending()) vk_exec_turn(), vk_contact_flush(ex);
    switch (op) {
    case OP_REPORT_COUNTER: case OP_RESET_COUNTER: {
        int type = op == OP_REPORT_COUNTER ? (int)((const p_report_counter *)payload)->type
                                           : (int)((const p_reset_counter *)payload)->type;
        if (type != NVN_COUNTER_TYPE_SAMPLES_PASSED || !oq_ready()) break;
        oq_close(ex);   // into the segment this op ends; the next queries go to the next one
        XW.seg++;
        if (op == OP_RESET_COUNTER) XW.counting = 1, XW.reported = 0;
        else {
            if (XW.reported) XW.keep = 1;   // as oq_report
            XW.reported = 1;
            if (!XW.keep) XW.counting = 0;
        }
        break;
    }
    case OP_RENDER_ENABLE: ex->render_disabled = !((const p_flag *)payload)->value; break;
    case OP_RENDER_COND:
        ex->render_cond = *(const p_render_cond *)payload;
        ex->has_render_cond = 1;
        if (XW.i >= XW.ncond) rt_fatal("vk: a replay worker met more render conditions than its front pass decided");
        ex->render_disabled = !XW.cond[XW.i++];
        break;
    default: rt_fatal("vk: op %s in a parallel group", nvn_op_name(op));
    }
}

static void op_xfer_impl(VkExec *ex, uint32_t op, const void *payload) {
    if (XW.on) return xfer_worker(ex, op, payload);
    if (op != OP_RENDER_ENABLE && op != OP_SIGNAL_EVENT) vk_contact_flush(ex);
    switch (op) {
    case OP_COPY_B2T: rt_fatal("vk: CopyBufferToTexture reached replay without its ordered receipt");
    case OP_COPY_T2B: rt_fatal("vk: CopyTextureToBuffer reached replay without its ordered receipt");
    case OP_COPY_T2T: rt_fatal("vk: CopyTextureToTexture reached replay without ordered source selection");
    case OP_CLEAR_TEX: clear_tex(ex, payload); break;
    case OP_SIGNAL_EVENT: signal_event(ex, payload); break;
    case OP_REPORT_COUNTER: report_counter(ex, payload); break;
    case OP_RESET_COUNTER: reset_counter(ex, payload); break;
    case OP_COPY_B2B: case OP_CLEAR_BUF: {
        if (vk_exec_recording_failed()) break;
        VkExecBufferTransfer *transfer = vk_exec_effects_buffer_transfer(ex, op, payload);
        if (transfer) buffer_transfer_record(ex, transfer);
        else vk_exec_required_graphics_unavailable(ex, op, "ordered buffer transfer receipt missing");
        break;
    }
    case OP_RENDER_ENABLE: ex->render_disabled = !((const p_flag *)payload)->value; break;
    case OP_RENDER_COND:
        ex->render_cond = *(const p_render_cond *)payload;
        ex->has_render_cond = 1;
        ex->render_disabled = !cond_enabled(payload);
        break;
    default: rt_fatal("vk: op %s is not a transfer", nvn_op_name(op));
    }
}

// NX2NAVI_REPLAY_TIME: see the note over vk_exec_op_draw in vk_exec_draw.c; transfers are the third kind.
void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload) {
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    static int on = -1;
    if (on < 0) on = cfg_bool("NX2NAVI_REPLAY_TIME");
    if (!on) { op_xfer_impl(ex, op, payload); return; }
    uint64_t t0 = plat_now_ns();
    op_xfer_impl(ex, op, payload);
    perf_replay_op_at(PERF_RP_XFER, op, plat_now_ns() - t0);
}

// Metadata-only output declarations. Texture outputs conservatively cover their entire original storage; this
// creates authority, not a claim that every byte changed. Buffer image copies use their actual row-pitch formula.
static int effect_emit_range(VkExecEffectEmit emit, void *ctx, uint64_t addr, uint64_t size,
                             RcRegistryWriterKind kind) {
    if (!size) return 1;
    if (!addr || addr > UINT64_MAX - size) return 0;
    emit(ctx, addr, size, kind);
    return 1;
}
static int effect_emit_texture(VkExecEffectEmit emit, void *ctx, const tex_t *t) {
    return !t || effect_emit_range(emit, ctx, (uint64_t)(uintptr_t)t->mem, t->L.size, RC_REGISTRY_IMAGE);
}
int vk_exec_xfer_effects(VkExec *ex, uint32_t op, const void *payload, VkExecEffectEmit emit, void *ctx) {
    switch (op) {
    case OP_UPDATE_UB: {
        const p_update_ub *p = payload;
        return effect_emit_range(emit, ctx, p->dst, p->size, RC_REGISTRY_BUFFER);
    }
    case OP_COPY_B2T:
        return effect_emit_texture(emit, ctx, ((const p_copy_b2t *)payload)->tex);
    case OP_COPY_T2T:
        return effect_emit_texture(emit, ctx, ((const p_copy_t2t *)payload)->dst);
    case OP_CLEAR_TEX:
        return effect_emit_texture(emit, ctx, ((const p_clear_tex *)payload)->tex);
    case OP_COPY_T2B: {
        const p_copy_b2t *p = payload;
        if (!p->tex) return 1;
        int bw, bh, bpb = nvn_fmt_bpb(p->tex->d.format, &bw, &bh);
        // A defaulted extent depends on the resolved representation, which this metadata pass cannot assume.
        // Positive explicit extents determine the buffer bytes independently; scaled copies are skipped by emission.
        if (!bpb || bw <= 0 || bh <= 0 || p->r.width <= 0 || p->r.height <= 0) return 0;
        BufferImagePitch pitch;
        int result = buffer_image_pitch((uint32_t)p->r.width, (uint32_t)p->r.height,
            p->r.depth > 0 ? (uint32_t)p->r.depth : 1, bw, bh, bpb, ex->copy_stride, &pitch);
        if (result == COPY_PITCH_INVALID) return 1;
        if (result != COPY_PITCH_OK || pitch.size > UINT64_MAX - p->src) return 0;
        uint64_t runs = buffer_image_runs(&pitch), bytes = buffer_image_run_bytes(&pitch);
        for (uint64_t i = 0; i < runs; i++)
            if (!effect_emit_range(emit, ctx, p->src + i * pitch.stride, bytes, RC_REGISTRY_BUFFER)) return 0;
        return 1;
    }
    case OP_REPORT_COUNTER: {
        const p_report_counter *p = payload;
        if (nvn_gpu_only(p->addr)) return 1;
        return effect_emit_range(emit, ctx, p->addr, sizeof(NVNcounterData), RC_REGISTRY_BUFFER);
    }
    case OP_SIGNAL_EVENT:
        return effect_emit_range(emit, ctx, (uint64_t)(uintptr_t)((const p_signal_event *)payload)->val,
                                 sizeof(uint32_t), RC_REGISTRY_BUFFER);
    case OP_RESET_COUNTER: case OP_RENDER_ENABLE: case OP_RENDER_COND: return 1;
    default: return 0;
    }
}

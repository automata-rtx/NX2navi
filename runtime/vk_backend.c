// The Vulkan graphics back end: gfx_vulkan_backend wires the memory and
// texture layer (vk_memory.c, vk_texture.c), submission and fences (vk_sync.c) and the present. Every entry starts the
// device on first use.
//
// Present: the game texture is resolved on the calling thread and its blit is handed to a present thread, so
// nvnQueuePresentTexture returns at once (section 5). Queue order does the rest: the submissions that wrote the texture
// are already on gfx_queue, and every call that may write or free GPU state the blit reads (submit, finish, pool and
// texture teardown, texel writes) first waits until the queued blit was submitted (vk_present_drain).
//
// Back-edge pacing: the present thread submits the blit at once, then holds the host present (vkQueuePresentKHR) until
// the frame's display time from the pacer (nvn_api_obj.c), so present-to-present intervals stay even whatever the
// replay took. The blit is already on the queue, so the wait blocks no submission.
//
// Asynchronous replay (NX2NAVI_REPLAY_ASYNC, default 0; 1 turns it on): nvnQueueSubmitCommands
// and nvnQueuePresentTexture only queue. Each item gets its timeline value at enqueue (vk_sync_reserve), in queue
// order; the replay thread runs the synchronous submit and present bodies in that order, so every value is signalled
// by the submission it was handed out for. The game runs at most one frame ahead: queueing present N waits until the
// replay thread handled present N-1. Calls that read or free GPU state that queued work may use drain the queue first
// (replay_drain), each cause counted.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "flight.h"
#include "gfx.h"
#include "motion_owner.h"
#include "host.h"
#include "nvn_api.h"
#include "perfmon.h"
#include "vk.h"
#include "replay_phase_trace.h"
#include "vk_exec.h"
#include "platform/platform.h"
#include "vk_dlaa.h"
#include "vk_dlssg.h"
#include "vk_fg.h"
#include "vk_mesh_fb.h"

#define WEAK __attribute__((weak))

typedef struct {
    VkTexturePresentSource *source;
    VkDlaaFgSizeUse *temporal;
    VkDlaaHudSource *hud;
    VkImage image;
    VkFormat format;
    int w, h, has_crop, crop[4];
    uint64_t frame, at, queued, value;
} PresentJob;

static PlatMutex pl = PLAT_MUTEX_INIT;
static PlatCond pcv = PLAT_COND_INIT;   // job queued, job taken, job submitted
static PresentJob job;
static int pending, busy, present_thread;

// Sleeps on a high-resolution timer to 1 ms before at, then yields until at: the timer alone wakes up to ~0.5 ms late.
static void wait_until(uint64_t at) {
    uint64_t now = kern_now_ns();
    if (at > now + 1500000) plat_wait_ns(at - now - 1000000);
    while (kern_now_ns() < at) plat_yield();
}

// NX2NAVI_PACE_SMOOTH (default 1; 0 = off): a picture goes out no sooner than seven eighths of the recent interval
// after the one before it. The game's frames reach the present thread in uneven steps although each costs about the
// same: with the replay on the submitting thread, work held for an event is replayed with the next frame, so pictures
// came 13 to 31 ms apart around a mean of 22 ms in a village (5th to 95th percentile), which shows as judder at a
// frame rate that would look even. Only the early ones wait, a few milliseconds each, so the rate stays what it was;
// the recent interval is a moving average over about four pictures of intervals under 100 ms, so the floor follows
// a scene that gets lighter within a second. Not with frame generation, which paces its own presents.
static uint64_t pace_avg_ns;   // the present thread's

static uint64_t pace_smooth(uint64_t at, uint64_t last, uint64_t now) {
    static int on = -1;
    if (on < 0) on = cfg_bool("NX2NAVI_PACE_SMOOTH");
    if (!on || !last || !pace_avg_ns || vk_fg_active() || vk_dlssg_active()) return at;
    uint64_t floor = last + pace_avg_ns / 8 * 7;
    return floor > at && floor < now + 50000000ull ? floor : at;
}

static void pace_shown(uint64_t interval_ns) {
    if (interval_ns >= 100000000ull) return;   // a load or a hitch is not the rate
    pace_avg_ns = pace_avg_ns ? pace_avg_ns + (uint64_t)(((int64_t)interval_ns - (int64_t)pace_avg_ns) / 4) : interval_ns;
}

static void present_main(void *arg) {
    (void)arg;
    uint64_t last = 0, last_delivered = 0, last_queued = 0;
    plat_mutex_lock(&pl);
    for (;;) {
        while (!pending) plat_cond_wait(&pcv, &pl);
        PresentJob j = job;
        job = (PresentJob){0};
        pending = 0;
        busy = 1;
        plat_cond_broadcast(&pcv);
        plat_mutex_unlock(&pl);
        if (vk_sync_stopping() || vk_device_is_lost()) {
            vk_texture_present_release(j.source);
            if (j.temporal) vk_dlaa_fg_release(j.temporal);
            vk_dlaa_hud_source_release(j.hud);
            plat_mutex_lock(&pl);
            busy = 0;
            plat_cond_broadcast(&pcv);
            continue;
        }
        if (flight_on || vk_dump_white) vk_thumbs_frame(j.frame);   // NX2NAVI_DUMP_WHITE reads them too
        vk_dlssg_present_frame(j.frame);
        if (j.hud) {
            // H can still own the sealed source. Flush its real native work
            // before the composition records reads of these images.
            vk_queue_lock();
            vk_queue_unlock();
        }
        VkPresentSourceHandoff handed;
        do {
            VkPresentConfigRetry retry = {0};
            handed = vk_present_blit(j.image, VK_IMAGE_LAYOUT_GENERAL, j.format, j.w, j.h,
                                     j.has_crop ? j.crop : NULL, j.frame, &j.source, &j.temporal, &j.hud, &retry);
            if (handed != VK_PRESENT_SOURCE_PENDING) break;
            if (vk_sync_stopping() || vk_device_is_lost()) {
                handed = VK_PRESENT_SOURCE_UNSENT;
                break;
            }
            if (retry.older_config) {
                VkResult r = vk_present_config_drain(&retry);
                vk_device_lost_check(r, "retiring previous Streamline configuration");
                if (r != VK_SUCCESS) rt_fatal("vk: cannot retire earlier Streamline configuration (%d)", (int)r);
                resource_core_budget_refresh();
            } else if (!retry.reprepare) {
                RcBudgetProgress progress;
                resource_core_budget_service(0, &retry.peak, &progress);
                resource_core_budget_wait_since(progress.event, 100);
            }
        } while (handed == VK_PRESENT_SOURCE_PENDING);
        if (j.temporal) vk_dlaa_fg_release(j.temporal);
        vk_dlaa_hud_source_release(j.hud);
        if (handed == VK_PRESENT_SOURCE_UNSENT) vk_texture_present_release(j.source);
        else if (j.source) rt_fatal("vk: present did not adopt its source");
        if (handed == VK_PRESENT_SOURCE_UNCERTAIN) {
            if (!vk_device_is_lost()) rt_fatal("vk: the present source submission could not be proven");
            plat_mutex_lock(&pl);
            busy = 0;
            plat_cond_broadcast(&pcv);
            continue;
        }
        // Guest cadence uses its reserved timeline value. The graphics fence owns the submitted source.
        uint64_t f = vk_sync_present_complete();
        vk_dlssg_submitted(j.frame);
        if (j.value && f != j.value && !vk_sync_stopping() && !vk_device_is_lost())
            rt_fatal("vk: the present of frame %llu signalled %llu, queued as %llu", (unsigned long long)j.frame,
                     (unsigned long long)f, (unsigned long long)j.value);
        plat_mutex_lock(&pl);
        busy = 0;
        plat_cond_broadcast(&pcv);
        plat_mutex_unlock(&pl);
        uint64_t t0 = kern_now_ns();
        wait_until(pace_smooth(j.at, last, t0));
        uint64_t t1 = kern_now_ns();
        // Pace present-call starts as before. Including time inside the driver here
        // adds it to the next spacing target and can slow otherwise uncapped rendering.
        // Delivery accounting below independently measures successful call returns.
        if (last) pace_shown(t1 - last);
        int shown = vk_present_flip();
        uint64_t delivered = kern_now_ns();
        if (shown) {
            perfmon_shown(delivered); // successful host presents, including time inside the presentation call
        }
        // Optional compact diagnostic. This measures host delivery, not physical scan-out.
        // Together with the frame number it detects a faster producer building a display backlog.
        static FILE *delivery;
        static int delivery_opened;
        static unsigned delivery_rows;
        if (!delivery_opened) {
            delivery_opened = 1;
            const char *path = cfg_get("NX2NAVI_PRESENT_TIMING");
            if (path && *path) {
                delivery = fopen(path, "wb");
                if (delivery) fprintf(delivery, "frame\tqueued_ns\tflip_start_ns\tflip_end_ns\tshown\n");
            }
        }
        if (delivery) {
            fprintf(delivery, "%llu\t%llu\t%llu\t%llu\t%d\n", (unsigned long long)j.frame,
                (unsigned long long)j.queued, (unsigned long long)t1, (unsigned long long)delivered, shown);
            if (++delivery_rows % 60 == 0) fflush(delivery);
        }
        // NX2NAVI_TRACE=present: one line per present: the host present's interval, the interval of the game's present
        // calls (when the host present went out before back-edge pacing), the wait and how far past its target it went
        if (rt_trace & RT_TRACE_PRESENT)
            rt_log("present: frame %llu interval %.3f ms, game %.3f ms, waited %.3f ms, %.3f ms past target",
                   (unsigned long long)j.frame, last_delivered ? (delivered - last_delivered) / 1e6 : 0.0,
                   last_queued ? (j.queued - last_queued) / 1e6 : 0.0, (t1 - t0) / 1e6,
                   j.at && t1 > j.at ? (t1 - j.at) / 1e6 : 0.0);
        last = t1;
        if (shown) last_delivered = delivered;
        last_queued = j.queued;
        vk_pipeline_autosave();   // starts the autosave thread once
        plat_mutex_lock(&pl);
    }
    return;
}

void vk_present_drain(void) {
    uint64_t p0 = perf_begin();
    plat_mutex_lock(&pl);
    while (pending || busy) plat_cond_wait(&pcv, &pl);
    plat_mutex_unlock(&pl);
    perf_end(PERF_DRAIN, p0);
}

// NX2NAVI_FRAME_HOLD_MS (default 120; 0 = off): a frame whose draws were skipped because a shader or pipeline was
// still building (vk_exec_draw.c pending_draws) is not shown: the previous picture stays on screen until a frame is
// complete again. Such a frame lacks what was skipped, which showed for a frame or two as a black or unlit picture with
// the interface intact (a lighting pass), a light going out, or ground and walls missing and then back. The game runs
// on meanwhile, so nothing waits. The hold comes from an allowance of this many milliseconds, used up by the time
// held and filled again at an eighth of the clock's rate: a pipeline that takes longer, or a run of them on a first
// visit with nothing recorded, shows its incomplete frames as before and does not freeze the picture. Not with DLSS
// Frame Generation, which is told about every frame.
// The same hold covers a change of the scene's size (a DLSS or FSR 3.1 mode picked in the menu, vk_dlaa.c
// vk_dlaa_resized_take): the first frames drawn into the game's new, empty scene targets come out dark or black, so the
// previous picture stays for RESIZE_HOLD presents from the first frame of the new size. That is once per change of a
// setting and does not use the allowance. It holds with DLSS Frame Generation too: a present left out is a longer gap
// between two rendered frames to it (its constants and tags are set at the host present, gfx_present.c), and a dark
// frame it is given comes out as several generated ones.
// And it covers an area load (vk_exec_draw.c vk_exec_pool_loading): while the texture pool still makes the area's
// textures, the frames show the ground and the walls as black stand-ins, so the picture before the load stays for up
// to LOAD_HOLD_NS, also beside the allowance.
#define RESIZE_HOLD 6
#define LOAD_HOLD_NS 2000000000ull
static struct {
    int64_t limit_ns, left_ns;
    uint64_t last_ns, held, runs, shown, load_t0, load_held;
    int holding, resize_left;
} FH = {.limit_ns = -1};

static void frame_hold_refill(uint64_t now) {
    if (FH.limit_ns < 0) {
        long long ms = cfg_int("NX2NAVI_FRAME_HOLD_MS");
        FH.limit_ns = (int64_t)(ms < 0 ? 0 : ms > 500 ? 500 : ms) * 1000000;
        FH.left_ns = FH.limit_ns, FH.last_ns = now;
        if (FH.limit_ns)
            rt_log("vk: a frame with draws skipped for a pipeline is not shown, the previous picture stays up to %lld ms "
                   "(NX2NAVI_FRAME_HOLD_MS)", (long long)(FH.limit_ns / 1000000));
    }
    if (!FH.holding) FH.left_ns += (int64_t)(now - FH.last_ns) / 8;
    else FH.left_ns -= (int64_t)(now - FH.last_ns);
    if (FH.left_ns > FH.limit_ns) FH.left_ns = FH.limit_ns;
    FH.last_ns = now;
}

// Whether a frame that skipped draws now would be held (vk_exec_draw.c: a full-screen draw is skipped only then).
int vk_frame_hold_left(void) {
    frame_hold_refill(kern_now_ns());
    return FH.left_ns > 0 && !vk_dlssg_active();
}

// 1: this frame is not shown. skipped: the draws it lacks; resized: its scene is the first of a new size.
static int frame_hold(uint32_t skipped, int resized, uint64_t frame) {
    frame_hold_refill(kern_now_ns());
    if (resized && FH.limit_ns > 0) {
        FH.resize_left = RESIZE_HOLD;
        rt_log("vk: frame %llu: the scene has a new size after a setting changed; the previous picture stays for %d "
               "frames", (unsigned long long)frame, RESIZE_HOLD);
    }
    if (FH.resize_left) {   // beside the allowance: nothing below counts these frames
        FH.resize_left--;
        return 1;
    }
    if (FH.limit_ns > 0 && vk_exec_pool_loading()) {
        uint64_t now = kern_now_ns();
        if (!FH.load_t0) FH.load_t0 = now, FH.load_held = 0;
        if (now - FH.load_t0 < LOAD_HOLD_NS) {
            FH.load_held++;
            return 1;
        }
    } else if (FH.load_t0) {
        rt_log("vk: frame %llu: the previous picture stayed for %llu frames, %.0f ms, while an area's textures were made",
               (unsigned long long)frame, (unsigned long long)FH.load_held, (double)(kern_now_ns() - FH.load_t0) / 1e6);
        FH.load_t0 = 0;
    }
    int hold = skipped && FH.left_ns > 0 && !vk_dlssg_active();
    if (hold && !FH.holding) FH.runs++;
    if (skipped && !hold) FH.shown++;
    FH.holding = hold;
    if (hold) FH.held++;
    static uint64_t logged_at, logged_held, logged_shown;
    if (!hold && FH.held + FH.shown != logged_held + logged_shown && frame - logged_at >= 600) {
        rt_log("vk: frame %llu: %llu frames with skipped draws held back so far in %llu runs, %llu shown incomplete "
               "(allowance used up)", (unsigned long long)frame, (unsigned long long)FH.held,
               (unsigned long long)FH.runs, (unsigned long long)FH.shown);
        logged_at = frame, logged_held = FH.held, logged_shown = FH.shown;
    }
    if (hold && (rt_trace & RT_TRACE_PRESENT))
        rt_log("present: frame %llu held back, %u draws skipped", (unsigned long long)frame, skipped);
    return hold;
}

// The present after the frame count: resolves the image and hands the blit to the present thread. queued: when the game
// presented; value: the timeline value queued for this present (asynchronous replay), 0 for the next one (synchronous).
static void present_image(const tex_t *t, const int crop[4], uint64_t at_ns, uint64_t queued, uint64_t frame,
                          uint64_t value) {
    if (vk_sync_stopping() || vk_device_is_lost()) return;
    static volatile int32_t n_skipped;
    static int vram_probe;
    if (!vram_probe) vram_probe = 1, perfmon_set_vram_game(vk_mem_device_used_mb);   // the overlay's "game" beside VRAM
    vk_exec_frame_log(t, frame);
    dynres_frame();   // before vk_dlaa_frame reads the DLSS mode it may override
    vk_exec_barrier_log(frame);
    vk_dlaa_hud_frame(frame);
    vk_dlaa_frame();
    vk_sampler_aniso_live();   // NX2NAVI_ANISO changed in the menu: the surface textures' samplers take the new level
    vk_mvsurvey_frame();
    vk_mvprev_frame();
    vk_mem_census(frame);
    vk_gpu_passes_frame(frame);
    if (vk_dump_armed) vk_dump_present(frame);
    __atomic_store_n(&gfx_counters.validation_errors, (uint64_t)vk_validation_errors_total(), __ATOMIC_RELAXED);
    int held = frame_hold(vk_exec_pending_draws_take(), vk_dlaa_resized_take(), frame);
    VkTexImage img;
    VkTexturePresentSource *source = NULL;
    if (held) (void)vk_texture_get(t);
    else source = vk_texture_present_capture(t, &img);
    if (!source) {
        vk_dlaa_hud_take(frame, NULL);
        vk_dlaa_fg_discard_current();
        if (!held && plat_inc32(&n_skipped) <= 8)
            rt_log("vk: present of texture %p (format %#x) skipped: no colour image", (const void *)t, t ? t->d.format : 0);
        if (value) {   // the queued value is still signalled, in order
            vk_present_drain();
            uint64_t f = vk_sync_submit(NULL, 0);
            if (f != value && !vk_sync_stopping() && !vk_device_is_lost()) rt_fatal("vk: skipped present signalled %llu, queued as %llu", (unsigned long long)f, (unsigned long long)value);
        }
        vk_dlssg_submitted(frame);
        return;
    }
    PresentJob j = {.source = source, .image = img.image, .format = img.format, .w = (int)img.width, .h = (int)img.height, .frame = frame,
                    .temporal = vk_dlaa_fg_capture(), .hud = vk_dlaa_hud_take(frame, &img),
                    .at = at_ns, .queued = queued, .value = value};
    if (crop) {
        j.has_crop = 1;
        for (int i = 0; i < 4; i++) j.crop[i] = crop[i];
    }
    vk_upload_flush(0);   // an upload vk_texture_get just recorded goes before the blit
    plat_mutex_lock(&pl);
    while (pending) plat_cond_wait(&pcv, &pl);   // one present in waiting at most
    if (vk_sync_stopping() || vk_device_is_lost()) {
        plat_mutex_unlock(&pl);
        vk_texture_present_release(j.source);
        if (j.temporal) vk_dlaa_fg_release(j.temporal);
        vk_dlaa_hud_source_release(j.hud);
        return;
    }
    if (!present_thread) {
        if (!plat_thread_spawn(present_main, NULL, "vk present", 0)) rt_fatal("vk: cannot start the present thread");
        present_thread = 1;
    }
    job = j;
    pending = 1;
    plat_cond_broadcast(&pcv);
    plat_mutex_unlock(&pl);
    if (rdoc_active()) { vk_present_drain(); rdoc_after_present(frame); }
}

// ---- Asynchronous replay
enum { RI_SUBMIT, RI_PRESENT, RI_RELEASE };
typedef struct {
    int kind, n, has_crop, crop[4];
    uint64_t value, submit_no, at, queued;
    uint64_t *h;
    RcBudgetTicket handle_budget;
    VkReplayCaptureBundle captures;
    VkTextureRegistrationScope *registrations;
    tex_t *tex;   // RI_PRESENT and RI_RELEASE own the captured guest descriptor until replay consumes it.
} ReplayItem;

static void replay_handles_release(ReplayItem *item) {
    vk_replay_metadata_free(item->h, &item->handle_budget);
    item->h = NULL;
}

static tex_t *replay_copy_texture(const tex_t *texture) {
    if (!texture) return NULL;
    tex_t *copy = malloc(sizeof *copy);
    if (!copy) rt_fatal("vk: texture descriptor capture allocation failed");
    *copy = *texture;
    return copy;
}
#define RQ_CAP 1024   // a frame queues 1 submit and 1 present, and up to a few hundred releases
enum { DR_POOL_FINI, DR_MAP_VIRTUAL, DR_TEX_RELEASE, DR_PROG_RELEASE, DR_FINISH, DR_READ_TEXELS, DR_WRITE_TEXELS,
       DR_GUEST_READ, DR_FRAME, DR_FULL, DR_CHECK, DR_GATE, DR_CAPTURE, DR_REGISTRATION, DR_N };
static const char *const dr_name[DR_N] = {
    "pool_fini", "map_virtual", "texture_release (queued)", "program_release", "finish", "read_texels (writer fence)",
    "write_texels", "guest_read (writer fence)", "frame bound", "queue full", "check drain", "snapshot leading event",
    "older queued capture", "queued texture registration"};

static PlatMutex rl = PLAT_MUTEX_INIT;
static PlatMutex replay_registration_order = PLAT_MUTEX_INIT;
static PlatCond rcv = PLAT_COND_INIT;   // item queued, item taken, item done
static ReplayItem rq[RQ_CAP];
static int rq_head, rq_n, r_busy, r_started, r_frontend_calls;
// Projection of successful FIFO admissions. The NVN frontend serializes asynchronous submit calls.
static VkUsnapBindingState r_uniforms;
static uint64_t r_presents_queued, r_presents_done, r_submit_done, game_frames;
// Accepted registration-bearing scopes are in publication order because enqueue validates under the order lock.
static uint64_t r_registration_issued, r_registration_done, r_registration_publication;
// Under rl: calls, calls that waited and the time waited per cause, since the last replay_log line and over the run.
static uint64_t dr_calls[DR_N], dr_waits[DR_N], dr_ns[DR_N], dr_tcalls[DR_N], dr_twaits[DR_N], dr_tns[DR_N];
// NX2NAVI_REPLAY_CHECK: ReadTexels compared with a read after a full drain (under rl): checked, differing, differing
// although no submission recorded a write to the texture in between, differing with the queue idle at the first read.
static uint64_t ck_reads, ck_diff, ck_diff_unexplained, ck_diff_idle;

int vk_replay_async(void) {
    static int on = -1;
    if (on < 0) {
        // Off by default. Where the game waits for the GPU it overlaps the replay with the game threads (a GTX 1650 in
        // one scene went from 450-686 frames over 50 ms a minute to 10-13, issue #88), the only place a gain was
        // measured. While it was the default, a GTX 1660 Ti at the 60 fps cap with the GPU at 26 % reported uneven
        // motion that 0.3.4 did not have (#659). It adds up to a frame of latency.
        on = cfg_bool("NX2NAVI_REPLAY_ASYNC");
        rt_log("vk: replay %s (NX2NAVI_REPLAY_ASYNC=%d)",
               on ? "on its own thread, at most one frame behind the game" : "on the submitting thread", on);
    }
    return on;
}

// With the replay on its own thread a submission's value covers its replay too, and the wait stays the fence's.
static int vk_be_exec_wait(uint64_t value, uint64_t timeout_ns) {
    return vk_replay_async() ? vk_sync_fence_wait(value, timeout_ns) : vk_sync_exec_wait(value, timeout_ns);
}

// The front end's view (nvn_api_core.c, nvn_api_cmd.c): replay is asynchronous only on this back end.
int gfx_replay_async(void) { return gfx_backend() == &gfx_vulkan_backend && vk_replay_async(); }

// The front end's nvnEventSignal, before it writes (vk_exec_xfer.c "events the CPU signals").
void gfx_event_signalling(const void *value, uint32_t to) {
    vk_exec_event_cpu(value);
    vk_exec_evt('c', value, to);
}

// The newest submit number (nvn_submit_count) whose replay is done: recording copies it used may be freed.
uint64_t gfx_replayed_submits(void) {
    if (!gfx_replay_async()) return nvn_submit_count();
    plat_mutex_lock_shared(&rl);
    uint64_t v = r_submit_done;
    plat_mutex_unlock_shared(&rl);
    return v;
}

// ---- a present that waits for held work (event order, vk_exec.c)
// On the console a present is queued behind its queue's work and goes out once the GPU has run the frame, including
// work that waited for an event. The replay holds such work (vk_exec.c, "event order across the game's queues") until
// the wait holds, and the present went out at once: in Kokiri Forest dialogue (Saria), several presents a second came
// with the frame's whole submission held (4 held submissions, nothing drawn since the last present) and showed the
// picture from before, a flash of the whole picture; the frame was then drawn after its present, in the present
// thread's empty submission, next to the following frame. So with work held at a present, what can go on now goes
// first, in a submission of its own (held_resume); if work is still held, the present waits and goes out right after
// the submission that replays the last of it, at the next present, or after PH_MAX submissions at the latest. Only with
// the replay on the submitting thread, as the event order. NX2NAVI_PRESENT_HELD=0: every present goes out at once.
#define PH_MAX 8
static struct {
    int on;                      // a present waits
    tex_t texture;
    int has_texture, crop[4], has_crop;
    uint64_t at_ns, queued, frame, since;   // since: vk_exec_submit_seq when it began to wait
    uint64_t waited, ran, after, capped, next, logged;   // presents that waited; held work run first at a present;
                                                         // presents out after the held work, at PH_MAX, at the next present
} PH;

static int present_held_on(void) {
    static int on = -1;
    if (on < 0) on = !gfx_replay_async() && rt_knob("PRESENT_HELD", "1");
    return on;
}

static uint64_t submit_sync(const uint64_t *handles, int n);
static uint64_t submit_sync_capture(const uint64_t *handles, int n, VkReplayCaptureBundle *queued,
                                    VkTextureRegistrationScope *registrations);

// The held work that can go on now, in a submission of its own (vk_exec_record syncs it like any other: held_sync).
// The present thread first finishes what it was handed: its empty submission replays held work and frees what it
// replayed (vk_exec_record), which this walk reads. A present ph_flush handed it a moment ago is such a case.
static int held_resume(void) {
    vk_present_drain();
    if (!vk_exec_held() || !vk_exec_held_ready()) return 0;
    submit_sync(NULL, 0);
    PH.ran++;
    return 1;
}

static void ph_flush(uint64_t *why) {
    if (!PH.on) return;
    PH.on = 0, (*why)++;
    if (PH.logged < 16)
        PH.logged++, rt_log("vk: the present of frame %llu went out %llu submission(s) after the game's present call, %d "
                            "held submissions left (%s)", (unsigned long long)PH.frame,
                            (unsigned long long)(vk_exec_submit_seq - PH.since), vk_exec_held(),
                            why == &PH.after ? "the held work was replayed" : why == &PH.capped ? "PH_MAX reached"
                                                                                         : "the next present came");
    present_image(PH.has_texture ? &PH.texture : NULL, PH.has_crop ? PH.crop : NULL, PH.at_ns, PH.queued, PH.frame, 0);
    if (PH.waited % 600 == 0)
        rt_log("vk: presents that waited for held work: %llu (held work run at the present %llu times; out after it %llu, "
               "at %d submissions %llu, at the next present %llu)", (unsigned long long)PH.waited,
               (unsigned long long)PH.ran, (unsigned long long)PH.after, PH_MAX, (unsigned long long)PH.capped,
               (unsigned long long)PH.next);
}

static uint64_t submit_sync_capture(const uint64_t *handles, int n, VkReplayCaptureBundle *queued,
                                    VkTextureRegistrationScope *registrations) {
    RP_SCOPE(rp_request, RP_REQUEST);
    RP_VALUE(RPK_HANDLES, n);
    vk_sync_start();
    if (PH.on && n) {   // a present waits: what can go on goes first, then the present, then this submission
        held_resume();
        if (!vk_exec_held()) ph_flush(&PH.after);
    }
    RP_CALL(RP_PRESENT_DRAIN, vk_present_drain());
    if (!queued) (void)RP_CALL(RP_GATE_WAIT, vk_exec_gate_wait(handles, n)); // a closed gate can pause synchronous replay
    RP_CALL(RP_TEXTURE_SUBMIT, vk_texture_new_submit());
    VkTextureRegistrationScope *prior = vk_exec_texture_registration_select(registrations);
    uint64_t f = queued ? vk_sync_submit_bundle(handles, n, *queued) : vk_sync_submit(handles, n);
    if (queued) *queued = (VkReplayCaptureBundle){0};
    vk_exec_texture_registration_select(prior);
    vk_texture_registration_scope_end(registrations);
    // The submission stopped at a wait whose signal the CPU gave while it was being replayed (vk_exec_xfer.c "events
    // the CPU signals"): it goes on now, in a submission of its own with copies taken after the signal, and that
    // submission's value is the one the game's fence stands for.
    if (n && vk_exec_event_at_sync() && vk_exec_held() && vk_exec_held_ready()) f = submit_sync(NULL, 0);
    if (PH.on && n) {
        if (!vk_exec_held()) ph_flush(&PH.after);
        else if (vk_exec_submit_seq - PH.since >= PH_MAX) ph_flush(&PH.capped);
    }
    return f;
}

static uint64_t submit_sync(const uint64_t *handles, int n) {
    return submit_sync_capture(handles, n, NULL, NULL);
}

int vk_sub_in_frame;

static void replay_main(void *arg) {
    (void)arg;
    int sub_idx = 0;   // submissions since the last present, in queue order
    perf_replay_thread();
    plat_mutex_lock(&rl);
    for (;;) {
        while (!rq_n) plat_cond_wait(&rcv, &rl);
        ReplayItem it = rq[rq_head];
        rq[rq_head] = (ReplayItem){0};
        rq_head = (rq_head + 1) % RQ_CAP;
        rq_n--;
        r_busy = 1;
        plat_cond_broadcast(&rcv);
        plat_mutex_unlock(&rl);
        if (it.kind == RI_SUBMIT) {
            if (vk_sync_stopping() || vk_device_is_lost()) {
                vk_usnap_queue_bundle_cancel(&it.captures);
                vk_texture_registration_scope_end(it.registrations);
            } else {
                vk_sub_in_frame = sub_idx++;
                if (!vk_sub_in_frame) vk_dlssg_render_start(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1);
                uint64_t f = submit_sync_capture(it.h, it.n, &it.captures, it.registrations);
                if (f != it.value && !vk_sync_stopping() && !vk_device_is_lost())
                    rt_fatal("vk: replayed submission signalled %llu, queued as %llu", (unsigned long long)f,
                             (unsigned long long)it.value);
            }
            replay_handles_release(&it);
        } else if (it.kind == RI_RELEASE) {
            vk_present_drain();
            vk_texture_release_replayed(it.tex);
        } else if (!vk_sync_stopping() && !vk_device_is_lost()) {
            uint64_t frame = __atomic_add_fetch(&gfx_counters.frames, 1, __ATOMIC_RELAXED);
            vk_dlssg_render_start(frame);   // the blit can be the first submission of an otherwise empty frame
            present_image(it.tex, it.has_crop ? it.crop : NULL, it.at, it.queued, frame, it.value);
            sub_idx = 0;
        }
        if (it.kind != RI_SUBMIT) free(it.tex);
        plat_mutex_lock(&rl);
        if (it.kind == RI_SUBMIT) {
            r_submit_done = it.submit_no;
            if (it.n) r_registration_done++;
        }
        else r_presents_done++;
        r_busy = 0;
        plat_cond_broadcast(&rcv);
    }
    return;
}

static void wait_counted(int why, uint64_t t0) {   // rl held
    uint64_t d = kern_now_ns() - t0;
    dr_waits[why]++, dr_twaits[why]++;
    dr_ns[why] += d, dr_tns[why] += d;
    perf_end(PERF_DRAIN, t0);
}

static int replay_frontend_begin(void) {
    plat_mutex_lock(&rl);
    int admitted = !vk_sync_stopping() && !vk_device_is_lost();
    if (admitted) r_frontend_calls++;
    plat_mutex_unlock(&rl);
    return admitted;
}

static void replay_frontend_end(void) {
    plat_mutex_lock(&rl);
    r_frontend_calls--;
    plat_cond_broadcast(&rcv);
    plat_mutex_unlock(&rl);
}

// A guest thread waiting for the replay gives its virtual core up, as around any blocking host wait (NVN_BLOCKING):
// guest_read (memcpy, pool invalidate), pool and texture finalize run without it, and a drain lasts up to a replay.
// Called and returning with rl held; rl is not held while the core is taken back.
static int core_away_unlocked(void) {
    plat_mutex_unlock(&rl);
    int away = kern_core_away();
    plat_mutex_lock(&rl);
    return away;
}
static void core_back_unlocked(int *away) {
    if (!*away) return;
    plat_mutex_unlock(&rl);
    kern_core_back(away);
    plat_mutex_lock(&rl);
}

// Queues an item (waits while the queue is full) and returns its value, reserved under rl so values follow queue order.
static uint64_t replay_enqueue(ReplayItem *it) {
    plat_mutex_lock(&rl);
    if (!r_started) {
        if (!plat_thread_spawn(replay_main, NULL, "vk replay", 0)) rt_fatal("vk: cannot start the replay thread");
        r_started = 1;
    }
    // rl is dropped while the core is taken back, and another thread may fill the queue again meanwhile: check again.
    while (rq_n == RQ_CAP) {
        uint64_t p0 = perf_begin();
        dr_calls[DR_FULL]++, dr_tcalls[DR_FULL]++;
        int away = core_away_unlocked();
        while (rq_n == RQ_CAP) plat_cond_wait(&rcv, &rl);
        wait_counted(DR_FULL, p0);
        core_back_unlocked(&away);
    }
    if (it->kind == RI_SUBMIT && (vk_sync_stopping() || vk_device_is_lost())) {
        plat_mutex_unlock(&rl);
        return 0;
    }
    if (it->kind == RI_SUBMIT && it->n && !vk_texture_registration_scope_current(it->registrations)) {
        plat_mutex_unlock(&rl);
        return 0; // This attempt has not promised either capture to the guest.
    }
    it->value = it->kind == RI_RELEASE ? 0 : vk_sync_reserve();   // a release signals nothing
    rq[(rq_head + rq_n) % RQ_CAP] = *it;
    if (it->kind == RI_SUBMIT && it->n) {
        r_registration_issued++;
        r_registration_publication = vk_texture_registration_scope_publication(it->registrations);
    }
    it->h = NULL;
    it->handle_budget = (RcBudgetTicket){0};
    it->captures = (VkReplayCaptureBundle){0};
    it->registrations = NULL;
    rq_n++;
    plat_cond_broadcast(&rcv);
    plat_mutex_unlock(&rl);
    return it->value;
}

// Called with registration publication serialized. Only older admitted submissions can require a wait.
static uint64_t replay_registration_cutoff(uint64_t *publication, uint64_t *recording) {
    plat_mutex_lock(&rl);
    uint64_t cutoff = r_registration_issued > r_registration_done ? r_registration_issued : 0;
    *publication = cutoff ? r_registration_publication : 0;
    *recording = vk_texture_registration_record_revision();
    plat_mutex_unlock(&rl);
    return cutoff;
}

void vk_backend_registration_record_progress(void) {
    plat_mutex_lock(&rl);
    plat_cond_broadcast(&rcv);
    plat_mutex_unlock(&rl);
}

static int replay_registration_pending(uint64_t cutoff, uint64_t recording) {
    return r_registration_done < cutoff || (recording && recording == vk_texture_registration_record_revision());
}

static void replay_registration_wait(uint64_t cutoff, uint64_t recording) {
    plat_mutex_lock(&rl);
    dr_calls[DR_REGISTRATION]++, dr_tcalls[DR_REGISTRATION]++;
    if (replay_registration_pending(cutoff, recording)) {
        uint64_t start = perf_begin();
        int away = core_away_unlocked();
        while (replay_registration_pending(cutoff, recording)) plat_cond_wait(&rcv, &rl);
        wait_counted(DR_REGISTRATION, start);
        core_back_unlocked(&away);
    }
    plat_mutex_unlock(&rl);
}

// Returns once every queued item was replayed and submitted. Back-end entries only: the replay thread never calls them.
static void replay_drain(int why) {
    if (!gfx_replay_async()) return;
    plat_mutex_lock(&rl);
    dr_calls[why]++, dr_tcalls[why]++;
    if (rq_n || r_busy) {
        uint64_t p0 = perf_begin();
        int away = core_away_unlocked();
        while (rq_n || r_busy) plat_cond_wait(&rcv, &rl);
        wait_counted(why, p0);
        core_back_unlocked(&away);
    }
    plat_mutex_unlock(&rl);
}

// What made the game wait for the replay: since the last line (every 1800 game frames), or over the run (at exit).
static void replay_log(uint64_t frame, int total) {
    char buf[1024];
    int k = 0;
    plat_mutex_lock(&rl);
    const uint64_t *c = total ? dr_tcalls : dr_calls, *w = total ? dr_twaits : dr_waits, *t = total ? dr_tns : dr_ns;
    for (int i = 0; i < DR_N && k < (int)sizeof buf - 96; i++)
        if (c[i] || w[i])
            k += snprintf(buf + k, sizeof buf - (size_t)k, "%s %s %llu calls, %llu waited %.1f ms", k ? ";" : "", dr_name[i],
                          (unsigned long long)c[i], (unsigned long long)w[i], t[i] / 1e6);
    if (ck_reads && k < (int)sizeof buf - 128)
        k += snprintf(buf + k, sizeof buf - (size_t)k, "; check (run): %llu reads, %llu differ, %llu without a recorded write, %llu with the queue idle",
                      (unsigned long long)ck_reads, (unsigned long long)ck_diff, (unsigned long long)ck_diff_unexplained,
                      (unsigned long long)ck_diff_idle);
    memset(dr_calls, 0, sizeof dr_calls);
    memset(dr_waits, 0, sizeof dr_waits);
    memset(dr_ns, 0, sizeof dr_ns);
    plat_mutex_unlock(&rl);
    rt_log("vk: replay queue %s frame %llu:%s", total ? "over the run to" : "at", (unsigned long long)frame,
           k ? buf : " no waits");
}

// The run total, once: the game exits through prof_write (boot.c _exit, the window close), probes through
// vk_backend_shutdown.
void gfx_exit_log(void) {
    static int32_t done;
    rec_check_log();
    if (gfx_replay_async() && r_started && !plat_xchg32(&done, 1)) replay_log(game_frames, 1);
}

static void vk_be_present(const tex_t *t, const int crop[4], uint64_t at_ns) {
    if (!replay_frontend_begin()) return;
    motion_owner_present();
    vk_sync_start();
    gfx_tl_present();
    vk_dlssg_game_present();
    if (!gfx_replay_async()) {
        uint64_t frame = __atomic_add_fetch(&gfx_counters.frames, 1, __ATOMIC_RELAXED);
        vk_dlssg_render_start(frame);
        vk_exec_evt_frame(frame);
        perf_present(frame);   // before vk_exec_frame_log, which resets the frame's exec counts the flight recorder reads
        uint64_t queued = kern_now_ns();
        if (present_held_on()) {   // present_held
            vk_present_drain();    // the present thread's empty submission may run held work: done before it is read
            if (PH.on) held_resume(), ph_flush(&PH.next);   // the last present still waits: out now
            if (vk_exec_held()) held_resume();
            if (vk_exec_held()) {
                PH.on = 1, PH.has_texture = t != NULL, PH.has_crop = crop != NULL;
                if (t) PH.texture = *t;
                PH.at_ns = at_ns, PH.queued = queued, PH.frame = frame;
                if (crop) memcpy(PH.crop, crop, sizeof PH.crop);
                PH.since = vk_exec_submit_seq, PH.waited++;
                replay_frontend_end();
                return;
            }
        }
        present_image(t, crop, at_ns, queued, frame, 0);
        replay_frontend_end();
        return;
    }
    // Frame bound: the replay thread handled the previous present before this one is queued.
    plat_mutex_lock(&rl);
    dr_calls[DR_FRAME]++, dr_tcalls[DR_FRAME]++;
    if (r_presents_done < r_presents_queued) {
        uint64_t p0 = perf_begin();
        while (r_presents_done < r_presents_queued) plat_cond_wait(&rcv, &rl);
        wait_counted(DR_FRAME, p0);
    }
    r_presents_queued++;
    plat_mutex_unlock(&rl);
    uint64_t frame = ++game_frames;
    perf_present(frame);
    if (frame % 1800 == 0) replay_log(frame, 0);
    ReplayItem it = {.kind = RI_PRESENT, .tex = replay_copy_texture(t), .at = at_ns, .queued = kern_now_ns()};
    if (crop) {
        it.has_crop = 1;
        for (int i = 0; i < 4; i++) it.crop[i] = crop[i];
    }
    replay_enqueue(&it);
    replay_frontend_end();
}

// vk_pipeline.c drops a released program's pipelines here.
WEAK void vk_program_release(const prog_t *p) { (void)p; }

static int vk_be_pool_init(pool_t *p, const pool_t *setup) {
    vk_sync_start();
    if (p->magic == NVN_POOL_MAGIC) {
        replay_drain(DR_POOL_FINI);
        vk_present_drain();
    }
    return vk_pool_init_setup(p, setup);
}
static void vk_be_pool_fini(pool_t *p) { vk_sync_start(); replay_drain(DR_POOL_FINI); vk_present_drain(); vk_pool_fini(p); }
static int vk_be_pool_map_virtual(pool_t *v, int n, const void *requests) {
    vk_sync_start();
    replay_drain(DR_MAP_VIRTUAL);
    vk_present_drain();
    return vk_pool_map_virtual(v, n, requests);
}
static void vk_be_texture_register(uint64_t pool, int id, const tex_t *t, const view_t *v) {
    if (!replay_frontend_begin()) return;
    vk_sync_start();
    for (;;) {
        plat_mutex_lock(&replay_registration_order);
        uint64_t publication, recording;
        uint64_t cutoff = replay_registration_cutoff(&publication, &recording);
        int published = vk_texture_register_in_pool_ordered(pool, id, t, v, publication);
        plat_mutex_unlock(&replay_registration_order);
        if (published) break;
        replay_registration_wait(cutoff, recording);
    }
    replay_frontend_end();
}
static void vk_be_texture_pool_release(uint64_t pool) {
    if (!replay_frontend_begin()) return;
    for (;;) {
        plat_mutex_lock(&replay_registration_order);
        uint64_t publication, recording;
        uint64_t cutoff = replay_registration_cutoff(&publication, &recording);
        int released = vk_texture_pool_release_ordered(pool, publication);
        plat_mutex_unlock(&replay_registration_order);
        if (released) break;
        replay_registration_wait(cutoff, recording);
    }
    replay_frontend_end();
}
static void vk_be_sampler_register(int id, const samp_par *s) { vk_sync_start(); vk_sampler_register(id, s); }
// Asynchronous: the release is queued behind the submissions before it, so their replay still resolves the texture's
// slots to its image (a released slot follows another live texture at the storage), and the ones after it see it
// released, the order of the synchronous replay. vk_texture_release reads only the object's key fields: a copy.
static void vk_be_texture_release(const tex_t *t) {
    if (!t || !t->mem || !replay_frontend_begin()) return;
    vk_sync_start();
    int asynchronous = gfx_replay_async();
    if (!asynchronous) vk_present_drain();
    ReplayItem it = {.kind = RI_RELEASE, .tex = asynchronous ? replay_copy_texture(t) : NULL};
    for (;;) {
        plat_mutex_lock(&replay_registration_order);
        uint64_t publication, recording;
        uint64_t cutoff = replay_registration_cutoff(&publication, &recording);
        int released = asynchronous ? vk_texture_finalize_registration_ordered(t, publication) :
            vk_texture_release_ordered(t, publication);
        if (released && asynchronous) {
            plat_mutex_lock(&rl);
            dr_calls[DR_TEX_RELEASE]++, dr_tcalls[DR_TEX_RELEASE]++;
            plat_mutex_unlock(&rl);
            replay_enqueue(&it);
        }
        plat_mutex_unlock(&replay_registration_order);
        if (released) break;
        replay_registration_wait(cutoff, recording);
    }
    replay_frontend_end();
}
static void vk_be_program_release(const prog_t *p) { replay_drain(DR_PROG_RELEASE); vk_program_release(p); }

static uint64_t replay_capture(ReplayItem *item, const VkUsnapInputs *inputs, uint64_t max_update) {
    int serialize_registration = 0;
    while (!vk_sync_stopping() && !vk_device_is_lost()) {
        // FIFO work can open the leading event, and another CPU signal can rearm it
        // while prior capture capacity drains. Observe it again before each attempt.
        if (!vk_exec_gate_wait(inputs->handles, inputs->nhandles)) {
            replay_drain(DR_GATE);
            vk_present_drain();
            vk_sync_finish();
            if (vk_sync_stopping() || vk_device_is_lost()) return 0;
            if (!vk_exec_gate_wait(inputs->handles, inputs->nhandles))
                rt_fatal("vk: asynchronous replay cannot capture a submission before its leading event opens");
        }
        if (serialize_registration) plat_mutex_lock(&replay_registration_order);
        if (item->n) item->registrations = vk_texture_registration_scope_begin();
        VkReplayCaptureWait wait;
        VkResult result = vk_usnap_queue_bundle(inputs, max_update, &item->captures, &wait);
        if (result == VK_SUCCESS) {
            if ((item->n && !item->captures.snapshot) || (max_update && !item->captures.scratch))
                rt_fatal("vk: successful queue admission omitted a required capture");
            if (!serialize_registration) plat_mutex_lock(&replay_registration_order);
            uint64_t value = replay_enqueue(item);
            plat_mutex_unlock(&replay_registration_order);
            if (value) return value;
            vk_usnap_queue_bundle_cancel(&item->captures);
            vk_texture_registration_scope_end(item->registrations);
            item->registrations = NULL;
            // A concurrent publication invalidated the provisional combination. Serialize the next
            // capture with publication instead of repeatedly discarding successful native allocations.
            serialize_registration = 1;
            continue;
        }
        if (serialize_registration) plat_mutex_unlock(&replay_registration_order);
        vk_texture_registration_scope_end(item->registrations);
        item->registrations = NULL;
        if (vk_sync_stopping() || vk_device_is_lost()) return 0;
        if (result != VK_NOT_READY) rt_fatal("vk: queuing submission capture failed (%d)", (int)result);
        if (wait.kind == VK_REPLAY_CAPTURE_WAIT_DRAIN_OLDER) {
            replay_drain(DR_CAPTURE);
            vk_present_drain();
            vk_sync_finish();
        } else if (wait.kind == VK_REPLAY_CAPTURE_WAIT_PRODUCER) {
            vk_usnap_wait_producer(wait.producer_epoch, 16);
        } else if (wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET) {
            vk_queue_lock();
            vk_queue_unlock();
            RcBudgetProgress progress;
            resource_core_budget_service(1, &wait.peak, &progress);
            RcBudgetReadiness now;
            resource_core_budget_readiness(1, &now);
            if (!resource_core_budget_peak_progress(&wait.peak, &wait.before, &now))
                resource_core_budget_wait_since(progress.event, 16);
        } else rt_fatal("vk: queuing submission capture omitted its retry cause");
    }
    return 0;
}

static int replay_handles_clone(ReplayItem *item, const uint64_t *handles) {
    if (!item->n) return 1;
    if (item->n < 0 || !handles || (size_t)item->n > SIZE_MAX / sizeof *item->h)
        rt_fatal("vk: invalid replay handle array");
    const size_t bytes = (size_t)item->n * sizeof *item->h;
    RcAdmissionVector peak = {0};
    if (!vk_replay_metadata_peak_add(&peak, bytes))
        rt_fatal("vk: replay handle allocation size overflow");
    while (!vk_sync_stopping() && !vk_device_is_lost()) {
        void *copy;
        VkResult result = vk_replay_metadata_allocate(bytes, 0, RC_BUDGET_SNAPSHOT, &item->handle_budget, &copy);
        if (result == VK_SUCCESS) {
            item->h = copy;
            if (vk_sync_stopping() || vk_device_is_lost()) return 0;
            memcpy(item->h, handles, bytes);
            return 1;
        }
        if (vk_sync_stopping() || vk_device_is_lost()) return 0;
        if (result != VK_NOT_READY) rt_fatal("vk: replay handle allocation failed (%d)", (int)result);
        RcBudgetProgress progress;
        resource_core_budget_service(1, &peak, &progress);
        RcBudgetReadiness now;
        resource_core_budget_readiness(1, &now);
        if (!resource_core_budget_peak_ready(&peak, &now) &&
            !vk_sync_stopping() && !vk_device_is_lost())
            resource_core_budget_wait_since(progress.event, 16);
    }
    return 0;
}

// Asynchronous: the handles (nvn_submit_prepare_async ran under the front end's submit lock) are copied and queued;
// the value returned is the one the replay will signal.
static uint64_t replay_submit(const uint64_t *handles, int n) {
    if (!gfx_replay_async()) {
        vk_sub_in_frame = __atomic_load_n(&nvn_submits_in_frame, __ATOMIC_RELAXED);   // the front end counts it up after the submit returns
        if (!vk_sub_in_frame) vk_dlssg_render_start(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1);
        return submit_sync(handles, n);
    }
    vk_sync_start();
    if (n > 0 && !vk_sync_prepare_draw()) return 0;
    ReplayItem it = {.kind = RI_SUBMIT, .n = n, .submit_no = nvn_submit_count()};
    if (!replay_handles_clone(&it, handles)) {
        replay_handles_release(&it);
        return 0;
    }
    VkUsnapBindingState next_uniforms = r_uniforms;
    uint64_t max_update;
    if (!nvn_submission_ordered_scratch_max(it.h, n, &max_update)) {
        replay_handles_release(&it);
        rt_fatal("vk: asynchronous replay cannot determine the retained command scratch requirement");
    }
    if (!vk_usnap_project(it.h, n, &next_uniforms)) {
        replay_handles_release(&it);
        rt_fatal("vk: asynchronous replay cannot determine the inherited uniform bindings");
    }
    VkUsnapInputs inputs = {.handles = it.h, .nhandles = n};
    if (n) {
        inputs.inherited = r_uniforms.slots;
        inputs.ninherited = sizeof r_uniforms.slots / sizeof r_uniforms.slots[0];
    }
    uint64_t value = replay_capture(&it, &inputs, max_update);
    if (value) r_uniforms = next_uniforms;
    else {
        vk_usnap_queue_bundle_cancel(&it.captures);
        vk_texture_registration_scope_end(it.registrations);
        replay_handles_release(&it);
    }
    return value;
}

static uint64_t vk_be_submit(const uint64_t *handles, int n) {
    if (!replay_frontend_begin()) return 0;
    uint64_t value = replay_submit(handles, n);
    replay_frontend_end();
    return value;
}
static void vk_be_finish(void) { vk_sync_start(); replay_drain(DR_FINISH); vk_present_drain(); vk_sync_finish(); }
static int replay_check(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("NX2NAVI_REPLAY_CHECK");
        on = e && strcmp(e, "0");
        if (on) rt_log("vk: NX2NAVI_REPLAY_CHECK: every ReadTexels is read again after a full replay drain and compared");
    }
    return on;
}

// Ordered projected reads need every older asynchronous item to declare its potential writes. Legacy
// ReadTexels with the replay queued waits for the newest submission recorded to write the texture (replayed and
// completed), not for the whole queue. The readback copy goes through the upload batch on the same queue, after every
// submission the replay has submitted. Submissions still queued or being recorded run after the copy, as on hardware
// when the GPU has not reached them yet: the game only sees their texels after waiting on a sync, and every sync
// value covers replay and GPU, so a read the game synchronised sees them. NX2NAVI_REPLAY_CHECK compares with a full drain.
static void vk_be_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                              uint64_t img) {
    vk_sync_start();
    const int async = gfx_replay_async();
    const int candidate = vk_texture_cpu_read_candidate(t, v);
    // Preserve the descriptor and requested box across the asynchronous declaration drain.
    tex_t texture;
    view_t view;
    NVNcopyRegion box;
    if (candidate) {
        texture = *t;
        t = &texture;
        if (v) { view = *v; v = &view; }
        box = *r;
        r = &box;
    }
    uint64_t selected_fence = 0;
    VkTextureCpuReadResult selected = VK_TEXTURE_CPU_READ_OUTSIDE;
    int idle = 0;
    if (async) {
        plat_mutex_lock(&rl);
        idle = !rq_n && !r_busy;
        plat_mutex_unlock(&rl);
        if (candidate) replay_drain(DR_READ_TEXELS);
    }
    if (candidate) {
        selected = vk_texture_cpu_read_texels(t, v, r, dst, row, img, !vk_exec_held(), &selected_fence);
        if (selected == VK_TEXTURE_CPU_READ_STOPPED) return;
        if (selected == VK_TEXTURE_CPU_READ_REFUSED) {
            rt_fatal("vk: ReadTexels could not complete its required ordered texture read");
            return;
        }
    }
    if (selected == VK_TEXTURE_CPU_READ_OUTSIDE) {
        if (async) {
            const uint64_t f = vk_texture_last_write(t);
            if (!candidate) {
                plat_mutex_lock(&rl);
                dr_calls[DR_READ_TEXELS]++, dr_tcalls[DR_READ_TEXELS]++;
                plat_mutex_unlock(&rl);
            }
            if (f && vk_sync_fence_done() < f) {
                const uint64_t t0 = kern_now_ns();
                vk_sync_fence_wait(f, UINT64_MAX);
                if (!candidate) {
                    const uint64_t elapsed = kern_now_ns() - t0;
                    plat_mutex_lock(&rl);
                    dr_waits[DR_READ_TEXELS]++, dr_twaits[DR_READ_TEXELS]++;
                    dr_ns[DR_READ_TEXELS] += elapsed, dr_tns[DR_READ_TEXELS] += elapsed;
                    plat_mutex_unlock(&rl);
                }
            }
        }
        vk_read_texels(t, v, r, dst, row, img);
    }
    if (!async || !replay_check()) return;
    size_t n = vk_texture_texel_span(t, v, r, row, img);
    uint8_t *again = n ? malloc(n) : NULL;
    if (!again) return;
    memcpy(again, dst, n);   // bytes between the rows stay the caller's
    uint64_t f0 = selected == VK_TEXTURE_CPU_READ_READY ? selected_fence : vk_texture_last_write(t);
    replay_drain(DR_CHECK);
    uint64_t f1;
    if (selected == VK_TEXTURE_CPU_READ_READY) {
        if (vk_texture_cpu_read_texels(t, v, r, again, row, img, !vk_exec_held(), &f1) != VK_TEXTURE_CPU_READ_READY) {
            free(again);
            return;
        }
    } else {
        f1 = vk_texture_last_write(t);
        vk_read_texels(t, v, r, again, row, img);
    }
    int differ = memcmp(again, dst, n) != 0;
    static volatile int32_t n_logged;
    if (differ && plat_inc32(&n_logged) <= 40) {
        size_t at = 0;
        while (at < n && again[at] == dst[at]) at++;
        rt_log("vk: replay check: ReadTexels of %p (format %#x, %dx%dx%d at %d,%d,%d) differs at byte %zu of %zu; writer "
               "fence %llu at the read, %llu after the drain, done %llu; queue %s at the read",
               (void *)t->mem, t->d.format, r->width, r->height, r->depth, r->xoffset, r->yoffset, r->zoffset, at, n,
               (unsigned long long)f0, (unsigned long long)f1, (unsigned long long)vk_sync_fence_done(), idle ? "idle" : "busy");
    }
    plat_mutex_lock(&rl);
    ck_reads++;
    if (differ) {
        ck_diff++;
        if (f1 == f0) ck_diff_unexplained++;
        if (idle) ck_diff_idle++;
    }
    plat_mutex_unlock(&rl);
    free(again);
}
static void vk_be_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row,
                               uint64_t img) {
    vk_sync_start();
    replay_drain(DR_WRITE_TEXELS);
    vk_present_drain();
    vk_write_texels(t, v, r, src, row, img);
}
static void vk_be_write_packaged_texture(const tex_t *t, const void *src) {
    vk_sync_start();
    vk_write_packaged_texture(t, src);
}
static void vk_be_guest_written(uint64_t addr, uint64_t size) { vk_sync_start(); vk_guest_written(addr, size); }
// A CPU read of GPU-written mirrored storage (memcpy, pool invalidate). With the replay queued it waits for the newest
// submission recorded to write the textures there, not for the whole queue: the write-back copy then runs after their
// renders, and clears gpu_dirty only for content that is on the queue. Submissions not replayed yet render after the
// copy, as on hardware when the GPU has not reached them; they mark the texture dirty again for a later read.
static void vk_be_guest_read(uint64_t addr, uint64_t size) {
    if (!vk_texture_cpu_stale(addr, size)) return;
    vk_sync_start();
    if (gfx_replay_async()) {
        uint64_t f = vk_texture_dirty_fence(addr, size);
        plat_mutex_lock(&rl);
        dr_calls[DR_GUEST_READ]++, dr_tcalls[DR_GUEST_READ]++;
        plat_mutex_unlock(&rl);
        if (f && vk_sync_fence_done() < f) {
            uint64_t t0 = kern_now_ns();
            NVN_BLOCKING();
            vk_sync_fence_wait(f, UINT64_MAX);
            uint64_t d = kern_now_ns() - t0;
            plat_mutex_lock(&rl);
            dr_waits[DR_GUEST_READ]++, dr_twaits[DR_GUEST_READ]++;
            dr_ns[DR_GUEST_READ] += d, dr_tns[DR_GUEST_READ] += d;
            plat_mutex_unlock(&rl);
        }
    }
    vk_present_drain();
    vk_texture_write_back(addr, size);
}
// After a fence wait: the value is done, so its replay is too; a later submission the replay records meanwhile marks
// its targets with a later fence, which write_back_small leaves alone. No drain.
// The count of such waits that came back while event-ordered work was still held (vk_exec.c): a submission stopped at an
// event wait returns its value with its remainder held for a later replay, so the game may take a wait for that value
// as the whole submission done and reuse what the held part still reads. How often that happens decides whether the
// value should be signalled only after the remainder (audit finding F-2-05). The count is unlocked: a diagnostic.
static uint64_t synced_n, synced_held;
static void synced_held_count(uint64_t fence) {
    synced_n++;
    int held = vk_exec_held();
    if (!held) return;
    synced_held++;
    if (synced_held <= 10 || synced_held % 600 == 0)
        rt_log("vk: a game's wait for %llu returned with %d event-ordered submission(s) still held (%llu of %llu such "
               "waits so far)", (unsigned long long)fence, held, (unsigned long long)synced_held,
               (unsigned long long)synced_n);
}

static void vk_be_cpu_synced(uint64_t fence) {
    synced_held_count(fence);
    if (!vk_texture_small_dirty()) return;
    vk_sync_start();
    vk_present_drain();
    vk_texture_write_back_small(fence);
}

const GfxBackend gfx_vulkan_backend = {
    .name = "vulkan",
    .pool_init = vk_be_pool_init,
    .pool_fini = vk_be_pool_fini,
    .pool_map_virtual = vk_be_pool_map_virtual,
    .texture_register = vk_be_texture_register,
    .texture_pool_release = vk_be_texture_pool_release,
    .sampler_register = vk_be_sampler_register,
    .texture_release = vk_be_texture_release,
    .program_release = vk_be_program_release,
    .submit = vk_be_submit,
    .fence_wait = vk_sync_fence_wait,
    .exec_wait = vk_be_exec_wait,
    .fence_done = vk_sync_fence_done,
    .present = vk_be_present,
    .frame_start = vk_dlssg_frame_start,
    .finish = vk_be_finish,
    .read_texels = vk_be_read_texels,
    .write_texels = vk_be_write_texels,
    .write_packaged_texture = vk_be_write_packaged_texture,
    .guest_written = vk_be_guest_written,
    .guest_read = vk_be_guest_read,
    .cpu_synced = vk_be_cpu_synced,
};

void vk_backend_shutdown(void) {
    // Unblock required preparation before joining replay; queued, unissued work is canceled explicitly.
    vk_sync_stop_submissions();
    vk_pipeline_stop_preparation();
    plat_mutex_lock(&rl);
    while (r_frontend_calls) plat_cond_wait(&rcv, &rl);
    plat_mutex_unlock(&rl);
    vk_exec_draw_mfb_stop();
    replay_drain(DR_FINISH);
    vk_exec_discard_held();
    gfx_exit_log();
    vk_present_drain();
    vk_gpu_passes_stop_capture();
    vk_sync_shutdown();
    if (vk_device_is_lost()) return; // no proof that possibly executing allocations can be destroyed
    vk_pipeline_shutdown();
    if (vk_device_is_lost()) return;
    gfx_window_release_vk();
    if (vk_device_is_lost()) return;
    VkResult mesh_idle = vk_meshfb_shutdown();
    if (mesh_idle != VK_SUCCESS) {
        vk_device_lost_check(mesh_idle, "draining mesh fallback resources");
        rt_fatal("vk: cannot retire mesh fallback resources without idle proof (%d)", (int)mesh_idle);
    }
    vk_exec_draw_mfb_shutdown();
    vk_exec_draw_native_mesh_shutdown();
    vk_dlaa_size_shutdown();
    vk_dlaa_hud_shutdown();
    vk_usnap_shutdown();
    vk_exec_draw_scratch_shutdown();
    vk_exec_draw_descriptor_shutdown();
    vk_exec_draw_bindless_shutdown();
    vk_exec_draw_fallback_shutdown();
    vk_exec_draw_fixed_shutdown();
    vk_mem_shutdown();
    vk_shutdown();
}

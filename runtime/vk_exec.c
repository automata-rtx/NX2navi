// The op-stream replay core of the Vulkan back end: walks the command handles of
// one submission into its command buffer, keeps render targets and fixed-function state, opens and closes dynamic
// rendering passes and records clears and barriers. Draws, binds and dispatches go to vk_exec_op_draw
// (vk_exec_draw.c); copies, counters, events and conditional rendering to vk_exec_op_xfer (vk_exec_xfer.c). Images
// stay in VK_IMAGE_LAYOUT_GENERAL (vk_texture.c); full memory barriers order passes, clears and whatever those two
// record between them.
//
// The op loop handles END / BEGIN / JUMP / CALL and the pure-state ops, applies the state defaults, keeps render targets
// until they change and skips clears while rendering is disabled. A pass has GENERAL-layout attachments, LOAD/STORE and
// a render area from the smallest attachment, with a full barrier before it.
// Contains code Copyright (c) 2026 jhaizhou, Copyright (c) 2026 Peiikai (MIT; see THIRD_PARTY_NOTICES.md).
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "platform/platform.h"
#include "prof.h"
#include "resource_prep.h"
#include "resource_image_write.h"
#include "resource_command_effects.h"
#include "resource_core_vma.h"
#include "tex.h"
#include "vk.h"
#include "replay_phase_trace.h"
#include "replay_work_trace.h"
#include "vk_exec.h"
#include "vk_mesh_fb.h"
#include "vk_gpa.h"
#include "vk_target_frames.h"
#include "vk_dlaa.h"

#define WEAK __attribute__((weak))
#define CALL_DEPTH 8

typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    VkImageAspectFlags aspect;
    uint32_t level, base, layers;
    VkImageView view;
} ViewEnt;

static VkExec X;
static int x_ready;
static void (*probe_pass_record)(VkExec *, void *);
static void *probe_pass_context;
static const void *probe_mesh_partial;

void vk_exec_probe_mesh_partial(const void *payload) {
    __atomic_store_n(&probe_mesh_partial, payload, __ATOMIC_RELEASE);
}

int vk_exec_probe_mesh_partial_allowed(const void *payload) {
    return payload && __atomic_load_n(&probe_mesh_partial, __ATOMIC_ACQUIRE) == payload;
}

void vk_exec_probe_pass_arm(void (*record)(VkExec *, void *), void *context) {
    if (!record || __atomic_load_n(&probe_pass_record, __ATOMIC_ACQUIRE))
        rt_fatal("vk: ordered graphics pass probe already armed");
    probe_pass_context = context;
    __atomic_store_n(&probe_pass_record, record, __ATOMIC_RELEASE);
}

static void resource_target_prepare(const VkExecTarget *t) {
    if (t->image && t->img && t->tex)
        vk_texture_image_write_prepare(t->image, t->img->addr, t->tex->L.size);
}

static void resource_target_written(const VkExecTarget *t) {
    if (!t->image || !t->img || t->image != t->img->image) return;
    VkImage image = t->image;
    uint64_t addr = t->img->addr;
    if (!vk_texture_render_target_written(image, addr)) resource_image_written((uint64_t)(uintptr_t)image, addr);
}

static void resource_target_capture(VkExec *ex, unsigned slot, const VkExecTarget *t) {
    if (!t->image || !t->img || !t->tex || t->image != t->img->image) return;
    RcRegistryImageBinding binding = {0};
    RcRegistryImageWriteExtent *extents = NULL;
    uint32_t count = 0;
    uint64_t addr = t->img->addr, size = t->tex->L.size;
    if (!vk_texture_image_binding(t->image, addr, size, &binding, &extents, &count))
        binding = (RcRegistryImageBinding){.image = (uint64_t)(uintptr_t)t->image};
    ex->target_bindings[slot] = (VkExecTargetBinding){.binding = binding, .range = {1, addr, size},
        .extents = extents, .extent_count = count};
    ex->target_binding_mask |= 1u << slot;
}

static void resource_target_pin(VkExec *ex, unsigned slot, const VkExecTarget *t) {
    if (!t->img || t->image != t->img->image) return;
    VkPinnedTextureWrite *capture = &ex->target_bindings[slot].pinned;
    vk_texture_pinned_write_capture(t->tex, t->image, capture);
    if (capture->pin) {
        ex->target_binding_mask |= 1u << slot;
        vk_texture_pinned_write_use(capture);
    }
}

static VkPinnedTextureWrite *resource_target_pinned(VkExec *ex, unsigned slot) {
    VkPinnedTextureWrite *capture = &ex->target_bindings[slot].pinned;
    return (ex->target_binding_mask & (1u << slot)) && capture->pin ? capture : NULL;
}

static void resource_targets_forget(VkExec *ex) {
    for (uint32_t bits = ex->target_binding_mask | ex->bc_target_mask; bits; bits &= bits - 1) {
        VkExecTargetBinding *target = &ex->target_bindings[__builtin_ctz(bits)];
        vk_texture_pinned_write_release(&target->pinned);
        vk_texture_alias_read_cancel(target->load_seed);
        free(target->extents);
        *target = (VkExecTargetBinding){0};
    }
    ex->target_binding_mask = 0;
    ex->bc_target_mask = ex->bc_written_mask = 0;
}

static const VkImageSubresourceRange *attachment_load_planes(const VkExec *ex, const VkExecTarget *target,
    unsigned slot, VkImageSubresourceRange *range) {
    const VkImageAspectFlags aspect = slot == 8 ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    if (!target->img || target->image != target->img->image || target->aspect != aspect ||
        target->img->aspect != aspect || target->img->scale > 1 || target->is3d || target->blocks || target->bc_backing ||
        (slot == 8 && target->format != VK_FORMAT_D16_UNORM && target->format != VK_FORMAT_D32_SFLOAT) ||
        target->format != target->img->format || target->samples != 1 || !ex->pass_layers ||
        ex->pass_layers > target->layers || target->level >= target->img->levels ||
        target->base_layer >= target->img->layers || ex->pass_layers > target->img->layers - target->base_layer) return NULL;
    *range = (VkImageSubresourceRange){target->aspect, target->level, 1, target->base_layer, ex->pass_layers};
    return range;
}

static void resource_attachment_prepare(VkExec *ex, unsigned slot) {
    VkPinnedTextureWrite *capture = resource_target_pinned(ex, slot);
    if (capture) {
        vk_texture_pinned_write_prepare(capture);
        return;
    }
    VkExecTargetBinding *binding = &ex->target_bindings[slot];
    binding->loaded_planes = (VkImageSubresourceRange){0};
    const VkImage image = (VkImage)(uintptr_t)binding->binding.image;
    const VkExecTarget *selected = slot == 8 ? &ex->depth : &ex->color[slot];
    VkImageSubresourceRange range;
    const VkImageSubresourceRange *planes = selected->image == image && !ex->tonemap_extent.width ?
        attachment_load_planes(ex, selected, slot, &range) : NULL;
    if (vk_texture_alias_target_replacement(binding->load_seed)) {
        if (vk_texture_alias_target_prepare_replacement(binding->load_seed, planes)) binding->loaded_planes = *planes;
        else
            vk_exec_required_graphics_unavailable(ex, 0, "mixed LOAD destination changed before issue");
        return;
    }
    // An exact LOAD/STORE preserves untouched texels in the native attachment planes.
    if (planes) {
        if (vk_texture_alias_target_load_current(binding->load_seed, image, planes) ||
            vk_texture_render_target_load_current(image, binding->range.offset, planes)) binding->loaded_planes = *planes;
        else if (selected->img->preserve_pool_bytes) {
            vk_exec_required_graphics_unavailable(ex, 0, "attachment LOAD contents changed before issue");
            return;
        }
    }
    vk_texture_image_write_prepare(image, binding->range.offset, binding->range.size);
}

static int resource_attachment_written(VkExec *ex, unsigned slot) {
    VkPinnedTextureWrite *capture = resource_target_pinned(ex, slot);
    if (capture) {
        vk_texture_pinned_write_recorded(capture);
        return 0;
    }
    const VkExecTargetBinding *binding = &ex->target_bindings[slot];
    if (vk_texture_alias_target_replacement(binding->load_seed)) {
        vk_texture_alias_target_record_replacement(binding->load_seed, &binding->loaded_planes);
        return 1;
    }
    const VkImage image = (VkImage)(uintptr_t)binding->binding.image;
    int written = binding->loaded_planes.aspectMask ? vk_texture_render_target_loaded_written(image, binding->range.offset,
        binding->load_seed, &binding->loaded_planes) : vk_texture_render_target_written(image, binding->range.offset);
    if (!written) {
        resource_image_written(binding->binding.image, binding->range.offset);
        resource_effects_image_record_extents(binding->binding.image, binding->extents, binding->extent_count);
    }
    return written == 2;
}

void vk_exec_targets_prepare(VkExec *ex) {
    if (!ex->in_pass) return;
    for (uint32_t bits = ex->target_binding_mask; bits; bits &= bits - 1)
        resource_attachment_prepare(ex, (unsigned)__builtin_ctz(bits));
}

void vk_exec_targets_written(VkExec *ex) {
    if (!ex->in_pass) return;
    ex->bc_written_mask |= ex->bc_target_mask;
    // Reuse the pass's selected images and holdings when recording the actual guest output.
    for (uint32_t bits = ex->target_binding_mask; bits; bits &= bits - 1)
        resource_attachment_written(ex, (unsigned)__builtin_ctz(bits));
}
// The replay state this thread records with now (inside vk_exec_record, or a test's vk_exec_probe_inline), NULL when
// it records none: vk_exec_inline_begin hands out its command buffer.
static _Thread_local VkExec *x_cur;
// The serialized recorder and its workers jointly own the active native tail.
enum { RECORDING_RESOURCE_UNAVAILABLE = 1, RECORDING_CANCELLED };
static int recording_failed;

int vk_exec_recording_failed(void) {
    return __atomic_load_n(&recording_failed, __ATOMIC_ACQUIRE) != 0;
}

int vk_exec_recording_missing_resource(void) {
    return __atomic_load_n(&recording_failed, __ATOMIC_ACQUIRE) == RECORDING_RESOURCE_UNAVAILABLE;
}

static void selected_sampled_failure_log(const VkExecSampledFailure *, uint32_t opcode, const char *path);

static void fail_texture_read(VkExec *ex, int logical, const char *reason, uint32_t opcode,
                              const VkExecSampledFailure *sampled) {
    if (__atomic_exchange_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_ACQ_REL)) return;
    if (sampled) selected_sampled_failure_log(sampled, opcode, "recording-bind");
    int bindless = logical < 0;
    if (bindless) logical = -logical - 1;
    rt_log("vk: required %s %s (pool %llu, slot %d, program %llu, submission %llu, operation %llu)",
           bindless ? "bindless texture" : "texture", reason, (unsigned long long)ex->tex_pool, logical,
           (unsigned long long)(ex->prog ? ex->prog->serial : 0),
           (unsigned long long)ex->fence, (unsigned long long)ex->resource_op);
    vk_texture_registration_report_unavailable(logical);
}

void vk_exec_registration_gone(VkExec *ex, int logical) {
    fail_texture_read(ex, logical, "history is unavailable", 0, NULL);
}

void vk_exec_sample_source_refused(VkExec *ex, int logical) {
    fail_texture_read(ex, logical, "source read was refused", 0, NULL);
}

void vk_exec_sampled_binding_refused(VkExec *ex, int logical, uint32_t opcode, const char *reason,
                                     const VkExecSampledFailure *sampled) {
    if (reason) {
        if (__atomic_exchange_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_ACQ_REL)) return;
        selected_sampled_failure_log(sampled, opcode, "recording-bind");
        uint64_t operation = ex->resource_op ? ex->resource_op : ex->resource_next + 1;
        rt_log("vk: required graphics preparation failed: %s (op %u, submission %llu, operation %llu)", reason, opcode,
               (unsigned long long)ex->fence, (unsigned long long)operation);
    } else {
        fail_texture_read(ex, logical, "source read was refused", opcode, sampled);
    }
}

void vk_exec_required_temporal_unavailable(VkExec *ex, const char *reason) {
    if (__atomic_exchange_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_ACQ_REL)) return;
    rt_log("vk: required reconstruction resources: %s (submission %llu, operation %llu)", reason,
           (unsigned long long)ex->fence, (unsigned long long)ex->resource_op);
}

static void cancel_recording(void) {
    int active = 0;
    __atomic_compare_exchange_n(&recording_failed, &active, RECORDING_CANCELLED, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

void vk_exec_cancel_temporal_preparation(void) { cancel_recording(); }

void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *reason) {
    if (__atomic_exchange_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_ACQ_REL)) return;
    uint64_t operation = ex->resource_op ? ex->resource_op : ex->resource_next + 1;
    rt_log("vk: required graphics preparation failed: %s (op %u, submission %llu, operation %llu)", reason, opcode,
           (unsigned long long)ex->fence, (unsigned long long)operation);
}

// ---- parallel groups (NX2NAVI_REPLAY_WORKERS; "replay workers" below). A chunk: a part of a group's ops that one
// thread records into a command buffer of its own, from the replay state the front pass left at its first op.
#define GROUP_CHUNKS 17                 // chunks a group is cut into at most: RW_MAX workers and the submitting thread
#define LIVE_GROUPS 4                   // groups handed out whose command buffers are not in their part yet, at most
typedef struct Group Group;
typedef struct {
    void (*run)(void *, unsigned);
    void *arg;
    int done;
} ChunkPair;
typedef struct {
    VkExec ex;                          // the state at its first op, then the state it records with
    VkExecPoolView pool;                // private to this recording chunk, unlike the immutable group publication
    const op_hdr *h;                    // its first op
    NvnRecordPlan plan;                 // immutable executable metadata under the command copy's replay lifetime
    uint64_t resource_begin;            // first logical operation, independent of repeated command addresses
    uint32_t nops;                      // the ops it records (JUMPs not counted)
    uint32_t cond0, ncond;              // its render conditions' decisions in its group's cond
    uint64_t oq;                        // the occlusion counting at its first op (vk_exec_xfer_chunk_state)
    uint32_t slot0, nslots;             // its block of occlusion query slots (vk_exec_xfer_slots)
    uint32_t brk;                       // its ops that are not state or a draw (vk_exec_brk_now)
    int idx;                            // in its group: 0 runs the group's pass hooks
    Group *g;                           // its group
    VkCommandBuffer cb;
    int recorded;
    volatile int32_t done;
} Chunk;
struct Group {
    ChunkPair *pair;                    // optional independent memory query offered to a waiting recorder
#if NX2NAVI_REPLAY_WORK
    uint64_t work_replay;
#endif
    struct GroupOp {
        void *effect;
        VkExecGraphicsDraw *graphics;
        int noop;
    } *ops;                            // per-operation native selections and effects, owned through worker recording
    uint32_t neffects, ceffects;
    uint64_t resource_begin;
    Chunk c[GROUP_CHUNKS];
    VkExecPoolView pool;                // one completed pool state, published before peers may consume it
    VkExecPoolRequirement requirement;
    int n;
    uint8_t *cond;                      // the decisions of the group's render conditions, in stream order
    uint32_t ncond, ccond;
    volatile int32_t hooks;             // chunk 0 began its first pass (or ended): the others may start
    volatile int32_t ndone;             // its chunks done
    volatile uint64_t seq;              // its number: Q.out when it was handed out (vk_exec_stream_pos)
    uint32_t upd;                       // the submission's inline uniform updates before it (vk_exec_upd_top)
    int at;                             // its chunks' places in the part's list (vk_sync_append_later)
    uint64_t out_ns;                    // when it was handed out
};
// Group number s is GQ[s % LIVE_GROUPS]. Its chunks wait in Q in the order of the stream once it is handed out; the
// workers (and the submitting thread, while it waits) take them in that order.
static Group GQ[LIVE_GROUPS];

static _Thread_local VkTextureRegistrationScope *incoming_registration_scope;
VkTextureRegistrationScope *vk_exec_texture_registration_select(VkTextureRegistrationScope *scope) {
    VkTextureRegistrationScope *prior = incoming_registration_scope;
    incoming_registration_scope = scope;
    return prior;
}
VkTextureRegistrationScope *vk_exec_texture_registration_scope(void) {
    return x_cur ? x_cur->registration_scope : incoming_registration_scope;
}
uint64_t vk_exec_texture_pool(void) { return x_cur ? x_cur->tex_pool : 0; }

RcRegistryPosition vk_exec_resource_position(void) {
    return x_cur && x_cur->resource_op ?
        (RcRegistryPosition){x_cur->fence, x_cur->resource_op * 2} : (RcRegistryPosition){0};
}
static void resource_seal(VkExec *ex) {
    if (rc_registry_declare_through(rc_registry_runtime(),
        (RcRegistryPosition){ex->fence, ex->resource_next * 2 + 1}) != RC_REGISTRY_OK)
        rt_fatal("vk: resource declaration prefix could not be sealed");
}
static void resource_assign(VkExec *ex, uint32_t count) {
    if (!count || ex->resource_next > (UINT64_MAX - 1) / 2 - count)
        rt_fatal("vk: resource operation identity exhausted");
    ex->resource_op = ex->resource_next + 1;
    ex->resource_next += count;
}
static void *resource_declare(VkExec *ex, const op_hdr *op, uint32_t count, VkExecDispatch *compat_dispatch) {
    RWT_CHAIN(work_declare, RWT_REGION_EFFECT_DECLARE);
    resource_assign(ex, count);
    // Effect inputs belong before this operation's outputs. Prior operations are already completely declared.
    if (rc_registry_declare_through(rc_registry_runtime(),
        (RcRegistryPosition){ex->fence, ex->resource_op * 2}) != RC_REGISTRY_OK)
        rt_fatal("vk: resource input prefix could not be sealed");
    void *effect = vk_exec_effects_declare(ex, op->op, nvn_op_payload(op),
        (RcRegistryPosition){ex->fence, ex->resource_op * 2 + 1}, compat_dispatch);
    resource_seal(ex);
    return effect;
}
static void *resource_declare_texture_write(VkExec *ex, uint32_t opcode, const VkTextureWriteTarget *target) {
    resource_assign(ex, 1);
    void *effect = vk_exec_effects_declare_texture_write(ex, opcode, target,
        (RcRegistryPosition){ex->fence, ex->resource_op * 2 + 1});
    resource_seal(ex);
    return effect;
}
static void *resource_declare_projected_transfer(VkExec *ex, VkExecProjectedTransfer *transfer) {
    resource_assign(ex, 1);
    const RcRegistryPosition output = {ex->fence, ex->resource_op * 2 + 1};
    void *effect = vk_exec_effects_declare_projected_transfer(ex, transfer, output);
    if (rc_registry_declare_through(rc_registry_runtime(), output) != RC_REGISTRY_OK) {
        vk_exec_effects_release(effect);
        vk_exec_projected_transfer_cancel(transfer);
        rt_fatal("vk: projected transfer declaration prefix could not be sealed");
    }
    return effect;
}

static void resource_declare_noop(VkExec *ex) {
    resource_assign(ex, 1);
    resource_seal(ex);
}

static void resource_group_append(Group *g, VkExec *ex, const op_hdr *op, uint32_t count, VkExecGraphicsDraw *draw,
                                  int noop) {
    if (count > UINT32_MAX - g->neffects) rt_fatal("vk: resource declaration group overflow");
    uint32_t needed = g->neffects + count;
    if (needed > g->ceffects) {
        uint32_t cap = needed > UINT32_MAX / 2 ? needed : needed * 2;
        struct GroupOp *items = realloc(g->ops, (size_t)cap * sizeof *items);
        if (!items) rt_fatal("vk: resource declaration storage unavailable");
        g->ops = items, g->ceffects = cap;
    }
    memset(g->ops + g->neffects, 0, count * sizeof *g->ops);
    VkExecGraphicsDraw *prior = vk_exec_graphics_draw_enter(draw);
    void *effect = NULL;
    if (noop) resource_declare_noop(ex);
    else effect = resource_declare(ex, op, count, NULL);
    g->ops[g->neffects] = (struct GroupOp){effect, draw, noop};
    vk_exec_graphics_draw_enter(prior);
    g->neffects = needed;
}

#define QN (LIVE_GROUPS * GROUP_CHUNKS)
static struct {
    Chunk *q[QN];                       // chunk number i of those handed out is q[i % QN]
    volatile uint64_t put, take;        // chunks handed out, taken
    volatile uint64_t done;             // every group up to this number has all its chunks done
    uint64_t out, drained;              // the submitting thread: groups handed out, and put into their parts
} Q;
static PlatMutex done_lock = PLAT_MUTEX_INIT;   // Q.done moves on (groups_done_advance)
static _Thread_local Chunk *rw_chunk;   // the chunk the calling thread records, NULL outside one
static _Thread_local Chunk *temporal_hook_chunk; // first chunk recorded before any of its peers are published
// The submitting thread records on past groups whose chunks may not all be done yet (NX2NAVI_REPLAY_PIPE).
static _Thread_local int rw_behind;
// Why it waited for them (RW.pw_*): room for the next group, an op chunks never record, a pass's hooks, an effect that
// waits for its turn (vk_exec_turn), a texture entry, a part split, the submission's end; and the waits that found
// every group done already.
enum { PW_NEXT, PW_OP, PW_HOOKS, PW_TURN, PW_TEX, PW_SPLIT, PW_END, PW_DONE, PW_N };
static void group_wait(int why, int help);
static void group_drain(VkExec *ex, int why);
static _Thread_local const op_hdr *rw_op;   // in a chunk: the op it records (vk_exec_mesh_run reads on from it)
static _Thread_local int rw_ordered;    // every earlier chunk of its group is done
static PlatMutex tfr_lock = PLAT_MUTEX_INIT;   // target frames (vk_target_frames.h) noted by several chunks at once

// ---- NX2NAVI_TRACE=gfx frame counters

VkExecFrame vk_exec_frame;
static struct { const uint8_t *mem; uint32_t w, h, fmt, n; } fw[64];   // textures written this frame
static int nfw;

uint64_t vk_exec_recording_fence(void) { return x_cur ? x_cur->fence : 0; }

void vk_exec_frame_written(const tex_t *t) {
    if (!(rt_trace & RT_TRACE_GFX) || !t) return;
    for (int i = 0; i < nfw; i++)
        if (fw[i].mem == t->mem) { fw[i].n++; return; }
    if (nfw == 64) return;
    fw[nfw].mem = t->mem, fw[nfw].w = t->d.width, fw[nfw].h = t->d.height, fw[nfw].fmt = (uint32_t)t->d.format;
    fw[nfw++].n = 1;
}

// ---- draw runs (NX2NAVI_REPLAY_RUNS=1, diagnostics for recording a submission on several threads)
// How the draws of a frame fall into runs that a recording thread of their own could take: a run is the draws between
// two ops that have to stay in the replay's order on one thread (dispatches, copies, texture and buffer clears, events,
// the game's barriers, a held submission going on, the end of a submission). The ops a run carries that a serial front
// pass would have to settle in order before the threads record (counter reports and resets, render conditions, inline
// uniform updates, render target changes, attachment clears) are counted apart. Every 600 presented frames one line:
// the draws per frame, the share of them in runs of at least 16, 64 and 256 draws, the runs, and the breaks by kind.
enum { RB_DISPATCH, RB_COPY, RB_CLEAR, RB_EVENT, RB_BARRIER, RB_HELD, RB_SUBMIT, RB_N };
enum { RO_COUNTER, RO_COND, RO_UB, RO_RT, RO_ATT_CLEAR, RO_N };
static const char *const rb_name[RB_N] = {"dispatches", "copies", "clears", "events", "barriers", "held", "submits"};
static const char *const ro_name[RO_N] = {"counters", "conditions", "uniform updates", "target changes",
                                          "attachment clears"};
static struct {
    int on;
    uint32_t draws;                      // of the run open now
    uint64_t total, in16, in64, in256, runs, longest, brk[RB_N], carried[RO_N], frame0;
} RR = {.on = -1};

static void runs_break(int why) {
    uint32_t d = RR.draws;
    RR.brk[why]++;
    if (!d) return;
    RR.runs++, RR.total += d, RR.draws = 0;
    if (d >= 16) RR.in16 += d;
    if (d >= 64) RR.in64 += d;
    if (d >= 256) RR.in256 += d;
    if (d > RR.longest) RR.longest = d;
}

static void runs_op(uint32_t op) {
    switch (op) {
    case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND: case OP_DRAW_MESH: case OP_DRAW_MESH_IND: RR.draws++; break;
    case OP_DISPATCH: case OP_DISPATCH_IND: runs_break(RB_DISPATCH); break;
    case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: runs_break(RB_COPY); break;
    case OP_CLEAR_TEX: case OP_CLEAR_BUF: runs_break(RB_CLEAR); break;
    case OP_SIGNAL_EVENT: case OP_WAIT_EVENT: runs_break(RB_EVENT); break;
    case OP_BARRIER: runs_break(RB_BARRIER); break;
    case OP_REPORT_COUNTER: case OP_RESET_COUNTER: RR.carried[RO_COUNTER]++; break;
    case OP_RENDER_ENABLE: case OP_RENDER_COND: RR.carried[RO_COND]++; break;
    case OP_UPDATE_UB: RR.carried[RO_UB]++; break;
    case OP_SET_RT: RR.carried[RO_RT]++; break;
    case OP_CLEAR_COLOR: case OP_CLEAR_DS: RR.carried[RO_ATT_CLEAR]++; break;
    default: break;
    }
}

static int runs_on(void) {
    if (RR.on < 0) RR.on = rt_knob("REPLAY_RUNS", "0");
    return RR.on;
}

static void runs_frame(uint64_t frame) {
    if (!runs_on() || frame % 600) return;
    if (RR.frame0 && frame > RR.frame0) {
        double f = (double)(frame - RR.frame0), t = RR.total ? (double)RR.total : 1.0;
        char b[256], c[256];
        int o = 0;
        for (int k = 0; k < RB_N; k++)
            o += snprintf(b + o, sizeof b - (size_t)o, "%s %s %.1f", k ? "," : "", rb_name[k], RR.brk[k] / f);
        o = 0;
        for (int k = 0; k < RO_N; k++)
            o += snprintf(c + o, sizeof c - (size_t)o, "%s %s %.1f", k ? "," : "", ro_name[k], RR.carried[k] / f);
        rt_log("vk_exec: frame %llu: draw runs: %.0f draws a frame in %.1f runs; in runs of 16 or more %.0f%%, "
               "64 or more %.0f%%, 256 or more %.0f%%, longest %llu; breaks a frame:%s; carried a frame:%s",
               (unsigned long long)frame, RR.total / f, RR.runs / f, 100.0 * RR.in16 / t, 100.0 * RR.in64 / t,
               100.0 * RR.in256 / t, (unsigned long long)RR.longest, b, c);
    }
    uint32_t open = RR.draws;
    memset(&RR.total, 0, sizeof RR - offsetof(typeof(RR), total));
    RR.draws = open, RR.frame0 = frame;
}

uint64_t vk_exec_submit_seq;

void vk_exec_frame_log(const tex_t *p, uint64_t frame) {
    vk_exec_cond_frame_end(frame);   // a frame boundary in replay order (present_image runs on the replay thread)
    runs_frame(frame);
    vk_exec_draw_prepared_stats_collect();
    VkExecFrame *f = &vk_exec_frame;
    if ((f->prep_builds || f->prep_cache_hits || f->prep_cache_misses) &&
        ((rt_trace & RT_TRACE_GFX) || ((rt_trace & RT_TRACE_PERF) && frame % 60 == 0))) {
        rt_log("resource prep: frame %llu: builds %llu descriptors %llu aggregate-skipped %llu reused %llu snapshot-reused %llu "
               "resolved %llu ordered %llu checked %llu proof-descriptors %llu renewals %llu renewed-descriptors %llu",
               (unsigned long long)frame, (unsigned long long)f->prep_builds,
               (unsigned long long)f->prep_descriptors, (unsigned long long)f->prep_aggregate_skipped,
               (unsigned long long)f->prep_reused,
               (unsigned long long)f->prep_snapshot_reused, (unsigned long long)f->prep_resolved,
               (unsigned long long)f->prep_ordered, (unsigned long long)f->prep_checked,
               (unsigned long long)f->prep_proof_descriptors, (unsigned long long)f->prep_renewals,
               (unsigned long long)f->prep_renewed_descriptors);
        rt_log("prepared cache: frame %llu: hits %llu misses %llu evictions %llu lookup-probes %llu",
               (unsigned long long)frame, (unsigned long long)f->prep_cache_hits,
               (unsigned long long)f->prep_cache_misses, (unsigned long long)f->prep_cache_evictions,
               (unsigned long long)f->prep_cache_probes);
        f->prep_builds = f->prep_descriptors = f->prep_aggregate_skipped = f->prep_reused = f->prep_snapshot_reused =
            f->prep_resolved = f->prep_ordered = f->prep_checked = f->prep_proof_descriptors =
            f->prep_renewals = f->prep_renewed_descriptors = 0;
        f->prep_cache_hits = f->prep_cache_misses = f->prep_cache_evictions = f->prep_cache_probes = 0;
    }
    if (!(rt_trace & RT_TRACE_GFX)) return;
    uint64_t ew, eu, el, ed;
    vk_exec_event_stats(&ew, &eu, &el, &ed);
    int hit = 0;
    char list[512] = "";
    size_t o = 0;
    for (int i = 0; i < nfw; i++) {
        if (p && fw[i].mem == p->mem) hit = 1;
        if (o < sizeof list - 48)
            o += (size_t)snprintf(list + o, sizeof list - o, " %p:%ux%u/%#x*%u", (const void *)fw[i].mem, fw[i].w,
                                  fw[i].h, fw[i].fmt, fw[i].n);
    }
    rt_log("frame %llu: submits %llu passes %llu draws %llu/%llu clears %llu copies %llu dispatches %llu; present %p "
           "%ux%u/%#x written %d; feedback draws %llu barriers %llu restarts %llu layouts %llu read-before-write %llu; "
           "uniform updates %llu (%llu ended a pass); binds %llu in %.2f ms (textures %.2f, buffers %.2f, sig %.2f, push %.2f ms; "
           "descriptors reused %llu, resolved %llu); dynamic state calls %llu (%llu unchanged, left out); "
           "event waits %llu (%llu before their signal: %llu "
           "signalled later, %.1f submits later on average); targets%s",
           (unsigned long long)frame, (unsigned long long)f->submits,
           (unsigned long long)f->passes, (unsigned long long)f->draws, (unsigned long long)f->draws_seen,
           (unsigned long long)f->clears, (unsigned long long)f->copies, (unsigned long long)f->dispatches,
           p ? (const void *)p->mem : NULL, p ? p->d.width : 0, p ? p->d.height : 0, p ? (uint32_t)p->d.format : 0,
           hit, (unsigned long long)f->fb_draws, (unsigned long long)f->fb_barriers, (unsigned long long)f->fb_restarts,
           (unsigned long long)f->fb_layouts, (unsigned long long)f->fb_war_barriers, (unsigned long long)f->ub_updates,
           (unsigned long long)f->ub_pass_breaks, (unsigned long long)f->bind_calls,
           (double)(f->bind_ns) / 1e6, (double)(f->bind_tex_ns) / 1e6, (double)(f->bind_buf_ns) / 1e6,
           (double)(f->bind_sig_ns) / 1e6, (double)(f->bind_push_ns) / 1e6, (unsigned long long)f->bind_reused,
           (unsigned long long)f->bind_resolved, (unsigned long long)f->dyn_calls, (unsigned long long)f->dyn_left,
           (unsigned long long)ew, (unsigned long long)eu, (unsigned long long)el,
           el ? (double)ed / (double)el : 0.0, list);
    memset(f, 0, sizeof *f);
    nfw = 0;
}

WEAK void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex, (void)payload;
    rt_fatal("vk: op %s is not replayed yet", nvn_op_name(op));
}

WEAK void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex, (void)payload;
    rt_fatal("vk: op %s is not replayed yet", nvn_op_name(op));
}

enum { W_UNRENDERABLE, W_LEVEL, W_LAYER, W_VIEW_FORMAT, W_BLOCK_VIEW, W_BLOCK_LAYERS, W_CLEAR_TARGET, W_CLEAR_MASK, W_STENCIL_MASK, W_COUNT };

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

// NVN defaults of the replay state (nvn_mtl.m).
static void state_reset(VkExec *ex) {
    memset(ex, 0, sizeof *ex);
    ex->st_vmask[0] = ex->st_vmask[1] = ex->st_wmask[0] = ex->st_wmask[1] = 0xff;
    ex->depth_range[1] = 1.0f;
    vk_pipe_state_defaults(&ex->pipe);
    ex->barrier_at = -1;
}

// ---- render targets

static uint32_t level_extent(uint32_t v, uint32_t level) {
    v >>= level;
    return v ? v : 1;
}

// A view format other than the texture's is used when both map directly and have the same texel size (the image is
// created MUTABLE_FORMAT for colour formats). blocks: t is a BC texture with block_views, and the view an uncompressed
// format whose texel is one of its blocks.
static int view_format(const tex_t *t, const view_t *v, int blocks, VkFormat *out) {
    TexVkFormat a, b;
    int bw, bh, vbw, vbh;
    if (!tex_vk_format(t->d.format, &a) || a.vk != a.ideal || !tex_vk_format(v->format, &b) || b.vk != b.ideal ||
        b.depth || b.stencil)
        return 0;
    int bpb = nvn_fmt_bpb(t->d.format, &bw, &bh), vbpb = nvn_fmt_bpb(v->format, &vbw, &vbh);
    if (!bpb || bpb != vbpb || (!blocks && (bw != 1 || bh != 1)) || vbw != 1 || vbh != 1) return 0;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, b.vk, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) return 0;
    *out = b.vk;
    return 1;
}

// ---- 3D BC render targets
// A 3D BC texture has no block views (vk_texture.c create_tex, NX2NAVI_BLOCK_VIEWS_3D): on RDNA2 cards such an image
// made BLOCK_TEXEL_VIEW_COMPATIBLE and 2D_ARRAY_COMPATIBLE took GPU writes past the end of its memory (issue #5). Draws
// into it through an uncompressed view of its blocks (agl's BC4 encoder fills 32x32x32 and 64x64x64 volumes and their
// levels at the logo) go to a stand-in: a 2D array of the blocks of one level in the view's format, one layer per
// depth plane. It starts with the level's content, so a pass that writes part of a plane keeps the rest, and each pass
// end copies the planes it drew into the level (a copy between a compressed and a size-compatible uncompressed format).
typedef struct {
    VkImage bc, img;
    VmaAllocation alloc;
    VkFormat fmt;
    uint32_t level, w, h, d;
    uint64_t pending;
} BcStandIn;
#define BC_STANDINS 64   // one per level drawn: 6 volumes of up to 7 levels at the logo
static BcStandIn bcs[BC_STANDINS];
static int nbcs;
static uint64_t bc_birth;
static PlatMutex bcs_lock = PLAT_MUTEX_INIT;   // the replay adds and reads; destroy_tex (any thread) removes

static void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, uint32_t layers) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                .oldLayout = from, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
}

static void memory_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                           VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static BcStandIn *bc_standin_find(VkImage image, uint32_t level, VkFormat format) {
    for (int i = 0; i < nbcs; ++i)
        if (bcs[i].bc == image && bcs[i].level == level && bcs[i].fmt == format) return &bcs[i];
    return NULL;
}

static void bc_upload_release(void *opaque) {
    vk_texture_target_loan_release(opaque);
}

// Only one allocation can prepare a given native image/level/format. The original
// target can retire while capacity is being obtained or the upload is being built.
static VkResult bc_standin(const tex_t *t, const VkTexImage *img, uint32_t level, VkFormat fmt, BcStandIn *out) {
    const VkImage backing = img->image;
    plat_mutex_lock(&bcs_lock);
    BcStandIn *entry = bc_standin_find(backing, level, fmt);
    if (entry && !entry->pending) {
        *out = *entry;
        plat_mutex_unlock(&bcs_lock);
        return VK_SUCCESS;
    }
    // Workers can use an existing stand-in before their turn. Creating one, or
    // encountering a pending one, uses the same ordered retry as cold textures.
    const int turn_deferred = vk_texture_defer_until_turn();
    if (turn_deferred || entry || nbcs == BC_STANDINS) {
        plat_mutex_unlock(&bcs_lock);
        return turn_deferred || entry ? VK_NOT_READY : VK_ERROR_TOO_MANY_OBJECTS;
    }
    if (bc_birth == UINT64_MAX) rt_fatal("vk: 3D BC stand-in ownership exhausted");
    const uint64_t pending = ++bc_birth;
    bcs[nbcs++] = (BcStandIn){.bc=backing,.fmt=fmt,.level=level,.pending=pending};
    plat_mutex_unlock(&bcs_lock);

    VkTextureTargetLoan *loan = vk_texture_target_loan(t, backing, 1);
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    BcStandIn s = {.bc=backing,.fmt=fmt,.level=level};
    if (loan) {
        const VkTexImage *source = vk_texture_target_loan_image(loan);
        const uint32_t lw = level_extent(source->width, level), lh = level_extent(source->height, level);
        s.w = (lw + 3) / 4;
        s.h = (lh + 3) / 4;
        s.d = level_extent(source->depth, level);
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                                .extent = {s.w, s.h, 1}, .mipLevels = 1, .arrayLayers = s.d, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        const VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        const RcVmaRequest request = {.producer = RC_BUDGET_TEXTURE, .required = 1};
        result = resource_core_vma_create_image(&request, vk_ctx()->vma, &ci, &ai, &s.img, &s.alloc, NULL, NULL);
        if (result == VK_ERROR_DEVICE_LOST) vk_device_lost_check(result, "admitting a 3D block attachment");
        if (result == VK_SUCCESS && !vk_texture_target_loan_current(loan))
            result = VK_ERROR_INITIALIZATION_FAILED;
        if (result == VK_SUCCESS) {
            // Queue initialization before any pass can see this image. Both the old
            // physical source and any unadopted destination survive the actual upload.
            VkCommandBuffer cb = vk_upload_begin();
            image_barrier(cb, s.img, VK_IMAGE_LAYOUT_UNDEFINED, s.d);
            memory_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
            const VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                                    .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, s.d},
                                    .extent = {lw, lh, s.d}};
            vkCmdCopyImage(cb, backing, VK_IMAGE_LAYOUT_GENERAL, s.img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
            memory_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
            const uint32_t w = source->width, h = source->height, d = source->depth;
            vk_upload_defer_release(vk_upload_recording_sequence(cb), bc_upload_release, loan);
            loan = NULL;
            vk_upload_end();
            rt_log("vk: 3D BC render target %ux%ux%u level %u drawn through a %ux%u x %u stand-in (format %d)", w,
                   h, d, level, s.w, s.h, s.d, (int)fmt);
        }
    }

    const int recorded = result == VK_SUCCESS;
    plat_mutex_lock(&bcs_lock);
    entry = bc_standin_find(backing, level, fmt);
    if (entry && entry->pending == pending) {
        if (recorded) {
            *entry = s;
            *out = s;
            s.img = VK_NULL_HANDLE;
        } else *entry = bcs[--nbcs];
    } else if (recorded) result = VK_ERROR_INITIALIZATION_FAILED;
    plat_mutex_unlock(&bcs_lock);

    if (s.img) {
        if (recorded) vk_defer_image(s.img, s.alloc);
        else resource_core_vma_destroy_image(vk_ctx()->vma, s.img, s.alloc);
    }
    vk_texture_target_loan_release(loan);
    return result;
}

// vk_exec_draw.c vk_exec_image_gone: the stand-in of a BC image that goes, freed once the GPU is done with it.
void vk_exec_bc_gone(VkImage image) {
    plat_mutex_lock(&bcs_lock);
    for (int i = 0; i < nbcs; i++)
        if (bcs[i].bc == image) {
            vk_defer_image(bcs[i].img, bcs[i].alloc);
            bcs[i--] = bcs[--nbcs];
        }
    plat_mutex_unlock(&bcs_lock);
}

// At a pass end: the planes the pass drew into a stand-in go to its texture.
static void bc_copy_back(VkExec *ex, const VkExecTarget *t, const VkExecTargetBinding *binding) {
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    VkImage image = t->bc_backing;
    uint64_t addr = t->bc_address;
    if (!binding->private_detached && t->tex) {
        if (binding->extent_count) {
            RcInterval *intervals = calloc(binding->extent_count, sizeof *intervals);
            if (!intervals) rt_fatal("vk: could not retain BC copyback backing intervals");
            for (uint32_t i = 0; i < binding->extent_count; ++i) intervals[i] = binding->extents[i].interval;
            resource_image_prepare_mapped((uint64_t)(uintptr_t)image, addr, t->tex->L.size, intervals, binding->extent_count);
            free(intervals);
        } else vk_texture_image_write_prepare(image, addr, t->tex->L.size);
    }
    memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                   VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, t->base_layer, t->layers},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, t->bc_level, 0, 1},
                      .dstOffset = {0, 0, (int32_t)t->base_layer},
                      .extent = {t->width, t->height, t->layers}};
    vk_gpu_gap_note(VK_GAP_BC_BACK, 0);
    vkCmdCopyImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    ex->recorded++;
    if (binding->private_detached) return;
    resource_image_written((uint64_t)(uintptr_t)image, addr);
    if (!vk_texture_render_target_written(image, addr))
        resource_effects_image_record_extents((uint64_t)(uintptr_t)image, binding->extents, binding->extent_count);
}

static VkMemoryImageHold *bc_target_hold(const VkExecTarget *target) {
    VkMemoryImageHold *hold = NULL;
    plat_mutex_lock(&bcs_lock);
    for (int i = 0; i < nbcs; ++i) {
        if (bcs[i].img != target->image || bcs[i].bc != target->bc_backing || bcs[i].level != target->bc_level) continue;
        hold = vk_mem_image_hold(target->image);
        break;
    }
    plat_mutex_unlock(&bcs_lock);
    return hold;
}

// A draw into a 3D BC image without block views through an uncompressed view of its blocks: the target is the
// image's stand-in (bc_standin). 1 when o is filled (target or left out), 0 for any other target.
static int bc_target(const tex_t *t, const view_t *v, int has_v, const VkTexImage *img, VkExecTarget *o) {
    int bw, bh;
    if (img->type != VK_IMAGE_TYPE_3D || img->block_views || !nvn_fmt_bpb(t->d.format, &bw, &bh) || bw != 4 || bh != 4)
        return 0;
    VkFormat fmt;
    BcStandIn s;
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    if (level >= img->levels) {
        warn_once(W_LEVEL, "render target level %u of a %u-level texture: left out", level, img->levels);
        return 1;
    }
    if (!has_v || !v->set_format || !view_format(t, v, 1, &fmt)) {
        warn_once(W_BLOCK_VIEW, "render target format %#x (%ux%u) through view format %#x: no block view, left out",
                  t->d.format, t->d.width, t->d.height, has_v && v->set_format ? v->format : 0);
        return 1;
    }
    const uint32_t depth = level_extent(img->depth, level);
    uint32_t base = 0, layers = depth;
    if (v->set_layers) {
        base = v->min_layer;
        layers = v->nlayers ? v->nlayers : (base < depth ? depth - base : 0);
    }
    if (base >= depth || !layers) return 1;
    o->bc_status = bc_standin(t, img, level, fmt, &s);
    if (o->bc_status != VK_SUCCESS) return 1;
    if (base >= s.d) return 1;
    if (layers > s.d - base) layers = s.d - base;
    o->image = s.img;   // not img->image: view_of makes this command buffer's view of the stand-in
    o->bc_backing = s.bc, o->bc_address = img->addr, o->bc_level = s.level;
    o->img = img;
    o->format = fmt;
    o->view_type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    o->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    o->level = 0, o->base_layer = base, o->layers = layers;
    o->width = s.w, o->height = s.h, o->samples = 1;
    o->nvn_format = v->format;
    return 1;
}

static void resolve_target_image(const tex_t *t, const view_t *v, int has_v, int depth_slot,
                                 const VkTexImage *img, int allow_standin, VkExecTarget *o) {
    memset(o, 0, sizeof *o);
    o->tex = t;
    if (!t) return;
    if (allow_standin && img && !depth_slot && bc_target(t, v, has_v, img, o)) return;
    VkImageUsageFlags need = depth_slot ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    int is3d = img && img->type == VK_IMAGE_TYPE_3D && !depth_slot;   // slices drawn through 2D views (2D_ARRAY_COMPATIBLE)
    if (!img || !(img->usage & need) || (img->type != VK_IMAGE_TYPE_2D && !is3d)) {
        warn_once(W_UNRENDERABLE, "render target format %#x target %d (%ux%u) is not renderable here: left out",
                  t->d.format, t->d.target, t->d.width, t->d.height);
        return;
    }
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    if (level >= img->levels) {
        warn_once(W_LEVEL, "render target level %u of a %u-level texture: left out", level, img->levels);
        return;
    }
    // A 3D level's depth planes take the place of layers (Vulkan views them as the array layers of a 2D array view).
    uint32_t nlay = is3d ? level_extent(img->depth, level) : img->layers;
    uint32_t base = 0, layers = nlay;
    if (has_v && v->set_layers) {
        base = v->min_layer;
        layers = v->nlayers ? v->nlayers : (base < nlay ? nlay - base : 0);
    }
    if (base >= nlay || !layers) {
        warn_once(W_LAYER, "render target layer %u of a %u-layer texture: left out", base, nlay);
        return;
    }
    if (base + layers > nlay) layers = nlay - base;
    VkFormat fmt = img->format;
    int32_t nvn_fmt = t->d.format;
    int blocks = !depth_slot && img->block_views;
    if (blocks) {   // a BC texture takes draws only through an uncompressed view of its blocks
        if (!has_v || !v->set_format || !view_format(t, v, 1, &fmt)) {
            warn_once(W_BLOCK_VIEW, "render target format %#x (%ux%u) through view format %#x: no block view, left out",
                      t->d.format, t->d.width, t->d.height, has_v && v->set_format ? v->format : 0);
            return;
        }
        nvn_fmt = v->format;
        if (layers > 1) {   // an uncompressed view of a BC image has one layer (no maintenance6 here)
            warn_once(W_BLOCK_LAYERS, "render target format %#x: block view of %u layers, first layer only", t->d.format,
                      layers);
            layers = 1;
        }
    } else if (!depth_slot && has_v && v->set_format && v->format != t->d.format) {
        if (view_format(t, v, 0, &fmt)) nvn_fmt = v->format;
        else warn_once(W_VIEW_FORMAT, "render target view format %#x of a format %#x texture: texture format used",
                       v->format, t->d.format);
    }
    o->image = img->image;
    o->img = img;
    o->scale = img->scale ? img->scale : 1;
    o->format = fmt;
    o->view_type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    o->aspect = depth_slot ? img->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                           : VK_IMAGE_ASPECT_COLOR_BIT;
    o->level = level, o->base_layer = base, o->layers = layers;
    o->is3d = (uint8_t)is3d;
    o->blocks = (uint8_t)blocks;
    o->width = level_extent(img->width, level), o->height = level_extent(img->height, level);
    if (blocks) o->width = (o->width + 3) / 4, o->height = (o->height + 3) / 4;   // the view's extent: 4x4 blocks
    o->samples = img->samples ? img->samples : 1;
    o->nvn_format = nvn_fmt;
    if (!o->aspect) o->image = VK_NULL_HANDLE;
}

static void resolve_target(const tex_t *t, const view_t *v, int has_v, int depth_slot, VkExecTarget *o) {
    // A render target may have shared immutable contents until this write intent. Detach before caching any view.
    const VkTexImage *img = t ? vk_texture_get_target(t) : NULL;
    if (!img && vk_texture_deferred()) {
        *o = (VkExecTarget){.tex = t};
        return;
    }
    resolve_target_image(t, v, has_v, depth_slot, img, 1, o);
}

/* The size of a pass without attachments: the far corner of viewport 0, else of the scissor, 1..16384.
   NX2navi draws a fragment shader that only writes a storage buffer into a 1x1
   viewport; a pass sized by the viewport's width alone puts an offset viewport outside the scissor, and the cave rock
   loses its ambient light. */
void vk_exec_noatt_dims(const VkExec *ex, uint32_t *w, uint32_t *h) {
    double x1 = 0, y1 = 0;
    if (ex->has_vp) {
        x1 = ceil(fmax(ex->vp[0][0], ex->vp[0][0] + ex->vp[0][2]));
        y1 = ceil(fmax(ex->vp[0][1], ex->vp[0][1] + ex->vp[0][3]));
    }
    if ((x1 < 1 || y1 < 1) && ex->has_scissor) x1 = (double)ex->scissor.x + ex->scissor.w, y1 = (double)ex->scissor.y + ex->scissor.h;
    *w = x1 < 1 ? 1 : x1 > 16384 ? 16384 : (uint32_t)x1;
    *h = y1 < 1 ? 1 : y1 > 16384 ? 16384 : (uint32_t)y1;
}

static uint32_t rt_color_count(const p_set_rt *r) {
    return r->n < 0 ? 0 : r->n > 8 ? 8 : (uint32_t)r->n;
}

static void target_shape(const VkExec *ex, VkExecGraphicsPassShape *shape) {
    uint32_t n = rt_color_count(&ex->rt);
    uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = UINT32_MAX, samples = 0, scale = 1;
    for (unsigned i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &shape->color[i] : &shape->depth;
        if (i < 8) {
            shape->pass_color[i] = t->image ? t->format : VK_FORMAT_UNDEFINED;
            shape->pipe_color_fmt[i] = t->image ? t->nvn_format : 0;
        } else {
            shape->pass_depth = t->image && (t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? t->format : VK_FORMAT_UNDEFINED;
            shape->pass_stencil = t->image && (t->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? t->format : VK_FORMAT_UNDEFINED;
            shape->pipe_depth_fmt = t->image ? t->nvn_format : 0;
        }
        if (!t->image) continue;
        if (t->width < w) w = t->width;
        if (t->height < h) h = t->height;
        if (t->layers < layers) layers = t->layers;
        if (!samples) samples = t->samples;
        if (t->scale > scale) scale = t->scale;
    }
    shape->pass_noatt = w == UINT32_MAX;
    if (shape->pass_noatt) {
        vk_exec_noatt_dims(ex, &w, &h);
        layers = 1;
    }
    shape->pass_w = w, shape->pass_h = h, shape->pass_layers = layers;
    shape->pass_scale = shape->pass_noatt ? 1 : scale;
    shape->pass_ncolor = n;
    shape->pass_samples = samples ? samples : 1;
}

static void apply_target_shape(VkExec *ex, const VkExecGraphicsPassShape *shape) {
    memcpy(ex->color, shape->color, sizeof ex->color);
    ex->depth = shape->depth;
    memcpy(ex->pass_color, shape->pass_color, sizeof ex->pass_color);
    ex->pass_depth = shape->pass_depth, ex->pass_stencil = shape->pass_stencil;
    memcpy(ex->pipe.color_fmt, shape->pipe_color_fmt, sizeof ex->pipe.color_fmt);
    ex->pipe.depth_fmt = shape->pipe_depth_fmt;
    ex->pass_w = shape->pass_w, ex->pass_h = shape->pass_h, ex->pass_layers = shape->pass_layers;
    ex->pass_scale = shape->pass_scale, ex->pass_noatt = shape->pass_noatt;
    ex->pass_ncolor = shape->pass_ncolor, ex->pass_samples = shape->pass_samples;
    ex->pipe.samples = (uint8_t)shape->pass_samples;
    ex->rt_resolved = 1;
}

// Resolve this command buffer's images and derive the same shape used by retained passes.
static void resolve_rt(VkExec *ex) {
    if (ex->rt_resolved) return;
    ex->rt_resolved = 1;
    const p_set_rt *r = &ex->rt;
    VkExecGraphicsPassShape shape = {0};
    uint32_t n = rt_color_count(r);
    for (unsigned i = 0; i < n; i++)
        resolve_target(r->color[i], &r->cview[i], r->has_cview[i], 0, &shape.color[i]);
    resolve_target(r->depth, &r->dview, r->has_dview, 1, &shape.depth);
    target_shape(ex, &shape);
    apply_target_shape(ex, &shape);
    if (shape.pass_noatt) {
        static int logged;
        uint32_t w = shape.pass_w, h = shape.pass_h;
        if (logged < 8 && ex->has_vp && (ex->vp[0][0] != 0 || ex->vp[0][1] != 0 || ex->vp[0][2] != w || ex->vp[0][3] != h))
            logged++, rt_log("vk_exec: pass without attachments %ux%u for viewport %g,%g %gx%g", w, h, ex->vp[0][0],
                             ex->vp[0][1], ex->vp[0][2], ex->vp[0][3]);
    }
}

struct VkExecGraphicsPass {
    VkExecGraphicsPassShape shape;
    VkTextureTargetLoan *loan[9];
    VkMemoryImageHold *standin[8];
    VkExecTargetBinding bindings[9];
    p_set_rt rt;
    uint64_t fence;
    VkCommandBuffer hook_cb;
    uint32_t hook_pass;
    VkCommandBuffer barrier_cb;
    uint32_t barrier_pass;
    int barrier_was_open;
    int check_selected_hooks;
    uint32_t selected_hook_work;
    VkDlaaPreparedObserver *observer;
    int reopen;
    int activated;
};

static int target_same(const VkExecTarget *a, const VkExecTarget *b) {
    return a->image == b->image && (!a->image ||
        (a->format == b->format && a->view_type == b->view_type && a->aspect == b->aspect &&
         a->level == b->level && a->base_layer == b->base_layer && a->layers == b->layers &&
         a->width == b->width && a->height == b->height && a->samples == b->samples &&
         a->scale == b->scale && a->nvn_format == b->nvn_format && a->blocks == b->blocks && a->is3d == b->is3d &&
         a->bc_backing == b->bc_backing && a->bc_address == b->bc_address && a->bc_level == b->bc_level));
}

int vk_exec_graphics_pass_reopen_match(const VkExecGraphicsPass *before, const VkExecGraphicsPass *after) {
    if (!before || !after || before->fence != after->fence || memcmp(&before->rt, &after->rt, sizeof before->rt))
        return 0;
    const VkExecGraphicsPassShape *a = &before->shape, *b = &after->shape;
    if (!a->in_pass || b->in_pass ||
        memcmp(a->pass_color, b->pass_color, sizeof a->pass_color) || a->pass_depth != b->pass_depth ||
        a->pass_stencil != b->pass_stencil || memcmp(a->pipe_color_fmt, b->pipe_color_fmt, sizeof a->pipe_color_fmt) ||
        a->pipe_depth_fmt != b->pipe_depth_fmt || a->pass_w != b->pass_w || a->pass_h != b->pass_h ||
        a->pass_layers != b->pass_layers || a->pass_scale != b->pass_scale || a->pass_ncolor != b->pass_ncolor ||
        a->pass_samples != b->pass_samples || a->pass_noatt != b->pass_noatt ||
        a->tonemap_extent.width != b->tonemap_extent.width || a->tonemap_extent.height != b->tonemap_extent.height ||
        a->hud_alpha_attachment != b->hud_alpha_attachment || a->feedback_count != b->feedback_count ||
        a->fbl_mask != b->fbl_mask || memcmp(a->feedback_images, b->feedback_images, sizeof a->feedback_images) ||
        memcmp(a->feedback_aspects, b->feedback_aspects, sizeof a->feedback_aspects)) return 0;
    for (unsigned i = 0; i < 8; ++i)
        if (!target_same(&a->color[i], &b->color[i]) || a->color[i].exact_snapshot != b->color[i].exact_snapshot)
            return 0;
    return target_same(&a->depth, &b->depth) && a->depth.exact_snapshot == b->depth.exact_snapshot;
}

int vk_exec_graphics_pass_reopen_allowed(const VkExecGraphicsPass *pass) {
    return pass && pass->shape.in_pass && !pass->hook_cb && !pass->barrier_cb &&
        !pass->shape.tonemap_extent.width && !pass->shape.tonemap_extent.height &&
        !pass->shape.hud_alpha_attachment;
}

static int graphics_pass_hooks_active(void) {
    return vk_contact_strength() || vk_dlaa_available() || vk_dlaa_swap_img;
}

static uint32_t graphics_pass_selected_hooks(const VkExec *ex, const VkExecGraphicsPassShape *shape) {
    // Contact shadow retains its separate compatibility route when explicitly active.
    const uint32_t contact = vk_contact_strength() ? (1u << 16) : 0;
    uint32_t work = vk_dlaa_graphics_pass_work(ex, shape);
    // Reusing an already open, exact pass cannot publish its begin-time observation again.
    if (shape->in_pass) work &= ~VK_DLAA_GRAPHICS_PASS_OBSERVE;
    return contact | work;
}

static int has_image(const VkImage *list, int n, VkImage img);

static void graphics_pass_feedback(VkExecGraphicsPassShape *shape) {
    for (unsigned i = 0; i < 9; ++i) {
        const VkExecTarget *target = i < 8 ? &shape->color[i] : &shape->depth;
        if (!target->image || !target->img->fb_learned) continue;
        shape->fbl_mask |= 1u << i;
        if (has_image(shape->feedback_images, shape->feedback_count, target->image)) continue;
        int index = shape->feedback_count++;
        shape->feedback_images[index] = target->image;
        shape->feedback_aspects[index] = i < 8 ? VK_IMAGE_ASPECT_COLOR_BIT : target->img->aspect;
    }
}

void vk_exec_graphics_pass_cancel(VkExecGraphicsPass **out) {
    VkExecGraphicsPass *pass = *out;
    if (!pass) return;
    vk_dlaa_prepared_observer_cancel(&pass->observer);
    for (unsigned i = 0; i < 9; ++i) {
        vk_texture_target_loan_release(pass->loan[i]);
        if (i < 8) vk_mem_image_release(pass->standin[i]);
        vk_texture_alias_read_cancel(pass->bindings[i].load_seed);
        free(pass->bindings[i].extents);
    }
    free(pass);
    *out = NULL;
}

enum {
    LOAD_ATTACHMENT_SEED_IMAGE = 0x100,
    LOAD_ATTACHMENT_SEED_CURRENT,
    LOAD_ATTACHMENT_OPEN_TARGET,
    LOAD_ATTACHMENT_PEER_DEFERRED,
    LOAD_ATTACHMENT_PEER_LOAN,
    LOAD_ATTACHMENT_STANDIN,
    LOAD_ATTACHMENT_VIEW,
    LOAD_ATTACHMENT_FEEDBACK,
    LOAD_ATTACHMENT_NO_CURRENT_SOURCE,
    LOAD_ATTACHMENT_HOOK_WORK = 0x100000,
    LOAD_ATTACHMENT_HOOK_OBSERVER = 0x200000,
};

static VkExecGraphicsPassResult graphics_attachment_failure(VkExecPreparationFailure *failure,
    VkExecPreparationReason reason, VkExecGraphicsPassResult result, unsigned slot, int detail);

static VkExecGraphicsPassResult bc_target_refused(VkExecPreparationFailure *failure, unsigned slot, VkResult status) {
    const VkExecPreparationReason reason = status == VK_ERROR_INITIALIZATION_FAILED ? VK_EXEC_PREPARATION_PLACEMENT :
                                                                                     VK_EXEC_PREPARATION_PRESSURE;
    return graphics_attachment_failure(failure, reason, VK_EXEC_GRAPHICS_PASS_UNAVAILABLE, slot, status);
}

static VkExecGraphicsPassResult graphics_pass_capture(VkExec *ex, int allow_ordered, int after_barrier,
    VkTextureAliasRead *const *seeds, VkExecGraphicsPass **out, VkExecPreparationFailure *failure) {
    *out = NULL;
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_GRAPHICS_PASS_STOPPED;
    if (vk_exec_turn_due()) return allow_ordered ? VK_EXEC_GRAPHICS_PASS_UNAVAILABLE : VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED;
    if (after_barrier && !allow_ordered) return VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED;
    uint32_t w, h;
    vk_exec_noatt_dims(ex, &w, &h);
    int reopen = !after_barrier && ex->in_pass && ex->pass_noatt && (w > ex->pass_w || h > ex->pass_h);
    unsigned first_seed = 0;
    while (first_seed < 9 && (!seeds || !seeds[first_seed])) first_seed++;
    const int selected_hooks = first_seed < 9 || after_barrier;
    const int compatibility_hooks = !selected_hooks && graphics_pass_hooks_active();
    if (after_barrier && (!ex->cb || ex->tonemap_extent.width || ex->hud_alpha_attachment))
        return VK_EXEC_GRAPHICS_PASS_COMPAT;
    // An existing compatibility hook can open its actual pass here. A selected LOAD must
    // establish its effective views and vendor role first, before any rendering begins.
    if ((!ex->in_pass || reopen) && compatibility_hooks) {
        if (!allow_ordered) return VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED;
        if (reopen) vk_exec_pass_end(ex);
        if (!ex->in_pass && ex->pass_noatt) ex->rt_resolved = 0;
        vk_exec_pass_begin(ex);
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed())
            return VK_EXEC_GRAPHICS_PASS_STOPPED;
        if (!ex->in_pass) return VK_EXEC_GRAPHICS_PASS_UNAVAILABLE;
        reopen = 0;
    } else if (first_seed == 9 && !after_barrier && allow_ordered && !ex->in_pass) {
        if (ex->pass_noatt) ex->rt_resolved = 0;
        resolve_rt(ex);
        if (vk_texture_deferred()) return VK_EXEC_GRAPHICS_PASS_UNAVAILABLE;
    }
    VkExecGraphicsPass *pass = calloc(1, sizeof *pass);
    if (!pass) rt_fatal("vk: exact render pass receipt allocation failed");
    pass->rt = ex->rt;
    pass->fence = ex->fence;
    pass->check_selected_hooks = selected_hooks;
    VkExecGraphicsPassShape *shape = &pass->shape;
    pass->reopen = reopen;
    if (after_barrier) {
        pass->barrier_cb = ex->cb;
        pass->barrier_pass = ex->pass_serial;
        pass->barrier_was_open = ex->in_pass;
    }
    shape->in_pass = !after_barrier && ex->in_pass && !pass->reopen;
    shape->feedback_dirty = shape->in_pass ? ex->fb_dirty : 0;
    const int materialized = !reopen && (ex->in_pass || (allow_ordered && ex->rt_resolved));
    if (shape->in_pass && (compatibility_hooks || ex->tonemap_extent.width || ex->hud_alpha_attachment)) {
        // The pass already references the exact hook-selected views. Its current command buffer has the scene
        // obligation, and a compiler park cannot hand off or change it. A native split invalidates this receipt.
        pass->hook_cb = ex->cb;
        pass->hook_pass = ex->pass_serial;
        shape->tonemap_extent = ex->tonemap_extent;
        shape->hud_alpha_attachment = ex->hud_alpha_attachment;
    }
    const p_set_rt *r = &pass->rt;
    const uint32_t n = rt_color_count(r);
    unsigned refused_slot = first_seed;
    int detail = LOAD_ATTACHMENT_VIEW;
    const int resolve_peers = first_seed < 9 && allow_ordered && !ex->in_pass && !ex->rt_resolved;
    for (unsigned i = 0; i < 9; ++i) {
        const int depth = i == 8;
        const tex_t *tex = depth ? r->depth : i < n ? r->color[i] : NULL;
        if (!tex) continue;
        VkExecTarget *target = depth ? &shape->depth : &shape->color[i];
        const VkExecTarget *open = depth ? &ex->depth : &ex->color[i];
        const view_t *view = depth ? &r->dview : &r->cview[i];
        const int has_view = depth ? r->has_dview : r->has_cview[i];
        VkTextureAliasRead *seed = seeds ? seeds[i] : NULL;
        VkExecTargetBinding *binding = &pass->bindings[i];
        const VkTexImage *image;
        refused_slot = i;
        if (seed) {
            // The prepared seed owns the native image. Looking up this slot again can resume
            // or rebind its dormant predecessor before the original LOAD issues.
            image = vk_texture_alias_target_image(seed);
            detail = LOAD_ATTACHMENT_SEED_IMAGE;
            if (!image) goto compatibility;
            detail = LOAD_ATTACHMENT_SEED_CURRENT;
            if (!vk_texture_alias_target_current(seed, image->image)) goto compatibility;
            resolve_target_image(tex, view, has_view, depth, image, 0, target);
            detail = LOAD_ATTACHMENT_OPEN_TARGET;
            if (shape->in_pass && !target_same(target, open)) goto compatibility;
            binding->load_seed = vk_texture_alias_read_retain(seed);
        } else {
            VkExecTarget peer;
            if (resolve_peers) {
                resolve_target(tex, view, has_view, depth, &peer);
                detail = LOAD_ATTACHMENT_PEER_DEFERRED;
                if (vk_texture_deferred()) goto compatibility;
                open = &peer;
            }
            const int peer_materialized = materialized || resolve_peers;
            if (peer_materialized && open->bc_status != VK_SUCCESS) {
                vk_exec_graphics_pass_cancel(&pass);
                return bc_target_refused(failure, i, open->bc_status);
            }
            if (peer_materialized && !open->image) continue;
            VkImage native = peer_materialized ? (open->bc_backing ? open->bc_backing : open->image) : VK_NULL_HANDLE;
            VkTextureTargetLoan *loan = vk_texture_target_loan(tex, native, peer_materialized);
            detail = LOAD_ATTACHMENT_PEER_LOAN;
            if (!loan) goto compatibility;
            pass->loan[i] = loan;
            image = vk_texture_target_loan_image(loan);
            if (!depth && peer_materialized && open->bc_backing) {
                detail = LOAD_ATTACHMENT_STANDIN;
                if (!shape->in_pass && !(pass->standin[i] = bc_target_hold(open))) goto compatibility;
                *target = *open;
                target->img = image;
            } else resolve_target_image(tex, view, has_view, depth, image, 0, target);
            detail = LOAD_ATTACHMENT_OPEN_TARGET;
            if (peer_materialized && !target_same(target, open)) goto compatibility;
        }
        detail = LOAD_ATTACHMENT_VIEW;
        if (!target->image) goto compatibility;
        target->exact_snapshot = 1;
        binding->range = (RcSourceRange){1, image->addr, tex->L.size};
        if (!vk_texture_image_binding(image->image, image->addr, tex->L.size, &binding->binding,
                                      &binding->extents, &binding->extent_count))
            binding->binding = (RcRegistryImageBinding){.image = (uint64_t)(uintptr_t)image->image};
    }
    target_shape(ex, shape);
    if (shape->in_pass) {
        shape->pass_w = ex->pass_w;
        shape->pass_h = ex->pass_h;
    }
    graphics_pass_feedback(shape);
    if (shape->in_pass && ex->tonemap_extent.width) {
        shape->fbl_mask = 0;
        shape->feedback_count = 0;
    }
    refused_slot = first_seed;
    detail = LOAD_ATTACHMENT_FEEDBACK;
    if (shape->in_pass && ex->fbl_mask != shape->fbl_mask) goto compatibility;
    if (pass->check_selected_hooks) {
        const uint32_t work = graphics_pass_selected_hooks(ex, shape);
        if (work) {
            detail = LOAD_ATTACHMENT_HOOK_WORK | (int)work;
            if (work != VK_DLAA_GRAPHICS_PASS_OBSERVE || first_seed == 9 || !allow_ordered ||
                shape->in_pass || (rw_chunk && rw_chunk->idx)) goto compatibility;
            detail = LOAD_ATTACHMENT_HOOK_OBSERVER;
            if (!vk_dlaa_prepare_observer(ex, shape, pass->bindings[0].load_seed, pass->bindings[8].load_seed,
                                          &pass->observer))
                goto compatibility;
            pass->selected_hook_work = work;
        }
    }
    *out = pass;
    return VK_EXEC_GRAPHICS_PASS_READY;

compatibility:
    vk_exec_graphics_pass_cancel(&pass);
    if (first_seed < 9) {
        const VkExecGraphicsPassResult result = allow_ordered ? VK_EXEC_GRAPHICS_PASS_UNAVAILABLE :
            VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED;
        if (failure && !failure->owner) {
            uint32_t count;
            const RcInterval *range = vk_texture_alias_read_intervals(seeds[first_seed], &count);
            graphics_attachment_failure(failure, VK_EXEC_PREPARATION_ATTACHMENT, result, refused_slot, detail);
            failure->role = refused_slot == 8 ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            if (count) failure->range = range[0];
        }
        return result;
    }
    return allow_ordered ? VK_EXEC_GRAPHICS_PASS_COMPAT : VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED;
}

VkExecGraphicsPassResult vk_exec_graphics_pass_capture(VkExec *ex, int allow_ordered, VkExecGraphicsPass **out) {
    return graphics_pass_capture(ex, allow_ordered, 0, NULL, out, NULL);
}

static int prepare_bound_texture_relocation(VkExec *ex, const tex_t *destination, int ordered);

typedef struct { const VkExec *ex; p_set_rt rt; VkTextureAliasRead *seed[9]; } AttachmentSeedScope;

static int graphics_attachment_seeds_current(void *opaque) {
    const AttachmentSeedScope *scope = opaque;
    if (memcmp(&scope->rt, &scope->ex->rt, sizeof scope->rt)) return 0;
    for (unsigned i = 0; i < 9; ++i)
        if (scope->seed[i] && !vk_texture_alias_read_current(scope->seed[i])) return 0;
    return 1;
}

static VkExecGraphicsPassResult graphics_attachment_failure(VkExecPreparationFailure *failure,
    VkExecPreparationReason reason, VkExecGraphicsPassResult result, unsigned slot, int detail) {
    if (failure && !failure->owner) {
        VkExecGraphicsResult disposition = result == VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED ?
            VK_EXEC_GRAPHICS_NEEDS_ORDERED : VK_EXEC_GRAPHICS_UNAVAILABLE;
        *failure = (VkExecPreparationFailure){
            .owner = VK_EXEC_PREPARATION_IMAGE, .reason = reason, .result = disposition,
            .detail = detail, .role = slot == 8 ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, .slot = slot,
        };
    }
    return result;
}

static int graphics_attachment_peer_failure(VkExecPreparationFailure *failure, unsigned slot,
    VkExecPreparationReason reason, int detail, RcInterval target, RcInterval peer) {
    if (failure && !failure->owner) {
        graphics_attachment_failure(failure, reason, VK_EXEC_GRAPHICS_PASS_UNAVAILABLE, slot, detail);
        failure->range = target;
        failure->peer = peer;
    }
    return 0;
}

int vk_exec_graphics_pass_attachment_peers(const VkExecGraphicsPass *pass, VkExecPreparationFailure *failure) {
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTargetBinding *binding = &pass->bindings[i];
        if (!vk_texture_alias_target_replacement(binding->load_seed)) continue;
        uint32_t count;
        const RcInterval *target = vk_texture_alias_read_intervals(binding->load_seed, &count);
        if (!count) return graphics_attachment_peer_failure(failure, i, VK_EXEC_PREPARATION_MAPPING,
                                                            RC_REGISTRY_STALE, (RcInterval){0}, (RcInterval){0});
        for (uint32_t j = 0; j < 9; j++) {
            if (i == j) continue;
            const VkExecTarget *attachment = j < 8 ? &pass->shape.color[j] : &pass->shape.depth;
            if (!attachment->image) continue;
            const VkExecTargetBinding *other = &pass->bindings[j];
            uint32_t peer_count;
            const RcInterval *seed = vk_texture_alias_read_intervals(other->load_seed, &peer_count);
            if (!peer_count && vk_texture_target_loan_backing_current(pass->loan[j], other->extents, other->extent_count))
                peer_count = other->extent_count;
            if (!peer_count) return graphics_attachment_peer_failure(failure, i, VK_EXEC_PREPARATION_MAPPING,
                                                                     j, target[0], (RcInterval){0});
            for (uint32_t p = 0; p < peer_count; p++) {
                RcInterval peer = seed ? seed[p] : other->extents[p].interval;
                if (!rc_range_valid(peer.range)) return graphics_attachment_peer_failure(failure, i,
                    VK_EXEC_PREPARATION_MAPPING, j, target[0], peer);
                for (uint32_t a = 0; a < count; a++) {
                    const VkExecTarget *selected = i < 8 ? &pass->shape.color[i] : &pass->shape.depth;
                    if (attachment->image == selected->image || rc_range_overlap(target[a].range, peer.range))
                        return graphics_attachment_peer_failure(failure, i, VK_EXEC_PREPARATION_ATTACHMENT,
                                                                 j, target[a], peer);
                }
            }
        }
    }
    return 1;
}

int vk_exec_graphics_pass_attachment_access(const VkExecGraphicsPass *pass, VkExecAttachmentAccessCheck check,
    const void *context, VkExecPreparationFailure *failure) {
    for (uint32_t i = 0; i < 9; ++i) {
        const VkExecTarget *target = i == 8 ? &pass->shape.depth : &pass->shape.color[i];
        if (!target->image) continue;
        const VkExecTargetBinding *binding = &pass->bindings[i];
        uint32_t count = 0;
        const RcInterval *seed = vk_texture_alias_read_intervals(binding->load_seed, &count);
        if (!seed && vk_texture_target_loan_backing_current(pass->loan[i], binding->extents, binding->extent_count))
            count = binding->extent_count;
        if (!count)
            return graphics_attachment_peer_failure(failure, i, VK_EXEC_PREPARATION_MAPPING,
                                                     RC_REGISTRY_STALE, (RcInterval){0}, (RcInterval){0});
        for (uint32_t j = 0; j < count; ++j) {
            const RcInterval interval = seed ? seed[j] : binding->extents[j].interval;
            RcInterval overlap = {0};
            const RcRegistryResult result = check(context, target->bc_backing ? target->bc_backing : target->image,
                                                  interval, &overlap);
            if (result != RC_REGISTRY_OK)
                return graphics_attachment_peer_failure(failure, i, VK_EXEC_PREPARATION_ATTACHMENT, result,
                                                         interval, overlap);
        }
    }
    return 1;
}

int vk_exec_graphics_pass_attachment_read(const VkExecGraphicsPass *pass, VkImage image,
    const RcInterval *peer, uint32_t peer_count, uint32_t slot, VkExecAttachmentRead mode, VkExecPreparationFailure *failure) {
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTargetBinding *binding = &pass->bindings[i];
        if (!vk_texture_alias_target_replacement(binding->load_seed)) continue;
        uint32_t count;
        const RcInterval *target = vk_texture_alias_read_intervals(binding->load_seed, &count);
        if (!count || !peer_count)
            return graphics_attachment_peer_failure(failure, i, VK_EXEC_PREPARATION_MAPPING, slot,
                                                     count ? target[0] : (RcInterval){0}, (RcInterval){0});
        const VkExecTarget *selected = i < 8 ? &pass->shape.color[i] : &pass->shape.depth;
        int original_read = 0;
        for (uint32_t p = 0; p < peer_count; p++) {
            if (!rc_range_valid(peer[p].range))
                return graphics_attachment_peer_failure(failure, i, VK_EXEC_PREPARATION_MAPPING, slot, target[0], peer[p]);
            for (uint32_t a = 0; a < count; a++) {
                if (image != selected->image && !rc_range_overlap(target[a].range, peer[p].range)) continue;
                if (!original_read && mode != VK_EXEC_ATTACHMENT_READ_DISJOINT)
                    original_read = vk_texture_alias_target_read_prior(binding->load_seed, selected->image, image,
                        mode == VK_EXEC_ATTACHMENT_READ_FRAGMENT);
                if (original_read) continue;
                if (failure && !failure->owner) failure->image = selected->image;
                return graphics_attachment_peer_failure(failure, i, VK_EXEC_PREPARATION_FEEDBACK,
                                                         slot, target[a], peer[p]);
            }
        }
    }
    return 1;
}

// A geometry-only shader still accesses the native descriptor. Prove that neither
// an active target nor a retained replacement can change its image layout.
int vk_exec_graphics_pass_geometry_read(const VkExec *ex, const VkExecGraphicsPass *pass,
    VkImage image, RcInterval read, int active) {
    if (!image || !read.mapping || !rc_range_valid(read.range) ||
        (active ? !ex->in_pass : !pass || pass->activated)) return 0;
    for (unsigned i = 0; i < 9; ++i) {
        const VkExecTarget *target = active ? (i == 8 ? &ex->depth : &ex->color[i]) :
            (i == 8 ? &pass->shape.depth : &pass->shape.color[i]);
        if (!target->image) continue;
        if (image == target->image || image == target->bc_backing || target->bc_backing) return 0;
        const VkExecTargetBinding *binding = active ? &ex->target_bindings[i] : &pass->bindings[i];
        if ((active && !(ex->target_binding_mask & (1u << i))) ||
            binding->binding.image != (uint64_t)(uintptr_t)target->image) return 0;
        uint32_t count = 0;
        const RcInterval *seed = vk_texture_alias_read_intervals(binding->load_seed, &count);
        if (!binding->load_seed) {
            if (!active && !vk_texture_target_loan_backing_current(pass->loan[i], binding->extents,
                                                                  binding->extent_count)) return 0;
            if (!binding->extents) return 0;
            count = binding->extent_count;
        }
        if (!count || (binding->load_seed && !seed)) return 0;
        for (uint32_t j = 0; j < count; ++j) {
            RcInterval target_range = seed ? seed[j] : binding->extents[j].interval;
            if (!target_range.mapping || !rc_range_valid(target_range.range) ||
                rc_range_overlap(read.range, target_range.range)) return 0;
        }
    }
    return 1;
}

static VkExecGraphicsPassResult capture_attachment_load_seeds(VkExec *ex, int allow_ordered, int relocate_color,
    VkTextureAliasRead *seeds[9], VkExecPreparationFailure *failure) {
    AttachmentSeedScope scope = {.ex = ex, .rt = ex->rt};
    const VkTexImage *images[9];
    uint32_t count = 0;
    uint32_t layers = ex->in_pass ? ex->pass_layers : 0;
    if (!layers) for (unsigned i = 0; i < 9; ++i) {
        const int depth = i == 8;
        if (!depth && i >= rt_color_count(&scope.rt)) continue;
        const tex_t *target = depth ? scope.rt.depth : scope.rt.color[i];
        const view_t *view = depth ? (scope.rt.has_dview ? &scope.rt.dview : NULL) :
            (scope.rt.has_cview[i] ? &scope.rt.cview[i] : NULL);
        VkImageSubresourceRange planes;
        if (vk_texture_alias_attachment_range(target, view, depth, &planes) && (!layers || planes.layerCount < layers))
            layers = planes.layerCount;
    }
    for (unsigned i = 0; i < 9; ++i) {
        const int depth = i == 8;
        if (!depth && i >= rt_color_count(&scope.rt)) continue;
        if (seeds[i]) {
            if (!vk_texture_alias_read_current(seeds[i]))
                return graphics_attachment_failure(failure, VK_EXEC_PREPARATION_MAPPING,
                    VK_EXEC_GRAPHICS_PASS_UNAVAILABLE, i, RC_REGISTRY_STALE);
            continue;
        }
        const tex_t *target = depth ? scope.rt.depth : scope.rt.color[i];
        const view_t *view = depth ? (scope.rt.has_dview ? &scope.rt.dview : NULL) :
            (scope.rt.has_cview[i] ? &scope.rt.cview[i] : NULL);
        if (relocate_color && !depth && target && !view && !graphics_pass_hooks_active()) {
            NVNcopyRegion whole = {0, 0, 0, target->d.width, target->d.height, 1};
            if (vk_texture_alias_region_transfer(target, NULL, &whole)) {
                int preserved = prepare_bound_texture_relocation(ex, target, allow_ordered);
                if (preserved > 0)
                    return graphics_attachment_failure(failure, VK_EXEC_PREPARATION_ORDER,
                        VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED, i, preserved);
                if (preserved < 0) {
                    if (vk_sync_stopping() || vk_device_is_lost()) return VK_EXEC_GRAPHICS_PASS_STOPPED;
                    return graphics_attachment_failure(failure, VK_EXEC_PREPARATION_PLACEMENT,
                        VK_EXEC_GRAPHICS_PASS_UNAVAILABLE, i, preserved);
                }
            }
        }
        VkTextureAliasFailure refused = {0};
        int captured = vk_texture_alias_target_capture(target, view, layers, &seeds[i], allow_ordered, &refused);
        if (captured == 2)
            return graphics_attachment_failure(failure, VK_EXEC_PREPARATION_ORDER,
                VK_EXEC_GRAPHICS_PASS_NEEDS_ORDERED, i, captured);
        if (captured < 0) {
            static const VkExecPreparationReason reasons[] = {VK_EXEC_PREPARATION_ORDER, VK_EXEC_PREPARATION_MAPPING,
                VK_EXEC_PREPARATION_ATTACHMENT, VK_EXEC_PREPARATION_ORDER, VK_EXEC_PREPARATION_PLACEMENT,
                VK_EXEC_PREPARATION_PRESSURE};
            graphics_attachment_peer_failure(failure, i, reasons[refused.phase], refused.status,
                                             refused.receiver, (RcInterval){0});
            if (failure) {
                failure->registration = refused.registration;
                failure->material_serial = refused.material_serial;
                failure->image = refused.image;
                failure->coverage = refused.coverage;
                failure->stage = "attachment-load-capture";
                failure->load.position = vk_exec_resource_position();
                failure->load.present = 1;
                failure->load.seeded = seeds[i] != NULL;
                if (target) {
                    failure->load.address = (uintptr_t)target->mem;
                    failure->load.bytes = target->L.size;
                    failure->load.guest_format = target->d.format;
                    failure->load.width = target->d.width;
                    failure->load.height = target->d.height;
                    failure->load.samples = target->d.samples;
                }
            }
            return VK_EXEC_GRAPHICS_PASS_UNAVAILABLE;
        }
        if (captured == 1) scope.seed[count++] = seeds[i];
    }
    if (count) {
        VkExecRestoreResult result = vk_exec_prepare_alias_texture_read(ex, scope.seed, count, graphics_attachment_seeds_current,
                                                                       &scope, images);
        if (result == VK_EXEC_RESTORE_STOPPED) return VK_EXEC_GRAPHICS_PASS_STOPPED;
        if (result != VK_EXEC_RESTORE_READY)
            return graphics_attachment_failure(failure, result == VK_EXEC_RESTORE_STALE ?
                VK_EXEC_PREPARATION_MAPPING : VK_EXEC_PREPARATION_PRESSURE,
                VK_EXEC_GRAPHICS_PASS_UNAVAILABLE, UINT32_MAX, result);
        ex->rt_resolved = 0;
    }
    return VK_EXEC_GRAPHICS_PASS_READY;
}

VkExecGraphicsPassResult vk_exec_graphics_pass_capture_seeded(VkExec *ex, int allow_ordered,
    VkTextureAliasRead *seeds[9], VkExecGraphicsPass **out, VkExecPreparationFailure *failure) {
    *out = NULL;
    VkExecGraphicsPassResult result = capture_attachment_load_seeds(ex, allow_ordered, 1, seeds, failure);
    if (result != VK_EXEC_GRAPHICS_PASS_READY) return result;
    return graphics_pass_capture(ex, allow_ordered, 0, seeds, out, failure);
}

VkExecGraphicsPassResult vk_exec_graphics_pass_capture_reopen(VkExec *ex, int allow_ordered, VkExecGraphicsPass **out) {
    return graphics_pass_capture(ex, allow_ordered, 1, NULL, out, NULL);
}

const VkExecGraphicsPassShape *vk_exec_graphics_pass_shape(const VkExecGraphicsPass *pass) {
    return &pass->shape;
}

int vk_exec_graphics_pass_current(const VkExec *ex, const VkExecGraphicsPass *pass) {
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed() ||
        !pass || pass->activated || ex->fence != pass->fence ||
        memcmp(&ex->rt, &pass->rt, sizeof ex->rt)) return 0;
    const VkExecGraphicsPassShape *shape = &pass->shape;
    if (pass->barrier_cb && (ex->cb != pass->barrier_cb || ex->pass_serial != pass->barrier_pass ||
        (!pass->barrier_was_open && ex->in_pass) || ex->tonemap_extent.width || ex->hud_alpha_attachment)) return 0;
    if (pass->hook_cb && (ex->cb != pass->hook_cb || ex->pass_serial != pass->hook_pass ||
        ex->tonemap_extent.width != shape->tonemap_extent.width || ex->tonemap_extent.height != shape->tonemap_extent.height ||
        ex->hud_alpha_attachment != shape->hud_alpha_attachment)) return 0;
    if (!pass->hook_cb && (pass->check_selected_hooks ?
        graphics_pass_selected_hooks(ex, shape) != pass->selected_hook_work : graphics_pass_hooks_active())) return 0;
    if (pass->observer && !vk_dlaa_prepared_observer_current(ex, shape, pass->observer)) return 0;
    if (shape->in_pass && !ex->in_pass) return 0;
    if (ex->in_pass && !pass->reopen) {
        int projected_noatt = pass->barrier_cb && shape->pass_noatt;
        if ((!projected_noatt && (ex->pass_w != shape->pass_w || ex->pass_h != shape->pass_h)) ||
            ex->pass_layers != shape->pass_layers ||
            ex->pass_samples != shape->pass_samples || ex->pass_ncolor != shape->pass_ncolor ||
            ex->fbl_mask != shape->fbl_mask || !target_same(&ex->depth, &shape->depth)) return 0;
        for (unsigned i = 0; i < shape->pass_ncolor; ++i)
            if (!target_same(&ex->color[i], &shape->color[i])) return 0;
    }
    if (shape->pass_noatt) {
        uint32_t w, h;
        vk_exec_noatt_dims(ex, &w, &h);
        if (ex->in_pass && !pass->reopen && !pass->barrier_cb) {
            if (w > shape->pass_w || h > shape->pass_h) return 0;
        } else if (w != shape->pass_w || h != shape->pass_h) return 0;
    }
    int mixed = 0;
    for (unsigned i = 0; i < 9 && !mixed; ++i)
        mixed = vk_texture_alias_target_replacement(pass->bindings[i].load_seed) != NULL;
    for (unsigned i = 0; i < 9; ++i) {
        const VkTextureTargetLoan *loan = pass->loan[i];
        VkTextureAliasRead *seed = pass->bindings[i].load_seed;
        if (mixed && loan && !seed && !vk_texture_target_loan_backing_current(loan, pass->bindings[i].extents,
                                                                              pass->bindings[i].extent_count)) return 0;
        if (seed && !vk_texture_alias_target_current(seed, (VkImage)(uintptr_t)pass->bindings[i].binding.image)) return 0;
        if (!loan || vk_texture_target_loan_current(loan)) continue;
        if (i == 8 || !shape->color[i].bc_backing || !vk_texture_target_loan_private(loan)) return 0;
    }
    return 1;
}

// The attachment view of a target. A registered texture's view lives as long as its image (vk_exec_target_view); any
// other image's view, and that of a texture destroyed since the target was resolved (gone: its record is freed only
// after this submission, vk_defer_free), is made for this command buffer and freed at its fence.
static VkImageView view_of(VkExec *ex, const VkExecTarget *t) {
    if (!t->exact_snapshot && t->img && t->img->image == t->image && !t->img->gone)
        return vk_exec_target_view(t->img, t->format, t->view_type, t->aspect, t->level, t->base_layer, t->layers, t->blocks);
    ViewEnt *v = ex->views;
    for (int i = 0; i < ex->nviews; i++)
        if (v[i].image == t->image && v[i].format == t->format && v[i].type == t->view_type && v[i].aspect == t->aspect &&
            v[i].level == t->level && v[i].base == t->base_layer && v[i].layers == t->layers)
            return v[i].view;
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->image,
                                .viewType = t->view_type, .format = t->format,
                                .subresourceRange = {t->aspect, t->level, 1, t->base_layer, t->layers}};
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    if (t->blocks) ci.pNext = &use;   // not the BC image's sampling and transfer usages
    VkImageView view;
    if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS)
        rt_fatal("vk: render target view (format %d, %ux%u) could not be created", (int)t->format, t->width, t->height);
    if (ex->nviews == ex->cap_views) {
        ex->cap_views = ex->cap_views ? ex->cap_views * 2 : 16;
        ex->views = realloc(ex->views, sizeof(ViewEnt) * (size_t)ex->cap_views);
        if (!ex->views) rt_fatal("vk: out of memory");
        v = ex->views;
    }
    v[ex->nviews++] = (ViewEnt){t->image, t->format, t->view_type, t->aspect, t->level, t->base_layer, t->layers, view};
    return view;
}

typedef struct { int n; VkImageView v[]; } ViewList;

static void free_views(void *arg) {   // completion thread, once the submission that used them is done
    ViewList *l = arg;
    VkDevice dev = vk_ctx()->device;
    for (int i = 0; i < l->n; i++) vkDestroyImageView(dev, l->v[i], NULL);
    free(l);
}

// ---- target history (vk_target_frames.h: the table, what recent and steady mean, and the pass verdict)
uint64_t vk_exec_steady_passes;   // lookups a steady target alone made per-frame (vk_exec_targets_recent)

static TfrTarget tfr_of(const VkExecTarget *t) {
    return (TfrTarget){(uint64_t)(uintptr_t)t->image, t->level, t->base_layer, t->img ? t->img->addr : 0};
}

static void target_frames_written(VkExec *ex, uint64_t stamp) {
    TfrTarget targets[9];
    unsigned count = 0;
    for (unsigned i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        targets[count++] = tfr_of(t);
    }
    int lk = vk_exec_shared();
    if (lk) plat_mutex_lock(&tfr_lock);
    for (unsigned i = 0; i < count; i++) tfr_note(&targets[i], stamp);
    if (lk) plat_mutex_unlock(&tfr_lock);
}

static int steady_on(void) {
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("NX2NAVI_PIPE_SKIP_STEADY");
        rt_log("vk: a pass whose targets the game has rendered in %d frames or more counts as per-frame after a "
               "pause too, so its draws are skipped, not waited for, while a pipeline builds: %s (NX2NAVI_PIPE_SKIP_STEADY)",
               TFR_STEADY, on ? "on" : "off");
    }
    return on;
}

// tfr_pass_recent for the current pass; vk_exec_steady_passes counts the lookups that only a steady target made
// per-frame.
int vk_exec_targets_recent(VkExec *ex, int any) {
    resolve_rt(ex);
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    TfrTarget c[8], d = tfr_of(&ex->depth);
    for (int i = 0; i < 8; i++) c[i] = tfr_of(&ex->color[i]);
    int steady = steady_on(), lk = vk_exec_shared();
    if (lk) plat_mutex_lock(&tfr_lock);
    int by_steady = 0, r = tfr_pass_recent(c, 8, &d, stamp, any, steady, &by_steady);
    vk_exec_steady_passes += (uint64_t)by_steady;
    if (lk) plat_mutex_unlock(&tfr_lock);
    return r;
}

// One line naming each target of the current pass: format, extent, layer and the two frames it was last rendered in
// (prepare_draw's pipeline wait log).
void vk_exec_targets_describe(VkExec *ex, char *buf, size_t cap) {
    resolve_rt(ex);
    TfrTarget targets[9];
    TargetFrames history[9] = {0};
    for (int i = 0; i < 9; i++) targets[i] = tfr_of(i < 8 ? &ex->color[i] : &ex->depth);
    int lk = vk_exec_shared();
    if (lk) plat_mutex_lock(&tfr_lock);
    for (int i = 0; i < 9; i++) {
        if (!targets[i].image) continue;
        const TargetFrames *e = tfr_find(&targets[i], 0);
        if (e) history[i] = *e;
    }
    if (lk) plat_mutex_unlock(&tfr_lock);
    size_t n = 0;
    buf[0] = 0;
    for (int i = 0; i < 9 && n < cap; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!targets[i].image) continue;
        const TargetFrames *e = &history[i];
        int w = snprintf(buf + n, cap - n, "%s%c%d fmt %d %ux%u L%u seen %llu/%llu in %u frames", n ? "; " : "",
                         i < 8 ? 'c' : 'd', i < 8 ? i : 0, (int)t->format, t->width, t->height, t->base_layer,
                         (unsigned long long)e->last, (unsigned long long)e->prev,
                         e->owner == targets[i].owner ? e->n : 0);
        if (w < 0) break;
        n += (size_t)w;
    }
    if (!n) snprintf(buf, cap, "no targets");
}

// ---- passes and barriers

static int has_image(const VkImage *list, int n, VkImage img) {
    for (int i = 0; i < n; i++)
        if (list[i] == img) return 1;
    return 0;
}

static VkImageMemoryBarrier2 layout_barrier(VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to) {
    return (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                                   .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
}

// Full barrier. The images in the feedback layout become want[] (pass begin) or none (every other barrier): the others
// go back to GENERAL and the new ones leave it, in the same command.
// NX2NAVI_BARRIER_GAME (1; 0: every pass begin and dispatch a full barrier; ab: in the B blocks): the game's own
// barriers order its work, as on the Switch. A pass that
// samples what an earlier one drew, a dispatch that reads what another dispatch wrote, an indirect draw whose arguments
// a dispatch wrote: on the Switch each needs nvnCommandBufferBarrier (ORDER_* and INVALIDATE_* bits) in the game's
// command stream, which the replay keeps as a full barrier (OP_BARRIER). What the GPU orders by itself there is kept: a
// switch between graphics and compute work drains the one before (the 3D and compute classes share the engine). With
// every pass begin and every dispatch a full barrier, 2.3 ms of a 9.7 ms GPU frame in Kokiri Forest was barriers.
// With the switch:
//  - a pass begin after an earlier pass and no dispatch since the last full barrier gets an attachment barrier
//    (colour and depth/stencil output before the next pass's loads and writes: Vulkan orders attachment access only
//    inside one pass instance), after neither nothing;
//  - a dispatch after other dispatches only gets none;
//  - a pass after a dispatch, and a dispatch after a pass, get a full barrier, the drain of the switch;
//  - either gets a full barrier as before after one the replay itself asked for (vk_exec_barrier: its copies,
//    clears, fills, uniform updates and stand-in copies call it before their commands, which sets bar_dirty), and
//    when feedback-loop targets change layout.
// Copies, clears and the game's barriers keep their full barriers.
//
// The default is 1 on NVIDIA GPUs and 0 on every other. The game placed its barriers for an NVIDIA GPU, and what it
// left out is what such a GPU orders by itself; Vulkan asks for a barrier wherever a pass samples what an earlier pass
// drew or a dispatch reads what an earlier one wrote, whether the game recorded one or not. On an RTX 5090 the picture
// was the same with the switch on and off (34 screenshots across A/B blocks in Kokiri Forest) and the GPU time fell by
// 0.5 to 0.6 ms of 5.4 a frame; no other vendor's driver has run it, so there the setting has to be turned on by hand.
static int barrier_game(void) {
    static int mode = -1;
    if (mode < 0) {
        const VkCtx *c = vk_ctx();
        int nvidia = c && c->props.vendorID == 0x10de;
        mode = rt_knob("BARRIER_GAME", nvidia ? "1" : "0");
        if (mode)
            rt_log("vk_exec: NX2NAVI_BARRIER_GAME=%s: passes and dispatches are ordered by the game's barriers",
                   mode == 2 ? "ab" : "1");
    }
    return RT_KNOB_ON(mode);
}

// NX2NAVI_TRACE=perf: barriers per frame by kind (vk_exec_barrier_log)
enum { BS_FULL, BS_GAME, BS_PASS_FULL, BS_PASS_ATT, BS_PASS_NONE, BS_DISP_FULL, BS_DISP_NONE, BS_N };
static uint64_t bstat[BS_N];
static const char *const bs_name[BS_N] = {"replay full", "game", "pass full", "pass attachment", "pass none",
                                          "dispatch full", "dispatch none"};
enum { BR_OTHER, BR_GAME, BR_PASS, BR_DISPATCH };

static void barrier_to_why(VkExec *ex, const VkImage *want, const VkImageAspectFlags *wasp, int nwant, int why);
static void barrier_to(VkExec *ex, const VkImage *want, const VkImageAspectFlags *wasp, int nwant) {
    barrier_to_why(ex, want, wasp, nwant, BR_OTHER);
}

static void attachment_barrier(VkExec *ex) {
    const VkPipelineStageFlags2 st = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = st,
                           .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           .dstStageMask = st,
                           .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
}

static void barrier_to_why(VkExec *ex, const VkImage *want, const VkImageAspectFlags *wasp, int nwant, int why) {
    vk_exec_pass_end(ex);
    if (why == BR_OTHER) ex->bar_dirty = 1;   // also when it falls together with the last one: its commands follow
    VkImageMemoryBarrier2 ib[18];
    uint32_t n = 0;
    for (int i = 0; i < ex->nfbl; i++)
        if (!has_image(want, nwant, ex->fbl_img[i]))
            ib[n++] = layout_barrier(ex->fbl_img[i], ex->fbl_aspect[i], VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
                                     VK_IMAGE_LAYOUT_GENERAL);
    for (int i = 0; i < nwant; i++)
        if (!has_image(ex->fbl_img, ex->nfbl, want[i]))
            ib[n++] = layout_barrier(want[i], wasp[i], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT);
    for (int i = 0; i < nwant; i++) ex->fbl_img[i] = want[i], ex->fbl_aspect[i] = wasp[i];
    ex->nfbl = nwant;
    if (ex->barrier_at == ex->recorded && !n) return;
    if (!n && !ex->bar_dirty && ((why == BR_PASS && !ex->bar_dispatches) || (why == BR_DISPATCH && !ex->bar_passes)) &&
        barrier_game()) {
        if (why == BR_DISPATCH) {
            bstat[BS_DISP_NONE]++;
        } else if (ex->bar_passes) {
            attachment_barrier(ex);
            bstat[BS_PASS_ATT]++;
        } else {
            bstat[BS_PASS_NONE]++;
        }
        return;
    }
    bstat[why == BR_GAME ? BS_GAME : why == BR_PASS ? BS_PASS_FULL : why == BR_DISPATCH ? BS_DISP_FULL : BS_FULL]++;
    if (why != BR_OTHER) ex->bar_dirty = 0;   // after BR_OTHER the replay's own commands follow: the next pass or
    ex->bar_passes = ex->bar_dispatches = 0;    // dispatch orders after them
    vk_exec_frame.fb_layouts += n;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = n, .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->barrier_at = ++ex->recorded;
}

void vk_exec_barrier(VkExec *ex) { barrier_to(ex, NULL, NULL, 0); }

void vk_exec_barrier_dispatch(VkExec *ex) {
    barrier_to_why(ex, NULL, NULL, 0, BR_DISPATCH);
    ex->bar_dispatches++;
}

void vk_exec_barrier_log(uint64_t frame) {
    static uint64_t last;
    if (!(rt_trace & RT_TRACE_PERF) || frame % 600) return;
    double f = frame > last ? (double)(frame - last) : 1.0;
    last = frame;
    char s[320];
    int o = 0;
    for (int k = 0; k < BS_N; k++) {
        o += snprintf(s + o, sizeof s - o, "%s%s %.1f", k ? ", " : "", bs_name[k], bstat[k] / f);
        bstat[k] = 0;
    }
    rt_log("vk_exec: frame %llu: barriers per frame: %s", (unsigned long long)frame, s);
}

static void graphics_target_loan_done(void *arg) {
    vk_texture_target_loan_release(arg);
}

static void graphics_target_loan_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;
    graphics_target_loan_done(arg);
}

static void graphics_standin_done(void *arg) {
    vk_mem_image_release(arg);
}

static void graphics_standin_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;
    graphics_standin_done(arg);
}

static void graphics_pass_record_target(VkExec *ex, unsigned slot, VkExecGraphicsPass *pass) {
    VkTextureTargetLoan *loan = pass->loan[slot];
    VkTextureAliasRead *seed = pass->bindings[slot].load_seed;
    if (!loan && !seed) return;
    VkExecTargetBinding *binding = &ex->target_bindings[slot];
    *binding = pass->bindings[slot];
    if (seed) vk_texture_alias_read_use(seed, ex->fence);
    if (loan) binding->pinned = vk_texture_target_loan_take_pin(loan);
    if (binding->pinned.pin) vk_texture_pinned_write_use(&binding->pinned);
    if (slot < 8 && pass->standin[slot]) {
        ex->bc_target_mask |= 1u << slot;
        vk_completion_add_owned(ex->fence, graphics_standin_done, graphics_standin_cancel, pass->standin[slot]);
        pass->standin[slot] = NULL;
    } else ex->target_binding_mask |= 1u << slot;
    if (loan) vk_completion_add_owned(ex->fence, graphics_target_loan_done, graphics_target_loan_cancel, loan);
    pass->bindings[slot] = (VkExecTargetBinding){0};
    pass->loan[slot] = NULL;
}

static int bc_pass_capture(VkExec *ex, unsigned slot) {
    const VkExecTarget *target = &ex->color[slot];
    VkTextureTargetLoan *backing = vk_texture_target_loan(target->tex, target->bc_backing, 1);
    VkMemoryImageHold *standin = backing ? bc_target_hold(target) : NULL;
    if (!standin) {
        vk_texture_target_loan_release(backing);
        return 0;
    }
    VkExecTargetBinding *binding = &ex->target_bindings[slot];
    binding->range = (RcSourceRange){1, target->bc_address, target->tex->L.size};
    if (!vk_texture_image_binding(target->bc_backing, target->bc_address, target->tex->L.size, &binding->binding,
                                  &binding->extents, &binding->extent_count))
        binding->binding = (RcRegistryImageBinding){.image = (uint64_t)(uintptr_t)target->bc_backing};
    ex->bc_target_mask |= 1u << slot;
    vk_completion_add_owned(ex->fence, graphics_target_loan_done, graphics_target_loan_cancel, backing);
    vk_completion_add_owned(ex->fence, graphics_standin_done, graphics_standin_cancel, standin);
    return 1;
}

static void graphics_preparation_unavailable(VkExec *, uint32_t opcode, VkExecGraphicsResult,
    const VkExecPreparationFailure *, const char *reason);

static VkRenderingAttachmentInfo pass_attachment(VkImageView view, int feedback) {
    if (!view) return (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    return (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = view, .imageLayout = feedback ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                  : VK_IMAGE_LAYOUT_GENERAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
}

static VkRenderingInfo ordinary_rendering(const VkExec *ex, VkRenderingAttachmentInfo colors[8],
    VkRenderingAttachmentInfo *depth, VkRenderingAttachmentInfo *stencil) {
    for (uint32_t i = 0; i < ex->pass_ncolor; i++)
        colors[i] = pass_attachment(ex->pass_views[i], (ex->fbl_mask >> i) & 1);
    const VkRenderingAttachmentInfo ds = pass_attachment(ex->pass_views[8], (ex->fbl_mask >> 8) & 1);
    *depth = ds.imageView && (ex->depth.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? ds : (VkRenderingAttachmentInfo){0};
    *stencil = ds.imageView && (ex->depth.aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? ds : (VkRenderingAttachmentInfo){0};
    return (VkRenderingInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, {ex->pass_w, ex->pass_h}}, .layerCount = ex->pass_layers,
        .colorAttachmentCount = ex->pass_ncolor, .pColorAttachments = colors,
        .pDepthAttachment = ex->pass_depth ? depth : NULL, .pStencilAttachment = ex->pass_stencil ? stencil : NULL};
}

static void pass_begin(VkExec *ex, VkExecGraphicsPass *admitted, uint32_t opcode) {
    if (ex->in_pass) return;
    if (admitted && (vk_texture_deferred() || vk_exec_recording_failed())) return;
    if (admitted && admitted->check_selected_hooks &&
        (graphics_pass_selected_hooks(ex, &admitted->shape) != admitted->selected_hook_work ||
         (admitted->selected_hook_work && (!admitted->observer || (rw_chunk && rw_chunk->idx))))) {
        vk_exec_required_graphics_unavailable(ex, 0, "selected LOAD pass hook changed before rendering");
        return;
    }
    resource_targets_forget(ex);
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    if (admitted) apply_target_shape(ex, &admitted->shape);
    else {
        resolve_rt(ex);
        if (vk_texture_deferred()) return;   // a chunk's lookup waits for its turn (chunk_record): the op goes again
    }
    for (unsigned i = 0; i < ex->pass_ncolor; ++i) {
        if (ex->color[i].bc_status == VK_SUCCESS) continue;
        VkExecPreparationFailure failure = {0};
        bc_target_refused(&failure, i, ex->color[i].bc_status);
        graphics_preparation_unavailable(ex, opcode, VK_EXEC_GRAPHICS_UNAVAILABLE, &failure,
                                         "required 3D block attachment unavailable");
        return;
    }
    // This pass always LOADs its selected native views. A current ordinary seed is owned
    // by the admitted bindings; otherwise a protected placement needs a whole native producer.
    for (unsigned i = 0; i < 9; ++i) {
        if (i < 8 && i >= ex->pass_ncolor) continue;
        const VkExecTarget *target = i == 8 ? &ex->depth : &ex->color[i];
        const VkTexImage *image = target->img;
        VkTextureAliasRead *seed = admitted ? admitted->bindings[i].load_seed : NULL;
        if (!target->image || !image || (!seed && !image->preserve_pool_bytes)) continue;
        VkImageSubresourceRange range;
        const VkImageSubresourceRange *planes = attachment_load_planes(ex, target, i, &range);
        if (seed ? vk_texture_alias_target_load_current(seed, target->image, planes) :
                   vk_texture_attachment_load_ready(image, planes)) continue;
        const VkExecPreparationFailure failure = {
            .owner = VK_EXEC_PREPARATION_IMAGE, .reason = VK_EXEC_PREPARATION_ATTACHMENT,
            .result = VK_EXEC_GRAPHICS_UNAVAILABLE,
            .detail = seed ? LOAD_ATTACHMENT_SEED_CURRENT : LOAD_ATTACHMENT_NO_CURRENT_SOURCE,
            .role = target->aspect, .slot = i, .image = target->image,
            .material_serial = target->tex ? target->tex->serial : 0,
            .stage = seed ? "load-seed-current" : "load-native-current",
            .load = {.position = vk_exec_resource_position(), .present = 1, .seeded = seed != NULL,
                .native_checked = 1, .preserve = image->preserve_pool_bytes,
                .address = target->tex ? (uintptr_t)target->tex->mem : image->addr,
                .bytes = target->tex ? target->tex->L.size : 0,
                .guest_format = target->tex ? target->tex->d.format : 0, .native_format = target->format,
                .width = target->width, .height = target->height, .samples = target->samples,
                .level = target->level, .layer = target->base_layer, .layers = target->layers},
        };
        graphics_preparation_unavailable(ex, opcode, failure.result, &failure,
                                         "attachment LOAD has no preserved current source");
        return;
    }
    // A parallel group's targets do not change: its pass hooks, which follow the frame pass by pass, run in its first
    // chunk, which the others start after (chunk_record).
    if (!rw_chunk || !rw_chunk->idx) {
        // the submitting thread past a group: the hooks read what its first chunk's left, and wait for the whole
        // group where they change what its draws read (vk_exec_hook_turn)
        if (!rw_chunk) vk_exec_hooks_wait();
        if (!admitted) {
            vk_contact_pass_begin(ex);
            vk_dlaa_pass(ex);   // DLAA runs before the bloom chain's first pass (vk_dlaa.c)
        } else if (admitted->check_selected_hooks) {
            if (graphics_pass_selected_hooks(ex, &admitted->shape) != admitted->selected_hook_work) {
                vk_exec_required_graphics_unavailable(ex, 0, "selected LOAD vendor work changed at recording turn");
                return;
            }
            if (admitted->observer && !vk_dlaa_pass_prepared(ex, &admitted->observer)) {
                vk_exec_required_graphics_unavailable(ex, 0, "selected LOAD scene observer changed before recording");
                return;
            }
        }
        if (vk_exec_recording_failed()) return;
    }
    // Learned targets (VkTexImage.fb_learned) are attached in the feedback layout
    VkImage want[9];
    VkImageAspectFlags wasp[9];
    int nwant = 0;
    uint32_t fbl = 0;
    // The output-sized tonemap binds private attachments. Its guest targets stay in GENERAL for the copy back.
    int tonemap = !admitted && vk_dlaa_tonemap_target(ex);
    if (!admitted && !tonemap) for (unsigned i = 0; i < ex->pass_ncolor; ++i) {
        if (!ex->color[i].bc_backing || bc_pass_capture(ex, i)) continue;
        vk_exec_required_graphics_unavailable(ex, 0, "3D BC attachment retired before its pass opened");
        return;
    }
    for (uint32_t i = 0; !tonemap && i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || !t->image || !t->img || !t->img->fb_learned) continue;
        fbl |= 1u << i;
        if (has_image(want, nwant, t->image)) continue;
        want[nwant] = t->image, wasp[nwant++] = i < 8 ? VK_IMAGE_ASPECT_COLOR_BIT : t->img->aspect;
    }
    barrier_to_why(ex, want, wasp, nwant, BR_PASS);   // passes, clears, copies and dispatches see each other's results
    ex->bar_passes++;
    ex->fbl_mask = fbl;
    ex->fb_dirty = 0;
    ex->fb_read = 0;
    ex->pass_serial++;
    for (uint32_t i = 0; i < ex->pass_ncolor; i++)
        ex->pass_views[i] = ex->color[i].image ? view_of(ex, &ex->color[i]) : VK_NULL_HANDLE;
    ex->pass_views[8] = ex->depth.image ? view_of(ex, &ex->depth) : VK_NULL_HANDLE;
    VkRenderingAttachmentInfo ca[8], da, sa;
    VkRenderingInfo ri = ordinary_rendering(ex, ca, &da, &sa);
    // The start mark goes between the barrier and vkCmdBeginRendering, its end right after vkCmdEndRendering
    // (vk_exec_pass_end): outside the pass, and without breaking it.
    VkPassMark *m = ex->mark = rw_chunk ? NULL : vk_pass_mark_begin(ex->cb, VK_MARK_PASS, ex->fence);   // one ring
    if (m) {
        m->step = ex->mark_passes++;
        m->w = ex->pass_w, m->h = ex->pass_h, m->d = ex->pass_layers;
        m->layer = ex->depth.image ? ex->depth.base_layer : 0;
        m->ncolor = (uint8_t)(ex->pass_ncolor < 8 ? ex->pass_ncolor : 8);
        for (int i = 0; i < m->ncolor; i++) m->color[i] = ex->pass_color[i];
        m->depth = ex->pass_depth ? ex->pass_depth : ex->pass_stencil;
        // The shaders bound now are the first draw's when it repeats the pipeline state; when it does not,
        // vk_exec_draw.c prepare_draw replaces them as it records the new state.
        vk_exec_draw_shaders(&m->first[0], &m->first[1]);
        ex->mark_draws0 = vk_exec_frame.draws;
    }
    if (tonemap) vk_dlaa_tonemap_begin(ex, ca, &da, &ri);
    ex->hud_alpha_attachment = !admitted && !tonemap && vk_dlaa_hud_attachment(ex, &ca[1]);
    if (ex->hud_alpha_attachment) ri.colorAttachmentCount = 2;
    // Retain the actual guest attachments for load/store and later commands. Opening a load pass publishes no new
    // guest writer. Private tonemap and block-format stand-ins report their own actual copy-back.
    if (admitted) {
        for (unsigned i = 0; i < 9; ++i) graphics_pass_record_target(ex, i, admitted);
    } else if (!tonemap) {
        for (uint32_t i = 0; i < ex->pass_ncolor; i++) {
            const VkExecTarget *t = &ex->color[i];
            resource_target_pin(ex, i, t);
        }
        resource_target_pin(ex, 8, &ex->depth);
    }
    vkCmdBeginRendering(ex->cb, &ri);
    if (!admitted && !tonemap) {
        for (uint32_t i = 0; i < ex->pass_ncolor; i++) {
            const VkExecTarget *t = &ex->color[i];
            if (resource_target_pinned(ex, i)) continue;
            resource_target_capture(ex, i, t);
        }
        if (!resource_target_pinned(ex, 8)) resource_target_capture(ex, 8, &ex->depth);
    }
    vk_exec_frame.passes++;
    ex->recorded++;
    ex->in_pass = 1;
    if (rw_chunk && !rw_chunk->idx) __atomic_store_n(&rw_chunk->g->hooks, 1, __ATOMIC_RELEASE);
}

void vk_exec_pass_begin(VkExec *ex) {
    pass_begin(ex, NULL, 0);
}

int vk_exec_record_between_renderings(VkExec *ex, int (*record)(VkCommandBuffer, void *), void *context) {
    if (!ex->cb) return 0;
    if (!ex->in_pass) {
        const int recorded = record(ex->cb, context);
        ex->recorded += recorded;
        return recorded;
    }
    if (ex->tonemap_extent.width || ex->tonemap_extent.height || ex->hud_alpha_attachment) return 0;
    VkRenderingAttachmentInfo color[8], depth, stencil;
    const VkRenderingInfo rendering = ordinary_rendering(ex, color, &depth, &stencil);
    // The existing views and write receipts stay with the logical pass. End only the Vulkan rendering and
    // occlusion segment, leaving temporal work, BC copy-back and target retirement for their existing owner.
    vk_exec_oq_pass_end(ex);
    vkCmdEndRendering(ex->cb);
    ex->in_pass = 0;
    ex->recorded++;
    const int recorded = record(ex->cb, context);
    ex->recorded += recorded;
    // STORE becomes visible to the next LOAD and any feedback reads. Earlier reads finish before later writes.
    memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    vkCmdBeginRendering(ex->cb, &rendering);
    ex->in_pass = 1;
    ex->fb_dirty = ex->fb_read = 0;
    ex->recorded += 2;
    return recorded;
}

int vk_exec_graphics_pass_activate(VkExec *ex, VkExecGraphicsPass *pass) {
    if (!vk_exec_graphics_pass_current(ex, pass)) return 0;
    if (pass->barrier_cb && ex->in_pass) return 0;
    if (pass->reopen) vk_exec_pass_end(ex);
    for (unsigned i = 0; i < pass->shape.pass_ncolor; ++i) {
        const VkTextureTargetLoan *loan = pass->loan[i];
        if (!pass->shape.color[i].bc_backing || !loan || !vk_texture_target_loan_private(loan) ||
            vk_texture_target_loan_current(loan)) continue;
        if (!ex->in_pass) pass->bindings[i].private_detached = 1;
        else if (!(ex->bc_written_mask & (1u << i))) ex->target_bindings[i].private_detached = 1;
    }
    if (!ex->in_pass) {
        ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
        pass_begin(ex, pass, 0);
    }
    pass->activated = ex->in_pass;
    return pass->activated;
}

void vk_exec_pass_end(VkExec *ex) {
    if (!ex->in_pass) return;
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    vk_exec_oq_pass_end(ex);
    vkCmdEndRendering(ex->cb);
    ex->recorded++;
    ex->in_pass = 0;
    if (ex->tonemap_extent.width) vk_dlaa_tonemap_end(ex);
    if (ex->mark) {
        // vk_exec_frame_log resets the frame counters at a present (NX2NAVI_TRACE=gfx): then only the draws since it count
        uint64_t d = vk_exec_frame.draws, d0 = ex->mark_draws0;
        ex->mark->draws = (uint32_t)(d >= d0 ? d - d0 : d);
        if (ex->mark->draws) vk_exec_draw_shaders(&ex->mark->last[0], &ex->mark->last[1]);
        vk_pass_mark_end(ex->cb, ex->mark);
        ex->mark = NULL;
    }
    for (uint32_t bits = ex->bc_written_mask; bits; bits &= bits - 1) {
        unsigned slot = (unsigned)__builtin_ctz(bits);
        bc_copy_back(ex, &ex->color[slot], &ex->target_bindings[slot]);
    }
    if (vk_dump_armed) vk_dump_pass_end(ex);
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    target_frames_written(ex, stamp);
    for (unsigned i = 0; i < 9; ++i) {
        VkExecTarget *target = i < 8 ? &ex->color[i] : &ex->depth;
        if (!target->exact_snapshot) continue;
        // A later pass resolves or activates its own attachments after the current view and use are sealed.
        target->img = NULL;
        ex->rt_resolved = 0;
    }
    resource_targets_forget(ex);
}

VkCommandBuffer vk_exec_inline_begin(int *reopen) {
    VkExec *ex = x_cur;
    *reopen = 0;
    if (!ex || !ex->cb) return VK_NULL_HANDLE;
    *reopen = ex->in_pass;
    vk_exec_barrier(ex);   // ends the pass; feedback-layout targets go back to GENERAL
    return ex->cb;
}

void vk_exec_inline_end(int reopen) {
    VkExec *ex = x_cur;   // set: the caller's vk_exec_inline_begin handed out its command buffer
    ex->recorded++;
    if (reopen) vk_exec_pass_begin(ex);
}

void vk_exec_probe_inline(VkCommandBuffer cb, uint64_t fence) {
    X.cb = cb, X.fence = fence;
    x_cur = cb != VK_NULL_HANDLE ? &X : NULL;
}

VkRect2D vk_exec_scissor(const VkExec *ex) {
    int64_t w = ex->pass_w, h = ex->pass_h;
    if (!ex->has_scissor) {
        VkExtent2D size = ex->tonemap_extent.width ? ex->tonemap_extent : (VkExtent2D){(uint32_t)w, (uint32_t)h};
        return (VkRect2D){{0, 0}, size};
    }
    int64_t k = ex->pass_scale > 1 ? ex->pass_scale : 1;
    int64_t x0 = ex->scissor.x * k, y0 = ex->scissor.y * k;
    int64_t x1 = x0 + (int64_t)ex->scissor.w * k, y1 = y0 + (int64_t)ex->scissor.h * k;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > w) x1 = w;
    if (y1 > h) y1 = h;
    if (x1 <= x0 || y1 <= y0) return (VkRect2D){{0, 0}, {0, 0}};
    if (ex->tonemap_extent.width) {
        x0 = x0 * ex->tonemap_extent.width / w, x1 = x1 * ex->tonemap_extent.width / w;
        y0 = y0 * ex->tonemap_extent.height / h, y1 = y1 * ex->tonemap_extent.height / h;
    }
    return (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
}

// ---- clears

// Outside a pass, a clear of a whole target (no narrower scissor, the target as large as the render area) clears the
// image directly instead of opening a pass for it.
static int whole_target(const VkExec *ex, const VkExecTarget *t) {
    // vkCmdClear*Image addresses a 3D level whole, not one depth plane: slices of a 3D target clear in a pass; it
    // cannot clear a compressed image: views of BC blocks clear in a pass too
    if (ex->in_pass || t->is3d || t->blocks || t->bc_backing || t->width != ex->pass_w || t->height != ex->pass_h) return 0;
    if (!ex->has_scissor) return 1;
    const p_scissor *s = &ex->scissor;
    return s->x <= 0 && s->y <= 0 && (int64_t)s->x + s->w >= t->width && (int64_t)s->y + s->h >= t->height;
}

static VkPinnedTextureWrite clear_target_prepare(const VkExecTarget *target) {
    VkPinnedTextureWrite capture = {0};
    if (target->img && target->image == target->img->image)
        vk_texture_pinned_write_capture(target->tex, target->image, &capture);
    if (capture.pin) vk_texture_pinned_write_prepare(&capture);
    else resource_target_prepare(target);
    return capture;
}

static void clear_target_written(const VkExecTarget *target, VkPinnedTextureWrite *capture) {
    if (capture->pin) {
        vk_texture_pinned_write_recorded(capture);
        vk_texture_pinned_write_release(capture);
    } else {
        resource_target_written(target);
    }
}

static void clear_image_fragment(const VkExecTarget *target, VkRect2D region, uint32_t layers, VkImageAspectFlags aspect) {
    if (!target->tex || !target->img || target->image != target->img->image || target->level ||
        target->base_layer || layers != 1 || target->img->scale > 1 || target->blocks ||
        target->bc_backing || target->format != target->img->format || target->aspect != aspect)
        return;
    const NVNcopyRegion box = {region.offset.x, region.offset.y, 0,
                               (int)region.extent.width, (int)region.extent.height, 1};
    if (aspect == VK_IMAGE_ASPECT_COLOR_BIT)
        vk_texture_alias_transfer_written(target->tex, target->image, NULL, &box);
    else if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT && target->format == VK_FORMAT_D32_SFLOAT)
        vk_texture_alias_depth_clear_written(target->tex, target->image, &box);
}

static void clear_attachment(VkExec *ex, const VkClearAttachment *a) {
    if (ex->has_scissor && (ex->scissor.w <= 0 || ex->scissor.h <= 0)) return;
    pass_begin(ex, NULL, a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT ? OP_CLEAR_COLOR : OP_CLEAR_DS);
    if (!ex->in_pass || vk_texture_deferred() || vk_exec_recording_failed()) return;
    VkClearRect cr = {vk_exec_scissor(ex), 0, ex->pass_layers};
    if (!cr.rect.extent.width || !cr.rect.extent.height) return;
    uint32_t bit = a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT ? 1u << a->colorAttachment : 0x100u;
    vk_exec_feedback_war(ex, bit);   // a draw of this pass sampled it: its reads finish before the clear writes
    unsigned slot = (unsigned)__builtin_ctz(bit);
    const VkExecTarget *target = ex->target_binding_mask & bit ? (slot == 8 ? &ex->depth : &ex->color[slot]) : NULL;
    if (target) resource_attachment_prepare(ex, slot);
    if (vk_exec_recording_failed()) return;
    vkCmdClearAttachments(ex->cb, 1, a, 1, &cr);
    ex->bc_written_mask |= ex->bc_target_mask & bit;
    if (target && !resource_attachment_written(ex, slot))
        clear_image_fragment(target, cr.rect, cr.layerCount, a->aspectMask);
    ex->recorded++;
    ex->fb_dirty |= bit;
}

// NX2NAVI_LOG_RCOND=1: clears issued under a render condition (the first 400), run or skipped, with their target.
static void clear_cond_log(const VkExec *ex, const char *what, int index) {
    static int on = -1, n;
    if (on < 0) on = rt_knob("LOG_RCOND", "0");
    if (on == 2 && index < 0 && ex->has_render_cond && ex->depth.width >= 1600) {   // full-size depth clears, all of them
        char why[256];
        vk_exec_cond_why(&ex->render_cond, why, sizeof why);
        rt_log("SEQ depth clear %#llx %s; %s", ex->depth.img ? (unsigned long long)ex->depth.img->addr : 0ull,
               ex->render_disabled ? "skipped" : "RUN", why);
        return;
    }
    if (!on || !ex->has_render_cond || n >= 400) return;
    n++;
    const VkExecTarget *t = index < 0 ? &ex->depth : index < (int)ex->pass_ncolor ? &ex->color[index] : NULL;
    rt_log("vk: %s clear %d under the condition at %#llx (mode %d): %s, target %ux%u", what, index,
           (unsigned long long)ex->render_cond.addr, ex->render_cond.mode, ex->render_disabled ? "skipped" : "run",
           t ? t->width : 0, t ? t->height : 0);
}

static int clear_attachment_extent(const tex_t *t, const view_t *view, int depth, uint32_t *width, uint32_t *height) {
    VkImageSubresourceRange planes;
    if (vk_texture_alias_attachment_range(t, view, depth, &planes)) {
        *width = level_extent(t->d.width, planes.baseMipLevel);
        *height = level_extent(t->d.height, planes.baseMipLevel);
        return 1;
    }
    if (!vk_texture_attachment_base_view(t, view)) return 0;
    *width = t->d.width;
    *height = t->d.height;
    return 1;
}

static const tex_t *clear_whole_target_requested(const VkExec *ex, unsigned slot) {
    const p_set_rt *rt = &ex->rt;
    const tex_t *t = slot == 8 ? rt->depth : slot < rt_color_count(rt) ? rt->color[slot] : NULL;
    const view_t *view = slot == 8 ? (rt->has_dview ? &rt->dview : NULL) :
        slot < rt_color_count(rt) && rt->has_cview[slot] ? &rt->cview[slot] : NULL;
    uint32_t width, height;
    if (!t || !clear_attachment_extent(t, view, slot == 8, &width, &height)) return NULL;
    if (ex->has_scissor && (ex->scissor.x > 0 || ex->scissor.y > 0 ||
        (int64_t)ex->scissor.x + ex->scissor.w < width || (int64_t)ex->scissor.y + ex->scissor.h < height)) return NULL;
    for (unsigned j = 0; j <= 8; ++j) {
        const tex_t *other = j == 8 ? rt->depth : j < rt_color_count(rt) ? rt->color[j] : NULL;
        if (!other || j == slot) continue;
        const view_t *peer = j == 8 ? (rt->has_dview ? &rt->dview : NULL) : rt->has_cview[j] ? &rt->cview[j] : NULL;
        uint32_t peer_width, peer_height;
        if (!clear_attachment_extent(other, peer, j == 8, &peer_width, &peer_height) ||
            peer_width < width || peer_height < height) return NULL;
    }
    return t;
}

static int partial_clear_slot(const VkExec *ex, const op_hdr *op) {
    if (ex->in_pass || ex->render_disabled ||
        (ex->has_scissor && (ex->scissor.w <= 0 || ex->scissor.h <= 0))) return -1;
    if (op->op == OP_CLEAR_DS && op->size >= sizeof *op + sizeof(p_clear_ds)) {
        const p_clear_ds *clear = nvn_op_payload(op);
        VkImageSubresourceRange planes;
        return clear->depth_mask && vk_texture_alias_depth_attachment_range(ex->rt.depth,
            ex->rt.has_dview ? &ex->rt.dview : NULL, &planes) &&
            !clear_whole_target_requested(ex, 8) ? 8 : -1;
    }
    if (op->op == OP_CLEAR_COLOR && op->size >= sizeof *op + sizeof(p_clear_color)) {
        const p_clear_color *clear = nvn_op_payload(op);
        if ((clear->mask & 0xf) != 0xf || clear->index < 0 || clear->index >= (int)rt_color_count(&ex->rt) ||
            clear_whole_target_requested(ex, (unsigned)clear->index)) return -1;
        VkImageSubresourceRange planes;
        if (vk_texture_alias_attachment_range(ex->rt.color[clear->index],
            ex->rt.has_cview[clear->index] ? &ex->rt.cview[clear->index] : NULL, 0, &planes)) return clear->index;
    }
    return -1;
}

static const tex_t *depth_clear_alias_destination(const VkExec *ex, const p_clear_ds *clear) {
    if (!clear->depth_mask || ex->render_disabled) return NULL;
    const tex_t *t = clear_whole_target_requested(ex, 8);
    return vk_texture_alias_depth_clear_candidate(t) ? t : NULL;
}

static int clear_ordinary_attachment(const tex_t *target, int depth) {
    VkImageSubresourceRange planes;
    if (depth) return vk_texture_alias_depth_attachment_range(target, NULL, &planes);
    const NVNcopyRegion whole = {0, 0, 0, target->d.width, target->d.height, 1};
    return vk_texture_alias_region_transfer(target, NULL, &whole);
}

static int clear_direct_attachment(VkExec *ex, unsigned slot, VkExecTarget *selected) {
    if (ex->in_pass) return 0;
    const tex_t *target = clear_whole_target_requested(ex, slot);
    if (!target || !clear_ordinary_attachment(target, slot == 8)) return 0;
    const p_set_rt *rt = &ex->rt;
    // A standalone clear selects only its own actual plane. It never LOADs a sibling.
    for (unsigned i = 0; i <= 8; ++i) {
        const tex_t *peer = i == 8 ? rt->depth : i < rt_color_count(rt) ? rt->color[i] : NULL;
        if (peer && i != slot && !clear_ordinary_attachment(peer, i == 8)) return 0;
    }
    const view_t *view = slot == 8 ? &rt->dview : &rt->cview[slot];
    const int has_view = slot == 8 ? rt->has_dview : rt->has_cview[slot];
    VkImageSubresourceRange planes = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (slot == 8 && (!vk_texture_alias_depth_attachment_range(target, has_view ? view : NULL, &planes) ||
                      planes.layerCount != 1)) return 0;
    const VkTexImage *image = vk_texture_get_alias_transfer_write(target);
    if (!image || vk_texture_deferred()) return 0;
    resolve_target_image(target, view, has_view, slot == 8, image, 0, selected);
    if (!selected->image || selected->level != planes.baseMipLevel || selected->base_layer != planes.baseArrayLayer ||
        selected->layers != 1 ||
        selected->scale != 1 || selected->blocks || selected->bc_backing ||
        selected->width != level_extent(target->d.width, planes.baseMipLevel) ||
        selected->height != level_extent(target->d.height, planes.baseMipLevel))
        return 0;
    ex->rt_resolved = 0;
    return 1;
}

static void clear_independent_target(VkExec *ex, unsigned slot) {
    if (ex->in_pass || ex->rt_resolved) return;
    const tex_t *t = clear_whole_target_requested(ex, slot);
    if (!t) return;
    const NVNcopyRegion whole = {0, 0, 0, t->d.width, t->d.height, 1};
    int supported = slot == 8 ? vk_texture_alias_depth_clear_candidate(t) :
                               vk_texture_alias_region_transfer(t, NULL, &whole);
    if (supported) vk_texture_get_alias_transfer_write(t);
}

static void clear_color(VkExec *ex, const p_clear_color *c) {
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    vk_exec_frame.clears++;
    clear_cond_log(ex, "colour", c->index);
    if (ex->has_render_cond) ex->render_disabled ? vk_cond_clears_skip++ : vk_cond_clears_run++;
    if (ex->render_disabled) return;
    const int i = c->index;
    const uint32_t mask = (uint32_t)c->mask & 0xf;
    VkExecTarget direct_target;
    const int direct = mask == 0xf && i >= 0 && i < 8 && clear_direct_attachment(ex, (unsigned)i, &direct_target);
    if (!direct) {
        if (mask == 0xf && i >= 0 && i < 8 && !vk_texture_deferred())
            clear_independent_target(ex, (unsigned)i);
        resolve_rt(ex);
    }
    if (vk_texture_deferred()) return;
    if (!direct && mask == 0xf && i >= 0 && i < (int)ex->pass_ncolor && ex->color[i].bc_status != VK_SUCCESS) {
        VkExecPreparationFailure failure = {0};
        bc_target_refused(&failure, (unsigned)i, ex->color[i].bc_status);
        graphics_preparation_unavailable(ex, OP_CLEAR_COLOR, VK_EXEC_GRAPHICS_UNAVAILABLE, &failure,
                                         "required 3D block clear attachment unavailable");
        return;
    }
    if (!direct && (i < 0 || i >= (int)ex->pass_ncolor || !ex->color[i].image)) {
        warn_once(W_CLEAR_TARGET, "colour clear of target %d, which is not bound or not renderable: skipped", i);
        return;
    }
    if (!mask) return;
    if (mask != 0xf) {   // vkCmdClearAttachments ignores write masks; a masked clear needs a draw (vk_exec_draw.c)
        warn_once(W_CLEAR_MASK, "colour clear with channel mask %#x is not replayed: skipped", mask);
        return;
    }
    const VkExecTarget *t = direct ? &direct_target : &ex->color[i];
    vk_dlaa_hud_write(t->image);
    VkClearColorValue cv;
    memcpy(cv.float32, c->rgba, sizeof cv.float32);   // integer targets: the payload's bits as the front end stored them
    if (direct || whole_target(ex, t)) {
        vk_exec_barrier(ex);
        VkImageSubresourceRange r = {VK_IMAGE_ASPECT_COLOR_BIT, t->level, 1, t->base_layer, t->layers};
        vk_gpu_gap_note(VK_GAP_CLEAR_TARGET, (uint64_t)t->width * t->height);
        VkPinnedTextureWrite capture = clear_target_prepare(t);
        vkCmdClearColorImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &r);
        ex->recorded++;
        clear_target_written(t, &capture);
        if (t->img && t->image == t->img->image && t->format == t->img->format && !t->blocks && !t->bc_backing)
            vk_texture_alias_clear_written(t->tex, t->image, &r);
        return;
    }
    VkClearAttachment a = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)i};
    a.clearValue.color = cv;
    clear_attachment(ex, &a);
}

static void clear_ds(VkExec *ex, const p_clear_ds *c) {
    RESOURCE_IMAGE_SCOPE(image_writes, ex->fence);
    clear_cond_log(ex, "depth", -1);
    if (ex->has_render_cond) ex->render_disabled ? vk_cond_clears_skip++ : (vk_cond_clears_run++, vk_cond_depth_run++);
    if (ex->render_disabled) return;
    VkExecTarget direct_target;
    const int direct = c->depth_mask && clear_direct_attachment(ex, 8, &direct_target);
    if (!direct) {
        if (c->depth_mask && !vk_texture_deferred()) clear_independent_target(ex, 8);
        resolve_rt(ex);
    }
    if (vk_texture_deferred()) return;
    const VkExecTarget *t = direct ? &direct_target : &ex->depth;
    if (!t->image) return;
    VkImageAspectFlags asp = 0;
    if (c->depth_mask && (t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
    uint32_t sm = (uint32_t)c->stencil_mask & 0xff;
    if (sm && (t->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) {
        if (sm == 0xff) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
        else warn_once(W_STENCIL_MASK, "stencil clear with mask %#x is not replayed: skipped", sm);
    }
    if (!asp) return;
    float d = c->depth < 0.0f ? 0.0f : c->depth > 1.0f ? 1.0f : c->depth;
    VkClearDepthStencilValue v = {d, (uint32_t)c->stencil & 0xff};
    if (direct || whole_target(ex, t)) {
        vk_exec_barrier(ex);
        VkImageSubresourceRange r = {asp, t->level, 1, t->base_layer, t->layers};
        vk_gpu_gap_note(VK_GAP_CLEAR_TARGET, (uint64_t)t->width * t->height);
        VkPinnedTextureWrite capture = clear_target_prepare(t);
        vkCmdClearDepthStencilImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        ex->recorded++;
        clear_target_written(t, &capture);
        if (t->img && t->image == t->img->image && t->format == t->img->format)
            vk_texture_alias_clear_written(t->tex, t->image, &r);
        return;
    }
    VkClearAttachment a = {.aspectMask = asp};
    a.clearValue.depthStencil = v;
    clear_attachment(ex, &a);
}

// ---- the op loop

static void set_rt(VkExec *ex, const p_set_rt *r) {
    if (!memcmp(&ex->rt, r, sizeof *r)) return;   // the same targets again: the pass goes on
    vk_exec_pass_end(ex);
    ex->rt = *r;
    ex->rt_resolved = 0;
}

// ---- command buffer parts
// NVN starts the GPU as soon as commands are submitted; one Vulkan command buffer per submission starts it only after
// the whole replay. Between passes, once NX2NAVI_SPLIT_DRAWS (64; 0 off) draws and dispatches were recorded since the
// last part, the part goes to the queue (vk_sync_split) and the replay continues in a new command buffer. Only
// command-buffer state is lost there (bindings and dynamic state: vk_exec_draw_new_cb, ex->dirty); barriers, layouts and
// queries are not: a barrier's first scope is everything earlier in submission order, image layouts belong to the image,
// and occlusion queries end with their pass. Fences, completion callbacks and reports stay per submission: only the last
// part signals the value. The texture pool set a part binds is sealed when the part goes (vk_exec_pool_split): later
// parts write and bind another, and a part is not split off while no other set is free of this submission.
// A part also spans at least NX2NAVI_SPLIT_US (1000) of replay time: a split costs about 40 us of the render thread
// (vkQueueSubmit2, a command buffer, the mirror prologue), and a fight frame of 2,000 draws made 15 parts on the 7900X;
// a millisecond of recording is a fraction of a millisecond of GPU work, so the GPU still starts early.
// With NX2NAVI_SPLIT_IDLE=1, neither limit holds the first part of a submission back while the GPU has run everything
// submitted: it goes at the first pass boundary after a draw or dispatch. Each frame the game's thread waits for the GPU
// to run the frame's first submission (nvn_api_core.c nvnQueueSubmitCommands, NX2NAVI_SUBMIT_FINISH), so the GPU is idle
// when the next one replays, and waited for the first 64 draws and 1 ms of it in Kokiri Forest (2026-11-09). Off by
// default: with it the foliage of Kokiri Forest flickered (RTX 5090, 2026-11-09), and the parts it cuts that early are
// the one thing it changes; why they matter is not known yet.
// Both limits also split inside a pass (NX2NAVI_SPLIT_IN_PASS, on): the pass ends there and the next draw begins it
// again in the new part, as after a barrier (attachments load and store; occlusion queries continue in a new segment).
// Kokiri Forest's two scene passes of ~500 draws take 2-4 ms each to replay and most of the frame's GPU time; split only
// between passes, the GPU waited for each whole pass, idle up to 3.6 ms a frame (2026-11-09).
// After the time limit held a part back, the clock is read again only 8 draws later (split_at): read after every
// command, it was 4% of the replay in Kokiri Forest.
static int split_n = -1, split_idle, split_in_pass, split_draws, split_at, part_start;
static uint64_t split_ns, part_t0;
static uint64_t n_parts, n_subs, n_idle_parts, n_pass_splits;

static int split_wanted(void) {
    if (split_draws >= split_at) return 1;
    // the first part only: later parts of the submission run on the GPU, while done and issued stay as they were
    return split_idle && split_draws && !part_start && vk_sync_fence_done() >= vk_sync_issued();
}

static void maybe_split(VkExec *ex) {
    if (split_n < 0) {
        split_n = cfg_int("NX2NAVI_SPLIT_DRAWS");
        if (split_n < 0) split_n = 0;
        int us = cfg_int("NX2NAVI_SPLIT_US");
        split_ns = us > 0 ? (uint64_t)us * 1000 : 0;
        split_idle = split_n && cfg_bool("NX2NAVI_SPLIT_IDLE");
        split_in_pass = split_n && cfg_bool("NX2NAVI_SPLIT_IN_PASS");
        rt_log("vk: replay command buffer split every %d draws %s, parts of at least %d us%s%s", split_n,
               split_in_pass ? "(inside passes too)" : "between passes", us > 0 ? us : 0, split_n ? "" : " (off)",
               split_idle ? "; a first part at once while the GPU is idle" : "");
        split_at = split_n ? split_n : INT_MAX;
    }
    if (!split_n) return;
    if (Q.drained < Q.out) {   // groups the submitting thread went on past go into the part first, once done
        if (__atomic_load_n(&Q.done, __ATOMIC_ACQUIRE) < Q.out) return;   // split_draws stays: a later op asks again
        group_drain(ex, PW_SPLIT);
    }
    if (vk_exec_recording_failed()) return;
    // the first part while the GPU is idle: at a pass boundary only, where a split costs no pass restart
    int idle = !ex->in_pass && split_idle && split_draws && !part_start && vk_sync_fence_done() >= vk_sync_issued();
    if (!idle) {
        if (ex->in_pass && !split_in_pass) return;
        if (split_draws < split_n) return;
        if (split_ns && kern_now_ns() - part_t0 < split_ns) {   // split_draws stays: a later draw asks again
            split_at = split_draws + 8;
            return;
        }
    }
    split_draws = 0;
    split_at = split_n;
    if (ex->recorded == part_start || !vk_exec_pool_split(ex->fence)) return;
    if (idle) n_idle_parts++;
    if (ex->in_pass) vk_exec_pass_end(ex), n_pass_splits++;   // the next draw begins the pass again, in the new part
    ex->cb = vk_sync_split();
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    n_parts++;
}

// ---- event order across the game's queues
// Ocarina runs work on a second queue that hands results back and forth with the main one by events: each frame the
// worker submits first, its command buffer waits (WaitEvent) for a signal the main queue gives in its submission a few
// submits later, does a step, signals, waits for the next point of the main queue's frame, and so on, while the main
// queue waits for the worker's signals before it uses the results (Kokiri Forest: about 13 waits a frame, 7 of them
// reached before their signal in submission order). On one Vulkan queue in submission order the worker's steps would
// run before the work they wait for and read the previous frame's results: Link and the other characters were
// drawn much too dark, at any time of day.
// So a top-level wait whose condition does not hold in replay order (vk_exec_wait_event: memory, or the signals
// replayed so far) stops its submission there: the rest becomes a continuation of that queue (Cont), and later
// submissions of the queue line up behind it. After every replayed signal the continuations whose wait now holds go
// on, inside the command buffer being recorded: the open pass ends, a full barrier orders them after what came before
// (the event's memory dependency), the queue's own state replaces the running one while they run, and another barrier
// and the running state come back after. One that would still wait after CONT_MAX submissions goes on regardless, as
// it did before (counted, logged). Only with the replay on the submitting thread: the recording copies a continuation
// reads stay alive there until the game recycles them, which it does only after the work ran on the console.
// NX2NAVI_EVENT_ORDER=0: every submission in the order it came.
#define CONT_MAX 36
#define EVO_QUEUES 16
typedef struct {
    RcBudgetTicket budget;
    unsigned refs;
    p_bind_buf bindings[sizeof X.ubo / sizeof(p_bind_buf)];
} ParkedUbo;

typedef struct Cont {
    struct Cont *next;      // the next submission of the same queue
    RcBudgetTicket budget;
    size_t bytes;
    int first, limit;       // admitted bounds in the incoming handle array
    int n, i;
    unsigned refs;         // readiness plan and, after parking, the held queue
    const op_hdr *at;       // the wait it stopped at in h[i] (NULL: not started)
    uint64_t since;         // vk_exec_submit_seq when it stopped
    int has_st;             // st holds the queue's state (else the running state carries over, as for any submission)
    uint8_t *st;
    ParkedUbo *parked_ubo; // logical uniform bindings at the first wait
    p_set_rt parked_rt;
    uint64_t parked_tex_pool;
    int32_t parked_tex_pool_count;
    VkTextureRegistrationScope *registration_scope;
    int registration_deferred;
    void **pin_leases;
    uint64_t h[];           // handles, pin leases, saved state and aligned command copies follow the header
} Cont;
static struct { uintptr_t q; Cont *head, *tail; } evq[EVO_QUEUES];
static int evo_on = -1, evo_in;     // knob; inside a continuation
static uintptr_t evo_queue;         // queue of the stream being walked (0: it cannot stop at a wait)
static int evo_pending;             // continuations held
static uint64_t evo_held, evo_resumed, evo_forced, evo_switches;   // since the last perf line
int vk_exec_held(void) { return __atomic_load_n(&evo_pending, __ATOMIC_ACQUIRE); }
#define ST_LO offsetof(VkExec, rt)
#define ST_HI offsetof(VkExec, views)
#define CONT_STATE_BYTES ((ST_HI - ST_LO + 7) & ~(size_t)7)
_Static_assert(sizeof(Cont) % 8 == 0 && sizeof(void *) % 8 == 0, "held command records stay aligned");

static void keep_rt_pins(const p_set_rt *targets, const tex_t **keep) {
    memcpy(keep, targets->color, sizeof targets->color);
    keep[8] = targets->depth;
}

static void sweep_recording_pins(const VkExec *ex) {
    const tex_t *keep[9 * (EVO_QUEUES + 1)];
    keep_rt_pins(&ex->rt, keep);
    int count = 9;
    for (unsigned i = 0; i < EVO_QUEUES; i++) {
        const Cont *c = evq[i].head;
        if (!c) continue;
        if (c->has_st) {
            p_set_rt targets;
            memcpy(&targets, c->st, sizeof targets); // ST_LO is the start of the target state
            keep_rt_pins(&targets, keep + count);
        } else if (c->parked_ubo) {
            keep_rt_pins(&c->parked_rt, keep + count);
        } else {
            continue;
        }
        count += 9;
    }
    nvn_pins_sweep(nvn_submit_count(), keep, count);
}

static const op_hdr *walk(VkExec *ex, const op_hdr *h, int depth, uint64_t owner);
static const op_hdr walk_stopped;

// ---- event order trace (NX2NAVI_EVO_TRACE=1, diagnostics)
// The order in which a frame's submissions, waits, signals and held parts were replayed, as a string of tokens:
// |<queue>.<n> a submission of n command buffers on that queue; p it lines up behind held work of its queue; m<e> a
// wait on event e that holds; u<e> one that does not; h the submission stops there; s<e>=<v> a replayed signal; r<queue>
// held work of that queue goes on; w it stops at a wait again; d it is through; F it went on without its signal;
// c<e>=<v> the CPU signals an event (nvnEventSignal, on whatever thread, where the call came between the others); L<e>
// a wait that holds only by a CPU signal that came after the submission took its copies (vk_exec_xfer.c "events the
// CPU signals"). Events and queues are numbered in the order they first appear; a value of 16 or more is written n,
// as a counter would make every frame's string its own. Every distinct string is logged once with its number, and the
// numbers of the frames are logged sixty to a line, so the frames that were replayed in another order than usual can
// be set against the pictures of the flight recorder.
#define EVT_KINDS 512
static struct {
    int on;
    PlatMutex lock;
    char buf[2048];
    int n, over;
    const void *ev[64];
    int nev;
    char *kind[EVT_KINDS];
    int nkind;
    uint64_t line_first;
    int line_n;
    char line[512];
    int line_len;
} ET = {.on = -1, .lock = PLAT_MUTEX_INIT};

static int evt_on(void) {
    if (ET.on < 0) ET.on = rt_knob("EVO_TRACE", "0");
    return ET.on;
}

void vk_exec_evt(char what, const void *event, uint32_t value) {
    if (!evt_on()) return;
    char t[40];
    int e = -1;
    plat_mutex_lock(&ET.lock);
    if (event) {
        for (e = 0; e < ET.nev && ET.ev[e] != event; e++) {}
        if (e == ET.nev && ET.nev < 64) ET.ev[ET.nev++] = event;
    }
    if (what == 's' || what == 'c') snprintf(t, sizeof t, value < 16 ? "%c%d=%u " : "%c%d=n ", what, e, value);
    else if (what == '|') snprintf(t, sizeof t, "|%d.%u ", e, value);
    else if (event) snprintf(t, sizeof t, "%c%d ", what, e);
    else snprintf(t, sizeof t, "%c ", what);
    for (const char *s = t; *s; s++) {
        if (ET.n >= (int)sizeof ET.buf - 1) { ET.over = 1; break; }
        ET.buf[ET.n++] = *s;
    }
    plat_mutex_unlock(&ET.lock);
}

void vk_exec_evt_frame(uint64_t frame) {
    if (!evt_on()) return;
    plat_mutex_lock(&ET.lock);
    ET.buf[ET.n] = 0;
    int k = 0;
    while (k < ET.nkind && strcmp(ET.kind[k], ET.buf)) k++;
    if (k == ET.nkind && ET.nkind < EVT_KINDS) {
        size_t len = strlen(ET.buf) + 1;
        if ((ET.kind[k] = malloc(len))) {
            memcpy(ET.kind[k], ET.buf, len);
            ET.nkind++;
            rt_log("evo: order %d first at frame %llu%s: %s", k, (unsigned long long)frame, ET.over ? " (cut)" : "", ET.buf);
        }
    }
    if (!ET.line_n) ET.line_first = frame, ET.line_len = 0;
    ET.line_len += snprintf(ET.line + ET.line_len, sizeof ET.line - (size_t)ET.line_len, " %d", k);
    if (++ET.line_n == 60 || ET.line_len > (int)sizeof ET.line - 16) {
        rt_log("evo: frames %llu-%llu:%s", (unsigned long long)ET.line_first, (unsigned long long)frame, ET.line);
        ET.line_n = 0;
    }
    ET.n = 0, ET.over = 0;
    plat_mutex_unlock(&ET.lock);
}

static int evo_enabled(void) {
    if (evo_on < 0) {
        const char *e = cfg_get("NX2NAVI_EVENT_ORDER");
        evo_on = !(e && !strcmp(e, "0"));
        if (evo_on && gfx_replay_async()) {
            evo_on = 0;
            rt_log("vk: event order across queues off: the replay runs on its own thread (NX2NAVI_REPLAY_ASYNC)");
        } else rt_log("vk: event order across queues %s (NX2NAVI_EVENT_ORDER)", evo_on ? "on" : "off");
    }
    return evo_on;
}

static int evq_find_slot(uintptr_t q) {
    int free_i = -1;
    for (int i = 0; i < EVO_QUEUES; i++) {
        if (evq[i].q == q) return i;
        if (free_i < 0 && !evq[i].head) free_i = i;
    }
    return free_i;
}

static int evq_slot(uintptr_t q) {
    int slot = evq_find_slot(q);
    if (slot >= 0) evq[slot].q = q;
    return slot;
}

static VkResult snapshot_metadata_allocate(size_t bytes, int clear, RcBudgetTicket *ticket, void **out,
                                           VkReplayCaptureWait *wait);

static size_t cont_command_bytes(uint64_t handle) {
    if (!handle) rt_fatal("vk: held submission has a null command handle");
    uint64_t indexed = nvn_submission_handle_bytes(handle);
    if (indexed) return indexed <= SIZE_MAX && !(indexed & 7) ? (size_t)indexed : 0;
    // Raw backend probe handles have no recording index. Walk their physical stream through OP_END.
    const op_hdr *op = nvn_op_first(handle);
    size_t bytes = 0;
    for (;;) {
        if (op->size < sizeof *op || (op->size & 7) || op->op >= OP_COUNT_)
            rt_fatal("vk: corrupt held command record at %p", (const void *)op);
        if (op->size > SIZE_MAX - bytes || op->size > UINTPTR_MAX - (uintptr_t)op) return 0;
        bytes += op->size;
        if (op->op == OP_END) return bytes;
        op = nvn_op_next(op);
    }
}

// Reserve the whole possible suffix before recording anything. The saved state is part of the same
// allocation so a subsequent wait can keep it without needing memory while recording.
static VkResult cont_prepare(const uint64_t *h, int n, int from, Cont **out, VkReplayCaptureWait *wait) {
    if (from < 0 || from >= n || *out) rt_fatal("vk: invalid held submission preparation");
    const size_t copies = (size_t)(n - from);
    size_t size = sizeof(Cont) + sizeof(uint64_t) + CONT_STATE_BYTES;
    if (copies > (SIZE_MAX - size) / (sizeof(uint64_t) + sizeof(void *))) return VK_ERROR_OUT_OF_HOST_MEMORY;
    size += copies * (sizeof(uint64_t) + sizeof(void *));
    for (int k = from; k < n; k++) {
        size_t command_bytes = cont_command_bytes(h[k]);
        if (!command_bytes || command_bytes > SIZE_MAX - size) return VK_ERROR_OUT_OF_HOST_MEMORY;
        size += command_bytes;
    }
    RcBudgetTicket budget = {0};
    void *allocation;
    VkResult result = snapshot_metadata_allocate(size, 0, &budget, &allocation, wait);
    if (result != VK_SUCCESS) return result;
    Cont *c = allocation;
    *c = (Cont){.budget = budget, .bytes = size, .first = from, .limit = n, .refs = 1};
    c->pin_leases = (void **)(c->h + copies + 1);
    c->st = (uint8_t *)(c->pin_leases + copies);
    *out = c;
    return VK_SUCCESS;
}

// Capture only when the submission actually parks. The guest's fence for this submission can signal before
// the rest of the commands run, so neither the original recording nor its target pins can be borrowed.
static Cont *cont_new(Cont *c, const uint64_t *h, int n, int from, const op_hdr *at) {
    if (!c || c->refs != 1 || c->n || n != c->limit || from < c->first || from >= n)
        rt_fatal("vk: held submission does not match its admitted suffix");
    uint8_t *copy = c->st + CONT_STATE_BYTES;
    const uint8_t *end = (const uint8_t *)c + c->bytes;
    for (int k = from; k < n; k++) {
        const void *original = (const void *)(uintptr_t)h[k];
        const size_t bytes = cont_command_bytes(h[k]);
        if (!bytes || bytes > (size_t)(end - copy)) rt_fatal("vk: held command size changed after admission");
        if (k == from && at) {
            const uintptr_t start = (uintptr_t)original, stop = (uintptr_t)at;
            if (stop < start || stop - start > bytes - sizeof *at || ((stop - start) & 7))
                rt_fatal("vk: held wait lies outside its original recording");
            c->at = (const op_hdr *)(copy + (stop - start));
        }
        c->pin_leases[k - from] = nvn_record_retain_pins(h[k]);
        memcpy(copy, original, bytes);
        c->h[k - from] = (uintptr_t)copy;
        copy += bytes;
    }
    c->n = n - from;
    c->h[c->n] = 0;
    c->since = vk_exec_submit_seq;
    c->registration_scope = vk_texture_registration_scope_retain(vk_exec_texture_registration_scope());
    c->refs++;
    return c;
}

// Both references are serialized by the recording owner. No command buffer reads this host allocation.
static void parked_ubo_release(ParkedUbo *parked) {
    if (!parked) return;
    if (!parked->refs) rt_fatal("vk: held uniform bindings lost their owner");
    if (--parked->refs) return;
    RcBudgetTicket budget = parked->budget;
    vk_replay_metadata_free(parked, &budget);
}

static void cont_free(Cont *c) {
    if (!c) return;
    if (!c->refs) rt_fatal("vk: held submission lost its budget owner");
    if (--c->refs) return;
    vk_texture_registration_scope_end(c->registration_scope);
    for (int k = 0; k < c->n; k++) nvn_record_release_pins(c->pin_leases[k]);
    parked_ubo_release(c->parked_ubo);
    // An UpdateUniformBuffer recorded on resume may still be borrowed later in this scene.
    if (c->n && vk_exec_draw_update_history_defer(c, c->bytes, &c->budget)) return;
    RcBudgetTicket budget = c->budget;
    vk_replay_metadata_free(c, &budget);
}

void vk_exec_discard_held_locked(void) {
    for (int i = 0; i < EVO_QUEUES; ++i) {
        while (evq[i].head) {
            Cont *c = evq[i].head;
            evq[i].head = c->next;
            cont_free(c);
        }
        evq[i].q = 0;
        evq[i].tail = NULL;
    }
    __atomic_store_n(&evo_pending, 0, __ATOMIC_RELEASE);
}

static void cont_push(int slot, Cont *c) {
    if (evq[slot].tail) evq[slot].tail->next = c;
    else evq[slot].head = c;
    evq[slot].tail = c;
    __atomic_add_fetch(&evo_pending, 1, __ATOMIC_RELEASE);
}

// Adopt the queue's saved state and bind its resources again.
static void st_adopt(VkExec *ex, const uint8_t *buf) {
    memcpy((uint8_t *)ex + ST_LO, buf, ST_HI - ST_LO);
    vk_exec_draw_state_adopted(ex);
    ex->rt_resolved = 0;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->tex_serial++;
    vk_exec_draw_new_cb();
}

static int cont_wait_timed_out(uint64_t sequence, uint64_t since) {
    return sequence - since > CONT_MAX;
}

static int cont_ready(const Cont *c) {
    return !c->at || vk_exec_event_met(nvn_op_payload(c->at)) || cont_wait_timed_out(vk_exec_submit_seq, c->since);
}

static void cont_registration_defer(Cont *c) {
    // The queue parked this follower before its first command and before any resource selection.
    vk_texture_registration_scope_end(c->registration_scope);
    c->registration_scope = NULL;
    c->registration_deferred = 1;
}

static VkTextureRegistrationScope *cont_registration_resume(Cont *c) {
    if (c->at || c->registration_deferred) {
        // A reached wait or an unstarted blocked follower opens a fresh horizon at the actual read.
        // Earlier recorded commands already own their selected providers through native batch holds.
        VkTextureRegistrationScope *fresh = vk_texture_registration_record_begin(NULL, 1);
        vk_texture_registration_scope_end(c->registration_scope);
        c->registration_scope = fresh;
        c->registration_deferred = 0;
    } else if (c->registration_scope) {
        vk_texture_registration_record_begin(c->registration_scope, 1);
    }
    return c->registration_scope;
}

static void cont_registration_select(VkExec *ex, VkTextureRegistrationScope *scope) {
    ex->registration_scope = scope;
    ex->tex_serial++;
    group_drain(ex, PW_SPLIT);
    if (vk_exec_recording_failed()) return;
    if (!vk_exec_pool_registration_needs_prefix(ex->fence)) return;
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    ex->cb = vk_sync_split();
    VkResult idle = vk_device_wait_idle(); // flushes any queued prefix and proves native descriptor reads are done
    if (idle != VK_SUCCESS) {
        vk_device_lost_check(idle, "retiring a bindless registration prefix");
        rt_fatal("vk: cannot reuse bindless descriptor sets after prefix wait (%d)", (int)idle);
    }
    vk_exec_pool_registration_prefix_complete(ex->fence);
    split_draws = 0;
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    ex->rt_resolved = 0;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    n_parts++;
}

// ---- the gate: a submission that begins with a wait for the CPU's signal (NX2NAVI_GATE_WAIT_MS, default 10; 0 off)
// Ocarina's main queue begins with a wait for an event the game signals from the CPU once it has written what the
// frame reads (vk_exec_xfer.c "events the CPU signals"). The render thread submits that queue without waiting for the
// signal, and polls the event afterwards, a millisecond at a time, before it presents. In Hyrule Field the signal
// came 1 to 2 ms after the submit call in about every other frame: the main thread first waits for the previous
// frame's fence and reads back what it needs. The submission then took its copies (the mirror sync and the uniform
// snapshot, 1.1 to 1.7 ms), met the wait unmet and was held; the render thread polled, presented, and the present
// replayed the frame (vk_backend.c held_resume) under a value the game's fence does not stand for, so the game's next
// wait for that fence returned with the GPU still at work and the frame after was replayed inside its submit call
// again: frames of 23 and 15 ms in turn where each takes 17.
// So when a submission would stop at such a wait before it has recorded anything, the submit call waits for the
// signal here, before the copies are taken, as the console's GPU waits at the wait itself: the copies are then
// taken after the game has finished writing, and the frame is replayed in the call the game's fence stands for.
// The leading prefix can span several recordings. Poll a closed wait only if the CPU has signalled that event
// before. When the prefix remains unresolved, synchronous replay can park it and asynchronous capture must defer.
static struct { uint64_t n, ns, late; } GW;   // since the last perf line: waits, their time, waits that ran out

// A missing entry is unresolved. In particular, raw CALL/JUMP targets have no retained extent here.
static const struct { size_t payload; int state; } gate_ops[OP_COUNT_] = {
    [OP_SET_RT] = {sizeof(p_set_rt), 1}, [OP_VIEWPORTS] = {sizeof(p_viewports), 1},
    [OP_SCISSOR] = {sizeof(p_scissor), 1}, [OP_DEPTH_RANGE] = {sizeof(p_depth_range), 1},
    [OP_BIND_PROGRAM] = {sizeof(p_bind_program), 1}, [OP_BIND_STATE] = {sizeof(p_bind_state), 1},
    [OP_BIND_VATTR] = {offsetof(p_bind_vattr, st), 1}, [OP_BIND_VSTREAM] = {offsetof(p_bind_vstream, st), 1},
    [OP_BIND_VBUF] = {offsetof(p_bind_vbuf, r), 1}, [OP_BIND_UBO] = {sizeof(p_bind_buf), 1},
    [OP_BIND_SSBO] = {sizeof(p_bind_buf), 1}, [OP_BIND_TEX] = {offsetof(p_bind_handles, h), 1},
    [OP_BIND_SAMP] = {offsetof(p_bind_handles, h), 1}, [OP_BIND_IMG] = {offsetof(p_bind_handles, h), 1},
    [OP_STENCIL] = {sizeof(p_stencil), 1}, [OP_POLY_OFFSET] = {sizeof(p_poly_offset), 1},
    [OP_SET_TEX_POOL] = {sizeof(p_set_pool), 1}, [OP_SET_SAMP_POOL] = {sizeof(p_set_pool), 1},
    [OP_COPY_STRIDE] = {sizeof(p_copy_stride), 1}, [OP_BLEND_COLOR] = {sizeof(p_blend_color), 1},
    [OP_DEPTH_CLAMP] = {sizeof(p_flag), 1}, [OP_RASTER_DISCARD] = {sizeof(p_flag), 1},
    [OP_RENDER_ENABLE] = {sizeof(p_flag), 1}, [OP_PRIM_RESTART] = {sizeof(p_prim_restart), 1},
    [OP_PUSH_CC] = {offsetof(p_push_cc, data), 1}, [OP_WAIT_EVENT] = {sizeof(p_wait_event), 0},
    [OP_SIGNAL_EVENT] = {sizeof(p_signal_event), 0}, [OP_REPORT_COUNTER] = {sizeof(p_report_counter), 0},
    [OP_COPY_B2B] = {sizeof(p_copy_b2b), 0}, [OP_UPDATE_UB] = {offsetof(p_update_ub, data), 0},
    [OP_CLEAR_COLOR] = {sizeof(p_clear_color), 0}, [OP_CLEAR_DS] = {sizeof(p_clear_ds), 0},
    [OP_DRAW] = {sizeof(p_draw), 0}, [OP_DRAW_IDX] = {sizeof(p_draw_idx), 0},
    [OP_DISPATCH] = {sizeof(p_dispatch), 0}, [OP_BARRIER] = {sizeof(p_barrier), 0},
    [OP_COPY_B2T] = {sizeof(p_copy_b2t), 0}, [OP_COPY_T2B] = {sizeof(p_copy_b2t), 0},
    [OP_COPY_T2T] = {sizeof(p_copy_t2t), 0}, [OP_RESET_COUNTER] = {sizeof(p_reset_counter), 0},
    [OP_CLEAR_TEX] = {sizeof(p_clear_tex), 0}, [OP_RENDER_COND] = {sizeof(p_render_cond), 0},
    [OP_DRAW_IDX_IND] = {sizeof(p_draw_idx_ind), 0}, [OP_DISPATCH_IND] = {sizeof(p_dispatch_ind), 0},
    [OP_CLEAR_BUF] = {sizeof(p_clear_buf), 0}, [OP_DRAW_MESH] = {sizeof(p_draw_mesh), 0},
    [OP_DRAW_MESH_IND] = {sizeof(p_draw_mesh_ind), 0},
};

// The fixed part of the payload has already been checked. Counted tails may be shorter than their C array type.
static int gate_payload_valid(const op_hdr *h) {
    const void *payload = nvn_op_payload(h);
    size_t tail = h->size - sizeof *h - gate_ops[h->op].payload;
    switch (h->op) {
    case OP_BIND_VATTR: {
        const p_bind_vattr *b = payload;
        return b->n >= 0 && (size_t)b->n <= tail / sizeof b->st[0];
    }
    case OP_BIND_VSTREAM: {
        const p_bind_vstream *b = payload;
        return b->n >= 0 && (size_t)b->n <= tail / sizeof b->st[0];
    }
    case OP_BIND_VBUF: {
        const p_bind_vbuf *b = payload;
        return b->n >= 0 && (size_t)b->n <= tail / sizeof b->r[0];
    }
    case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: {
        const p_bind_handles *b = payload;
        return b->n >= 0 && (size_t)b->n <= tail / sizeof b->h[0];
    }
    case OP_PUSH_CC: return ((const p_push_cc *)payload)->size <= tail;
    case OP_UPDATE_UB: return ((const p_update_ub *)payload)->size <= tail;
    case OP_WAIT_EVENT: {
        const p_wait_event *w = payload;
        return w->val && !((uintptr_t)w->val % _Alignof(uint32_t)) &&
            (w->mode == NVN_EVENT_WAIT_MODE_EQUAL || w->mode == NVN_EVENT_WAIT_MODE_GEQUAL_WRAP);
    }
    default: return 1;
    }
}

// Walk the retained prefix through the first operation that does work. Each valid step advances within
// its recording's extent, so a long state-only prefix never becomes a false proof that a later wait is open.
static int gate_scan(const uint64_t *handles, int n, const p_wait_event **blocked) {
    *blocked = NULL;
    for (int i = 0; i < n; i++) {
        uint64_t handle = handles[i];
        uintptr_t base = (uintptr_t)handle;
        uint64_t bytes = nvn_submission_handle_bytes(handle);
        if (!base || (uint64_t)base != handle || (base & 7) || (bytes & 7) || bytes < 2 * sizeof(op_hdr) ||
            bytes > UINTPTR_MAX - base) return 0;
        const op_hdr *begin = nvn_op_first(handle);
        const uint8_t *commands = (const uint8_t *)begin;
        size_t last = (size_t)bytes - sizeof(op_hdr);
        const op_hdr *end = (const op_hdr *)(commands + last);
        if (begin->op != OP_BEGIN || begin->size != sizeof *begin || end->op != OP_END || end->size != sizeof *end)
            return 0;
        for (size_t at = sizeof *begin; at < last;) {
            size_t remaining = last - at;
            if (remaining < sizeof(op_hdr)) return 0;
            const op_hdr *h = (const op_hdr *)(commands + at);
            if (h->size < sizeof *h || (h->size & 7) || h->size > remaining || h->op >= OP_COUNT_) return 0;
            size_t minimum = gate_ops[h->op].payload;
            if (!minimum || h->size - sizeof *h < minimum || !gate_payload_valid(h)) return 0;
            if (h->op == OP_WAIT_EVENT) {
                const p_wait_event *w = nvn_op_payload(h);
                if (!vk_exec_event_holds_now(w)) { *blocked = w; return 1; }
            } else if (!gate_ops[h->op].state) return 1;
            at += h->size;
        }
    }
    return 1;
}

int vk_exec_gate_wait(const uint64_t *handles, int n) {
    // Only the guest submitter has a queue here. The asynchronous replay thread
    // receives the completed snapshot and must never wait a second time.
    if (!gfx_submit_queue) return 1;
    if (n < 0 || (n && !handles)) return 0;
    int async = gfx_replay_async();
    if (!async && !evo_enabled()) return 1;
    if (vk_sync_stopping() || vk_device_is_lost()) return 0;
    const p_wait_event *w;
    if (!gate_scan(handles, n, &w)) return 0;
    if (!w) return 1;
    if (!vk_exec_event_cpu_known((const void *)w->val)) return 0;
    static int64_t cap_ns = -1;
    if (cap_ns < 0) {
        int ms = cfg_int("NX2NAVI_GATE_WAIT_MS");
        cap_ns = ms > 0 ? (int64_t)ms * 1000000 : 0;
    }
    if (!cap_ns) return 0;
    uint64_t t0 = kern_now_ns(), now = t0;
    for (unsigned i = 0; !vk_exec_event_holds_now(w); i++) {
        if (!(i & 15) && (vk_sync_stopping() || vk_device_is_lost())) return 0;
        if ((i & 15) == 15 && (now = kern_now_ns()) - t0 >= (uint64_t)cap_ns) {
            GW.late++;
            break;
        }
        if (i < 20000) plat_pause();
        else plat_yield();
    }
    uint64_t elapsed = kern_now_ns() - t0;
    int opened = vk_exec_event_holds_now(w);
    // Recheck the ordered prefix once after polling. A later closed wait stays unresolved without starting
    // another timer, and earlier waits must still hold before the caller can take the copy.
    int ready = opened && !vk_sync_stopping() && !vk_device_is_lost() && gate_scan(handles, n, &w) && !w;
    if (async) {
        static unsigned reports;
        if (reports < 12) {
            ++reports;
            rt_log("vk: uniform snapshot after CPU gate: %s after %.3f ms, submit %llu",
                   ready ? "open" : opened ? "unresolved" : "timed out", elapsed / 1e6,
                   (unsigned long long)nvn_submit_count());
        }
    }
    GW.n++, GW.ns += elapsed;
    if (GW.n >= 600) {
        if (rt_trace & RT_TRACE_PERF)
            rt_log("perf: gate: %llu submissions waited for the CPU's signal before their copies, %.2f ms on average; "
                   "%llu waited %lld ms without it and were held", (unsigned long long)GW.n, (double)GW.ns / (double)GW.n / 1e6,
                   (unsigned long long)GW.late, (long long)(cap_ns / 1000000));
        GW.n = GW.ns = GW.late = 0;
    }
    return ready;
}

int vk_exec_held_ready_locked(void) {
    for (int s = 0; evo_pending && s < EVO_QUEUES; s++)
        if (evq[s].head && cont_ready(evq[s].head)) return 1;
    return 0;
}

// Held work that goes on in a submission without commands of its own (the present thread's empty one after its blit,
// vk_backend.c present_main, or the one a waiting present makes, present_held) gets what any submission's recording
// gets: the VRAM mirror sync before and the write-back after (vk_mem_mirror_begin/end), and a uniform copy (vk_usnap).
// Without them its draws read mirrors that missed the CPU's writes since the last submission, and its GPU writes never
// went back to guest memory: the next vk_mem_mirror_begin dropped the write-back list. NX2NAVI_HELD_SYNC=0: as before.
static int held_sync(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("HELD_SYNC", "1");
    return on;
}

// Walks the continuation from where it stopped; 1 when it stopped at another wait (c->i, c->at updated), 0 when done.
static int cont_walk(VkExec *ex, Cont *c) {
    RP_SCOPE(rp_cont_walk, RP_CONT_WALK);
    RP_VALUE(RPK_CONT_SINCE, c->since);
    RP_VALUE(RPK_CONT_INDEX, c->i);
    RP_VALUE(RPK_HANDLES, c->n);
    if (c->at && !vk_exec_event_met(nvn_op_payload(c->at))) {
        evo_forced++;
        vk_exec_evt('F', NULL, 0);
        static int logged;
        if (logged++ < 8)
            rt_log("vk: event order: a held submission goes on after %llu submissions without its signal",
                   (unsigned long long)(vk_exec_submit_seq - c->since));
    }
    for (; c->i < c->n; c->i++) {
        const op_hdr *from = c->at ? nvn_op_next(c->at) : nvn_op_first(c->h[c->i]);
        c->at = NULL;
        const op_hdr *stop = walk(ex, from, 0, c->h[c->i]);
        if (stop == &walk_stopped) return -1;
        if (stop) {
            c->at = stop;
            c->since = vk_exec_submit_seq;
            return 1;
        }
    }
    return 0;
}

// Runs the held continuations whose wait holds now, in the command buffer being recorded, until none can go on.
// The projection and recorder iterate ready queue slots in the same ordered passes.
static int evo_scan(void *context, int (*resume_slot)(void *, int)) {
    int ran = 0;
    for (int progress = 1; progress;) {
        progress = 0;
        for (int slot = 0; slot < EVO_QUEUES; ++slot) {
            int result = resume_slot(context, slot);
            if (result < 0) return -1;
            progress |= result;
        }
        ran |= progress;
    }
    return ran;
}

static int evo_resume_slot(void *context, int s) {
    VkExec *ex = context;
    uintptr_t run_q = evo_queue;
    int stopped = 0;
    Cont *c = evq[s].head;
    if (!c || !cont_ready(c)) return 0;
    evo_switches++;
    vk_exec_brk++;
    if (RR.on > 0) runs_break(RB_HELD);
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);
    // The recorder serializes queue switches and evo_in prevents recursive resume.
    static uint8_t mine[ST_HI - ST_LO];
    memcpy(mine, (uint8_t *)ex + ST_LO, sizeof mine);
    if (c->has_st) st_adopt(ex, c->st);
    else if (c->parked_ubo) {
        memcpy(ex->ubo, c->parked_ubo->bindings, sizeof ex->ubo);
        ex->rt = c->parked_rt;
        ex->rt_resolved = 0;
        ex->tex_pool = c->parked_tex_pool;
        ex->tex_pool_count = c->parked_tex_pool_count;
        parked_ubo_release(c->parked_ubo);
        c->parked_ubo = NULL;
        vk_exec_draw_state_adopted(ex);
    }
    VkTextureRegistrationScope *run_scope = ex->registration_scope;
    evo_in = 1, evo_queue = evq[s].q;
    for (;;) {
        cont_registration_select(ex, cont_registration_resume(c));
        evo_resumed++;
        vk_exec_evt('r', (const void *)evq[s].q, 0);
        int outcome = cont_walk(ex, c);
        group_drain(ex, PW_SPLIT);
        if (c->registration_scope) vk_texture_registration_record_end(c->registration_scope, 1);
        if (vk_exec_recording_failed()) outcome = -1;
        if (outcome < 0) { stopped = 1; break; }
        if (outcome) {   // waits again: the queue's state stays with it
            vk_exec_evt('w', NULL, 0);
            vk_exec_pass_end(ex);
            memcpy(c->st, (uint8_t *)ex + ST_LO, ST_HI - ST_LO);
            c->has_st = 1;
            break;
        }
        evq[s].head = c->next;   // done: the queue's next submission starts with the state this one left
        if (!evq[s].head) evq[s].tail = NULL;
        __atomic_sub_fetch(&evo_pending, 1, __ATOMIC_RELEASE);
        vk_exec_evt('d', NULL, 0);
        ex->registration_scope = run_scope;
        cont_free(c);
        c = evq[s].head;
        if (!c) break;
        c->has_st = 0;
    }
    evo_in = 0, evo_queue = run_q;
    if (!stopped) { vk_exec_pass_end(ex); vk_exec_barrier(ex); }
    else resource_targets_forget(ex);
    memcpy((uint8_t *)ex + ST_LO, mine, ST_HI - ST_LO);   // the running stream's state back
    cont_registration_select(ex, run_scope);
    vk_exec_draw_state_adopted(ex);
    ex->rt_resolved = 0;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->tex_serial++;
    vk_exec_draw_new_cb();
    return stopped ? -1 : 1;
}

// Runs the held continuations whose wait holds now, in the command buffer being recorded, until none can go on.
static int evo_resume(VkExec *ex) {
    if (evo_in || !evo_pending) return 1;
    RP_SCOPE(rp_resume, RP_CONT_RESUME);
    RP_VALUE(RPK_HELD_BEFORE, evo_pending);
    vk_exec_signalled = 0;
    return evo_scan(ex, evo_resume_slot) >= 0;
}

// ---- looking ahead for pipelines (NX2NAVI_PIPE_AHEAD, default 1; 0 off)
// A draw whose pipeline is not there yet waits for its fast link or is skipped until a worker has built it, and the
// next draw with a new pipeline is only found when the replay reaches it: an area's first frame, or a turn that brings
// a dozen new materials into view, paid for them one after another (each wait a few milliseconds with the stage
// libraries in the driver's cache, 100 ms and more without), or showed frames with those draws missing while the
// workers that could have built them all at once sat idle.
// The submission is recorded before it is replayed, so what comes after the draw is known. The first draw of a
// submission that finds a shader or pipeline missing has the rest read here (vk_exec_ahead), from the op after it to
// the end of the submission's last command buffer: the scan applies the binds that decide a draw's shaders and its
// shader libraries (program, state objects, vertex attributes, textures and images, render targets, depth clamp) to a
// copy of the replay state and, at each draw, looks its shaders up as the draw will and names the state to
// vk_pipeline_hint, which queues the libraries not built yet. Nothing is recorded and no Vulkan object is touched;
// a shader not in memory is queued for its disk read or translation by the lookup itself. The draws then find their
// libraries built, or building on several workers at once.
// What the scan does not follow: the op stream past a wait for another queue's signal (what follows may be held), a
// change of the texture pool, mesh draws (their pipelines have no libraries), and whatever would take it past
// AHEAD_NS. Render targets are read from their descriptors (format, view format, sample count), which is what the
// replay resolves them to unless a target is left out; a guess that misses costs two libraries at most.
#define AHEAD_NS 3000000ull
#define AHEAD_OPS 400000u
static struct {
    int on, live;                      // on: -1 until read; live: vk_exec_record walks handles[i] now
    const uint64_t *handles;
    int n, i;
    const op_hdr *at[CALL_DEPTH + 1];  // the op being replayed at each call depth
    int depth;
    uint64_t seq;                      // the submission scanned last
    uint64_t scans, ops, cut, ns, max_ns, logged_scans;
} AH = {.on = -1};

static void ahead_set_rt(VkExec *sx, const p_set_rt *r) {
    sx->rt = *r;
    int n = r->n < 0 ? 0 : r->n > 8 ? 8 : r->n, samples = 0;
    for (int i = 0; i < 8; i++) {
        const tex_t *t = i < n ? r->color[i] : NULL;
        sx->pipe.color_fmt[i] = !t ? 0 : r->has_cview[i] && r->cview[i].set_format ? (int32_t)r->cview[i].format : t->d.format;
        if (t && !samples) samples = t->d.samples > 1 ? t->d.samples : 1;
    }
    sx->pipe.depth_fmt = r->depth ? r->depth->d.format : 0;
    if (r->depth && !samples) samples = r->depth->d.samples > 1 ? r->depth->d.samples : 1;
    sx->pipe.samples = (uint8_t)(samples ? samples : 1);
}

void vk_exec_ahead(VkExec *ex) {
    if (AH.on < 0) {
        AH.on = cfg_bool("NX2NAVI_PIPE_AHEAD");
        rt_log("vk: a draw that finds its pipeline missing has the rest of its submission read ahead for the stage "
               "libraries the later draws need: %s (NX2NAVI_PIPE_AHEAD)", AH.on ? "on" : "off");
    }
    if (!AH.on || !AH.live || evo_in || rw_chunk || AH.seq == vk_exec_submit_seq) return;   // not in a chunk
    AH.seq = vk_exec_submit_seq;
    uint64_t t0 = kern_now_ns();
    uint32_t ops = 0;
    static VkExec sx;   // a copy of the replay state: too large for the stack
    sx = *ex;
    vk_exec_ahead_begin();
    const op_hdr *ret[CALL_DEPTH + 1];
    int depth = AH.depth, i = AH.i;
    for (int d = 0; d < depth; d++) ret[d] = AH.at[d];
    const op_hdr *h = nvn_op_next(AH.at[depth]);
    for (;;) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_) break;   // the replay reports it when it gets there
        if (++ops >= AHEAD_OPS || (!(ops & 63) && kern_now_ns() - t0 > AHEAD_NS)) {
            AH.cut++;
            break;
        }
        const void *a = nvn_op_payload(h);
        int stop = 0;
        switch (h->op) {
        case OP_END:
            if (depth > 0) h = ret[--depth];   // back in the caller: on from the op after its call, below
            else if (++i < AH.n) {
                h = nvn_op_first(AH.handles[i]);
                continue;
            } else stop = 1;
            break;
        case OP_JUMP: h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target; continue;
        case OP_CALL:
            if (depth >= CALL_DEPTH) {
                stop = 1;
                break;
            }
            ret[depth++] = h;
            h = nvn_op_first(((const p_jump *)a)->target);
            continue;
        case OP_WAIT_EVENT: stop = 1; break;
        case OP_SET_TEX_POOL: stop = ((const p_set_pool *)a)->incarnation != sx.tex_pool; break;
        case OP_SET_RT:
            if (memcmp(&sx.rt, a, sizeof sx.rt)) ahead_set_rt(&sx, a);
            break;
        case OP_DEPTH_CLAMP: sx.pipe.depth_clamp = ((const p_flag *)a)->value != 0; break;
        case OP_RASTER_DISCARD: sx.pipe.raster_discard = ((const p_flag *)a)->value != 0; break;
        case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_TEX: case OP_BIND_IMG:
        case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND:
            vk_exec_ahead_op(&sx, h->op, a);
            break;
        default: break;
        }
        if (stop) break;
        h = nvn_op_next(h);
    }
    uint64_t ns = kern_now_ns() - t0;
    AH.scans++, AH.ops += ops, AH.ns += ns;
    if (ns > AH.max_ns) AH.max_ns = ns;
    if ((rt_trace & RT_TRACE_PERF) && AH.scans - AH.logged_scans >= 100) {
        rt_log("perf: looking ahead: %llu scans so far, %.0f ops and %.2f ms each (longest %.2f ms, %llu ended at their "
               "limit); %llu draws seen, %llu looked up", (unsigned long long)AH.scans, (double)AH.ops / (double)AH.scans,
               (double)AH.ns / 1e6 / (double)AH.scans, (double)AH.max_ns / 1e6, (unsigned long long)AH.cut,
               (unsigned long long)vk_exec_ahead_draws, (unsigned long long)vk_exec_ahead_lookups);
        AH.logged_scans = AH.scans;
    }
}

// Ops that only store state or draw: nothing they record runs outside a pass, and the walk goes on with the op after
// them. Counter reports and resets are among them: they close and open an occlusion query inside the pass, and the
// value reaches guest memory from the completion thread once the submission's fence has passed (vk_exec_xfer.c
// oq_cb, counter_cb), never from the GPU inside the run; so are the render conditions, which the replay decides on
// the CPU. Ocarina reports a counter after most of its indirect mesh draws. Any other op the walk replays bumps
// vk_exec_brk, and so does a held submission that goes on in between (evo_resume): what was prepared for the draws
// of a run of such ops (vk_exec_draw.c, the indirect mesh draws' records) holds only while it stays the same.
uint32_t vk_exec_brk;

static int op_linear(uint32_t op) {
    switch (op) {
    case OP_BEGIN: case OP_JUMP: case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_STENCIL:
    case OP_POLY_OFFSET: case OP_BLEND_COLOR: case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD: case OP_PRIM_RESTART:
    case OP_COPY_STRIDE: case OP_SET_SAMP_POOL: case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND:
    case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF: case OP_BIND_UBO:
    case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_PUSH_CC: case OP_DRAW_MESH:
    case OP_DRAW_MESH_IND: case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_ENABLE: case OP_RENDER_COND:
        return 1;
    default: return 0;
    }
}

#define MESH_RUN_OPS 2048
static uint32_t mesh_run_stop[OP_COUNT_];   // the op each run of vk_exec_mesh_run ended at, since the start

// For the perf log: the ops the runs ended at most often.
void vk_exec_mesh_run_stops(char *out, size_t n) {
    uint32_t c[OP_COUNT_];
    memcpy(c, mesh_run_stop, sizeof c);
    size_t at = 0;
    *out = 0;
    for (int k = 0; k < 4; k++) {
        int best = 0;
        for (int i = 1; i < OP_COUNT_; i++)
            if (c[i] > c[best]) best = i;
        if (!c[best]) break;
        int w = snprintf(out + at, n - at, "%s%u at %s", k ? ", " : "", c[best], nvn_op_name((uint32_t)best));
        if (w < 0 || (size_t)w >= n - at) break;
        at += (size_t)w, c[best] = 0;
    }
}

// The indirect mesh draws that follow the op being replayed (cur: its payload) with only ops of op_linear between
// them, MESH_RUN_OPS at most: their payloads into out, at most max. 0 when the walk is not at cur.
uint32_t vk_exec_brk_now(void) { return rw_chunk ? rw_chunk->brk : vk_exec_brk; }

int vk_exec_mesh_run(const void *cur, const void **out, int max) {
    const op_hdr *h = rw_chunk ? rw_op : AH.at[AH.depth];   // a chunk reads on from its own op (into the next ones)
    if (!h || nvn_op_payload(h) != cur) return 0;
    int n = 0;
    h = nvn_op_next(h);
    for (int ops = 0; ops < MESH_RUN_OPS && n < max; ops++) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_) break;
        if (!op_linear(h->op)) {
            mesh_run_stop[h->op]++;
            break;
        }
        if (h->op == OP_JUMP) {
            h = (const op_hdr *)(uintptr_t)((const p_jump *)nvn_op_payload(h))->target;
            continue;
        }
        if (h->op == OP_DRAW_MESH_IND) out[n++] = nvn_op_payload(h);
        h = nvn_op_next(h);
    }
    return n;
}

// One op of the stream other than those that steer the walk (END, JUMP, CALL and WAIT_EVENT: walk).
static inline void op_payload(VkExec *ex, uint32_t op, const void *a) {
    switch (op) {
    case OP_BEGIN: break;
    case OP_SET_RT: set_rt(ex, a); break;
    case OP_CLEAR_COLOR: clear_color(ex, a); break;
    case OP_CLEAR_DS: clear_ds(ex, a); break;
    case OP_VIEWPORTS: {
        const p_viewports *v = a;
        for (int i = 0; i < v->n && v->first + i < 16; i++)
            if (v->first + i >= 0) memcpy(ex->vp[v->first + i], v->v[i], sizeof ex->vp[0]);
        ex->has_vp = 1;
        ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
        break;
    }
    case OP_SCISSOR:
        ex->scissor = *(const p_scissor *)a;
        ex->has_scissor = 1;
        ex->dirty |= VK_EXEC_DIRTY_SCISSOR;
        break;
    case OP_DEPTH_RANGE: {
        const p_depth_range *r = a;
        ex->depth_range[0] = r->n, ex->depth_range[1] = r->f;
        ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
        break;
    }
    case OP_STENCIL: {
        const p_stencil *st = a;
        for (int f = 0; f < 2; f++) {
            if (!(st->face & (1 << f))) continue;
            if (st->which == 0) ex->st_vmask[f] = (uint32_t)st->value_mask;
            else if (st->which == 1) ex->st_wmask[f] = (uint32_t)st->mask;
            else ex->st_ref[f] = (uint32_t)st->ref;
        }
        ex->dirty |= VK_EXEC_DIRTY_STENCIL;
        break;
    }
    case OP_POLY_OFFSET:
        ex->poly_offset = *(const p_poly_offset *)a;
        ex->dirty |= VK_EXEC_DIRTY_DEPTH_BIAS;
        break;
    case OP_BLEND_COLOR:
        memcpy(ex->blend_color, ((const p_blend_color *)a)->rgba, sizeof ex->blend_color);
        ex->dirty |= VK_EXEC_DIRTY_BLEND_CONST;
        break;
    case OP_DEPTH_CLAMP: ex->pipe.depth_clamp = ((const p_flag *)a)->value != 0; break;
    case OP_RASTER_DISCARD: ex->pipe.raster_discard = ((const p_flag *)a)->value != 0; break;
    case OP_PRIM_RESTART: ex->prim_restart = *(const p_prim_restart *)a; break;
    case OP_COPY_STRIDE: ex->copy_stride = ((const p_copy_stride *)a)->stride; break;
    case OP_SET_TEX_POOL: {
        const p_set_pool *pool = a;
        if (ex->tex_pool != pool->incarnation) {
            ex->tex_pool = pool->incarnation;
            ex->tex_pool_count = pool->count;
            cont_registration_select(ex, ex->registration_scope);
        }
        break;
    }
    case OP_SET_SAMP_POOL: ex->samp_pool = ((const p_set_pool *)a)->pool; break;
    case OP_BARRIER: barrier_to_why(ex, NULL, NULL, 0, BR_GAME); break;
    case OP_DRAW: case OP_DRAW_IDX: case OP_DISPATCH: case OP_UPDATE_UB: case OP_DRAW_IDX_IND: case OP_DISPATCH_IND:
        if (!rw_chunk && (op == OP_DRAW || op == OP_DRAW_IDX || op == OP_DRAW_IDX_IND))
            vk_exec_frame.draws_seen++, split_draws++;
        if (op == OP_DISPATCH || op == OP_DISPATCH_IND) vk_exec_frame.dispatches++, split_draws++;
        // fall through
    case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
    case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_PUSH_CC:
        vk_exec_op_draw(ex, op, a);
        break;
    case OP_DRAW_MESH: case OP_DRAW_MESH_IND:
        if (!rw_chunk) vk_exec_frame.draws_seen++;
        vk_exec_op_draw(ex, op, a);
        break;
    case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX:
        if (op != OP_CLEAR_TEX) vk_exec_frame.copies++;
        if (vk_dump_armed) vk_dump_copy(ex, op, a);
        // fall through
    case OP_SIGNAL_EVENT: case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_ENABLE: case OP_RENDER_COND:
    case OP_CLEAR_BUF:
        vk_exec_op_xfer(ex, op, a);
        break;
    default: rt_fatal("vk: op %s has no replay handler", nvn_op_name(op));
    }
}

#if NX2NAVI_REPLAY_WORK && NX2NAVI_REPLAY_WORK_CHAIN
static unsigned work_operation_region(uint32_t opcode) {
    switch(opcode) {
    case OP_DRAW:case OP_DRAW_IDX:case OP_DRAW_IDX_IND:return RWT_REGION_OP_DRAW;
    case OP_DRAW_MESH:case OP_DRAW_MESH_IND:return RWT_REGION_OP_MESH;
    case OP_DISPATCH:case OP_DISPATCH_IND:return RWT_REGION_OP_COMPUTE;
    case OP_UPDATE_UB:return RWT_REGION_OP_UPDATE;
    case OP_COPY_B2B:case OP_COPY_B2T:case OP_COPY_T2B:case OP_COPY_T2T:case OP_CLEAR_TEX:
    case OP_SIGNAL_EVENT:case OP_REPORT_COUNTER:case OP_RESET_COUNTER:case OP_RENDER_ENABLE:
    case OP_RENDER_COND:case OP_CLEAR_BUF:case OP_CLEAR_COLOR:case OP_CLEAR_DS:return RWT_REGION_OP_TRANSFER;
    default:return RWT_REGION_OP_STATE;
    }
}
#endif
static void op_one(VkExec *ex, const op_hdr *h) {
    RWT_CHAIN(work_operation,work_operation_region(h->op));
    op_payload(ex,h->op,nvn_op_payload(h));
}

static void graphics_noop_record(VkExec *ex, const op_hdr *op) {
    if (ex->render_disabled) vk_cond_draws_skip += ex->has_render_cond;
    if (rw_chunk) return;
    vk_exec_frame.draws_seen++;
    if (op->op != OP_DRAW_MESH && op->op != OP_DRAW_MESH_IND) split_draws++;
}

// These functions are selected once while the immutable execution plan is built. The draw consumer supplies its
// own typed entry points; other ordered effects retain their authoritative implementation with a fixed operation.
#define EXEC_FIXED_OPS(X) \
    X(OP_BEGIN) X(OP_SET_RT) X(OP_CLEAR_COLOR) X(OP_CLEAR_DS) X(OP_VIEWPORTS) X(OP_SCISSOR) \
    X(OP_DEPTH_RANGE) X(OP_STENCIL) X(OP_POLY_OFFSET) X(OP_BLEND_COLOR) X(OP_DEPTH_CLAMP) \
    X(OP_RASTER_DISCARD) X(OP_PRIM_RESTART) X(OP_COPY_STRIDE) X(OP_SET_TEX_POOL) X(OP_SET_SAMP_POOL) \
    X(OP_BARRIER) X(OP_COPY_B2B) X(OP_COPY_B2T) X(OP_COPY_T2B) X(OP_COPY_T2T) X(OP_CLEAR_TEX) \
    X(OP_SIGNAL_EVENT) X(OP_REPORT_COUNTER) X(OP_RESET_COUNTER) X(OP_RENDER_ENABLE) X(OP_RENDER_COND) X(OP_CLEAR_BUF)
#define EXEC_FIXED_DEFINE(op) static void exec_##op(VkExec *ex, const void *a) { op_payload(ex, op, a); }
EXEC_FIXED_OPS(EXEC_FIXED_DEFINE)
#undef EXEC_FIXED_DEFINE

NvnExecFn gfx_exec_prepared_handler(uint32_t opcode) {
    NvnExecFn draw = vk_exec_draw_prepared_handler(opcode);
    if (draw) return draw;
    switch (opcode) {
#define EXEC_FIXED_CASE(op) case op: return exec_##op;
        EXEC_FIXED_OPS(EXEC_FIXED_CASE)
#undef EXEC_FIXED_CASE
    default: return NULL;
    }
}
#undef EXEC_FIXED_OPS

static int exec_plan_enabled(void) {
    // Per-operation diagnostics need the uncoalesced instruction sequence. Normal/checking execution consumes the
    // same prepared representation; its state oracle must never repeat draws or resource side effects.
    return resource_prep_mode() && !vk_dump_armed && RR.on <= 0 && !cfg_bool("NX2NAVI_REPLAY_TIME");
}

enum { EXEC_MAIN, EXEC_WORKER, EXEC_FRONT, EXEC_PATHS };
typedef struct { uint64_t blocks, state_ops, direct, reference, front_draws; unsigned path; } ExecPlanStats;
static ExecPlanStats exec_totals[EXEC_PATHS];
static void exec_stats_publish(ExecPlanStats *s) {
    ExecPlanStats *d = &exec_totals[s->path];
#define EXEC_PUBLISH(field) if (s->field) __atomic_add_fetch(&d->field, s->field, __ATOMIC_RELAXED)
    EXEC_PUBLISH(blocks); EXEC_PUBLISH(state_ops); EXEC_PUBLISH(direct); EXEC_PUBLISH(reference); EXEC_PUBLISH(front_draws);
#undef EXEC_PUBLISH
}
static void exec_stats_step(ExecPlanStats *s, const NvnExecStep *step) {
    if (!step || (step->opcode != NVN_EXEC_STATE && !step->run)) s->reference++;
    else if (step->opcode == NVN_EXEC_STATE) s->blocks++, s->state_ops += step->ops;
    else s->direct++;
}

static void exec_step_run(VkExec *ex, const NvnRecordPlan *plan, const NvnExecStep *step) {
    if (step->opcode == NVN_EXEC_STATE) {
        RWT_CHAIN(work_state,RWT_REGION_OP_STATE);
        vk_exec_draw_state_apply(ex, plan->commands, plan->state_data + step->state_offset);
        return;
    }
    const op_hdr *h = (const op_hdr *)(plan->commands + step->begin);
    if (!step->run) { op_one(ex, h); return; }
    RWT_CHAIN(work_operation,work_operation_region(h->op));
    // The submitting thread already counted a recording group's draws, including any that need a retry.
    if (!rw_chunk && (step->flags & (NVN_EXEC_DRAW | NVN_EXEC_MESH))) {
        vk_exec_frame.draws_seen++;
        if (step->flags & NVN_EXEC_DRAW) split_draws++;
    }
    if (step->flags & NVN_EXEC_DISPATCH) vk_exec_frame.dispatches++, split_draws++;
    step->run(ex, nvn_op_payload(h));
}

// ---- replay workers (NX2NAVI_REPLAY_WORKERS, default 3; 0: the replay records every op on the submitting thread)
// Default 3: on an RX 7900 XT with a Ryzen 7 5800X3D, Hyrule Field standing went from 41.2 to 50.4 fps (2026-11-12, one
// batch), with the same pictures there and in Kokiri Forest, Kakariko, Goron City, a graveyard house and Gerudo
// Valley; 5 or 7 workers measured the same as 3.
// A submission's draws come in runs between ops that have to stay in the replay's order on its own thread (dispatches,
// copies, texture and buffer clears, events, the game's barriers, calls). Hyrule Field, standing (NX2NAVI_REPLAY_RUNS,
// 2026-11-11): about 3,500 draws a frame in 82 runs, 96% of them in runs of 16 draws or more and 67% in runs of 256 or
// more, with a timestamp report beside nearly every draw and a render condition beside every second one.
// A run is cut into groups: a group ends where the render targets change after its first draw or clear, and before an
// inline uniform update (and a mesh draw with the mesh fallback or while NX2NAVI_GPU_PASSES copies their counts), which
// the submitting thread replays itself in the order of the stream like everything outside groups. A group of
// 2 * CHUNK_DRAWS draws or more is recorded by several threads at once:
// - the front pass, on the submitting thread, walks the group in the order of the stream before anything of it is
//   recorded: it applies the state and binds, settles the CPU side of counter reports and decides the render conditions
//   (vk_exec_xfer_front), and copies the replay state at the first op of each chunk;
// - the chunks are taken in order by the workers and the submitting thread, each recording into a command buffer of its
//   own pool with a draw context of its own, from its copy of the state, behind the full barrier that ended the
//   replay's command buffer before the group, and ending behind a full barrier of its own. vk_sync_append puts them
//   after the replay's command buffer in the same part, in the order of the stream, and the replay goes on in a new one.
// The frame's pass hooks (vk_contact_pass_begin, vk_dlaa_pass) follow the frame pass by pass: they run in chunk 0,
// which reaches the group's pass first, and the others start after that pass began; groups follow one another.
// What still depends on the order between chunks waits for its turn (vk_exec_turn: every earlier chunk of the group
// done): a texture lookup that would make or take back an entry (vk_texture.c entry_of) returns nothing and the chunk
// replays the op again once in turn, and so do the contact shadows' pass and the HDR pass's source. Per-object motion
// lists its draws under a lock (vk_exec_draw.c mvobj_draw), and the occlusion queries' slots and segments follow the
// stream (vk_exec_xfer.c vk_exec_xfer_slots, vk_exec_xfer_group_merge).
// NX2NAVI_REPLAY_PIPE=N (1-4, default 0): the submitting thread hands the group to the workers, takes chunks of it as
// they do until none is left to take, and goes on with the ops after it, in a new command buffer after the chunks'
// places in the part (vk_sync_append_later), while the last chunks are recorded; also into the front pass of the next
// group: up to N (at most LIVE_GROUPS) groups are handed out and not in their part yet, their chunks taken from one
// queue in the order of the stream (Q). Hyrule Field, 3 workers (2026-11-12): the frame's groups took 6.8 ms of the
// submitting thread and the ops between them 4.6. A group's chunk 0 runs its pass hooks after the group before began
// its pass, and waits for every earlier group where a hook acts; a turn (vk_exec_turn) waits for every earlier chunk,
// of earlier groups too.
// The submitting thread waits for the groups (group_wait), and puts their command buffers and query slots in place
// (group_drain, oldest first: groups_put_done), only where the order needs it:
// - for room when N groups are out, before a part is split off or submitted (a part due before a group goes to the
//   GPU first: its work would otherwise wait for the group's last chunk too), and before an op that chunks never
//   record (pipe_op: copies, clears of textures and buffers);
// - where a pass's hook acts: DLAA's run (vk_dlaa.c: the jitter and its end, and the per-object motion lists it takes)
//   and the contact shadows' targets (vk_contact.c) change what earlier draws read (vk_exec_hook_turn);
// - when a lookup would make a texture entry or take back a retired one: the op returns early and goes again once the
//   groups are done (op_again), as a chunk's does in turn (vk_texture.c lookup_quiet: what changes nothing, or takes
//   back a released render target that kept its image, needs no turn);
// - uniform snapshot drops count from where they are in the stream (vk_exec_epoch: a group's chunks do not see the
//   drops of ops after the group), the texture pool set syncs under the lock the chunks take, a chunk reads only the
//   inline uniform updates before its group (vk_exec_upd_top), the HDR present's source goes by stream position
//   (vk_exec_stream_pos), and the queries after a group's last report go into the next segment made after its front
//   pass (vk_exec_xfer.c GS).
// Locks a chunk takes, the submitting thread takes too while it is behind (vk_exec_shared). The chunks it records,
// at the hand-out or while it waits between two ops, use a draw context of its own (RW.help_dc).
// Off by default: Hyrule Field, 3 workers, standing (2026-11-12), it shortens the replay (10.9-11.5 against 11.7-12.1
// ms a frame) but not the frame: with 1, 2 or 4 groups out the fps stayed within one of the 50 the stage 2 mode makes,
// over three batches. A frame there is the replay (about 11.5 ms) and then the GPU's work after it (about 6.6 ms: the
// temporal upscaler about 4, the write-back copies 1.5), which the next frame waits for (nvn_api_core.c
// nvnQueueSubmitCommands, NX2NAVI_SUBMIT_FINISH; without that wait the game waits as long itself, 52.4 fps); during the
// replay the GPU keeps up with what is submitted. Sending the part's command buffers up to the first chunk place still
// being recorded while later groups record (tried) did so for at most 0.2 parts per submission and gained nothing.
#define RW_MAX VK_REPLAY_WORKER_LIMIT
#define CHUNK_DRAWS 32                  // draws a chunk takes at least
#define GROUP_PEEK 65536u               // ops read ahead for a group, at most

enum { GE_RUN, GE_TARGET, GE_UB, GE_INDEX, GE_MESH, GE_LIMIT, GE_N };

// The ops a run may hold; JUMP is followed.
static int run_op(uint32_t op) {
    return nvn_record_op_class(op) != NVN_PLAN_SERIAL;
}

typedef struct {
    PlatThread *t;
    int idx;                            // also its command pool (vk_sync_worker_begin); the submitting thread's is RW.n
    void *dc;                           // its draw context (vk_exec_draw_ctx_new)
    volatile int32_t sleeping;          // waits for wake rather than spinning
    PlatEvent wake;
} RwWorker;

static const char *const pw_name[PW_N] = {"next group", "op", "pass hooks", "turn", "texture entry", "split",
                                          "submission end", "done before"};

static struct {
    int n;                              // workers; 0 off, -1 not read yet
    int extra_chunks;
    int pipe;                           // NX2NAVI_REPLAY_PIPE: the groups out at once at most, 0 none
    void *help_dc;                      // the submitting thread's draw context for chunks it takes while it waits
    RwWorker *w[RW_MAX];
    const op_hdr *skip_to[CALL_DEPTH + 1];   // per call depth: the walk read ahead to this op, a group too small
    uint64_t groups, group_draws, chunks, serial_draws, wait_ns;   // since the last perf line
    uint64_t planned_groups, planned_draws, plan_fallbacks, front_planned_draws;
    uint64_t end_draws[GE_N];              // draws of groups by what ended them (group_peek's GE_*), since the last perf line
    uint64_t small_end[GE_N];              // ... of those too small to split
    uint64_t run_end[OP_COUNT_];        // draws of groups that an op outside groups ended, by that op
    // the submitting thread, since the last perf line: ops outside groups by kind (SER_*), and in groups the front
    // pass, its own chunks and the wait for the others'
    uint64_t ser_n[4], ser_ns[4], front_ns, own_ns, spin_ns;
    // NX2NAVI_REPLAY_PIPE, since the last perf line: waits by why (PW_*), the time from a group's hand-out to the end
    // of the wait for it, and the ops that waited (PW_OP)
    uint64_t pw_n[PW_N], pw_ns[PW_N], behind_ns, pd_op[OP_COUNT_];
    volatile uint64_t turns, turn_ns, redos, chunk_ns;   // chunk_ns: the chunks' recording, every thread's
} RW = {.n = -1};

static void group_take(int pool);

enum { SER_DRAW, SER_MESH, SER_DISPATCH, SER_OTHER };
static int ser_kind(uint32_t op) {
    switch (op) {
    case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND: return SER_DRAW;
    case OP_DRAW_MESH: case OP_DRAW_MESH_IND: return SER_MESH;
    case OP_DISPATCH: case OP_DISPATCH_IND: return SER_DISPATCH;
    default: return SER_OTHER;
    }
}

static Chunk *chunk_take(void);
static void chunk_record(Chunk *c, int pool);

// ---- side work (vk_exec_par): the replay thread hands the workers a loop over n items while they have no chunk to
// record, at a submission's start (vk_usnap.c copy_regions). One job at a time, in a static slot: seq is odd while it
// runs. A worker counts itself in (PJ.in) and checks seq again before it reads the job, and the caller, once every
// slice is done, makes seq even and waits until no worker is in, so none reads a job that has ended.
static struct {
    void (*fn)(void *, int, int);
    void *arg;
    int n, grain;
    volatile int32_t next, left, in;
    volatile uint32_t seq;
} PJ;
static PlatMutex pj_lock = PLAT_MUTEX_INIT;

// Only a thread already waiting on this group takes the second half. Claim the pointer before reading it.
// The publisher holds the generation lock. The callback must use only its own output and take no renderer locks.
static void chunk_pair_help(Group *g) {
    if (!__atomic_load_n(&g->pair, __ATOMIC_RELAXED)) return;
    ChunkPair *job = __atomic_exchange_n(&g->pair, NULL, __ATOMIC_ACQ_REL);
    if (!job) return;
    job->run(job->arg, 1);
    __atomic_store_n(&job->done, 1, __ATOMIC_RELEASE);
}

int vk_exec_chunk_pair(void (*run)(void *, unsigned), void *arg) {
    if (!rw_chunk || rw_chunk->g->n < 2) return 0;
    Group *g = rw_chunk->g;
    ChunkPair job = {.run = run, .arg = arg};
    ChunkPair *empty = NULL;
    if (!__atomic_compare_exchange_n(&g->pair, &empty, &job, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) return 0;
    // Busy peers keep recording. Withdraw an unclaimed offer and let the caller use a single OS query.
    for (unsigned i = 0; i < 16 && __atomic_load_n(&g->pair, __ATOMIC_RELAXED); i++) plat_pause();
    ChunkPair *offered = &job;
    if (__atomic_compare_exchange_n(&g->pair, &offered, NULL, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) return 0;
    run(arg, 0);
    while (!__atomic_load_n(&job.done, __ATOMIC_ACQUIRE)) plat_pause();
    return 1;
}

static void par_slices(void) {
    for (;;) {
        int lo = __atomic_fetch_add(&PJ.next, PJ.grain, __ATOMIC_ACQ_REL);
        if (lo >= PJ.n) return;
        int hi = lo + PJ.grain < PJ.n ? lo + PJ.grain : PJ.n;
        PJ.fn(PJ.arg, lo, hi);
        __atomic_sub_fetch(&PJ.left, hi - lo, __ATOMIC_ACQ_REL);
    }
}

static int par_pending(void) { return __atomic_load_n(&PJ.seq, __ATOMIC_ACQUIRE) & 1; }

static void par_help(void) {
    uint32_t s = __atomic_load_n(&PJ.seq, __ATOMIC_ACQUIRE);
    if (!(s & 1)) return;
    __atomic_add_fetch(&PJ.in, 1, __ATOMIC_SEQ_CST);
    if (__atomic_load_n(&PJ.seq, __ATOMIC_SEQ_CST) == s) par_slices();
    __atomic_sub_fetch(&PJ.in, 1, __ATOMIC_SEQ_CST);
}

void vk_exec_par(void (*fn)(void *, int, int), void *arg, int n, int grain) {
    if (grain < 1) grain = 1;
    if (RW.n <= 0 || n <= grain || Q.drained != Q.out) {
        if (n > 0) fn(arg, 0, n);
        return;
    }
    plat_mutex_lock(&pj_lock);
    PJ.fn = fn, PJ.arg = arg, PJ.n = n, PJ.grain = grain;
    __atomic_store_n(&PJ.next, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&PJ.left, n, __ATOMIC_RELAXED);
    __atomic_add_fetch(&PJ.seq, 1, __ATOMIC_SEQ_CST);   // odd: running
    for (int k = 0; k < RW.n; k++)
        if (__atomic_load_n(&RW.w[k]->sleeping, __ATOMIC_SEQ_CST)) plat_event_set(&RW.w[k]->wake);
    par_slices();
    while (__atomic_load_n(&PJ.left, __ATOMIC_ACQUIRE) > 0) plat_pause();
    __atomic_add_fetch(&PJ.seq, 1, __ATOMIC_SEQ_CST);   // even: ended
    while (__atomic_load_n(&PJ.in, __ATOMIC_SEQ_CST)) plat_pause();
    plat_mutex_unlock(&pj_lock);
}

static void rw_main(void *arg) {
    RwWorker *w = arg;
    plat_thread_priority(PLAT_PRIO_HIGHEST);   // as the recording thread (rec_claim)
    perf_lock_timing_here();   // NX2NAVI_LOCK_TIME, as on the recording thread
    vk_exec_draw_ctx_use(w->dc);
    for (;;) {
        par_help();
        Chunk *c = chunk_take();
        if (c) {
            chunk_record(c, w->idx);
            continue;
        }
        // spins about 100 us after a chunk (the next group of the frame is usually closer), then sleeps
        uint64_t t0 = kern_now_ns();
        for (unsigned i = 0; __atomic_load_n(&Q.take, __ATOMIC_SEQ_CST) >= __atomic_load_n(&Q.put, __ATOMIC_SEQ_CST);
             i++) {
            if (par_pending()) break;
            if ((i & 63) || kern_now_ns() - t0 < 100000) {
                plat_pause();
                continue;
            }
            __atomic_store_n(&w->sleeping, 1, __ATOMIC_SEQ_CST);
            if (__atomic_load_n(&Q.take, __ATOMIC_SEQ_CST) >= __atomic_load_n(&Q.put, __ATOMIC_SEQ_CST) &&
                !par_pending())
                plat_event_wait_ns(&w->wake, 2000000);
            __atomic_store_n(&w->sleeping, 0, __ATOMIC_RELAXED);
            t0 = kern_now_ns();
        }
    }
}

static void rw_init(void) {
    vk_exec_draw_pool_schedule_init();
    const char *waves = cfg_get("NX2NAVI_RECORD_EXTRA_CHUNKS");
    RW.extra_chunks = !waves || strcmp(waves, "0");
    int n = cfg_int("NX2NAVI_REPLAY_WORKERS");
    RW.n = n < 0 ? 0 : n > RW_MAX ? RW_MAX : n;
    for (int k = 0; k < RW.n; k++) {
        RwWorker *w = calloc(1, sizeof *w);
        if (!w) rt_fatal("vk: out of memory (replay worker)");
        w->idx = k, w->dc = vk_exec_draw_ctx_new();
        plat_event_init(&w->wake, 0, 0);
        char name[32];
        snprintf(name, sizeof name, "vk replay worker %d", k);
        if (!(w->t = plat_thread_new(rw_main, w, name))) rt_fatal("vk: cannot start a replay worker");
        RW.w[k] = w;
    }
    int pipe = cfg_int("NX2NAVI_REPLAY_PIPE");   // the groups out at once
    RW.pipe = RW.n <= 0 || pipe <= 0 ? 0 : pipe > LIVE_GROUPS ? LIVE_GROUPS : pipe;
    if (RW.pipe) RW.help_dc = vk_exec_draw_ctx_new();
    if (RW.pipe)
        rt_log("vk: groups of %d draws or more are recorded by %d replay worker%s while the submitting thread goes on "
               "past them, %d at once at most (NX2NAVI_REPLAY_WORKERS, NX2NAVI_REPLAY_PIPE)", 2 * CHUNK_DRAWS, RW.n,
               RW.n > 1 ? "s" : "", RW.pipe);
    else if (RW.n)
        rt_log("vk: groups of %d draws or more are recorded by %d replay worker%s and the submitting thread "
               "(NX2NAVI_REPLAY_WORKERS)", 2 * CHUNK_DRAWS, RW.n, RW.n > 1 ? "s" : "");
}

// Every group before g has all its chunks done.
static int groups_before_done(const Group *g) { return __atomic_load_n(&Q.done, __ATOMIC_ACQUIRE) + 1 >= g->seq; }

// Every chunk before c in the stream is done: those of earlier groups, and its group's earlier ones.
static int chunk_in_turn(const Chunk *c) {
    if (!groups_before_done(c->g)) return 0;
    for (int i = 0; i < c->idx; i++)
        if (!__atomic_load_n(&c->g->c[i].done, __ATOMIC_ACQUIRE)) return 0;
    return 1;
}

int vk_exec_turn_due(void) {
    if (!rw_chunk) {
        if (rw_behind && __atomic_load_n(&Q.done, __ATOMIC_ACQUIRE) >= Q.out) rw_behind = 0;
        return rw_behind;
    }
    if (!rw_ordered && chunk_in_turn(rw_chunk)) rw_ordered = 1;
    return !rw_ordered;
}
int vk_exec_in_chunk(void) { return rw_chunk != NULL; }
int vk_exec_shared(void) { return rw_chunk || rw_behind; }
int vk_exec_behind(void) { return !rw_chunk && rw_behind; }

VkExecPoolView *vk_exec_pool_view(void) { return rw_chunk ? &rw_chunk->g->pool : NULL; }
VkExecPoolView *vk_exec_pool_local_view(void) { return rw_chunk ? &rw_chunk->pool : NULL; }

VkExecPoolRequirement *vk_exec_pool_requirement_begin(void) {
    Chunk *c = rw_chunk;
    // A failed, skipped or non-pool attempt does not consume learning. Only completed preparation publishes a
    // valid family hint; chunk zero alone owns this record until retirement.
    if (!c || c->idx || c->g->requirement.valid) return NULL;
    return &c->g->requirement;
}

int vk_exec_pool_wait(uint64_t fence, uint32_t kinds, VkDescriptorSet *set) {
    Chunk *c = rw_chunk;
    if (!c) return 0;
    uint64_t t0 = kern_now_ns();
    for (;;) {
        // A preceding consumer may finish the shared refresh before it finishes recording all of its draws.
        uint64_t generation = vk_exec_pool_generation();
        if (vk_pool_view_get(&c->g->pool, fence, kinds, generation, set) &&
            generation == vk_exec_pool_generation()) return 1;
        if (rw_ordered || chunk_in_turn(c)) {
            if (!rw_ordered) {
                rw_ordered = 1;
                __atomic_add_fetch(&RW.turns, 1, __ATOMIC_RELAXED);
                __atomic_add_fetch(&RW.turn_ns, kern_now_ns() - t0, __ATOMIC_RELAXED);
            }
            return 0;   // this consumer may prepare and publish; never mark an unfinished refresh ready
        }
        chunk_pair_help(c->g);
        plat_pause();
    }
}

// A chunk of a group comes after the ops before the group and after its group's earlier chunks; the submitting
// thread after every chunk of the groups it handed out.
uint64_t vk_exec_stream_pos(void) { return rw_chunk ? rw_chunk->g->seq << 5 | (uint64_t)rw_chunk->idx : Q.out << 5 | 31; }

// Uniform snapshot drops (vk_usnap.c): a chunk's are its group's; the submitting thread's come after every group it
// handed out. A chunk sees the drops of epochs up to its group's, the submitting thread every one.
uint64_t vk_exec_epoch(int write) { return rw_chunk ? rw_chunk->g->seq : write ? Q.out + 1 : UINT64_MAX; }

// The inline uniform updates of the submission a draw may read (vk_exec_draw.c upd_read): a chunk only those before
// its group.
uint32_t vk_exec_upd_top(uint32_t all) { return rw_chunk ? rw_chunk->g->upd : all; }

void vk_exec_hooks_wait(void) {
    RWT_CHAIN(work_hooks,RWT_REGION_COORDINATION);
    if (!vk_exec_behind()) return;
    const Group *g = &GQ[Q.out % LIVE_GROUPS];   // the last handed out, whose first chunk ran after the others'
    while (!__atomic_load_n(&g->hooks, __ATOMIC_ACQUIRE)) plat_pause();
}

void vk_exec_hook_turn(void) {
    RWT_CHAIN(work_hooks,RWT_REGION_COORDINATION);
    if (!rw_chunk) {
        group_wait(PW_HOOKS, 0);
        return;
    }
    // a group's hooks run in its first chunk, after its group's earlier chunks; the earlier groups' draws read what
    // the hook changes
    while (!groups_before_done(rw_chunk->g)) plat_pause();
}

void vk_exec_turn(void) {
    RWT_CHAIN(work_turn,RWT_REGION_COORDINATION);
    Chunk *c = rw_chunk;
    if (!c) {   // the submitting thread: after every chunk of the groups it went on past
        group_wait(PW_TURN, 0);
        return;
    }
    if (rw_ordered) return;
    uint64_t t0 = kern_now_ns();
    while (!chunk_in_turn(c)) { chunk_pair_help(c->g); plat_pause(); }
    rw_ordered = 1;
    __atomic_add_fetch(&RW.turns, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&RW.turn_ns, kern_now_ns() - t0, __ATOMIC_RELAXED);
}

// The image views a command buffer made for its targets go when its submission is done.
static void views_release(VkExec *ex, uint64_t fence) {
    if (!ex->nviews) return;
    ViewList *l = malloc(sizeof *l + sizeof(VkImageView) * (size_t)ex->nviews);
    if (!l) rt_fatal("vk: out of memory");
    l->n = ex->nviews;
    for (int i = 0; i < ex->nviews; i++) l->v[i] = ((ViewEnt *)ex->views)[i].view;
    vk_completion_add(fence, free_views, l);
    ex->nviews = 0;
}

static const char *const ge_name[GE_N] = {
    "run end", "target change", "uniform update", "generated indices", "mesh draw", "read limit"
};

int vk_gpu_passes_capturing(void);   // vk_crumbs.c: NX2NAVI_GPU_PASSES is copying this frame's counts

// The draws of the group that starts at h (it ends at *end, for the reason *why), reading at most GROUP_PEEK ops.
static uint32_t group_peek(const VkExec *ex, const op_hdr *h, const op_hdr **end, int *why, uint32_t *end_op) {
    uint32_t draws = 0;
    *end_op = 0;
    int begun = 0;
    const p_set_rt *rt = &ex->rt;
    *why = GE_LIMIT;
    // mesh draws stay on the submitting thread with the mesh fallback (its buffers are shared) and while
    // NX2NAVI_GPU_PASSES copies their workgroup counts (one list)
    int mesh_serial = vk_ctx()->mesh_fallback || vk_gpu_passes_capturing();
    for (uint32_t ops = 0; ops < GROUP_PEEK; ops++) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_) { *why = GE_RUN; break; }   // the walk reports it
        const void *a = nvn_op_payload(h);
        if (h->op == OP_JUMP) {
            h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
            continue;
        }
        if (!run_op(h->op)) { *why = GE_RUN, *end_op = h->op; break; }
        if (h->op == OP_UPDATE_UB) { *why = GE_UB; break; }
        if (h->op == OP_DRAW) {
            if (h->size < sizeof *h + sizeof(p_draw)) { *why = GE_RUN; break; }
            if (nvn_generated_draw_index_bytes(a)) { *why = GE_INDEX; break; }
        }
        int mesh = h->op == OP_DRAW_MESH || h->op == OP_DRAW_MESH_IND;
        if (mesh && mesh_serial) { *why = GE_MESH; break; }
        if (h->op == OP_SET_RT && memcmp(rt, a, sizeof *rt)) {
            if (begun) { *why = GE_TARGET; break; }   // another pass: the next group
            rt = a;
        }
        if (h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND || mesh) draws++, begun = 1;
        if (h->op == OP_CLEAR_COLOR || h->op == OP_CLEAR_DS) begun = 1;
        h = nvn_op_next(h);
    }
    *end = h;
    return draws;
}

// Ordinary state/draw stretches have already been counted when the immutable command copy was made. Only the
// boundaries whose meaning depends on current replay state are inspected here. An interior resume or the read limit
// landing inside a span uses the original walker; no approximate group length changes scheduling or ordering.
static int group_peek_plan(const VkExec *ex, const NvnRecordPlan *plan, const op_hdr *h,
                           const op_hdr **end, int *why, uint32_t *end_op, uint32_t *draws) {
    if (!plan || !plan->nspans) return 0;
    uintptr_t base = (uintptr_t)plan->commands, at = (uintptr_t)h;
    if (at < base || at - base >= plan->bytes) return 0;
    size_t offset = (size_t)(at - base), lo = 0, hi = plan->nspans;
    while (lo < hi) {
        size_t m = lo + (hi - lo) / 2;
        if (plan->spans[m].begin < offset) lo = m + 1;
        else hi = m;
    }
    if (lo == plan->nspans || plan->spans[lo].begin != offset) return 0;
    uint32_t ops = 0;
    *draws = 0, *end_op = 0, *why = GE_LIMIT;
    int begun = 0, mesh_serial = vk_ctx()->mesh_fallback || vk_gpu_passes_capturing();
    const p_set_rt *rt = &ex->rt;
    for (size_t i = lo; i < plan->nspans; i++) {
        const NvnRecordSpan *s = &plan->spans[i];
        h = (const op_hdr *)(plan->commands + s->begin);
        if (ops == GROUP_PEEK) { *end = h; return 1; }
        if (s->kind == NVN_PLAN_SERIAL) {
            *why = GE_RUN, *end_op = h->op, *end = h;
            return 1;
        }
        if (s->kind == NVN_PLAN_UPDATE) { *why = GE_UB, *end = h; return 1; }
        if (s->kind == NVN_PLAN_GENERATED_INDEX) { *why = GE_INDEX, *end = h; return 1; }
        if (s->kind == NVN_PLAN_MESH && mesh_serial) { *why = GE_MESH, *end = h; return 1; }
        if (s->kind == NVN_PLAN_TARGET) {
            const p_set_rt *next = nvn_op_payload(h);
            if (memcmp(rt, next, sizeof *rt)) {
                if (begun) { *why = GE_TARGET, *end = h; return 1; }
                rt = next;
            }
        }
        if (s->ops > GROUP_PEEK - ops) return 0;
        ops += s->ops;
        *draws += s->draws;
        if (s->draws || s->kind == NVN_PLAN_CLEAR) begun = 1;
    }
    return 0;   // a valid recording has an explicit END boundary; unknown metadata keeps the original path
}

static Chunk *chunk_open(Group *g, const VkExec *ex, const op_hdr *h, int idx, const NvnRecordPlan *plan) {
    Chunk *c = &g->c[idx];
    memcpy(&c->ex, ex, sizeof *ex);
    c->ex.target_binding_mask = 0; // The worker opens its own rendering pass.
    c->ex.bc_target_mask = c->ex.bc_written_mask = 0;
    memset(c->ex.target_bindings, 0, sizeof c->ex.target_bindings);
    c->resource_begin = ex->resource_next + 1;
    c->h = h, c->nops = 0, c->cond0 = g->ncond, c->ncond = 0, c->idx = idx, c->g = g, c->oq = vk_exec_xfer_chunk_state();
    c->nslots = 0, c->brk = 0;
    c->plan = plan ? *plan : (NvnRecordPlan){0};
    return c;
}

// The groups whose every chunk is done, in order (Q.done), once a group's last chunk is.
static void groups_done_advance(void) {
    plat_mutex_lock(&done_lock);
    for (;;) {
        uint64_t d = Q.done;
        const Group *g = &GQ[(d + 1) % LIVE_GROUPS];
        if (__atomic_load_n(&g->seq, __ATOMIC_ACQUIRE) != d + 1 || __atomic_load_n(&g->ndone, __ATOMIC_ACQUIRE) != g->n)
            break;
        __atomic_store_n(&Q.done, d + 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&done_lock);
}

static Chunk *chunk_take(void) {
    uint64_t t = __atomic_load_n(&Q.take, __ATOMIC_ACQUIRE);
    while (t < __atomic_load_n(&Q.put, __ATOMIC_ACQUIRE))
        if (__atomic_compare_exchange_n(&Q.take, &t, t + 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return Q.q[t % QN];
    return NULL;
}

// Records chunk c with the command pool `pool` (a worker's index; the submitting thread's is RW.n).
static void chunk_record(Chunk *c, int pool) {
    Group *g = c->g;
    RWT_UNIT(work_chunk, pool == RW.n ? RWT_ROLE_HELP : RWT_ROLE_WORKER,
             c->ex.fence, g->work_replay, g->seq, (unsigned)c->idx);
    RWT_CHAIN(work_commands,RWT_REGION_CHUNK_BODY);
    RP_SCOPE(rp_hook_wait, RP_CHUNK_HOOK);
    RP_VALUE(RPK_WORK_UNIT, gfx_rwt_current_id());
    if (c->idx)
        while (!__atomic_load_n(&g->hooks, __ATOMIC_ACQUIRE)) { chunk_pair_help(g); plat_pause(); }
    else if (!groups_before_done(g)) {   // the pass hooks follow the frame pass by pass: after the group before's
        const Group *p = &GQ[(g->seq - 1) % LIVE_GROUPS];   // not done, so not put in its part yet: still its slot
        while (!__atomic_load_n(&p->hooks, __ATOMIC_ACQUIRE)) plat_pause();
    }
    RP_END(rp_hook_wait);
    RP_SCOPE(rp_chunk_setup, RP_CHUNK_SETUP);
    RP_VALUE(RPK_WORK_UNIT, gfx_rwt_current_id());
    uint64_t t0 = kern_now_ns();
    gfx_tl_note('c', t0, (uint64_t)c->idx, (uint64_t)c->nops);
    VkExec *ex = &c->ex, *was = x_cur;
    // the submitting thread records with its own (the state's), or one of its own past the group (RW.pipe)
    int help = pool == RW.n && RW.pipe;
    if (pool < RW.n) ex->dc = RW.w[pool]->dc;
    else if (help) ex->dc = RW.help_dc, vk_exec_draw_ctx_use(RW.help_dc);
    x_cur = ex, rw_chunk = c, rw_ordered = !c->idx && groups_before_done(g);
    vk_exec_draw_state_adopted(ex);
    vk_exec_xfer_worker(1, g->cond + c->cond0, c->ncond, c->oq, c->slot0, c->nslots, c->idx, (int)(g - GQ));
    ex->cb = vk_sync_worker_begin(pool);
    // the replay's command buffer before the group ended behind a full barrier, with every image in GENERAL
    ex->recorded = 0, ex->barrier_at = 0, ex->in_pass = 0, ex->nfbl = 0, ex->mark = NULL;
    ex->dirty = VK_EXEC_DIRTY_ALL, ex->rt_resolved = 0, ex->fb_fs = NULL;
    ex->views = NULL, ex->nviews = ex->cap_views = 0;
    vk_exec_draw_new_cb();
    const op_hdr *o = c->h;
    size_t exec_cursor = SIZE_MAX;
    int executable = exec_plan_enabled();
    ExecPlanStats stats __attribute__((cleanup(exec_stats_publish))) = {.path = EXEC_WORKER};
    RP_END(rp_chunk_setup);
    RP_SCOPE(rp_chunk_body, RP_CHUNK_BODY);
    RP_VALUE(RPK_WORK_UNIT, gfx_rwt_current_id());
    for (uint32_t k = 0; k < c->nops && !vk_exec_recording_failed();) {
        if (o->op == OP_JUMP) {
            o = (const op_hdr *)(uintptr_t)((const p_jump *)nvn_op_payload(o))->target;
            continue;
        }
        const NvnExecStep *step = executable ? nvn_record_exec_slice(&c->plan, o, NULL, c->nops - k, &exec_cursor) : NULL;
        if (step ? !(step->flags & NVN_EXEC_LINEAR) : !op_linear(o->op)) c->brk++;
        ex->resource_op = c->resource_begin + k;
        uint64_t effect_index = ex->resource_op - g->resource_begin;
        if (effect_index >= g->neffects) rt_fatal("vk: recording escaped its declared operation range");
        struct GroupOp *receipt = &g->ops[effect_index];
        if (receipt->graphics && !vk_exec_graphics_draw_current(ex, o->op, nvn_op_payload(o), receipt->graphics)) {
            vk_exec_required_graphics_unavailable(ex, o->op, "prepared recording inputs changed");
            break;
        }
        VkExecGraphicsDraw *prior_graphics = vk_exec_graphics_draw_enter(receipt->graphics);
        void *previous_effect = vk_exec_effects_enter(receipt->effect);
        rw_op = o;
        if (receipt->noop) graphics_noop_record(ex, o);
        else if (step) exec_step_run(ex, &c->plan, step);
        else op_one(ex, o);
        if (c->plan.nexec) exec_stats_step(&stats, step);
        if (!receipt->noop && vk_texture_deferred_take() && !vk_exec_recording_failed()) {   // retry a turn-dependent lookup
            vk_exec_pass_end(ex);
            ex->rt_resolved = 0, ex->dirty = VK_EXEC_DIRTY_ALL;
            vk_exec_draw_new_cb();
            vk_exec_turn();
            __atomic_add_fetch(&RW.redos, 1, __ATOMIC_RELAXED);
            if (step) exec_step_run(ex, &c->plan, step);
            else op_one(ex, o);
            if (c->plan.nexec) exec_stats_step(&stats, step);
        }
        vk_exec_effects_recorded(receipt->effect,0);
        vk_exec_effects_leave(previous_effect);
        vk_exec_graphics_draw_enter(prior_graphics);
        if (step) k += step->ops, o = (const op_hdr *)(c->plan.commands + step->end);
        else k++, o = nvn_op_next(o);
    }
    RP_END(rp_chunk_body);
    RP_SCOPE(rp_chunk_teardown, RP_CHUNK_TEARDOWN);
    RP_VALUE(RPK_WORK_UNIT, gfx_rwt_current_id());
    if (!c->idx) __atomic_store_n(&g->hooks, 1, __ATOMIC_RELEASE);   // a chunk 0 that began no pass
    vk_exec_barrier(ex);   // the next command buffer starts after it, every image back in GENERAL
    vk_sync_worker_end(ex->cb);
    views_release(ex, ex->fence);
    free(ex->views);
    ex->views = NULL;
    c->cb = ex->cb, c->recorded = ex->recorded;
    vk_exec_xfer_worker(0, NULL, 0, 0, 0, 0, 0, 0);
    vk_exec_draw_prepared_stats_flush(ex);
    x_cur = was, rw_chunk = NULL;
    if (help) vk_exec_draw_ctx_use(NULL);
    if (was) vk_exec_draw_state_adopted(was);   // helping may have used the original stream's DrawCtx
    __atomic_add_fetch(&RW.chunk_ns, kern_now_ns() - t0, __ATOMIC_RELAXED);
    gfx_tl_note('d', 0, (uint64_t)c->idx, 0);
    __atomic_store_n(&c->done, 1, __ATOMIC_RELEASE);
    if (__atomic_add_fetch(&g->ndone, 1, __ATOMIC_ACQ_REL) == g->n) groups_done_advance();
}

static void group_take(int pool) {
    for (Chunk *c; (c = chunk_take());) chunk_record(c, pool);
}

// The submitting thread past groups it handed out (RW.pipe): until every chunk of them is done. With `help` (only
// between two ops) it records the chunks no worker took yet itself. Then nothing of them runs beside it any more.
static void group_wait(int why, int help) {
    RWT_CHAIN(work_wait,RWT_REGION_COORDINATION);
    if (!rw_behind) return;
    uint64_t t0 = kern_now_ns();
    int done = __atomic_load_n(&Q.done, __ATOMIC_ACQUIRE) >= Q.out;
    if (!done) {
        gfx_tl_note('q', t0, (uint64_t)(uintptr_t)pw_name[why], 0);
        if (help) RP_CALL(RP_OWNER_HELP, group_take(RW.n));
        while (__atomic_load_n(&Q.done, __ATOMIC_ACQUIRE) < Q.out) plat_pause();
        gfx_tl_note('u', 0, 0, 0);
    }
    rw_behind = 0;
    uint64_t t1 = kern_now_ns();
    if (done) RW.pw_n[PW_DONE]++;
    else RW.pw_n[why]++, RW.pw_ns[why] += t1 - t0;
}

// A done group's command buffers into their places in the part, its closed query slots into their segments.
static void group_put(VkExec *ex, Group *g) {
    vk_exec_draw_pool_learn(&g->requirement);
    vk_exec_xfer_group_merge((int)(g - GQ), g->n);
    VkCommandBuffer cbs[GROUP_CHUNKS];
    int rec = 0;
    for (int i = 0; i < g->n; i++) cbs[i] = g->c[i].cb, rec += g->c[i].recorded;
    vk_sync_append_fill(g->at, cbs, g->n);
    for (uint32_t i = 0; i < g->neffects; i++) {
        vk_exec_effects_release(g->ops[i].effect);
        vk_exec_graphics_draw_cancel(&g->ops[i].graphics);
    }
    g->neffects = 0;
    if (ex->barrier_at == ex->recorded) ex->barrier_at += rec;   // nothing since the chunks' closing barrier
    ex->recorded += rec;
    RW.behind_ns += kern_now_ns() - g->out_ns;
    gfx_tl_note('h', 0, 0, 0);
}

// The done groups not in their parts yet, oldest first; when every one handed out is, the submitting thread is no
// longer behind.
static void groups_put_done(VkExec *ex) {
    uint64_t d = __atomic_load_n(&Q.done, __ATOMIC_ACQUIRE);
    while (Q.drained < d) Q.drained++, group_put(ex, &GQ[Q.drained % LIVE_GROUPS]);
    if (Q.drained == Q.out) rw_behind = 0;
}

static void group_drain(VkExec *ex, int why) {
    if (Q.drained == Q.out) return;
    group_wait(why, 1);
    groups_put_done(ex);
}

VkSyncProgressResult vk_exec_graphics_progress_prepare(VkExec *ex, int (*prepare)(void *), void *context) {
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    if (!ex || x_cur != ex || rw_chunk || !prepare) return VK_SYNC_PROGRESS_UNAVAILABLE;
    group_drain(ex, PW_OP);
    if (vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    return vk_sync_progress_prepare_pipeline(prepare, context);
}

static VkSyncProgressResult upload_staging_progress(VkExec *ex, uint64_t upload) {
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    if (!upload || !ex || x_cur != ex || rw_chunk) return VK_SYNC_PROGRESS_UNAVAILABLE;
    group_drain(ex, PW_OP);
    if (vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    return vk_sync_progress_wait_upload_staging(upload);
}

static void progress_adopt_tail(VkExec *ex, VkCommandBuffer cb);

VkSyncProgressResult vk_exec_graphics_pool_progress(VkExec *ex, uint64_t prior_scene) {
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    if (!ex || x_cur != ex || rw_chunk || !ex->fence || prior_scene >= ex->fence)
        return VK_SYNC_PROGRESS_UNAVAILABLE;
    group_drain(ex, PW_SPLIT);
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;

    if (prior_scene) {
        if (prior_scene > vk_sync_handed_fence()) return VK_SYNC_PROGRESS_UNAVAILABLE;
        if (prior_scene <= vk_sync_fence_done()) return VK_SYNC_PROGRESS_COMPLETE;
        return vk_sync_progress_wait_prior(prior_scene);
    }

    if (!ex->recorded) return VK_SYNC_PROGRESS_UNAVAILABLE;
    int empty_tail = ex->recorded == part_start;
    if (!empty_tail) vk_exec_pass_end(ex);
    if (vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    VkCommandBuffer tail = VK_NULL_HANDLE;
    VkSyncProgressResult result = vk_sync_progress_split(ex, &tail, empty_tail);
    if (tail) progress_adopt_tail(ex, tail);
    return result;
}

struct VkExecPointerReadback {
    VkPreparedStaging stage;
    uint32_t owners;
};

static void device_pointer_readback_drop(void *arg) {
    VkExecPointerReadback *readback = arg;
    if (__atomic_sub_fetch(&readback->owners, 1, __ATOMIC_ACQ_REL)) return;
    vk_staging_prepare_cancel(&readback->stage);
    free(readback);
}

static void device_pointer_readback_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed; // VkSync retains uncertain handed owners until actual completion or idle.
    device_pointer_readback_drop(arg);
}

static int device_pointer_current(const VkMemSpan *span, const VkMemSpanCapture *capture,
                                  const RcInterval *ranges, uint32_t count, RcRegistryPosition read) {
    return !rc_registry_position_compare(read, vk_exec_resource_position()) &&
        vk_mem_span_capture_current(capture, span) &&
        rc_registry_read_device_buffer_at(rc_registry_runtime(), ranges, count, read) == RC_REGISTRY_OK;
}

static void device_pointer_record_copy(VkExec *ex, const VkMemSpan *span, VkMemSpanCapture *capture,
                                       VkExecPointerReadback *readback) {
    VkMemSpanCapture *native = vk_mem_span_capture_retain(capture);
    vk_exec_record_native_input_provider(ex, &native);
    readback->owners++;
    vk_completion_add_owned(ex->fence, device_pointer_readback_drop, device_pointer_readback_cancel, readback);
    vk_exec_barrier(ex);
    VkBufferCopy copy = {span->offset, 0, sizeof(uint64_t)};
    vkCmdCopyBuffer(ex->cb, span->buffer, readback->stage.buffer, 1, &copy);
    VkBufferMemoryBarrier2 host = {.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask=VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask=VK_ACCESS_2_HOST_READ_BIT,
        .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
        .buffer=readback->stage.buffer, .size=sizeof(uint64_t)};
    VkDependencyInfo dependency = {.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount=1, .pBufferMemoryBarriers=&host};
    vkCmdPipelineBarrier2(ex->cb, &dependency);
    ex->recorded++;
}

VkSyncProgressResult vk_exec_read_device_pointer(VkExec *ex, uint64_t address, const VkMemSpan *span,
                                                 VkMemSpanCapture *capture, const RcInterval *ranges, uint32_t count,
                                                 RcRegistryPosition read, uint64_t *value) {
    if (vk_sync_stopping() || vk_device_is_lost()) return VK_SYNC_PROGRESS_STOPPED;
    if (!ex || x_cur != ex || rw_chunk || !value || !span || !capture || !span->alloc || !ranges || !count ||
        span->cls != VK_MCLS_DEVICE || !nvn_gpu_only(address) || (span->offset & 3) ||
        !rc_registry_provisional_read_current(rc_registry_runtime(), read)) return VK_SYNC_PROGRESS_UNAVAILABLE;
    group_drain(ex, PW_OP);
    if (vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    if (!device_pointer_current(span, capture, ranges, count, read)) return VK_SYNC_PROGRESS_UNAVAILABLE;
    VkExecPointerReadback *readback = calloc(1, sizeof *readback);
    if (!readback) return VK_SYNC_PROGRESS_UNAVAILABLE;
    readback->owners = 1;
    RcAdmissionVector peak;
    VkResult staged = vk_staging_prepare_plan(sizeof *value, &readback->stage, &peak);
    if (staged == VK_SUCCESS) staged = vk_staging_prepare_allocate(&readback->stage, NULL, 1);
    if ((staged == VK_NOT_READY || staged == VK_ERROR_OUT_OF_DEVICE_MEMORY || staged == VK_ERROR_OUT_OF_HOST_MEMORY) &&
        device_pointer_current(span, capture, ranges, count, read)) {
        vk_staging_prepare_cancel(&readback->stage);
        if (vk_staging_prepare_reuse(sizeof *value, &readback->stage)) staged = VK_SUCCESS;
        else {
            const uint64_t supplier = vk_staging_prepare_supplier(sizeof *value, 0);
            VkSyncProgressResult progress = upload_staging_progress(ex, supplier);
            if (progress == VK_SYNC_PROGRESS_STOPPED) {
                device_pointer_readback_drop(readback);
                return progress;
            }
            if (progress == VK_SYNC_PROGRESS_COMPLETE && device_pointer_current(span, capture, ranges, count, read) &&
                vk_staging_prepare_reuse(sizeof *value, &readback->stage)) staged = VK_SUCCESS;
        }
    }
    if (staged != VK_SUCCESS || !device_pointer_current(span, capture, ranges, count, read)) {
        if (staged == VK_ERROR_DEVICE_LOST) vk_device_lost_check(staged, "preparing ordered storage pointer readback");
        device_pointer_readback_drop(readback);
        return vk_sync_stopping() || vk_device_is_lost() ? VK_SYNC_PROGRESS_STOPPED : VK_SYNC_PROGRESS_UNAVAILABLE;
    }
    vk_exec_pass_end(ex);
    if (vk_exec_recording_failed() || !device_pointer_current(span, capture, ranges, count, read) ||
        !vk_mem_span_record_input(capture, span)) {
        device_pointer_readback_drop(readback);
        return vk_exec_recording_failed() ? VK_SYNC_PROGRESS_STOPPED : VK_SYNC_PROGRESS_UNAVAILABLE;
    }
    device_pointer_record_copy(ex, span, capture, readback);
    VkSyncProgressResult progress = vk_exec_graphics_pool_progress(ex, 0);
    if (progress == VK_SYNC_PROGRESS_COMPLETE) {
        VkResult visible = vmaInvalidateAllocation(vk_ctx()->vma, readback->stage.allocation, 0, sizeof *value);
        uint64_t decoded = 0;
        if (visible == VK_SUCCESS) memcpy(&decoded, readback->stage.map, sizeof decoded);
        else vk_device_lost_check(visible, "invalidating ordered storage pointer readback");
        if (visible == VK_SUCCESS && device_pointer_current(span, capture, ranges, count, read)) *value = decoded;
        else progress = VK_SYNC_PROGRESS_UNAVAILABLE;
        // Actual native prefix completion covers the copy independently of the scene's eventual final fence.
        vk_staging_prepare_cancel(&readback->stage);
    }
    device_pointer_readback_drop(readback);
    return progress;
}

// The caller owns the exact, coherent destination until this copy is completed or retired with the scene.
VkSyncProgressResult vk_exec_read_device_indices(VkExec *ex, uint64_t address, uint64_t bytes,
    const VkMemSpan *span, VkMemSpanCapture *capture, const RcInterval *ranges, uint32_t count,
    RcRegistryPosition read, VkBuffer destination, VkDeviceSize offset, int *issued) {
    *issued = 0;
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    if (!ex || x_cur != ex || rw_chunk || !span || !capture || !span->alloc || !ranges || !count ||
        !destination || destination == span->buffer || !bytes || bytes > span->avail ||
        offset > UINT64_MAX - bytes || span->offset > UINT64_MAX - bytes ||
        span->cls != VK_MCLS_DEVICE || !nvn_gpu_only(address) ||
        !rc_registry_provisional_read_current(rc_registry_runtime(), read)) return VK_SYNC_PROGRESS_UNAVAILABLE;
    group_drain(ex, PW_OP);
    if (vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    if (!device_pointer_current(span, capture, ranges, count, read)) return VK_SYNC_PROGRESS_UNAVAILABLE;
    vk_exec_pass_end(ex);
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    if (!ex->cb || ex->in_pass || !device_pointer_current(span, capture, ranges, count, read) ||
        !vk_mem_span_record_input(capture, span)) return VK_SYNC_PROGRESS_UNAVAILABLE;
    VkMemSpanCapture *native = vk_mem_span_capture_retain(capture);
    vk_exec_record_native_input_provider(ex, &native);
    vk_exec_barrier(ex);
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;

    VkBufferCopy copy = {span->offset, offset, bytes};
    *issued = 1;
    vkCmdCopyBuffer(ex->cb, span->buffer, destination, 1, &copy);
    VkBufferMemoryBarrier2 host = {.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask=VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask=VK_ACCESS_2_HOST_READ_BIT,
        .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
        .buffer=destination, .offset=offset, .size=bytes};
    VkDependencyInfo dependency = {.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount=1, .pBufferMemoryBarriers=&host};
    vkCmdPipelineBarrier2(ex->cb, &dependency);
    ex->recorded++;

    VkSyncProgressResult result = vk_exec_graphics_pool_progress(ex, 0);
    if (result == VK_SYNC_PROGRESS_COMPLETE &&
        !device_pointer_current(span, capture, ranges, count, read)) {
        *issued = 0; // The copy is complete even when its source was invalidated while waiting.
        return VK_SYNC_PROGRESS_UNAVAILABLE;
    }
    return result;
}

VkSyncProgressResult vk_exec_ordered_host_input_prefix(VkExec *ex, RcRegistryPosition read, VkBuffer buffer,
                                                      VkDeviceSize offset, VkDeviceSize size) {
    RcRegistry *registry=rc_registry_runtime();
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    if (!ex || x_cur!=ex || rw_chunk || rw_behind || !ex->cb || !buffer || !size || size>UINT64_MAX-offset ||
        read.ordinal<2 || rc_registry_position_compare(read,vk_exec_resource_position()) ||
        !rc_registry_provisional_read_current(registry,read)) return VK_SYNC_PROGRESS_UNAVAILABLE;
    group_drain(ex,PW_SPLIT);
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    vk_exec_pass_end(ex);
    if (vk_exec_recording_failed() || !rc_registry_provisional_read_current(registry,read))
        return VK_SYNC_PROGRESS_UNAVAILABLE;
    vk_exec_barrier(ex);
    VkBufferMemoryBarrier2 host={.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask=VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,.srcAccessMask=VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT,.dstAccessMask=VK_ACCESS_2_HOST_READ_BIT,
        .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
        .buffer=buffer,.offset=offset,.size=size};
    VkDependencyInfo dep={.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount=1,.pBufferMemoryBarriers=&host};
    vkCmdPipelineBarrier2(ex->cb,&dep);
    ex->recorded++;
    VkSyncProgressResult result=vk_exec_graphics_pool_progress(ex,0);
    if (result!=VK_SYNC_PROGRESS_COMPLETE) return result;
    if (vk_device_is_lost() || vk_sync_stopping() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED_AFTER_CUT;
    return rc_registry_provisional_read_current(registry,read) &&
        rc_registry_position_handed(registry,(RcRegistryPosition){read.scene,read.ordinal-1}) ?
        VK_SYNC_PROGRESS_COMPLETE : VK_SYNC_PROGRESS_UNAVAILABLE;
}

VkSyncProgressResult vk_exec_copy_native_inputs(VkExec *ex, RcRegistryPosition read, const VkExecNativeInputCopy *copies,
    uint32_t count, VkBuffer readback, VkExecNativeInputCopyRecord on_record, void *context) {
    if (vk_sync_stopping() || vk_device_is_lost()) return VK_SYNC_PROGRESS_STOPPED;
    if (!ex || x_cur != ex || rw_chunk || !copies || !count || count > VK_EXEC_NATIVE_INPUT_COPY_LIMIT ||
        !readback || !on_record || !rc_registry_provisional_read_current(rc_registry_runtime(), read))
        return VK_SYNC_PROGRESS_UNAVAILABLE;
    VkDeviceSize readback_size = 0;
    for (uint32_t i = 0; i < count; i++) {
        const VkExecNativeInputCopy *input = &copies[i];
        if (!input->span || !input->span->buffer || !input->capture || !input->ranges || !input->range_count ||
            input->bytes < 4 || input->bytes > input->span->avail || ((input->span->offset | input->bytes) & 3) ||
            input->span->offset > UINT64_MAX - input->bytes || input->span->buffer == readback ||
            (input->snapshot && (input->snapshot == input->span->buffer || input->snapshot == readback)) ||
            !input->words || !input->word_count || input->word_count > VK_EXEC_NATIVE_INPUT_WORD_LIMIT)
            return VK_SYNC_PROGRESS_UNAVAILABLE;
        for (uint32_t j = 0; j < input->word_count; j++) {
            const VkBufferCopy *region = &input->words[j];
            if (((region->srcOffset | region->size) & 3) || !region->size || region->dstOffset != readback_size ||
                region->srcOffset > input->bytes || region->size > input->bytes - region->srcOffset ||
                region->size > UINT64_MAX - readback_size) return VK_SYNC_PROGRESS_UNAVAILABLE;
            readback_size += region->size;
        }
    }
    group_drain(ex, PW_OP);
    if (vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    for (uint32_t i = 0; i < count; i++) {
        const VkExecNativeInputCopy *input = &copies[i];
        if (!device_pointer_current(input->span, input->capture, input->ranges, input->range_count, read))
            return VK_SYNC_PROGRESS_UNAVAILABLE;
    }
    vk_exec_pass_end(ex);
    if (vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    for (uint32_t i = 0; i < count; i++) {
        const VkExecNativeInputCopy *input = &copies[i];
        if (!device_pointer_current(input->span, input->capture, input->ranges, input->range_count, read) ||
            !vk_mem_span_record_input(input->capture, input->span)) return VK_SYNC_PROGRESS_UNAVAILABLE;
    }
    for (uint32_t i = 0; i < count; i++) {
        VkMemSpanCapture *native = vk_mem_span_capture_retain(copies[i].capture);
        vk_exec_record_native_input_provider(ex, &native);
    }
    on_record(ex->fence, context);
    vk_exec_barrier(ex);
    VkBufferMemoryBarrier2 barriers[VK_EXEC_NATIVE_INPUT_COPY_LIMIT];
    uint32_t snapshots = 0;
    for (uint32_t i = 0; i < count; i++) {
        const VkExecNativeInputCopy *input = &copies[i];
        if (!input->snapshot) continue;
        VkBufferCopy copy = {input->span->offset, 0, input->bytes};
        vkCmdCopyBuffer(ex->cb, input->span->buffer, input->snapshot, 1, &copy);
        barriers[snapshots++] = (VkBufferMemoryBarrier2){.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask=VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask=VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
            .dstAccessMask=VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
            .buffer=input->snapshot, .size=input->bytes};
    }
    VkDependencyInfo dependency = {.sType=VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount=snapshots, .pBufferMemoryBarriers=barriers};
    if (snapshots) vkCmdPipelineBarrier2(ex->cb, &dependency);
    for (uint32_t i = 0; i < count; i++) {
        const VkExecNativeInputCopy *input = &copies[i];
        if (input->snapshot) {
            vkCmdCopyBuffer(ex->cb, input->snapshot, readback, input->word_count, input->words);
        } else {
            VkBufferCopy direct[VK_EXEC_NATIVE_INPUT_WORD_LIMIT];
            for (uint32_t j = 0; j < input->word_count; j++) {
                direct[j] = input->words[j];
                direct[j].srcOffset += input->span->offset;
            }
            vkCmdCopyBuffer(ex->cb, input->span->buffer, readback, input->word_count, direct);
        }
    }
    VkBufferMemoryBarrier2 host = {.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask=VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask=VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask=VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask=VK_ACCESS_2_HOST_READ_BIT,
        .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
        .buffer=readback, .size=readback_size};
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &host;
    vkCmdPipelineBarrier2(ex->cb, &dependency);
    ex->recorded++;
    VkSyncProgressResult result = vk_exec_graphics_pool_progress(ex, 0);
    for (uint32_t i = 0; i < count && result == VK_SYNC_PROGRESS_COMPLETE; i++) {
        const VkExecNativeInputCopy *input = &copies[i];
        if (!device_pointer_current(input->span, input->capture, input->ranges, input->range_count, read))
            result = VK_SYNC_PROGRESS_UNAVAILABLE;
    }
    return result;
}

VkMemMirrorInputPolicy vk_exec_mirror_input_policy(const VkExec *ex, int ordered) {
    if (!ordered || !ex || x_cur != ex || vk_exec_shared() || Q.drained != Q.out ||
        ex->resource_op != ex->resource_next + 1 ||
        !rc_registry_provisional_read_current(rc_registry_runtime(), vk_exec_resource_position()))
        return VK_MEM_MIRROR_EXISTING;
    // A worker join alone leaves both prior open command buffers and handed parts capable of reading GPA.
    return !ex->in_pass && ex->recorded == part_start && vk_sync_mirror_publication_safe(ex->fence)
        ? VK_MEM_MIRROR_PREPARE : VK_MEM_MIRROR_IMPORT;
}

static int graphics_opcode(uint32_t op) {
    return op == OP_DRAW || op == OP_DRAW_IDX || op == OP_DRAW_IDX_IND || op == OP_DRAW_MESH || op == OP_DRAW_MESH_IND;
}

enum { GRAPHICS_BOUNDARY_NONE, GRAPHICS_BOUNDARY_ORDERED, GRAPHICS_BOUNDARY_REPLAN };

static VkExecGraphicsResult graphics_input(VkExec *ex, const op_hdr *op, int ordered, RcRegistryPosition attempted,
                                            VkExecGpaSampleScope **sample_pending, VkExecGraphicsDraw **draw) {
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_GRAPHICS_STOPPED;
    if (ex->resource_next >= (UINT64_MAX - 1) / 2) rt_fatal("vk: graphics input identity exhausted");
    ex->resource_op = ex->resource_next + 1;
    if (rc_registry_declare_through(rc_registry_runtime(), vk_exec_resource_position()) != RC_REGISTRY_OK)
        rt_fatal("vk: graphics input prefix could not be sealed");
    VkExecGraphicsResult result = vk_exec_graphics_draw_prepare(ex, op->op, nvn_op_payload(op), ordered,
                                                               attempted, sample_pending, draw);
    if ((result == VK_EXEC_GRAPHICS_READY || result == VK_EXEC_GRAPHICS_INVALIDATED) && !*draw) {
        vk_exec_required_graphics_unavailable(ex, op->op, "graphics input decision returned without an owned draw");
        result = VK_EXEC_GRAPHICS_UNAVAILABLE;
    } else if (result == VK_EXEC_GRAPHICS_READY && !vk_exec_graphics_draw_current(ex, op->op, nvn_op_payload(op), *draw)) {
        result = VK_EXEC_GRAPHICS_INVALIDATED;
    }
    ex->resource_op = 0;
    return result;
}

void vk_exec_graphics_storage_failure_log(const VkExecStorageFailure *s) {
    if (!s->gate) return;
    rt_log("vk: required graphics storage read=%llu/%llu gate=%s mode=%u stage=%u set=%u binding=%u marker=%u "
        "sites=%u descriptors=%u writers=%u revision=%u retained=%u compatibility=%u detail=%d registry=%d "
        "address=0x%llx bytes=0x%llx mapping=%llu backing=%llu offset=0x%llx size=0x%llx",
        (unsigned long long)s->read.scene, (unsigned long long)s->read.ordinal, s->gate, s->mode,
        s->stage, s->set, s->binding, s->marker, s->sites, s->descriptor_count, s->writer_count,
        s->revision, s->retained, s->compatibility, s->detail, s->registry,
        (unsigned long long)s->address, (unsigned long long)s->bytes, (unsigned long long)s->range.mapping,
        (unsigned long long)s->range.range.backing, (unsigned long long)s->range.range.offset,
        (unsigned long long)s->range.range.size);
    const VkMemNativeRmwBitmapFailure *b = &s->bitmap;
    if (b->gate)
        rt_log("vk: required graphics storage provider read=%llu/%llu gate=%s native-result=%d mirror-offset=0x%llx "
            "blocker=%llu/%llu blocker-offset=0x%llx blocker-bytes=0x%llx",
            (unsigned long long)s->read.scene, (unsigned long long)s->read.ordinal, b->gate, b->result,
            (unsigned long long)b->mirror_offset, (unsigned long long)b->blocker_read.scene,
            (unsigned long long)b->blocker_read.ordinal, (unsigned long long)b->blocker_offset,
            (unsigned long long)b->blocker_bytes);
}

static void sampled_receipt_hex(const uint8_t *bytes, char text[65]) {
    static const char hex[] = "0123456789abcdef";
    if (!bytes) {
        strcpy(text, "unavailable");
        return;
    }
    for (unsigned i = 0; i < 32; ++i) {
        text[i * 2] = hex[bytes[i] >> 4];
        text[i * 2 + 1] = hex[bytes[i] & 15];
    }
    text[64] = 0;
}

static int dispatch_trace_operand_text(char *text, size_t capacity, const VkExecDispatchTraceOperand *o) {
    const view_t *v = &o->view;
    const uint32_t view_flags = (v->set_levels ? 1u : 0u) | (v->set_layers ? 2u : 0u) |
        (v->set_format ? 4u : 0u) | (v->set_target ? 8u : 0u) | (v->set_swizzle ? 16u : 0u) |
        (v->set_dsmode ? 32u : 0u);
    return snprintf(text, capacity,
        "%s(known=%#x,set=%d,binding=%d,slot=%d,member=%d,stage=%d,kind=%d,descriptor=%d,count=%u,"
        "resource-type=%#x,flags=%#x,handle=%#llx,image=%#llx,address=%#llx,bytes=%#llx,extent=%ux%ux%u,"
        "shape=%u/%u/%#x/%u/%u/%u,registration=%llu,history=%d,holds=%#x,view=%d/%#x,"
        "mips=%u/%u,layers=%u/%u,format=%u,target=%u,depth-mode=%u,swizzle=%u/%u/%u/%u,planes=%#x/%u/%u/%u/%u)",
        o->role ? o->role : "unobserved", o->known, o->set, o->binding, o->slot, o->member, o->stage, o->kind,
        o->descriptor_type, o->count, o->resource_type, o->flags, (unsigned long long)o->handle,
        (unsigned long long)o->image, (unsigned long long)o->address, (unsigned long long)o->bytes,
        o->width, o->height, o->depth, o->type, o->format, o->aspect, o->levels, o->layers, o->samples,
        (unsigned long long)o->registration, o->history, o->holds, o->has_view, view_flags,
        v->base_level, v->nlevels, v->min_layer, v->nlayers, v->format, v->target, v->dsmode,
        v->swz[0], v->swz[1], v->swz[2], v->swz[3], o->planes.aspectMask, o->planes.baseMipLevel,
        o->planes.levelCount, o->planes.baseArrayLayer, o->planes.layerCount);
}

void vk_exec_dispatch_input_trace_log(const VkExecDispatch *dispatch, const char *stage, int status) {
    const VkExecDispatchInputTrace *trace = &dispatch->input_failure.trace;
    if (!trace->gate) return;
    enum { MAX_ROWS = 64, ROW_BYTES = 2040 }; // rt_log reserves [host], newline and a terminator.
    char output[768], input[768], row[ROW_BYTES], key[65];
    const int out = dispatch_trace_operand_text(output, sizeof output, &trace->output);
    const int in = dispatch_trace_operand_text(input, sizeof input, &trace->input);
    if (out < 0 || out >= sizeof output || in < 0 || in >= sizeof input) return;
    sampled_receipt_hex(trace->shader_known ? trace->shader_key : NULL, key);
    const RcRegistryPosition read = dispatch->read_position;
    const int length = snprintf(row, sizeof row,
        "vk: dispatch input trace read=%llu/%llu program=%llu op=%u path=%s stage=%s status=%d gate=%s detail=%s "
        "source-key=%s native-exact=%d output=%s input=%s",
        (unsigned long long)read.scene, (unsigned long long)read.ordinal, (unsigned long long)dispatch->program_serial,
        dispatch->opcode, dispatch->compatibility ? "compat" : "native", stage ? stage : "unrecorded", status,
        trace->gate, trace->detail ? trace->detail : "unobserved", key, trace->native_exact, output, input);
    if (length < 0 || length >= sizeof row) return;

    static PlatMutex lock = PLAT_MUTEX_INIT;
    static char (*seen)[ROW_BYTES];
    static unsigned count;
    plat_mutex_lock(&lock);
    if (!seen) seen = calloc(MAX_ROWS, sizeof *seen);
    int insert = seen && count < MAX_ROWS;
    for (unsigned i = 0; insert && i < count; ++i)
        if (!strcmp(seen[i], row)) insert = 0;
    if (insert) memcpy(seen[count++], row, (size_t)length + 1);
    plat_mutex_unlock(&lock);
    if (insert) rt_log("%s", row);
}

static void selected_sampled_failure_log(const VkExecSampledFailure *failure, uint32_t opcode, const char *path) {
    if (!failure->present) return;
    const VkTextureAliasFailure *alias = &failure->alias;
    const VkTextureSampleRefusal *legacy = &failure->legacy;
    const RcRegistryNativeImageFailure *coverage = &alias->coverage;
    uint8_t translated_hash[32];
    uint64_t words = 0;
    const uint32_t *module = failure->shader ? gfx_shader_spirv(failure->shader, &words) : NULL;
    if (module && words && words <= SIZE_MAX / sizeof *module)
        rt_sha256(module, (size_t)words * sizeof *module, translated_hash);
    else module = NULL;
    char key[65], translated[65];
    sampled_receipt_hex(failure->shader ? gfx_shader_key(failure->shader) : NULL, key);
    sampled_receipt_hex(module ? translated_hash : NULL, translated);
    const RcInterval range = coverage->interval.range.size ? coverage->interval :
        (alias->receiver.range.size ? alias->receiver : legacy->mapped_source);
    const VkTexImage *material = &legacy->material_shape;
    const char *row = coverage->interval.range.size ? "texel-demand" :
        alias->receiver.range.size ? "receiver" : legacy->mapped_source.range.size ? "retained-source-span" : "missing";
    const char *provider = coverage->selected.image ? "positioned-image" : coverage->blocker.id ? "blocked" :
        alias->raw_attempted ? "raw-attempted" : legacy->material ? "material" : "missing";
    const char *native = module && failure->native_exact ? translated : "unavailable";
    rt_log("vk: selected sampled failure read=%llu/%llu opcode=%u path=%s program=%llu pipeline=%p revision=%p "
        "stage=%d set=%d binding=%d translated-set=%d translated-binding=%d member=%d logical=%d sites=%u "
        "handle=%#llx status=%d gate=%s access=%s geometry=%s source-key=%s translated-sha256=%s "
        "final-sha256=%s module=%s registration=%llu material=%llu image=%p guest=%#llx extent=%ux%ux%u "
        "format=%u aspect=%#x levels=%u layers=%u samples=%u row=%s mapping=%llu backing=%llu "
        "offset=%#llx bytes=%#llx provider=%s phase=%d source-status=%d source-gate=%s "
        "source-read=%llu/%llu upload-retry=%llu raw=%u/%d/%u raw-location=%llu/%llu/%#llx "
        "image-winner=%#llx hold=%llu/%u author=%llu/%llu blocker=%llu/%u blocker-author=%llu/%llu "
        "coverage=%d/%d demand=%u writer=%u/%u guard=%u attachment-seed=%u",
        (unsigned long long)failure->read.scene, (unsigned long long)failure->read.ordinal, opcode, path,
        (unsigned long long)failure->program, (void *)failure->pipeline, (void *)failure->revision,
        failure->stage, failure->set, failure->binding, failure->translated_set, failure->translated_binding,
        failure->member, failure->logical, failure->matching_sites, (unsigned long long)failure->handle, failure->status,
        failure->gate ? failure->gate : "unrecorded", failure->geometry_only ? "geometry-proved" :
            failure->geometry ? "geometry-unavailable" : failure->table ? "table-texels" : "texel-path",
        failure->geometry ? failure->geometry : "unrecorded", key, translated, native,
        failure->native_exact && module ? "native-exact" : failure->pipeline ? "native-bytes-unavailable" : "unselected",
        (unsigned long long)(alias->registration ? alias->registration : legacy->registration ? legacy->registration : failure->registration),
        (unsigned long long)(alias->material_serial ? alias->material_serial : legacy->material),
        (void *)(alias->image ? alias->image : failure->image ? failure->image : legacy->image),
        (unsigned long long)(range.address ? range.address : legacy->address ? legacy->address : failure->address),
        failure->width ? failure->width : material->width, failure->height ? failure->height : material->height,
        failure->depth ? failure->depth : material->depth, failure->format ? failure->format : material->format,
        failure->aspect ? failure->aspect : material->aspect, failure->levels ? failure->levels : material->levels,
        failure->layers ? failure->layers : material->layers, failure->samples ? failure->samples : material->samples,
        row, (unsigned long long)range.mapping, (unsigned long long)range.range.backing,
        (unsigned long long)range.range.offset, (unsigned long long)range.range.size, provider,
        alias->status ? (int)alias->phase : -1, alias->status ? (int)alias->status : legacy->recorded ? (int)legacy->result : -1,
        alias->layout_gate ? alias->layout_gate : legacy->stage ? legacy->stage : "unrecorded",
        (unsigned long long)legacy->position.scene, (unsigned long long)legacy->position.ordinal,
        (unsigned long long)legacy->upload_retry, alias->raw_attempted, alias->raw_attempted ? (int)alias->raw_status : -1,
        alias->raw_spans, (unsigned long long)alias->raw_origin.allocation, (unsigned long long)alias->raw_origin.incarnation,
        (unsigned long long)alias->raw_origin.offset, (unsigned long long)coverage->selected.image,
        (unsigned long long)coverage->selected.hold.id, coverage->selected.hold.index,
        (unsigned long long)coverage->selected_output.scene, (unsigned long long)coverage->selected_output.ordinal,
        (unsigned long long)coverage->blocker.id, coverage->blocker.index,
        (unsigned long long)coverage->blocking_output.scene, (unsigned long long)coverage->blocking_output.ordinal,
        coverage->gate, coverage->cause, coverage->demand, coverage->writer_kind, coverage->writer_state,
        coverage->guard_reason, failure->attachment_seed);
    resource_effects_image_trace_blocker(failure->read, coverage);
}

static void graphics_preparation_unavailable(VkExec *ex, uint32_t opcode, VkExecGraphicsResult result,
    const VkExecPreparationFailure *failure, const char *reason) {
    char details[448];
    if (failure) {
        static const char *const owners[] = {"none", "image", "descriptor", "scratch", "pool-image", "pipeline"};
        static const char *const reasons[] = {"mapping", "order", "placement", "pressure", "attachment",
            "feedback", "descriptor-source", "descriptor-set"};
        uint64_t program = failure->sampled.present && failure->sampled.program ? failure->sampled.program :
            ex->prog ? ex->prog->serial : 0;
        snprintf(details, sizeof details,
            "%s/%s program=%llu result=%d stage-result=%d detail=%d role=%u slot=%u "
            "range={mapping=%llu backing=%llu offset=0x%llx size=0x%llx} "
            "peer={mapping=%llu backing=%llu offset=0x%llx size=0x%llx}",
            owners[failure->owner], reasons[failure->reason], (unsigned long long)program,
            result, failure->result, failure->detail,
            failure->role, failure->slot,
            (unsigned long long)failure->range.mapping, (unsigned long long)failure->range.range.backing,
            (unsigned long long)failure->range.range.offset, (unsigned long long)failure->range.range.size,
            (unsigned long long)failure->peer.mapping, (unsigned long long)failure->peer.range.backing,
            (unsigned long long)failure->peer.range.offset, (unsigned long long)failure->peer.range.size);
        reason = details;
        if (failure->issue.gate) {
            rt_log("vk: required graphics compatibility issue read=%llu/%llu actual=%llu/%llu gate=%s stage=%s "
                "prior-pipeline=%u pipeline=%p revision=%p marker=%u prior-pool=%u pool=%u finite=%u "
                "pool-index=%u pool-generation=%llu pool-kinds=%#x required-kinds=%#x pool-read=%llu/%llu "
                "prior-scratch=%u scratch=%u scratch-merges=%u",
                (unsigned long long)failure->issue.prepared.scene, (unsigned long long)failure->issue.prepared.ordinal,
                (unsigned long long)failure->issue.actual.scene, (unsigned long long)failure->issue.actual.ordinal,
                failure->issue.gate, failure->stage ? failure->stage : "unrecorded",
                failure->issue.prior_pipeline, (void *)failure->issue.pipeline, (void *)failure->issue.revision,
                failure->issue.marker, failure->issue.prior_pool, failure->issue.has_pool, failure->issue.finite_pool,
                failure->issue.pool_index, (unsigned long long)failure->issue.pool_generation, failure->issue.pool_kinds,
                failure->issue.required_kinds, (unsigned long long)failure->issue.pool_read.scene,
                (unsigned long long)failure->issue.pool_read.ordinal,
                failure->issue.prior_scratch, failure->issue.has_scratch, failure->issue.scratch_merges);
        }
        selected_sampled_failure_log(&failure->sampled, opcode, "graphics");
        if (failure->sampled.present && failure->reason == VK_EXEC_PREPARATION_FEEDBACK && failure->image)
            rt_log("vk: required graphics attachment feedback slot=%u target=%p sampled=%p",
                failure->slot, (void *)failure->image, (void *)failure->sampled.image);
        vk_exec_graphics_storage_failure_log(&failure->storage);
        if (failure->index.present) {
            rt_log("vk: required graphics index widening read=%llu/%llu actual=%llu/%llu stage=%s initial=%d/%s ordered=%u "
                "import=%u address=0x%llx bytes=0x%llx buffer=%p offset=0x%llx class=%d",
                (unsigned long long)failure->index.position.scene, (unsigned long long)failure->index.position.ordinal,
                (unsigned long long)failure->index.actual.scene, (unsigned long long)failure->index.actual.ordinal,
                failure->stage ? failure->stage : "unrecorded", failure->index.initial_result,
                failure->index.initial ? failure->index.initial : "unrecorded", failure->index.ordered,
                failure->index.import_used, (unsigned long long)failure->index.address,
                (unsigned long long)failure->index.bytes, (void *)failure->index.buffer,
                (unsigned long long)failure->index.offset, failure->index.selected_class);
        }
        if (failure->sample.present) {
            const VkTextureSampleRefusal *source = &failure->sampled_source;
            rt_log("vk: required graphics sampled source read=%llu/%llu stage=%u member=%u logical=%u status=%d "
                "selection=%s recorded=%u cause-read=%llu/%llu registry=%d registration=%llu material=%llu "
                "image=%p address=0x%llx bytes=0x%llx uploaded=%d upload-retry=%llu",
                (unsigned long long)failure->sample.position.scene, (unsigned long long)failure->sample.position.ordinal,
                failure->role, failure->sample.member, failure->slot, failure->sample.status,
                source->stage ? source->stage : "unrecorded", source->recorded,
                (unsigned long long)source->position.scene, (unsigned long long)source->position.ordinal,
                source->recorded ? (int)source->result : -1, (unsigned long long)source->registration,
                (unsigned long long)source->material, (void *)source->image,
                (unsigned long long)source->address, (unsigned long long)source->bytes,
                source->material ? source->uploaded : -1, (unsigned long long)source->upload_retry);
            if (source->tracking_failure != TEX_SOURCE_TRACKING_NONE)
                rt_log("vk: required graphics sampled host tracking read=%llu/%llu reason=%s "
                       "material-version=%llu material-payload=%llu material-provider=%u source-copy-version=%llu",
                       (unsigned long long)source->position.scene, (unsigned long long)source->position.ordinal,
                       texture_source_tracking_failure_name(source->tracking_failure),
                       (unsigned long long)source->material_version, (unsigned long long)source->material_payload,
                       source->material_provider, (unsigned long long)source->source_copy_version);
        }
        if (failure->load.present) {
            rt_log("vk: required graphics attachment load read=%llu/%llu stage=%s slot=%u seed=%u "
                "native-checked=%u preserve=%d address=0x%llx bytes=0x%llx guest-format=%u native-format=%u "
                "extent=%ux%u samples=%u mip=%u layer=%u/%u",
                (unsigned long long)failure->load.position.scene, (unsigned long long)failure->load.position.ordinal,
                failure->stage ? failure->stage : "prepare", failure->slot, failure->load.seeded,
                failure->load.native_checked, failure->load.native_checked ? failure->load.preserve : -1,
                (unsigned long long)failure->load.address, (unsigned long long)failure->load.bytes,
                failure->load.guest_format, failure->load.native_format, failure->load.width, failure->load.height,
                failure->load.samples, failure->load.level, failure->load.layer, failure->load.layers);
        }
        if (failure->registration || failure->image || failure->coverage.gate) {
            const RcRegistryNativeImageFailure *coverage = &failure->coverage;
            if (!failure->sampled.present && coverage->writer_kind == RC_REGISTRY_BUFFER)
                resource_effects_image_trace_blocker(failure->sampled.read, coverage);
            rt_log("vk: required graphics image selection stage=%s registration=%llu material=%llu image=%p "
                "coverage gate=%u cause=%u demand=%u winner=%llx hold=%llu/%u output=%llu/%llu "
                "blocker=%llu/%u writer=%u/%u guard=%u output=%llu/%llu "
                "blocker-range={mapping=%llu backing=%llu offset=0x%llx size=0x%llx}",
                failure->stage ? failure->stage : "prepare", (unsigned long long)failure->registration,
                (unsigned long long)failure->material_serial, (void *)failure->image,
                coverage->gate, coverage->cause, coverage->demand, (unsigned long long)coverage->selected.image,
                (unsigned long long)coverage->selected.hold.id, coverage->selected.hold.index,
                (unsigned long long)coverage->selected_output.scene, (unsigned long long)coverage->selected_output.ordinal,
                (unsigned long long)coverage->blocker.id, coverage->blocker.index, coverage->writer_kind,
                coverage->writer_state, coverage->guard_reason, (unsigned long long)coverage->blocking_output.scene,
                (unsigned long long)coverage->blocking_output.ordinal, (unsigned long long)coverage->blocking_interval.mapping,
                (unsigned long long)coverage->blocking_interval.range.backing,
                (unsigned long long)coverage->blocking_interval.range.offset,
                (unsigned long long)coverage->blocking_interval.range.size);
        }
    }
    vk_exec_required_graphics_unavailable(ex, opcode, reason);
}

void vk_exec_graphics_preparation_unavailable(VkExec *ex, uint32_t opcode, VkExecGraphicsResult result,
    const VkExecPreparationFailure *failure, const char *reason) {
    graphics_preparation_unavailable(ex, opcode, result, failure, reason);
}

static int graphics_input_finished(VkExec *ex, const op_hdr *op, VkExecGraphicsResult result,
                                   const VkExecGraphicsDraw *draw) {
    if (result == VK_EXEC_GRAPHICS_STOPPED) {
        cancel_recording();
        return 0;
    }
    if (result == VK_EXEC_GRAPHICS_INVALIDATED || result == VK_EXEC_GRAPHICS_UNAVAILABLE ||
        result == VK_EXEC_GRAPHICS_NEEDS_ORDERED) {
        const char *reason = result == VK_EXEC_GRAPHICS_INVALIDATED ? "could not safely replan invalidated draw inputs" :
                             result == VK_EXEC_GRAPHICS_NEEDS_ORDERED ? "ordered draw dependency has no progress receipt" :
                             "no valid prepared manifest";
        const VkExecPreparationFailure *failure = vk_exec_graphics_failure(draw);
        graphics_preparation_unavailable(ex, op->op, result, failure, reason);
        return 0;
    }
    return 1;
}

static VkExecGraphicsResult prepare_ordered_graphics(VkExec *ex, const op_hdr *op, int boundary,
                                                    RcRegistryPosition *attempted, RcRegistryPosition *replanned,
                                                    VkExecGraphicsDraw **draw) {
    VkExecGraphicsResult result = boundary == GRAPHICS_BOUNDARY_REPLAN ? VK_EXEC_GRAPHICS_INVALIDATED :
                                  boundary == GRAPHICS_BOUNDARY_ORDERED ? VK_EXEC_GRAPHICS_NEEDS_ORDERED :
                                  graphics_input(ex, op, 0, *attempted, NULL, draw);
    if (result != VK_EXEC_GRAPHICS_NEEDS_ORDERED && result != VK_EXEC_GRAPHICS_INVALIDATED) return result;
    group_drain(ex, PW_OP);
    if (ex->resource_next >= (UINT64_MAX - 1) / 2) rt_fatal("vk: graphics input identity exhausted");
    uint64_t next = ex->resource_next;
    RcRegistryPosition read = {ex->fence, (next + 1) * 2};
    for (;;) {
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_GRAPHICS_STOPPED;
        if (result == VK_EXEC_GRAPHICS_INVALIDATED) {
            if (!*draw || !rc_registry_position_compare(*replanned, read)) return result;
            *replanned = read;
            uint64_t saved_op = ex->resource_op;
            ex->resource_op = next + 1;
            int replan = vk_exec_graphics_draw_replan(ex, op->op, nvn_op_payload(op), *draw);
            ex->resource_op = saved_op;
            if (ex->resource_next != next) rt_fatal("vk: replanning a draw changed replay order");
            if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_GRAPHICS_STOPPED;
            if (!replan) return result;
        }

        VkExecGpaSampleScope *pending = NULL;
        result = graphics_input(ex, op, 1, *attempted, &pending, draw);
        if (pending && result != VK_EXEC_GRAPHICS_NEEDS_ORDERED) {
            vk_exec_gpa_sample_cancel(pending);
            pending = NULL;
        }
        if (result == VK_EXEC_GRAPHICS_INVALIDATED) continue;
        if (!pending) return result;
        if (!rc_registry_position_compare(*attempted, read)) {
            vk_exec_gpa_sample_cancel(pending);
            vk_exec_required_graphics_unavailable(ex, op->op, "sampled input remained pending after its exact prefix retry");
            return VK_EXEC_GRAPHICS_UNAVAILABLE;
        }
        if (*draw) vk_exec_graphics_draw_release_gpa_inputs_retry(*draw);
        VkSyncProgressResult progress = vk_exec_gpa_sample_settle(pending, attempted);
        if (progress == VK_SYNC_PROGRESS_STOPPED || progress == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT)
            return VK_EXEC_GRAPHICS_STOPPED;
        if (progress != VK_SYNC_PROGRESS_COMPLETE) {
            vk_exec_required_graphics_unavailable(ex, op->op, "sampled producer has no usable prefix proof");
            return VK_EXEC_GRAPHICS_UNAVAILABLE;
        }
        if (ex->resource_next != next || rc_registry_position_compare(*attempted, read))
            rt_fatal("vk: graphics prefix completed a different read position");
    }
}

VkSyncProgressResult vk_exec_temporal_prepare(VkExec *ex, int (*prepare)(void *), void *context) {
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    if (!ex || x_cur != ex || !prepare) return VK_SYNC_PROGRESS_UNAVAILABLE;
    if (rw_chunk) {
        // This chunk's part still has unpublished slots. Only earlier sealed scenes can make progress here.
        if (rw_chunk != temporal_hook_chunk || Q.out - Q.drained != 1) return VK_SYNC_PROGRESS_UNAVAILABLE;
    } else {
        group_drain(ex, PW_HOOKS);
        if (vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    }
    return vk_sync_progress_prepare_pipeline(prepare, context);
}

// Before the front pass of a new group: its slot holds a group put in its part (the oldest goes in first).
static void group_room(VkExec *ex) {
    RWT_CHAIN(work_room,RWT_REGION_COORDINATION);
    if (Q.out - Q.drained < (uint64_t)RW.pipe) return;
    uint64_t want = Q.drained + 1;
    if (__atomic_load_n(&Q.done, __ATOMIC_ACQUIRE) < want) {
        uint64_t t0 = kern_now_ns();
        gfx_tl_note('q', t0, (uint64_t)(uintptr_t)pw_name[PW_NEXT], 0);
        RP_CALL(RP_OWNER_HELP, group_take(RW.n));
        while (__atomic_load_n(&Q.done, __ATOMIC_ACQUIRE) < want) plat_pause();
        gfx_tl_note('u', 0, 0, 0);
        RW.pw_n[PW_NEXT]++, RW.pw_ns[PW_NEXT] += kern_now_ns() - t0;
    }
    groups_put_done(ex);
}

// The ops the submitting thread replays past a group it handed out; before any other it waits for the group. The
// ops a chunk records, the game's barriers, dispatches, and events (a wait reads memory, a signal's value goes there
// from the completion thread): their lookups go again once the group is done (op_again), and they leave what the
// chunks read as it was (vk_exec_shared, vk_exec_epoch, vk_exec_upd_top).
static int pipe_op(uint32_t op) {
    return run_op(op) || op == OP_BARRIER || op == OP_DISPATCH || op == OP_DISPATCH_IND || op == OP_WAIT_EVENT ||
           op == OP_SIGNAL_EVENT;
}

// A lookup of op h would have made or taken back a texture entry before the group it went on past was done (the
// submitting thread's vk_texture_deferred): once it is, h goes again, as a chunk's op does in turn.
static int op_texture_turn(VkExec *ex) {
    vk_exec_pass_end(ex);
    ex->rt_resolved = 0, ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    group_wait(PW_TEX, 1);
    if (vk_exec_recording_failed()) return 0;
    __atomic_add_fetch(&RW.redos, 1, __ATOMIC_RELAXED);
    return 1;
}

static void op_again(VkExec *ex, const op_hdr *h) {
    if (op_texture_turn(ex)) op_one(ex, h);
}

// Predict a known pool-consuming draw family from bounded immutable command metadata. No resolver, shader lookup
// or draw runs on this copy. The hint only chooses where the real draw executes; it proves no resource reuse.
static const op_hdr *pool_leader_draw(const VkExec *ex, const op_hdr *h, const op_hdr *end) {
    static VkExec state; // replay-thread owned; the large state does not consume a worker's stack
    state = *ex;
    // A pure render-enable operation may replace the already resolved outcome before the draw. New conditional
    // operations still refuse prediction and retain their ordinary execution point.
    for (unsigned count = 0; h != end && count < 128; count++, h = nvn_op_next(h)) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            return vk_exec_draw_pool_refused(&state, VK_POOL_REFUSE_INVALID), NULL;
        const void *payload = nvn_op_payload(h);
        if (h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND)
            return vk_exec_draw_pool_requirement_known(&state, 0) ? h : NULL;
        if (h->op == OP_DRAW_MESH && h->size >= sizeof *h + sizeof(p_draw_mesh) &&
            ((const p_draw_mesh *)payload)->count && vk_exec_draw_pool_requirement_known(&state, 1)) return h;
        if (h->op == OP_SET_RT) ahead_set_rt(&state, payload);
        else if (h->op == OP_RENDER_ENABLE) state.render_disabled = !((const p_flag *)payload)->value;
        // These effects do not select a shader family. Prediction performs none of them: the canonical prefix
        // executes their query/contact/clear work once, in stream order, before the consuming draw.
        else if (h->op == OP_REPORT_COUNTER || h->op == OP_RESET_COUNTER || h->op == OP_CLEAR_DS) continue;
        else if (h->op == OP_BEGIN || h->op == OP_SET_TEX_POOL || h->op == OP_SET_SAMP_POOL) continue;
        else if (nvn_record_exec_flags(h->op) & NVN_EXEC_STATE_INPUT)
            vk_exec_ahead_op(&state, h->op, payload); // state cases only; never its draw/translation case
        else {
            vk_exec_draw_pool_prefix_op(h->op);
            return vk_exec_draw_pool_refused(&state, VK_POOL_REFUSE_ORDERED), NULL;
        }
    }
    return vk_exec_draw_pool_refused(&state, VK_POOL_REFUSE_LIMIT), NULL;
}

static const op_hdr *pool_leader_run(VkExec *ex, const op_hdr *h, const op_hdr *end, uint32_t *draws) {
    if (*draws <= 2 * CHUNK_DRAWS) return h;
    if (Q.drained != Q.out) return vk_exec_draw_pool_refused(ex, 0), h;
    if (!vk_exec_draw_pool_leader_needed(ex)) return h;
    const op_hdr *leader = pool_leader_draw(ex, h, end);
    if (!leader) return h;
    // Preserve the original group-entry pass boundary before the first draw's pass-begin hooks. The remaining
    // workers receive a second boundary after this canonical draw has completed its resource preparation.
    vk_exec_barrier(ex);
    // Execute this draw, not a speculative refresh. Pass hooks, shader selection, resource resolution and uploads
    // retain their original order. The remaining draws use the same completed publication as any later group.
    const op_hdr *after = nvn_op_next(leader);
    for (const op_hdr *o = h; o != after; o = nvn_op_next(o)) {
        VkExecGraphicsDraw *draw = NULL;
        RcRegistryPosition attempted = {0}, replanned = {0};
        VkExecGraphicsResult result = graphics_opcode(o->op) ? prepare_ordered_graphics(ex, o, GRAPHICS_BOUNDARY_NONE,
            &attempted, &replanned, &draw) : VK_EXEC_GRAPHICS_COMPAT;
        if (!graphics_input_finished(ex, o, result, draw)) {
            vk_exec_graphics_draw_cancel(&draw);
            return o;
        }
        int noop = result == VK_EXEC_GRAPHICS_NOOP;
        if (noop) vk_exec_graphics_draw_cancel(&draw);
        VkExecGraphicsDraw *prior_graphics = vk_exec_graphics_draw_enter(draw);
        void *effect = NULL;
        if (noop) resource_declare_noop(ex);
        else effect = resource_declare(ex, o, 1, NULL);
        void *previous_effect = vk_exec_effects_enter(effect);
        if (noop) graphics_noop_record(ex, o);
        else op_one(ex, o);
        vk_exec_effects_recorded(effect,0);
        vk_exec_effects_leave(previous_effect);
        vk_exec_effects_release(effect);
        vk_exec_graphics_draw_enter(prior_graphics);
        vk_exec_graphics_draw_cancel(&draw);
        ex->resource_op = 0;
    }
    vk_exec_draw_pool_leader_finished(ex);
    (*draws)--;
    RW.serial_draws++;
    return after;
}

static int group_chunks(uint32_t draws) {
    int lanes = RW.n + 1;
    int n = (int)(draws / CHUNK_DRAWS);
    if (n <= lanes) return n;
    // At 256 draws per recorder a single slow chunk can hold the entire group. Leave a second
    // wave on the existing queue so recorders that finish early can take more of the same pass.
    if (RW.extra_chunks && draws / (uint32_t)lanes >= 256 && lanes < GROUP_CHUNKS) {
        int target = 2 * lanes < GROUP_CHUNKS ? 2 * lanes : GROUP_CHUNKS;
        if (vk_sync_append_room(target)) return target;
    }
    return lanes;
}

// The submitting thread, at op h of the walk: when a group large enough to split starts here, it is recorded by the
// workers and this thread, and the walk goes on with the op that ended it, which is returned; otherwise h (the walk
// replays the group itself, without reading ahead again before its end).
static const op_hdr *group_start(VkExec *ex, const op_hdr *h, int depth, const NvnRecordPlan *plan,
                                 VkExecGraphicsDraw **boundary, int *ordered_boundary) {
    RWT_CHAIN(work_front,RWT_REGION_GROUP_FRONT);
    const op_hdr *end;
    int why;
    uint32_t end_op;
    uint32_t draws;
    int planned = group_peek_plan(ex, plan, h, &end, &why, &end_op, &draws);
    if (!planned) {
        if (plan) RW.plan_fallbacks++;
        draws = group_peek(ex, h, &end, &why, &end_op);
    } else if (resource_prep_mode() == 2) {
        const op_hdr *ref_end;
        int ref_why;
        uint32_t ref_op, ref_draws = group_peek(ex, h, &ref_end, &ref_why, &ref_op);
        if (draws != ref_draws || end != ref_end || why != ref_why || end_op != ref_op)
            rt_fatal("vk: prepared command group differs from reference walk");
    }
    if (planned) RW.planned_groups++, RW.planned_draws += draws;
    if (why == GE_RUN && end_op < OP_COUNT_) RW.run_end[end_op] += draws;
    if (end == h) return h;   // an op the submitting thread replays itself
    RW.end_draws[why] += draws;
    int n = group_chunks(draws);
    if (n < 2 || vk_dump_armed || !vk_sync_append_room(n) || vk_gpa_write_scene_may_need_prefix(ex->fence)) {
        RW.skip_to[depth] = end, RW.serial_draws += draws, RW.small_end[why] += draws;
        return h;
    }
    if (vk_dlaa_temporal_hook_needs_recorder()) {
        group_drain(ex, PW_HOOKS);
        if (vk_exec_recording_failed()) return h;
    }
    if (RW.pipe) {
        // a part due goes to the GPU before the group, with the groups still recording in it: its GPU work would
        // otherwise wait for this group's chunks too
        if (split_draws && split_wanted()) group_drain(ex, PW_SPLIT), maybe_split(ex);
        group_room(ex);
    }
    h = pool_leader_run(ex, h, end, &draws);
    if (vk_exec_recording_failed()) return h;
    n = group_chunks(draws);
    Group *g = &GQ[(Q.out + 1) % LIVE_GROUPS];
#if NX2NAVI_REPLAY_WORK
    g->work_replay = gfx_rwt_replay();
#endif
    RP_SCOPE(rp_group_front, RP_GROUP_FRONT);
    RP_VALUE(RPK_WORK_UNIT, gfx_rwt_current_id());
    uint64_t t0 = kern_now_ns();
    gfx_tl_note('g', 0, draws, (uint64_t)n);
    vk_exec_barrier(ex);         // the command buffer so far ends behind a full barrier, every image back in GENERAL
    vk_pool_view_reset(&g->pool);   // group_room/ordered replay has retired this group's previous consumers
    g->requirement.valid = 0;
    ex->resource_op = 0; // pool preparation is outside a declared draw input phase
    g->upd = vk_exec_draw_group_begin(ex, &g->pool);
    // the front pass
    uint32_t per = (draws + (uint32_t)n - 1) / (uint32_t)n, cd = 0;
    int nc = 1, cut = 0;
    uint32_t mesh_draws = 0, actual_draws = 0;
    g->ncond = 0;
    if (g->neffects) rt_fatal("vk: resource declarations outlived their recording group");
    g->resource_begin = ex->resource_next + 1;
    vk_exec_xfer_group((int)(g - GQ));
    Chunk *c = chunk_open(g, ex, h, 0, plan);
    size_t exec_cursor = SIZE_MAX;
    int executable = exec_plan_enabled();
    ExecPlanStats stats __attribute__((cleanup(exec_stats_publish))) = {.path = EXEC_FRONT};
    for (const op_hdr *o = h; o != end;) {
        const void *a = nvn_op_payload(o);
        if (o->op == OP_JUMP) {
            o = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
            continue;
        }
        if (partial_clear_slot(ex, o) >= 0 ||
            (o->op == OP_CLEAR_DS && o->size >= sizeof *o + sizeof(p_clear_ds) && depth_clear_alias_destination(ex, a))) {
            end = o;
            break;
        }
        VkExecGraphicsDraw *draw = NULL;
        int graphics_noop = 0;
        if (graphics_opcode(o->op)) {
            VkExecGraphicsResult result = graphics_input(ex, o, 0, (RcRegistryPosition){0}, NULL, &draw);
            if (result == VK_EXEC_GRAPHICS_INVALIDATED || result == VK_EXEC_GRAPHICS_NEEDS_ORDERED ||
                (result == VK_EXEC_GRAPHICS_COMPAT && draw)) {
                *boundary = draw;
                *ordered_boundary = result == VK_EXEC_GRAPHICS_INVALIDATED ? GRAPHICS_BOUNDARY_REPLAN : GRAPHICS_BOUNDARY_ORDERED;
                end = o;
                break;
            }
            if (!graphics_input_finished(ex, o, result, draw)) {
                vk_exec_graphics_draw_cancel(&draw);
                end = o;
                break;
            }
            graphics_noop = result == VK_EXEC_GRAPHICS_NOOP;
            if (graphics_noop) vk_exec_graphics_draw_cancel(&draw);
        }
        if (cut) c = chunk_open(g, ex, o, nc++, plan), cd = 0, cut = 0;
        // Per-draw effects require intermediate state; final-state summaries cannot declare it correctly.
        const NvnExecStep *step = executable ? nvn_record_exec_slice(plan, o, end, UINT32_MAX, &exec_cursor) : NULL;
        if (step && step->opcode == NVN_EXEC_STATE) {
            resource_group_append(g, ex, o, step->ops, NULL, 0);
            exec_step_run(ex, plan, step);
            exec_stats_step(&stats, step);
            c->nops += step->ops;
            o = (const op_hdr *)(plan->commands + step->end);
            continue;
        }
        resource_group_append(g, ex, o, 1, draw, graphics_noop);
        void *previous_effect = vk_exec_effects_enter(g->ops[g->neffects - 1].effect);
        switch (o->op) {
        case OP_DRAW_MESH: case OP_DRAW_MESH_IND:
            mesh_draws++;
            // fall through
        case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND:
            actual_draws++;
            if (vk_exec_xfer_counting() && !ex->render_disabled) c->nslots++;   // at most one query each
            if (++cd == per && nc < n) cut = 1;
            break;
        case OP_CLEAR_COLOR: case OP_CLEAR_DS: break;   // recorded by the chunk
        case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_ENABLE: case OP_RENDER_COND: {
            int on = vk_exec_xfer_front(ex, o->op, a);
            if (o->op == OP_RENDER_COND) {
                if (g->ncond == g->ccond) {
                    g->ccond = g->ccond ? g->ccond * 2 : 1024;
                    if (!(g->cond = realloc(g->cond, g->ccond))) rt_fatal("vk: out of memory (render conditions)");
                }
                g->cond[g->ncond++] = (uint8_t)on;
                c->ncond++;
            }
            break;
        }
        default:
            if (step && step->opcode != NVN_EXEC_STATE) exec_step_run(ex, plan, step);
            else op_one(ex, o);
            if (plan && plan->nexec) exec_stats_step(&stats, step && step->opcode != NVN_EXEC_STATE ? step : NULL);
            break;   // state, binds, render targets
        }
        vk_exec_effects_leave(previous_effect);
        c->nops++;
        o = nvn_op_next(o);
    }
    ex->resource_op = 0;
    vk_exec_xfer_group(-1);   // the segments stay listed for the workers until the group is put in its part
    if (!g->neffects) return end;
    draws = actual_draws;
    // A group is indivisible for submission splitting. Count it once in stream order before publishing its
    // chunks. Recording workers can retry a draw and finish out of order without racing the split threshold.
    vk_exec_frame.draws_seen += draws;
    split_draws += (int)(draws - mesh_draws);
    for (int i = 0; i < nc; i++) g->c[i].slot0 = vk_exec_xfer_slots(g->c[i].nslots);   // in the order of the stream
    // handed out: the workers (and without RW.pipe this thread) take the chunks in order
    for (int i = 0; i < nc; i++) {
        g->c[i].done = 0;
        vk_pool_view_reset(&g->c[i].pool);
    }
    g->hooks = 0, g->ndone = 0, g->n = nc;
    __atomic_store_n(&g->seq, ++Q.out, __ATOMIC_RELEASE);
    // the replay's command buffer so far goes into the part, then the chunks' places, and the replay goes on in a new
    // one at once
    ex->cb = vk_sync_append_later(nc, &g->at);
    uint64_t tf = kern_now_ns();
    g->out_ns = tf;
    int first = 0;
    if (vk_dlaa_temporal_hook_needs_recorder()) {
        // Native targets first resolve inside the pass hook. Record chunk 0 in its usual part slot with the
        // submitter's park authority. Peer chunks cannot observe the generation until this recording returns.
        temporal_hook_chunk = &g->c[0];
        chunk_record(temporal_hook_chunk, RW.n);
        temporal_hook_chunk = NULL;
        first = 1;
    }
    for (int i = first; i < nc; i++) Q.q[(Q.put + (uint64_t)(i - first)) % QN] = &g->c[i];
    __atomic_store_n(&Q.put, Q.put + (uint64_t)(nc - first), __ATOMIC_SEQ_CST);
    for (int k = 0; k < RW.n; k++)
        if (__atomic_load_n(&RW.w[k]->sleeping, __ATOMIC_SEQ_CST)) plat_event_set(&RW.w[k]->wake);
    ex->barrier_at = ex->recorded;   // the chunks end with a full barrier
    ex->in_pass = 0, ex->nfbl = 0, ex->mark = NULL, ex->dirty = VK_EXEC_DIRTY_ALL, ex->rt_resolved = 0, ex->fb_fs = NULL;
    vk_exec_draw_new_cb();
    RW.groups++, RW.group_draws += draws, RW.chunks += (uint64_t)nc;
    RP_END(rp_group_front);
    if (RW.pipe) {
        // the submitting thread takes chunks too, as the workers wake up, and goes on once none is left to take: the
        // GPU waits for the group's part until its last chunk is done
        rw_behind = 1;
        RP_CALL(RP_OWNER_HELP, group_take(RW.n));
        uint64_t to = kern_now_ns();
        RW.front_ns += tf - t0, RW.own_ns += to - tf, RW.wait_ns += to - t0;
        return end;
    }
    RP_CALL(RP_OWNER_HELP, group_take(RW.n));
    uint64_t to = kern_now_ns();
    RP_SCOPE(rp_final_spin, RP_FINAL_SPIN);
    RP_VALUE(RPK_WORK_UNIT, gfx_rwt_current_id());
    {
        RWT_CHAIN(work_spin,RWT_REGION_COORDINATION);
        while (__atomic_load_n(&Q.done, __ATOMIC_ACQUIRE) < g->seq) { chunk_pair_help(g); plat_pause(); }
    }
    RP_END(rp_final_spin);
    uint64_t tw = kern_now_ns();
    RW.front_ns += tf - t0, RW.own_ns += to - tf, RW.spin_ns += tw - to;
    groups_put_done(ex);
    vk_exec_draw_new_cb();   // the submitting thread's own chunks used its draw context
    RW.wait_ns += kern_now_ns() - t0;
    return end;
}

static void progress_adopt_tail(VkExec *ex, VkCommandBuffer cb) {
    ex->cb = cb;
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    split_draws = 0;
    split_at = split_n > 0 ? split_n : INT_MAX;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->rt_resolved = 0;
    vk_exec_draw_new_cb();
}

typedef struct {
    uintptr_t shader, pipeline, info;
    VkMemSpanCapture *argument;
    VkMemSpan argument_span;
    int selected, native_argument;
} GpaDispatchBinding;

typedef struct {
    VkExec *ex;
    const void *effect;
    uint64_t registration;
    RcRegistryPosition attempted;
    VkGpaPrefixRequest request;
    const GpaDispatchBinding *prior_dispatch;
    GpaDispatchBinding dispatch;
    uint64_t indirect_address;
    VkMemSpanCapture *borrowed_argument;
    int binding_recorded;
    int compute, binding_checked, uniform;
    RcRegistryPosition provisional_read;
    int provisional_recorded;
} GpaReadPreparation;
static _Thread_local GpaReadPreparation *gpa_read_preparation;

struct VkExecGpaSampleScope {
    GpaReadPreparation prepare;
    int active;
};
static _Thread_local VkExecGpaSampleScope gpa_sample_scope;

VkExecGpaSampleScope *vk_exec_gpa_sample_enter(VkExec *ex, RcRegistryPosition attempted) {
    if (!ex || x_cur != ex || vk_exec_shared() || Q.drained != Q.out || gpa_read_preparation ||
        gpa_sample_scope.active || ex->resource_op != ex->resource_next + 1) return NULL;
    RcRegistryPosition read = vk_exec_resource_position();
    if (!rc_registry_position_valid(read)) return NULL;
    gpa_sample_scope = (VkExecGpaSampleScope){
        .prepare = {.ex = ex, .attempted = attempted, .provisional_read = read, .provisional_recorded = ex->recorded},
        .active = 1};
    gpa_read_preparation = &gpa_sample_scope.prepare;
    return &gpa_sample_scope;
}

int vk_exec_gpa_sample_leave(VkExecGpaSampleScope *scope) {
    if (scope != &gpa_sample_scope || !scope->active || gpa_read_preparation != &scope->prepare) return 0;
    gpa_read_preparation = NULL;
    return scope->prepare.request.guard.id != 0;
}

void vk_exec_gpa_sample_cancel(VkExecGpaSampleScope *scope) {
    if (scope != &gpa_sample_scope || !scope->active) return;
    if (gpa_read_preparation == &scope->prepare) gpa_read_preparation = NULL;
    if (scope->prepare.request.guard.id) vk_texture_deferred_take();
    *scope = (VkExecGpaSampleScope){0};
}

int vk_exec_gpa_dispatch_retrying(const VkExec *ex) {
    const GpaReadPreparation *prepare = gpa_read_preparation;
    return prepare && prepare->ex == ex && prepare->prior_dispatch;
}

static void gpa_dispatch_release(GpaDispatchBinding *binding) {
    vk_mem_span_capture_release(&binding->argument);
    *binding = (GpaDispatchBinding){0};
}

static int gpa_dispatch_same(const GpaDispatchBinding *expected, const GpaDispatchBinding *current,
                             const VkMemSpan *indirect, VkMemSpanCapture *argument) {
    if (expected->shader != current->shader || expected->pipeline != current->pipeline ||
        expected->info != current->info || expected->selected != current->selected ||
        !!indirect != !!expected->argument) return 0;
    // A previously exact input must reuse its actual receipt. A compatibility input may gain a native
    // receipt now that its mirror is installed, provided both prove the same physical bytes and mapping.
    if (expected->native_argument && expected->argument != argument) return 0;
    return !indirect || (vk_mem_span_capture_current(expected->argument, indirect) &&
                         (!argument || vk_mem_span_capture_current(argument, indirect)));
}

int vk_exec_gpa_dispatch_binding(VkExec *ex, const GfxShader *shader, VkPipeline pipeline, const VkPipeInfo *info,
                                 uint64_t indirect_address, const VkMemSpan *indirect, VkMemSpanCapture *argument,
                                 int repeatable, int selected) {
    GpaReadPreparation *prepare = gpa_read_preparation;
    if (!prepare || prepare->ex != ex || !prepare->compute) return 1;
    GpaDispatchBinding binding = {.shader = (uintptr_t)shader, .pipeline = (uintptr_t)pipeline,
        .info = (uintptr_t)info, .selected = selected, .native_argument = argument != NULL};
    if (prepare->prior_dispatch && !gpa_dispatch_same(prepare->prior_dispatch, &binding, indirect, argument)) {
        rt_log("vk: ordered GPA retry changed the native dispatch input provider");
        __atomic_store_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_RELEASE);
        return 0;
    }
    prepare->binding_checked = 1;
    if (repeatable) {
        prepare->dispatch = binding;
        prepare->indirect_address = indirect_address;
        prepare->borrowed_argument = argument;
        if (indirect) prepare->dispatch.argument_span = *indirect;
        prepare->binding_recorded = ex->recorded;
    }
    return 1;
}

int vk_exec_request_gpa_prefix(VkExec *ex, RcRegistryWrite guard, RcRegistryPosition producer,
                               RcRegistryPosition read, RcInterval source) {
    GpaReadPreparation *prepare = gpa_read_preparation;
    if (!prepare || prepare->request.guard.id || prepare->ex != ex || x_cur != ex || vk_exec_shared() || Q.drained != Q.out ||
        vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed() ||
        rc_registry_position_compare(read, vk_exec_resource_position()) ||
        !rc_registry_position_compare(read, prepare->attempted)) return 0;
    int provisional = rc_registry_position_valid(prepare->provisional_read);
    if (provisional) {
        if (rc_registry_position_compare(read, prepare->provisional_read) ||
            ex->resource_op != ex->resource_next + 1 || ex->recorded != prepare->provisional_recorded ||
            !rc_registry_provisional_read_current(rc_registry_runtime(), read)) return 0;
    } else if (ex->resource_op != ex->resource_next || !vk_exec_effects_unrecorded(prepare->effect)) return 0;
    if (prepare->compute) {
        if (!prepare->dispatch.shader || prepare->binding_recorded != ex->recorded) {
            rt_log("vk: ordered GPA read %llu/%llu has no replayable native compute binding (%s)",
                   (unsigned long long)read.scene, (unsigned long long)read.ordinal,
                   !prepare->binding_checked ? "native binder not reached" :
                   !prepare->dispatch.shader ? "specialized preparation already ran" : "binding recorded native work");
            return 0;
        }
    } else if (!provisional && (ex->tonemap_extent.width || ex->hud_alpha_attachment)) return 0;
    for (uint32_t i = 0; !provisional && !prepare->compute && i < ex->pass_ncolor && i < 8; ++i) {
        const VkExecTarget *target = &ex->color[i];
        if (target->image && target->img && target->image != target->img->image) return 0;
    }
    VkGpaPrefixRequest request = {.guard = guard, .producer = producer, .read = read, .source = source};
    if (!vk_gpa_write_prefix_ready(&request)) return 0;
    VkMemSpan *argument_span = &prepare->dispatch.argument_span;
    if (argument_span->buffer) {
        if (prepare->borrowed_argument) {
            if (vk_mem_span_capture_current(prepare->borrowed_argument, argument_span))
                prepare->dispatch.argument = vk_mem_span_capture_retain(prepare->borrowed_argument);
        } else {
            RcRegistryResult retained = RC_REGISTRY_UNSUPPORTED;
            uint64_t size = sizeof(VkDispatchIndirectCommand);
            if (prepare->dispatch.selected)
                retained = vk_mem_span_capture_bound_input(prepare->indirect_address, size, argument_span,
                                                          &prepare->dispatch.argument);
            if (retained == RC_REGISTRY_OK) prepare->dispatch.native_argument = 1;
            else if (retained == RC_REGISTRY_UNSUPPORTED)
                prepare->dispatch.argument = vk_mem_span_capture(prepare->indirect_address, size, argument_span);
        }
        if (!prepare->dispatch.argument) {
            rt_log("vk: ordered GPA read %llu/%llu could not retain its native indirect argument",
                   (unsigned long long)read.scene, (unsigned long long)read.ordinal);
            return 0;
        }
    }
    prepare->registration = vk_texture_registration_publication();
    prepare->request = request;
    return 1;
}

static int gpa_source_current(RcInterval source) {
    MemBackingCapture *capture = NULL;
    if (mem_backing_capture(source.address, source.range.size, &capture) != RC_BACKING_OK) return 0;
    uint32_t count;
    const RcInterval *ranges = mem_backing_intervals(capture, &count);
    int current = count == 1 && ranges[0].address == source.address && ranges[0].mapping == source.mapping &&
                  rc_range_equal(ranges[0].range, source.range) && mem_backing_current(capture);
    mem_backing_release(&capture);
    return current;
}

static int prepare_ordered_gpa_source(VkExec *ex, const GpaReadPreparation *prepare) {
    const VkGpaPrefixRequest *request = &prepare->request;
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return 0;
    int provisional = rc_registry_position_valid(prepare->provisional_read);
    if (provisional) {
        if (ex->recorded != prepare->provisional_recorded ||
            ex->resource_next * 2 + 2 != request->read.ordinal ||
            !rc_registry_provisional_read_current(rc_registry_runtime(), request->read)) goto refused;
    } else {
        vk_exec_draw_cancel_unrecorded(ex);
        if (rc_registry_reopen_unemitted(rc_registry_runtime(), request->read) != RC_REGISTRY_OK) goto refused;
    }
    if (!vk_gpa_write_prefix_ready(request)) goto refused;
    // The split hands off earlier outputs only. A prospective read has no output declaration or ordinal to rewind.
    if (!provisional) ex->resource_next--;
    vk_exec_barrier(ex);
    VkMemoryBarrier2 host_read = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1, .pMemoryBarriers = &host_read};
    vkCmdPipelineBarrier2(ex->cb, &dep);
    ex->recorded++;
    VkCommandBuffer tail = VK_NULL_HANDLE;
    VkSyncProgressResult result = vk_sync_progress_split(ex, &tail, 0);
    if (tail) progress_adopt_tail(ex, tail);
    if (result == VK_SYNC_PROGRESS_STOPPED || result == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT ||
        vk_sync_stopping() || vk_device_is_lost()) return 0;
    if (result == VK_SYNC_PROGRESS_COMPLETE && vk_gpa_write_prefix_complete(request) &&
        (prepare->uniform || prepare->registration == vk_texture_registration_publication()) &&
        gpa_source_current(request->source) &&
        (!prepare->dispatch.argument || vk_mem_span_capture_current(prepare->dispatch.argument, NULL))) {
        static unsigned reported;
        if (reported < 12) {
            reported++;
            rt_log("vk: ordered GPA prefix completed through %llu/%llu for %s input %llu/%llu",
                   (unsigned long long)request->producer.scene, (unsigned long long)request->producer.ordinal,
                   prepare->uniform ? "uniform" : "sampled",
                   (unsigned long long)request->read.scene, (unsigned long long)request->read.ordinal);
            if (prepare->compute)
                rt_log("vk: ordered GPA sampled input in %s%s native compute binder",
                       prepare->dispatch.selected ? "exact" : "compatibility-selected",
                       prepare->dispatch.argument ? " indirect" : " direct");
        }
        return 1;
    }
refused:
    rt_log("vk: required %s source %#llx+%#llx could not recover ordered GPA producer %llu/%llu",
           prepare->uniform ? "uniform" : "sampled", (unsigned long long)request->source.address, (unsigned long long)request->source.range.size,
           (unsigned long long)request->producer.scene, (unsigned long long)request->producer.ordinal);
    __atomic_store_n(&recording_failed, 1, __ATOMIC_RELEASE);
    return 0;
}

VkSyncProgressResult vk_exec_gpa_sample_settle(VkExecGpaSampleScope *scope, RcRegistryPosition *attempted) {
    if (scope != &gpa_sample_scope || !scope->active) return VK_SYNC_PROGRESS_UNAVAILABLE;
    GpaReadPreparation prepare = scope->prepare;
    int left = gpa_read_preparation != &scope->prepare;
    vk_exec_gpa_sample_cancel(scope);
    if (!prepare.request.guard.id) return VK_SYNC_PROGRESS_UNAVAILABLE;
    if (!left || !attempted) return VK_SYNC_PROGRESS_STOPPED;
    // Rendering work closed by the marker belongs to earlier operations, not this provisional input.
    uint64_t operation = prepare.ex->resource_op;
    prepare.ex->resource_op = 0;
    int complete = prepare_ordered_gpa_source(prepare.ex, &prepare);
    prepare.ex->resource_op = operation;
    if (!complete) return VK_SYNC_PROGRESS_STOPPED;
    *attempted = prepare.request.read;
    return VK_SYNC_PROGRESS_COMPLETE;
}

static int prepare_ordered_scratch(VkExec *ex, const op_hdr *h, uint64_t bytes, VkDrawScratchUpdate **grant) {
    const void *payload = nvn_op_payload(h);
    int update = h->op == OP_UPDATE_UB;
    const char *operation = update ? "UpdateUniformBuffer" : "generated draw indices";
    if (vk_sync_stopping() || vk_device_is_lost()) return 0;
    VkExecUpdateDestination destination = {0};
    if (update && vk_exec_draw_update_select(payload, &destination) != RC_REGISTRY_OK) {
        vk_exec_required_graphics_unavailable(ex, h->op, "inline uniform destination capture unavailable");
        return 0;
    }
    uint64_t observed_prior = 0;
    int checked_empty_prefix = 0;
    VkDrawScratchWaitContext wait = {.ex = ex, .bytes = bytes};
    for (;;) {
        if (vk_sync_stopping() || vk_device_is_lost()) break;
        RcVmaDependency dependency = {0};
        VkResult result = update ? vk_exec_draw_update_prepare(ex, payload, &destination, grant, &dependency)
                                 : vk_exec_draw_index_prepare(ex, payload, grant, &dependency);
        if (vk_sync_stopping() || vk_device_is_lost()) {
            if (update) vk_exec_draw_update_cancel(ex, *grant);
            else vk_exec_draw_index_finish(ex, *grant);
            *grant = NULL;
            break;
        }
        if (result == VK_SUCCESS) return 1;
        if (result != VK_NOT_READY) {
            int stale_destination = update && destination.hold &&
                !vk_mem_span_capture_current(destination.hold, &destination.span);
            vk_exec_draw_update_drop(&destination);
            if (stale_destination) {
                vk_exec_required_graphics_unavailable(ex, h->op, "inline uniform destination changed during scratch preparation");
                return 0;
            }
            rt_fatal("vk: required %s preparation failed (%llu bytes, result %d)",
                     operation, (unsigned long long)bytes, (int)result);
        }

        RcBudgetReadiness ready;
        resource_core_budget_readiness(1, &ready);
        uint64_t prior = vk_exec_draw_scratch_prior_useful(ex, bytes, &dependency, &ready);
        if (prior > observed_prior && prior < ex->fence) {
            observed_prior = prior;
            if (prior <= vk_sync_fence_done()) continue;
            VkSyncProgressResult advanced = vk_sync_progress_wait_prior(prior);
            if (advanced == VK_SYNC_PROGRESS_COMPLETE) continue;
            if (advanced == VK_SYNC_PROGRESS_STOPPED || vk_sync_stopping() || vk_device_is_lost()) break;
        }
        // The private cut waits for every earlier descriptor reader to finish. The ordinary asynchronous
        // split's requirement for another free set does not apply while this recorder stays parked.
        int empty_tail = ex->recorded == part_start;
        if (ex->recorded && (!empty_tail || !checked_empty_prefix) &&
            vk_exec_draw_scratch_prefix_useful(ex, bytes, &dependency, &ready)) {
            if (empty_tail) checked_empty_prefix = 1;
            else vk_exec_pass_end(ex);
            VkCommandBuffer tail = VK_NULL_HANDLE;
            VkSyncProgressResult cut = vk_sync_progress_split(ex, &tail, empty_tail);
            if (tail) progress_adopt_tail(ex, tail);
            if (cut == VK_SYNC_PROGRESS_COMPLETE) { checked_empty_prefix = 1; continue; }
            if (cut == VK_SYNC_PROGRESS_STOPPED || cut == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT ||
                vk_sync_stopping() || vk_device_is_lost()) break;
        }
        VkSyncProgressResult advanced = vk_sync_progress_wait_capacity(&dependency, vk_exec_draw_scratch_spare_ready, &wait);
        if (advanced == VK_SYNC_PROGRESS_COMPLETE) continue;
        if (advanced == VK_SYNC_PROGRESS_STOPPED) break;
        vk_exec_draw_update_drop(&destination);
        rt_fatal("vk: required %s cannot acquire preparation capacity (%llu bytes)",
                 operation, (unsigned long long)bytes);
    }
    vk_exec_draw_update_drop(&destination);
    return 0;
}

static void image_write_input_position(VkExec *ex) {
    if (ex->resource_next >= (UINT64_MAX - 1) / 2) rt_fatal("vk: image write input identity exhausted");
    // The prior outputs are sealed. Name this input position without advancing the operation ordinal or declaring
    // any new writer. The captured source is observed once here, before the first native attempt.
    ex->resource_op = ex->resource_next + 1;
    RcRegistryPosition position = vk_exec_resource_position();
    if (rc_registry_declare_through(rc_registry_runtime(), position) != RC_REGISTRY_OK)
        rt_fatal("vk: image write input prefix could not be sealed");
}

typedef struct {
    VkTextureWriteTarget *target;
    const VkExecBufferToTexture *buffer;
} TextureWriteCapacity;

typedef struct {
    VkExecGraphicsPass *pass;
    VkTextureAliasRead *seeds[9];
} AttachmentClearLoad;

static void attachment_clear_load_cancel(AttachmentClearLoad *load) {
    vk_exec_graphics_pass_cancel(&load->pass);
    for (unsigned i = 0; i < 9; ++i) {
        vk_texture_alias_target_cancel(load->seeds[i]);
        load->seeds[i] = NULL;
    }
}

static int prepare_attachment_clear_load(VkExec *ex, const op_hdr *op, unsigned slot, AttachmentClearLoad *load) {
    image_write_input_position(ex);
    VkExecPreparationFailure refusal = {0};
    const VkExecGraphicsPassResult selected = capture_attachment_load_seeds(ex, 1, 0, load->seeds, &refusal);
    int ready = selected == VK_EXEC_GRAPHICS_PASS_READY;
    unsigned first = 0;
    while (first < 9 && !load->seeds[first]) first++;
    // Inside rendering every bound view LOADs. Acquire the complete sibling cohort
    // before opening the pass; only the requested slot will be marked as written.
    if (ready && first < 9)
        ready = graphics_pass_capture(ex, 1, 0, load->seeds, &load->pass, &refusal) == VK_EXEC_GRAPHICS_PASS_READY &&
            vk_exec_graphics_pass_attachment_peers(load->pass, &refusal);
    if (!ready) {
        attachment_clear_load_cancel(load);
        if (!refusal.owner)
            refusal = (VkExecPreparationFailure){
                .owner = VK_EXEC_PREPARATION_IMAGE, .reason = VK_EXEC_PREPARATION_ATTACHMENT,
                .result = VK_EXEC_GRAPHICS_UNAVAILABLE, .detail = RC_REGISTRY_STALE, .slot = slot,
                .role = slot == 8 ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                .stage = "partial-clear-load",
            };
        if (!vk_sync_stopping() && !vk_device_is_lost())
            graphics_preparation_unavailable(ex, op->op, VK_EXEC_GRAPHICS_UNAVAILABLE, &refusal,
                                             "required partial attachment clear LOAD unavailable");
    }
    ex->resource_op = 0;
    return ready;
}

static int texture_write_input_current(const TextureWriteCapacity *input) {
    return vk_texture_write_target_current(input->target) &&
        (!input->buffer || vk_exec_buffer_to_texture_source_current(input->buffer));
}

static int texture_write_input_changed(void *opaque) {
    return !texture_write_input_current(opaque);
}

static VkSyncProgressResult prepare_ordered_write_target(VkExec *ex, VkTextureWriteTarget *target,
                                                        const VkExecBufferToTexture *buffer,
                                                        VkExecTextureToTexture *image, VkResult *failure) {
    uint64_t observed = 0;
    TextureWriteCapacity input = {target, buffer};
    for (;;) {
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
        if (!texture_write_input_current(&input)) {
            *failure = VK_ERROR_INITIALIZATION_FAILED;
            return VK_SYNC_PROGRESS_UNAVAILABLE;
        }
        RcVmaDependency dependency = {0};
        VkResult result = image && image->scratch_recipe.sType
            ? vk_exec_texture_to_texture_allocate(image, &dependency)
            : vk_texture_write_target_allocate(target, &dependency);
        if (result == VK_ERROR_DEVICE_LOST) vk_device_lost_check(result, "preparing a required image write");
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
        if (result == VK_SUCCESS) {
            if (texture_write_input_current(&input)) return VK_SYNC_PROGRESS_COMPLETE;
            *failure = VK_ERROR_INITIALIZATION_FAILED;
            return VK_SYNC_PROGRESS_UNAVAILABLE;
        }
        VkSyncProgressResult progress = vk_exec_texture_capacity(ex, &dependency, result,
            texture_write_input_changed, &input, &observed);
        if (progress == VK_SYNC_PROGRESS_COMPLETE) continue;
        *failure = result;
        return progress;
    }
}

static int prepare_ordered_clear(VkExec *ex, const p_clear_tex *op, VkExecClearTexture *clear) {
    if (vk_sync_stopping() || vk_device_is_lost()) return -1;
    image_write_input_position(ex);
    if (vk_exec_clear_alias_candidate(op) && prepare_bound_texture_relocation(ex, op->tex, 1) < 0) {
        ex->resource_op = 0;
        if (!vk_sync_stopping() && !vk_device_is_lost()) {
            rt_log("vk: required earlier bound image cannot preserve current bytes before ClearTexture");
            __atomic_store_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_RELEASE);
        }
        return -1;
    }
    int handled = vk_exec_clear_capture(op, clear);
    ex->resource_op = 0;
    if (handled < 0) rt_fatal("vk: required clear lost its selected texture version");
    if (!clear->target) return handled;
    VkResult result = VK_SUCCESS;
    VkSyncProgressResult ready = prepare_ordered_write_target(ex, clear->target, NULL, NULL, &result);
    if (ready == VK_SYNC_PROGRESS_COMPLETE) return 1;
    vk_texture_write_target_cancel(clear->target);
    clear->target = NULL;
    if (ready == VK_SYNC_PROGRESS_UNAVAILABLE)
        rt_fatal("vk: required clear of texture serial %llu cannot acquire private image capacity (%d)",
                 (unsigned long long)op->tex->serial, (int)result);
    return -1;
}

static int bound_relocation_input_changed(void *opaque) {
    return !vk_texture_bound_relocation_current(opaque);
}

// Complete the ownership-only copy before any selected write is declared or recorded.
static VkExecRestoreResult prepare_bound_relocation(VkExec *ex, VkTextureBoundRelocation *plan) {
    uint64_t observed_prior = 0;
    for (;;) {
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_RESTORE_STOPPED;
        if (ex->tonemap_extent.width || ex->bc_written_mask || vk_contact_pending()) return VK_EXEC_RESTORE_FAILED;
        if (!rc_registry_provisional_read_current(rc_registry_runtime(), vk_exec_resource_position()) ||
            !vk_texture_bound_relocation_current(plan)) return VK_EXEC_RESTORE_STALE;
        RcVmaDependency dependency = {0};
        VkResult allocated = vk_texture_bound_relocation_allocate(plan, &dependency);
        if (allocated == VK_ERROR_DEVICE_LOST) vk_device_lost_check(allocated, "preserving an earlier bound image");
        if (allocated != VK_SUCCESS) {
            VkSyncProgressResult capacity = vk_exec_texture_capacity(ex, &dependency, allocated,
                bound_relocation_input_changed, plan, &observed_prior);
            if (capacity == VK_SYNC_PROGRESS_COMPLETE) continue;
            if (capacity == VK_SYNC_PROGRESS_STOPPED || capacity == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT ||
                vk_sync_stopping() || vk_device_is_lost()) return VK_EXEC_RESTORE_STOPPED;
            return vk_texture_bound_relocation_current(plan) ? VK_EXEC_RESTORE_FAILED : VK_EXEC_RESTORE_STALE;
        }
        vk_exec_barrier(ex);
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_RESTORE_STOPPED;
        if (!vk_texture_bound_relocation_current(plan)) return VK_EXEC_RESTORE_STALE;
        vk_texture_bound_relocation_record(plan, ex->cb, ex->fence);
        ex->recorded++;
        VkSyncProgressResult completed = vk_exec_graphics_pool_progress(ex, 0);
        if (completed == VK_SYNC_PROGRESS_COMPLETE) return VK_EXEC_RESTORE_READY;
        if (completed == VK_SYNC_PROGRESS_STOPPED || completed == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT)
            return VK_EXEC_RESTORE_STOPPED;
        return VK_EXEC_RESTORE_FAILED;
    }
}

static int prepare_bound_texture_relocation(VkExec *ex, const tex_t *destination, int ordered) {
    VkTextureBoundRelocation *plan = NULL;
    int capture = vk_texture_bound_relocation_capture(destination, &plan);
    if (capture <= 0) return capture;
    int preserved = !ordered || (prepare_bound_relocation(ex, plan) == VK_EXEC_RESTORE_READY &&
                                 vk_texture_bound_relocation_commit(plan));
    vk_texture_bound_relocation_cancel(plan);
    if (preserved && ordered) ex->rt_resolved = 0;
    return preserved ? !ordered : -1;
}

VkExecRestoreResult vk_exec_prepare_storage_target_bound(VkExec *ex, VkTextureStorageTarget *target,
    VkTexImage *selected) {
    VkTextureBoundRelocation *plan = vk_texture_storage_target_bound_plan(target);
    if (!plan) return VK_EXEC_RESTORE_STALE;
    const VkExecRestoreResult result = prepare_bound_relocation(ex, plan);
    if (result != VK_EXEC_RESTORE_READY) return result;
    if (!vk_texture_storage_target_bound_commit(target, selected)) return VK_EXEC_RESTORE_STALE;
    ex->rt_resolved = 0;
    return VK_EXEC_RESTORE_READY;
}

static int prepare_ordered_buffer_to_texture(VkExec *ex, const p_copy_b2t *op, VkExecBufferToTexture *copy) {
    if (vk_sync_stopping() || vk_device_is_lost()) return -1;
    image_write_input_position(ex);
    if (vk_exec_buffer_to_texture_alias_candidate(ex, op) &&
        prepare_bound_texture_relocation(ex, op->tex, 1) < 0) {
        ex->resource_op = 0;
        if (!vk_sync_stopping() && !vk_device_is_lost()) {
            rt_log("vk: required earlier bound image cannot preserve current bytes before CopyBufferToTexture");
            __atomic_store_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_RELEASE);
        }
        return -1;
    }
    int handled = vk_exec_buffer_to_texture_capture(ex, op, copy);
    const uint64_t input_op = ex->resource_op;
    ex->resource_op = 0;
    if (handled >= 0 && !copy->target) return handled;
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    VkSyncProgressResult ready = handled > 0
        ? prepare_ordered_write_target(ex, copy->target, copy, NULL, &result) : VK_SYNC_PROGRESS_UNAVAILABLE;
    ex->resource_op = input_op;
    RcRegistryResult source = ready == VK_SYNC_PROGRESS_COMPLETE
        ? vk_exec_buffer_to_texture_prepare_source(ex, copy) : RC_REGISTRY_INVALID;
    ex->resource_op = 0;
    if (source == RC_REGISTRY_OK && !vk_sync_stopping() && !vk_device_is_lost() && !vk_exec_recording_failed())
        return 1;
    vk_exec_buffer_to_texture_cancel(copy);
    if (ready != VK_SYNC_PROGRESS_STOPPED && !vk_sync_stopping() && !vk_device_is_lost() && !vk_exec_recording_failed()) {
        rt_log("vk: required CopyBufferToTexture lost its retained input or private image capacity "
               "(serial %llu, result %d, source %d)", (unsigned long long)op->tex->serial, (int)result, source);
        vk_exec_required_graphics_unavailable(ex, OP_COPY_B2T, "ordered buffer-to-texture input unavailable");
    }
    return -1;
}

static int prepare_compat_buffer_to_texture(VkExec *ex, const p_copy_b2t *op, VkExecBufferToTexture *copy) {
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return 0;
    image_write_input_position(ex);
    const int captured = vk_exec_buffer_to_texture_capture_compat(ex, op, copy);
    const RcRegistryResult source = captured > 0 && copy->source_hold ?
        vk_exec_buffer_to_texture_prepare_source(ex, copy) : captured > 0 ? RC_REGISTRY_OK : RC_REGISTRY_INVALID;
    ex->resource_op = 0;
    if (source == RC_REGISTRY_OK && !vk_sync_stopping() && !vk_device_is_lost() && !vk_exec_recording_failed())
        return 1;
    vk_exec_buffer_to_texture_cancel(copy);
    if (!vk_sync_stopping() && !vk_device_is_lost() && !vk_exec_recording_failed()) {
        rt_log("vk: required compatible buffer-to-texture source refused registry=%d", source);
        vk_exec_required_graphics_unavailable(ex, OP_COPY_B2T, "compatible buffer-to-texture source unavailable");
    }
    return 0;
}

static int projected_transfer_changed(void *opaque) {
    return !vk_exec_projected_transfer_current(opaque);
}

static int prepare_ordered_projected_transfer(VkExec *ex, uint32_t opcode, const void *payload,
                                              VkExecProjectedTransfer *transfer) {
    image_write_input_position(ex);
    int result = vk_exec_projected_transfer_capture(ex, opcode, payload, transfer);
    if (opcode != OP_COPY_B2T && !transfer->image.input) ex->resource_op = 0;
    if (result > 0 && transfer->target) {
        int ready = 0;
        if (opcode == OP_COPY_B2T) {
            ready = vk_exec_buffer_to_texture_prepare_source(ex, &transfer->buffer) == RC_REGISTRY_OK &&
                    vk_exec_projected_transfer_current(transfer);
        } else if (!transfer->image.scratch_recipe.sType) {
            ready = vk_exec_projected_transfer_current(transfer);
        } else {
            uint64_t observed = 0;
            while (!vk_exec_recording_failed() && !vk_sync_stopping() && !vk_device_is_lost() &&
                   vk_exec_projected_transfer_current(transfer)) {
                RcVmaDependency dependency = {0};
                VkResult allocated = vk_exec_texture_to_texture_allocate(&transfer->image, &dependency);
                if (allocated == VK_SUCCESS) {
                    ready = vk_exec_projected_transfer_current(transfer);
                    break;
                }
                if (allocated == VK_ERROR_DEVICE_LOST) {
                    vk_exec_projected_transfer_cancel(transfer);
                    vk_device_lost_check(allocated, "allocating projected copy scratch");
                    return -1;
                }
                if (vk_exec_texture_capacity(ex, &dependency, allocated, projected_transfer_changed, transfer,
                                             &observed) != VK_SYNC_PROGRESS_COMPLETE) break;
            }
        }
        if (!ready) result = -1;
    }
    ex->resource_op = 0;
    if (vk_sync_stopping() || vk_device_is_lost()) cancel_recording();
    if (result < 0 || vk_exec_recording_failed()) {
        vk_exec_projected_transfer_cancel(transfer);
        if (!vk_exec_recording_failed())
            vk_exec_required_graphics_unavailable(ex, opcode, "projected image transfer input or capacity unavailable");
        return -1;
    }
    return result;
}

static int texture_transfer_target_current(void *opaque) {
    return vk_texture_write_target_current(opaque);
}

static int prepare_texture_to_texture_source(VkExec *, const p_copy_t2t *, VkTextureTransferInput *);

static int prepare_ordered_texture_to_texture(VkExec *ex, const p_copy_t2t *op, VkExecTextureToTexture *copy,
                                               VkTextureTransferInput *input) {
    if (vk_sync_stopping() || vk_device_is_lost()) return -1;
    image_write_input_position(ex);
    if (vk_exec_texture_to_texture_alias_candidate(op) &&
        prepare_bound_texture_relocation(ex, op->dst, 1) < 0) {
        ex->resource_op = 0;
        if (!vk_sync_stopping() && !vk_device_is_lost()) {
            rt_log("vk: required earlier bound image cannot preserve current bytes before CopyTextureToTexture");
            __atomic_store_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_RELEASE);
        }
        return -1;
    }
    const int capture = prepare_texture_to_texture_source(ex, op, input);
    if (capture < 0) { ex->resource_op = 0; return -1; }
    if (capture) copy->input = input;
    int handled = vk_exec_texture_to_texture_capture(ex, op, copy);
    ex->resource_op = 0;
    if (handled >= 0 && !copy->target) return handled;
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    if (handled > 0 && copy->restore) {
        const VkTexImage *restored;
        VkExecRestoreResult ready = vk_exec_prepare_texture_restore(ex, copy->restore,
            texture_transfer_target_current, copy->target, &restored, &result);
        if (ready == VK_EXEC_RESTORE_READY) copy->source.image = restored->image;
        else handled = -1;
    }
    if (handled > 0 && prepare_ordered_write_target(ex, copy->target, NULL, copy, &result) == VK_SYNC_PROGRESS_COMPLETE)
        return 1;
    vk_exec_texture_to_texture_cancel(copy);
    if (!vk_sync_stopping() && !vk_device_is_lost() && !vk_exec_recording_failed()) {
        rt_log("vk: required CopyTextureToTexture lost its exact source, destination or image capacity (result %d)",
               (int)result);
        __atomic_store_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_RELEASE);
    }
    return -1;
}

typedef struct {
    VkExecDispatch *dispatch;
    VkComputePipelinePending pending;
    VkComputePipelineResult result;
} OrderedDispatchWork;

// The oldest recorder still owns the op and its observation. The provider owns the exact compiler key and
// its single admission. No renderer command, writer declaration or guest fence is issued while parked.
static int prepare_dispatch_parked(void *arg) {
    OrderedDispatchWork *work = arg;
    VkExecDispatch *d = work->dispatch;
    do {
        work->result = vk_pipeline_compute_prepare(d->shader, 1, &d->pipeline, &d->info, &work->pending);
        if (work->result != VK_COMPUTE_PENDING) return work->result != VK_COMPUTE_STOPPED;
    } while (vk_pipeline_compute_wait(&work->pending));
    work->result = VK_COMPUTE_STOPPED;
    return 0;
}

static int restore_progress_ready(const RcVmaDependency *dependency, VkTextureRestore *restore) {
    if (vk_texture_restore_spare_ready(restore)) return 1;
    RcBudgetReadiness now;
    resource_core_budget_readiness(1, &now);
    return resource_core_vma_dependency_ready(dependency, &now);
}

typedef struct {
    VkTextureRestore *restore;
    int (*current)(void *);
    void *context;
} TextureRestoreWait;

static int restore_capacity_supplier(void *opaque) {
    TextureRestoreWait *wait = opaque;
    return !wait->current(wait->context) || vk_texture_restore_spare_ready(wait->restore);
}

VkSyncProgressResult vk_exec_texture_capacity(VkExec *ex, const RcVmaDependency *dependency, VkResult failure,
                                                int (*spare_ready)(void *), void *context, uint64_t *observed_prior) {
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    if (!dependency->selected || (failure != VK_NOT_READY && failure != VK_ERROR_OUT_OF_DEVICE_MEMORY &&
        failure != VK_ERROR_OUT_OF_HOST_MEMORY)) return VK_SYNC_PROGRESS_UNAVAILABLE;
    uint64_t prior = vk_mem_progress_prior(dependency, ex->fence);
    if (prior > *observed_prior) {
        *observed_prior = prior;
        if (prior > vk_sync_fence_done()) {
            VkSyncProgressResult progress = vk_sync_progress_wait_prior(prior);
            if (progress == VK_SYNC_PROGRESS_STOPPED || vk_sync_stopping() || vk_device_is_lost())
                return VK_SYNC_PROGRESS_STOPPED;
        }
    }
    // Completion can release the exact allocator's capacity between querying a useful prior and collecting.
    vk_mem_collect();
    RcBudgetReadiness now;
    resource_core_budget_readiness(1, &now);
    if ((spare_ready && spare_ready(context)) || resource_core_vma_dependency_ready(dependency, &now))
        return VK_SYNC_PROGRESS_COMPLETE;
    if (failure == VK_NOT_READY && (dependency->owner_full ||
        !resource_core_budget_peak_ready(&dependency->peak, &dependency->before)))
        return vk_sync_progress_wait_capacity(dependency, spare_ready, context);
    return VK_SYNC_PROGRESS_UNAVAILABLE;
}

VkExecRestoreResult vk_exec_prepare_texture_restore(VkExec *ex, VkTextureRestore *restore,
    int (*current)(void *), void *context, const VkTexImage **image, VkResult *failure) {
    *image = NULL;
    *failure = VK_SUCCESS;
    TextureRestoreWait wait = {restore, current, context};
    uint64_t observed_upload = 0, observed_prior = 0;
    for (;;) {
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_RESTORE_STOPPED;
        if (!current(context)) return VK_EXEC_RESTORE_STALE;
        RcVmaDependency dependency;
        VkResult result = vk_texture_restore_prepare(restore, &dependency);
        if (result == VK_ERROR_DEVICE_LOST) vk_device_lost_check(result, "preparing an exact texture restore");
        if (vk_sync_stopping() || vk_device_is_lost()) return VK_EXEC_RESTORE_STOPPED;
        if (!current(context)) return VK_EXEC_RESTORE_STALE;
        if (result == VK_SUCCESS) {
            *image = vk_texture_restore_publish(restore);
            if (vk_sync_stopping() || vk_device_is_lost()) return VK_EXEC_RESTORE_STOPPED;
            if (*image) return VK_EXEC_RESTORE_READY;
            result = VK_ERROR_INITIALIZATION_FAILED;
        }
        int pressure = result == VK_NOT_READY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
                       result == VK_ERROR_OUT_OF_HOST_MEMORY;
        if (pressure && dependency.selected) {
            if (restore_progress_ready(&dependency, restore)) continue;
            uint64_t upload = vk_texture_restore_upload_supplier(restore, observed_upload);
            if (upload) {
                observed_upload = upload;
                VkSyncProgressResult progress = upload_staging_progress(ex, upload);
                if (progress == VK_SYNC_PROGRESS_STOPPED) return VK_EXEC_RESTORE_STOPPED;
                if (!current(context)) return VK_EXEC_RESTORE_STALE;
                if (progress == VK_SYNC_PROGRESS_COMPLETE && restore_progress_ready(&dependency, restore)) continue;
            }
            VkSyncProgressResult progress = vk_exec_texture_capacity(ex, &dependency, result,
                restore_capacity_supplier, &wait, &observed_prior);
            if (progress == VK_SYNC_PROGRESS_COMPLETE) continue;
            if (progress == VK_SYNC_PROGRESS_STOPPED) return VK_EXEC_RESTORE_STOPPED;
        }
        *failure = result;
        return VK_EXEC_RESTORE_FAILED;
    }
}

typedef struct {
    const p_copy_b2t *operation;
    const VkExecRestoreCopy *copy;
} RestoreCopySource;

static int restore_copy_current(void *opaque) {
    const RestoreCopySource *source = opaque;
    return vk_exec_restore_copy_current(source->operation, source->copy);
}

static int alias_readback_position_current(const VkExec *ex, RcRegistryPosition read) {
    return !ex->tonemap_extent.width && !ex->bc_written_mask && !vk_contact_pending() &&
        !rc_registry_position_compare(read, vk_exec_resource_position()) &&
        rc_registry_provisional_read_current(rc_registry_runtime(), read);
}

static int alias_readback_source_current(VkExec *ex, RcRegistryPosition read, int (*current)(void *), void *scope) {
    return alias_readback_position_current(ex, read) && current(scope);
}

static VkExecRestoreResult alias_readback_raw_progress(VkExec *ex, RcRegistryPosition read,
    VkTextureAliasRead *const *aliases, uint32_t count, uint64_t *observed) {
    for (uint32_t i = 0; i < count; i++) {
        uint64_t upload;
        if (!vk_texture_alias_read_raw_progress(aliases[i], read, &upload)) return VK_EXEC_RESTORE_STALE;
        if (upload > *observed) *observed = upload;
    }
    if (!*observed) return VK_EXEC_RESTORE_READY;
    if (!alias_readback_position_current(ex, read)) return VK_EXEC_RESTORE_STALE;
    uint64_t submitted, completed;
    vk_upload_progress(&submitted, &completed);
    if (*observed > submitted) return VK_EXEC_RESTORE_STALE;
    if (*observed > completed) {
        VkSyncProgressResult progress = upload_staging_progress(ex, *observed);
        if (progress == VK_SYNC_PROGRESS_STOPPED || progress == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT)
            return VK_EXEC_RESTORE_STOPPED;
        if (progress != VK_SYNC_PROGRESS_COMPLETE) return VK_EXEC_RESTORE_FAILED;
        vk_upload_progress(&submitted, &completed);
    }
    if (*observed > completed || !alias_readback_position_current(ex, read)) return VK_EXEC_RESTORE_STALE;
    for (uint32_t i = 0; i < count; i++)
        if (!vk_texture_alias_read_raw_finish(aliases[i], read)) return VK_EXEC_RESTORE_STALE;
    return VK_EXEC_RESTORE_READY;
}

VkExecRestoreResult vk_exec_prepare_alias_readback(VkExec *ex, VkTextureAliasRead *const *aliases, uint32_t count,
    int (*current)(void *), void *scope) {
    // Closing an active attachment is legal before these reads only if no present or block-copy hook
    // could publish new source bytes between the proof and the actual readback.
    if (!count) return VK_EXEC_RESTORE_FAILED;
    const RcRegistryPosition read = vk_exec_resource_position();
    uint64_t observed_upload = 0, observed_scene = 0;
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_RESTORE_STOPPED;
    VkExecRestoreResult raw = alias_readback_raw_progress(ex, read, aliases, count, &observed_upload);
    if (raw != VK_EXEC_RESTORE_READY) return raw;
    for (;;) {
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_RESTORE_STOPPED;
        if (!alias_readback_source_current(ex, read, current, scope)) return VK_EXEC_RESTORE_STALE;
        VkTextureAliasStagingProgress upload;
        RcVmaDependency native;
        VkResult allocated = vk_texture_alias_read_prepare(aliases, count, 1, observed_upload, &upload, &native);
        if (allocated == VK_SUCCESS) break;
        if (allocated == VK_ERROR_DEVICE_LOST) vk_device_lost_check(allocated, "allocating an exact image-alias read");
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_RESTORE_STOPPED;
        if (!alias_readback_source_current(ex, read, current, scope)) return VK_EXEC_RESTORE_STALE;
        if (upload.upload > observed_upload) {
            observed_upload = upload.upload;
            VkSyncProgressResult progress = upload_staging_progress(ex, observed_upload);
            if (progress == VK_SYNC_PROGRESS_STOPPED) return VK_EXEC_RESTORE_STOPPED;
            if (!alias_readback_source_current(ex, read, current, scope)) return VK_EXEC_RESTORE_STALE;
            if (progress == VK_SYNC_PROGRESS_COMPLETE && vk_staging_prepare_available(upload.size)) {
                resource_core_budget_refresh();
                continue;
            }
        }
        if (native.selected) {
            uint64_t prior = vk_mem_progress_prior(&native, ex->fence);
            if (prior > observed_scene) {
                observed_scene = prior;
                VkSyncProgressResult progress = vk_exec_graphics_pool_progress(ex, prior);
                if (progress == VK_SYNC_PROGRESS_STOPPED) return VK_EXEC_RESTORE_STOPPED;
                if (!alias_readback_source_current(ex, read, current, scope)) return VK_EXEC_RESTORE_STALE;
                if (progress == VK_SYNC_PROGRESS_COMPLETE) {
                    vk_mem_collect();
                    resource_core_budget_refresh();
                    RcBudgetReadiness now;
                    resource_core_budget_readiness(1, &now);
                    if (resource_core_vma_dependency_ready(&native, &now)) continue;
                }
            }
        }
        return VK_EXEC_RESTORE_FAILED;
    }

    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed() ||
        !alias_readback_source_current(ex, read, current, scope)) {
        vk_texture_alias_read_unprepare(aliases, count);
        return vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()
            ? VK_EXEC_RESTORE_STOPPED : VK_EXEC_RESTORE_STALE;
    }
    vk_exec_barrier(ex);
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed() ||
        !alias_readback_source_current(ex, read, current, scope)) {
        vk_texture_alias_read_unprepare(aliases, count);
        return vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()
            ? VK_EXEC_RESTORE_STOPPED : VK_EXEC_RESTORE_STALE;
    }
    for (uint32_t i = 0; i < count; ++i) vk_texture_alias_read_record(aliases[i], ex->cb, ex->fence);
    ex->recorded++;
    VkSyncProgressResult progress = vk_exec_graphics_pool_progress(ex, 0);
    if (progress == VK_SYNC_PROGRESS_STOPPED || progress == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT)
        return VK_EXEC_RESTORE_STOPPED;
    if (progress != VK_SYNC_PROGRESS_COMPLETE || !alias_readback_source_current(ex, read, current, scope))
        return VK_EXEC_RESTORE_STALE;
    return VK_EXEC_RESTORE_READY;
}

VkExecRestoreResult vk_exec_prepare_alias_texture_read(VkExec *ex, VkTextureAliasRead *const *aliases, uint32_t count,
    int (*current)(void *), void *scope, const VkTexImage **image) {
    VkExecRestoreResult result = vk_exec_prepare_alias_readback(ex, aliases, count, current, scope);
    if (result != VK_EXEC_RESTORE_READY) return result;
    for (uint32_t i = 0; i < count; ++i) {
        image[i] = vk_texture_alias_read_upload(aliases[i], ex->cb, ex->fence);
        if (!image[i]) return VK_EXEC_RESTORE_STALE;
        ex->recorded++;
    }
    return VK_EXEC_RESTORE_READY;
}

static int prepare_ordered_texture_copy(VkExec *ex, const p_copy_b2t *op, VkExecRestoreCopy *copy) {
    if (vk_sync_stopping() || vk_device_is_lost()) return -1;
    if (ex->resource_next >= (UINT64_MAX - 1) / 2) rt_fatal("vk: texture copy input identity exhausted");
    ex->resource_op = ex->resource_next + 1;
    if (rc_registry_declare_through(rc_registry_runtime(), vk_exec_resource_position()) != RC_REGISTRY_OK)
        rt_fatal("vk: texture copy input prefix could not be sealed");
    VkTextureAliasFailure refused = {0};
    int captured = vk_exec_restore_copy_capture(ex, op, copy, &refused);
    VkTextureAliasRead *alias = copy->alias ? copy->alias : copy->input.alias;
    if (captured < 0 || (captured > 0 && alias)) {
        RestoreCopySource source = {op, copy};
        VkExecRestoreResult result = captured < 0 ? VK_EXEC_RESTORE_STALE :
            vk_exec_prepare_alias_texture_read(ex, &alias, 1, restore_copy_current, &source, &copy->image);
        ex->resource_op = 0;
        if (result == VK_EXEC_RESTORE_READY) return 1;
        if (captured > 0 && result != VK_EXEC_RESTORE_STOPPED) {
            uint32_t count;
            const RcInterval *intervals = vk_texture_alias_read_intervals(alias, &count);
            const int stale = result == VK_EXEC_RESTORE_STALE;
            refused = (VkTextureAliasFailure){.phase = stale ? VK_TEXTURE_ALIAS_MAPPING : VK_TEXTURE_ALIAS_CAPACITY,
                .status = stale ? RC_REGISTRY_STALE : RC_REGISTRY_CAPACITY, .receiver = count ? intervals[0] : (RcInterval){0}};
        }
        vk_exec_restore_copy_cancel(copy);
        if (result != VK_EXEC_RESTORE_STOPPED) {
            rt_log("vk: required current alias bytes unavailable for CopyTextureToBuffer "
                   "(submission %llu operation %llu, status %d phase %d registry %d mapping %llu backing %llu "
                   "offset 0x%llx size 0x%llx)", (unsigned long long)ex->fence,
                   (unsigned long long)(ex->resource_next + 1), result, refused.phase, refused.status,
                   (unsigned long long)refused.receiver.mapping, (unsigned long long)refused.receiver.range.backing,
                   (unsigned long long)refused.receiver.range.offset, (unsigned long long)refused.receiver.range.size);
            __atomic_store_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_RELEASE);
        }
        return -1;
    }
    ex->resource_op = 0;
    if (!captured) return 0;
    if (copy->input.texture) return 1;

    RestoreCopySource source = {op, copy};
    VkResult failure;
    VkExecRestoreResult result = vk_exec_prepare_texture_restore(ex, copy->texture, restore_copy_current, &source,
                                                               &copy->image, &failure);
    if (result == VK_EXEC_RESTORE_READY) return 1;
    vk_exec_restore_copy_cancel(copy);
    if (result == VK_EXEC_RESTORE_STALE) {
        rt_log("vk: required restored CopyTextureToBuffer lost its destination mapping");
        __atomic_store_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_RELEASE);
    }
    if (result == VK_EXEC_RESTORE_FAILED)
        rt_fatal("vk: required CopyTextureToBuffer restore cannot prepare texture serial %llu (%d)",
                 (unsigned long long)op->tex->serial, (int)failure);
    return -1;
}

static int texture_transfer_input_current(void *opaque) {
    return vk_texture_transfer_input_current(opaque);
}

static int prepare_texture_to_texture_source(VkExec *ex, const p_copy_t2t *op, VkTextureTransferInput *input) {
    VkTextureAliasFailure failure = {0};
    const int captured = vk_exec_transfer_image_input_capture(OP_COPY_T2T, op, input, &failure);
    VkExecRestoreResult result = captured < 0 ? VK_EXEC_RESTORE_STALE : VK_EXEC_RESTORE_READY;
    if (captured > 0 && input->alias) {
        const VkTexImage *image;
        result = vk_exec_prepare_alias_texture_read(ex, &input->alias, 1, texture_transfer_input_current, input, &image);
        if (result == VK_EXEC_RESTORE_READY) input->image = *image;
    }
    if (result == VK_EXEC_RESTORE_READY) return captured;
    if (result != VK_EXEC_RESTORE_STOPPED && !vk_exec_recording_failed()) {
        rt_log("vk: required CopyTextureToTexture source refused status=%d phase=%d registry=%d", result,
               failure.phase, failure.status);
        vk_exec_required_graphics_unavailable(ex, OP_COPY_T2T, "original image transfer source unavailable");
    }
    return -1;
}

static VkExecGpaSampleScope *prepare_dispatch_uniform_gpa(VkExec *ex, const VkExecDispatch *dispatch,
                                                            RcRegistryPosition read, RcRegistryPosition attempted) {
    if (!rc_registry_position_compare(read, attempted) ||
        (!vk_exec_shared() && !vk_gpa_write_scene_may_need_prefix(read.scene))) return NULL;
    VkGpaPrefixRequest request;
    if (!vk_exec_dispatch_uniform_gpa_blocker(ex, dispatch, &request)) return NULL;
    if (Q.drained != Q.out) {
        group_drain(ex, PW_OP);
        if (vk_exec_recording_failed() || !vk_exec_dispatch_uniform_gpa_blocker(ex, dispatch, &request)) return NULL;
    }
    VkExecGpaSampleScope *scope = vk_exec_gpa_sample_enter(ex, attempted);
    if (!scope) return NULL;
    scope->prepare.uniform = 1;
    if (!vk_exec_request_gpa_prefix(ex, request.guard, request.producer, request.read, request.source) ||
        !vk_exec_gpa_sample_leave(scope)) {
        vk_exec_gpa_sample_cancel(scope);
        vk_gpa_report_write_blocker(request.guard, request.source.range);
        return NULL;
    }
    return scope;
}

static VkExecDispatchResult prepare_dispatch_inputs(VkExec *ex, const op_hdr *h, VkExecDispatch *dispatch,
                                                    RcRegistryPosition input, RcRegistryPosition *attempted,
                                                    int compatibility) {
    const uint64_t operation = ex->resource_next + 1;
    int replanned = 0;
    if ((!compatibility && dispatch->info->pool) || vk_exec_dispatch_compat_storage_binding(dispatch))
        group_drain(ex, PW_OP);
    for (;;) {
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed())
            return VK_EXEC_DISPATCH_STOPPED;
        ex->resource_op = operation;
        if (rc_registry_position_compare(vk_exec_resource_position(), input))
            rt_fatal("vk: dispatch input changed its read position");
        dispatch->input_failure = (VkExecDispatchInputFailure){.binding = -1, .slot = -1, .member = -1};
        VkExecGpaSampleScope *pending = prepare_dispatch_uniform_gpa(ex, dispatch, input, *attempted);
        if (pending) {
            dispatch->input_failure.stage = "uniform-prefix";
            dispatch->input_failure.result = VK_EXEC_GRAPHICS_NEEDS_ORDERED;
        }
        VkExecGraphicsResult result = pending ? VK_EXEC_GRAPHICS_NEEDS_ORDERED :
            compatibility ? vk_exec_dispatch_prepare_compat_inputs(ex, dispatch, *attempted, &pending) :
                            vk_exec_dispatch_prepare_native_inputs(ex, dispatch, *attempted, &pending);
        ex->resource_op = 0;
        if (ex->resource_next + 1 != operation) rt_fatal("vk: preparing dispatch bindings changed replay order");
        if (pending && result == VK_EXEC_GRAPHICS_NEEDS_ORDERED &&
            rc_registry_position_compare(input, *attempted)) {
            VkSyncProgressResult progress = vk_exec_gpa_sample_settle(pending, attempted);
            if (progress == VK_SYNC_PROGRESS_COMPLETE && !rc_registry_position_compare(*attempted, input)) continue;
            if (progress == VK_SYNC_PROGRESS_STOPPED || progress == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT)
                return VK_EXEC_DISPATCH_STOPPED;
            pending = NULL;
        }
        if (pending) {
            vk_exec_gpa_sample_cancel(pending);
            result = VK_EXEC_GRAPHICS_UNAVAILABLE;
        }
        if (result == VK_EXEC_GRAPHICS_READY) return compatibility ? VK_EXEC_DISPATCH_COMPAT : VK_EXEC_DISPATCH_NATIVE;
        if (result == VK_EXEC_GRAPHICS_STOPPED) return VK_EXEC_DISPATCH_STOPPED;
        if (result == VK_EXEC_GRAPHICS_INVALIDATED && !replanned) {
            replanned = 1;
            continue;
        }
        const VkExecDispatchInputFailure *failure = &dispatch->input_failure;
        selected_sampled_failure_log(&failure->sampled, h->op, compatibility ? "compute-compat" : "compute-native");
        rt_log("vk: required dispatch input read=%llu/%llu program=%llu op=%u path=%s stage=%s status=%d stage-result=%d "
               "reason=%s subtype=%d binding=%d slot=%d member=%d handle=%#llx image=%#llx peer=%#llx addr=%#llx bytes=%#llx "
               "selection=%d pool=%d scratch=%d descriptor-vk=%d descriptor-set=%u retried=%d",
               (unsigned long long)input.scene, (unsigned long long)input.ordinal,
               (unsigned long long)dispatch->program_serial, h->op, compatibility ? "compat" : "native",
               failure->stage ? failure->stage : "unrecorded", result, failure->result,
               failure->reason ? failure->reason : "unrecorded", failure->subtype,
               failure->binding, failure->slot, failure->member, (unsigned long long)failure->handle,
               (unsigned long long)failure->image, (unsigned long long)failure->peer_image,
               (unsigned long long)failure->address, (unsigned long long)failure->bytes,
               dispatch->selection != NULL, dispatch->pool_grant != NULL, dispatch->scratch != NULL,
               dispatch->descriptors.refusal, dispatch->descriptors.refused_set, replanned);
        vk_exec_dispatch_input_trace_log(dispatch, failure->stage, result);
        __atomic_store_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_RELEASE);
        return VK_EXEC_DISPATCH_STOPPED;
    }
}

typedef struct {
    VkDrawScratchWaitContext spare;
    const VkExecBufferScratchWait *capacity;
    int ready;
} DispatchScratchWait;

static int dispatch_scratch_capacity_ready(DispatchScratchWait *wait) {
    const VkExecBufferScratchWait *capacity = wait->capacity;
    RcBudgetReadiness now;
    resource_core_budget_readiness(1, &now);
    if (capacity->host_only)
        return resource_core_budget_peak_progress(&capacity->host_peak, &capacity->host_before, &now);
    return resource_core_vma_dependency_ready(&capacity->native, &now) ||
        (resource_core_budget_peak_ready(&capacity->host_peak, &now) &&
         vk_exec_draw_graphics_scratch_spare_ready(&wait->spare));
}

static int dispatch_scratch_wait_parked(void *opaque) {
    DispatchScratchWait *wait = opaque;
    const VkExecBufferScratchWait *capacity = wait->capacity;
    vk_mem_collect();
    if (!dispatch_scratch_capacity_ready(wait) && !vk_sync_stopping() && !vk_device_is_lost()) {
        RcBudgetProgress progress;
        const RcAdmissionVector *peak = capacity->host_only ? &capacity->host_peak : &capacity->native.peak;
        resource_core_budget_service(1, peak, &progress);
        if (!dispatch_scratch_capacity_ready(wait) && !vk_sync_stopping() && !vk_device_is_lost())
            resource_core_budget_wait_since(progress.event, 100);
        vk_mem_collect();
    }
    wait->ready = dispatch_scratch_capacity_ready(wait);
    return 1;
}

static VkExecDispatchResult prepare_dispatch_scratch(VkExec *ex, VkExecDispatch *dispatch, RcRegistryPosition read) {
    for (unsigned attempt = 0; ; ++attempt) {
        if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_EXEC_DISPATCH_STOPPED;
        uint64_t next = ex->resource_next;
        ex->resource_op = next + 1;
        if (rc_registry_position_compare(vk_exec_resource_position(), read))
            rt_fatal("vk: compute scratch changed its read position");
        VkExecBufferScratchWait capacity;
        VkExecBufferScratchResult result = vk_exec_draw_buffer_scratch_prepare(ex, dispatch, &capacity);
        ex->resource_op = 0;
        if (ex->resource_next != next) rt_fatal("vk: preparing compute scratch changed replay order");
        if (result == VK_EXEC_SCRATCH_PASS || result == VK_EXEC_SCRATCH_READY) return VK_EXEC_DISPATCH_NATIVE;
        if (result == VK_EXEC_SCRATCH_STOPPED) return VK_EXEC_DISPATCH_STOPPED;
        if (!attempt && result == VK_EXEC_SCRATCH_NEED_CAPACITY && (capacity.host_only || capacity.native.selected)) {
            DispatchScratchWait wait = {.spare = {.ex = ex, .bytes = capacity.bytes}, .capacity = &capacity};
            VkSyncProgressResult progress = vk_exec_graphics_progress_prepare(ex, dispatch_scratch_wait_parked, &wait);
            if (progress == VK_SYNC_PROGRESS_COMPLETE && wait.ready) continue;
            if (progress == VK_SYNC_PROGRESS_STOPPED || vk_sync_stopping() || vk_device_is_lost())
                return VK_EXEC_DISPATCH_STOPPED;
        }
        const char *reason = result == VK_EXEC_SCRATCH_INVALIDATED ? "prepared compute scratch inputs changed" :
            result == VK_EXEC_SCRATCH_NEED_CAPACITY ? "required compute scratch capacity made no progress" :
            "required compute scratch has no usable placement";
        vk_exec_required_graphics_unavailable(ex, dispatch->opcode, reason);
        return VK_EXEC_DISPATCH_STOPPED;
    }
}

static VkExecDispatchResult prepare_ordered_dispatch(VkExec *ex, const op_hdr *h, VkExecDispatch *dispatch,
    RcRegistryPosition *attempted, const VkMemSpan *retry_span, VkMemSpanCapture *retry_capture) {
    if (vk_sync_stopping() || vk_device_is_lost()) return -1;
    if (ex->resource_next >= (UINT64_MAX - 1) / 2) rt_fatal("vk: dispatch input identity exhausted");

    // Name the exact current input position. Resource assignment and all output declaration remain pending.
    ex->resource_op = ex->resource_next + 1;
    RcRegistryPosition input = vk_exec_resource_position();
    if (rc_registry_declare_through(rc_registry_runtime(), input) != RC_REGISTRY_OK)
        rt_fatal("vk: dispatch input prefix could not be sealed");
    VkExecDispatchResult selected = vk_exec_dispatch_select(ex, h->op, nvn_op_payload(h), retry_span, retry_capture, dispatch);
    const int compatibility = selected == VK_EXEC_DISPATCH_COMPAT;
    const int prepare_compat = compatibility &&
        vk_exec_dispatch_compat_inputs_select(ex, h->op, nvn_op_payload(h), dispatch);
    ex->resource_op = 0;
    if (selected != VK_EXEC_DISPATCH_NATIVE && !prepare_compat) return selected;

    OrderedDispatchWork work = {.dispatch = dispatch};
    work.result = vk_pipeline_compute_prepare(dispatch->shader, 0, &dispatch->pipeline, &dispatch->info, &work.pending);
    if (work.result == VK_COMPUTE_PENDING) {
        group_drain(ex, PW_OP);
        if (vk_exec_recording_failed()) return -1;
        VkSyncProgressResult progress = vk_sync_progress_prepare_pipeline(prepare_dispatch_parked, &work);
        if (progress == VK_SYNC_PROGRESS_STOPPED) return -1;
        if (progress != VK_SYNC_PROGRESS_COMPLETE) {
            vk_exec_dispatch_cancel(dispatch);
            rt_fatal("vk: required dispatch has no active recorder for pipeline preparation");
        }
    }
    if (vk_sync_stopping() || vk_device_is_lost() || work.result == VK_COMPUTE_STOPPED) return -1;
    if (work.result != VK_COMPUTE_READY) {
        vk_exec_dispatch_cancel(dispatch);
        if (compatibility) return VK_EXEC_DISPATCH_COMPAT;
        rt_fatal("vk: required dispatch pipeline preparation failed (%s, Vulkan %d, admission %d)",
                 work.pending.failure ? work.pending.failure : "no ready pipeline", (int)work.pending.error,
                 (int)work.pending.admission);
    }
    if (compatibility) {
        VkExecDispatchResult ready = prepare_dispatch_inputs(ex, h, dispatch, input, attempted, 1);
        if (ready == VK_EXEC_DISPATCH_COMPAT &&
            prepare_dispatch_scratch(ex, dispatch, input) != VK_EXEC_DISPATCH_NATIVE)
            ready = VK_EXEC_DISPATCH_STOPPED;
        if (ready != VK_EXEC_DISPATCH_COMPAT || !vk_exec_dispatch_has_compat_inputs(dispatch))
            vk_exec_dispatch_cancel(dispatch);
        return ready;
    }
    VkResult standins = vk_exec_draw_storage_standins_prepare(ex, dispatch->info, 1);
    if (standins != VK_SUCCESS) {
        vk_exec_dispatch_cancel(dispatch);
        if (vk_sync_stopping() || vk_device_is_lost()) return VK_EXEC_DISPATCH_STOPPED;
        vk_exec_required_graphics_unavailable(ex, h->op, "selected compute storage stand-ins unavailable");
        return VK_EXEC_DISPATCH_STOPPED;
    }
    VkExecDispatchResult ready = prepare_dispatch_inputs(ex, h, dispatch, input, attempted, 0);
    return ready == VK_EXEC_DISPATCH_NATIVE ? prepare_dispatch_scratch(ex, dispatch, input) : ready;
}

static const op_hdr *walk(VkExec *ex, const op_hdr *h, int depth, uint64_t owner) {
    RWT_CHAIN(work_walk,RWT_REGION_COMMAND_WALK);
    NvnRecordPlan plan;
    int prepared = nvn_record_plan(owner, &plan);
    int executable = prepared && exec_plan_enabled();
    size_t exec_cursor = SIZE_MAX;
    ExecPlanStats stats __attribute__((cleanup(exec_stats_publish))) = {.path = EXEC_MAIN};
    RcRegistryPosition gpa_attempted = {0}, graphics_replanned = {0};
    VkExecGraphicsDraw *graphics __attribute__((cleanup(vk_exec_graphics_draw_cancel))) = NULL;
    int ordered_graphics_boundary = GRAPHICS_BOUNDARY_NONE;
    GpaDispatchBinding gpa_retry_dispatch __attribute__((cleanup(gpa_dispatch_release))) = {0};
    RW.skip_to[depth] = NULL;
    for (;;) {
        if (vk_exec_recording_failed()) return &walk_stopped;
        const NvnExecStep *step = executable ? nvn_record_exec_at(&plan, h, &exec_cursor) : NULL;
        if (!step && (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_))
            rt_fatal("vk: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        if (RW.skip_to[depth] == h) RW.skip_to[depth] = NULL;   // the small group read ahead is behind
        if (Q.drained < Q.out && Q.drained < __atomic_load_n(&Q.done, __ATOMIC_ACQUIRE)) groups_put_done(ex);
        if (RW.n > 0 && !RW.skip_to[depth] && !ordered_graphics_boundary && !evo_in && h->op != OP_JUMP &&
            (step ? (step->flags & NVN_EXEC_RUN) : run_op(h->op))) {
            const op_hdr *stop = group_start(ex, h, depth, prepared ? &plan : NULL,
                                             &graphics, &ordered_graphics_boundary);
            if (vk_exec_recording_failed()) return &walk_stopped;
            if (stop != h) {
                if (split_draws && split_wanted()) maybe_split(ex);
                h = stop;
                continue;
            }
        }
        const void *a = nvn_op_payload(h);
        VkDrawScratchUpdate *scratch_grant = NULL;
        int graphics_noop = 0;
        if (graphics_opcode(h->op)) {
            VkExecGraphicsResult ready = prepare_ordered_graphics(ex, h, ordered_graphics_boundary,
                &gpa_attempted, &graphics_replanned, &graphics);
            ordered_graphics_boundary = GRAPHICS_BOUNDARY_NONE;
            if (!graphics_input_finished(ex, h, ready, graphics)) return &walk_stopped;
            graphics_noop = ready == VK_EXEC_GRAPHICS_NOOP;
            if (graphics_noop) vk_exec_graphics_draw_cancel(&graphics);
        }
        uint64_t scratch_bytes = 0;
        if (h->op == OP_UPDATE_UB) {
            const p_update_ub *u = a;
            size_t header = sizeof *h + offsetof(p_update_ub, data);
            if (h->size < header || u->size > h->size - header)
                rt_fatal("vk: truncated UpdateUniformBuffer command payload");
            scratch_bytes = u->size;
        } else if (h->op == OP_DRAW && !graphics_noop) {
            if (h->size < sizeof *h + sizeof(p_draw)) rt_fatal("vk: truncated direct draw payload");
            scratch_bytes = nvn_generated_draw_index_bytes(a);
            if (scratch_bytes && step && (step->opcode != OP_DRAW || step->ops != 1))
                rt_fatal("vk: generated draw indices require a single draw step");
        }
        if (h->op == OP_UPDATE_UB || scratch_bytes) {
            group_drain(ex, PW_OP);
            if (vk_exec_recording_failed()) return &walk_stopped;
            if (!prepare_ordered_scratch(ex, h, scratch_bytes, &scratch_grant)) return &walk_stopped;
        }
        if (h->op == OP_CLEAR_DS && h->size >= sizeof *h + sizeof(p_clear_ds)) {
            const tex_t *depth = depth_clear_alias_destination(ex, a);
            if (depth) {
                group_drain(ex, PW_OP);
                if (vk_exec_recording_failed()) return &walk_stopped;
                if (vk_texture_alias_depth_clear_bound(depth)) {
                    image_write_input_position(ex);
                    int preserved = prepare_bound_texture_relocation(ex, depth, 1);
                    ex->resource_op = 0;
                    if (preserved < 0) {
                        if (!vk_sync_stopping() && !vk_device_is_lost())
                            vk_exec_required_graphics_unavailable(ex, OP_CLEAR_DS, "earlier bound image could not preserve depth-clear alias bytes");
                        return &walk_stopped;
                    }
                }
            }
        }
        AttachmentClearLoad clear_load = {0};
        const int clear_slot = partial_clear_slot(ex, h);
        if (clear_slot >= 0) {
            if (step && (step->ops != 1 || step->opcode != h->op))
                rt_fatal("vk: partial attachment clear requires one replay operation");
            group_drain(ex, PW_OP);
            if (vk_exec_recording_failed() || !prepare_attachment_clear_load(ex, h, (unsigned)clear_slot, &clear_load))
                return &walk_stopped;
        }
        VkExecClearTexture clear = {0};
        int exact_clear = 0;
        if (h->op == OP_CLEAR_TEX) {
            if (h->size < sizeof *h + sizeof(p_clear_tex)) rt_fatal("vk: truncated ClearTexture command payload");
            group_drain(ex, PW_OP);
            if (vk_exec_recording_failed()) return &walk_stopped;
            exact_clear = prepare_ordered_clear(ex, a, &clear);
            if (exact_clear < 0) return &walk_stopped;
        }
        const int is_buffer_transfer = h->op == OP_COPY_B2B || h->op == OP_CLEAR_BUF;
        VkExecBufferTransfer *buffer_transfer = NULL;
        if (is_buffer_transfer) {
            size_t size = h->op == OP_COPY_B2B ? sizeof(p_copy_b2b) : sizeof(p_clear_buf);
            if (h->size < sizeof *h + size || (step && (step->ops != 1 || step->opcode != h->op)))
                rt_fatal("vk: malformed ordered buffer transfer step");
            group_drain(ex, PW_OP);
            if (vk_exec_recording_failed()) return &walk_stopped;
            image_write_input_position(ex);
            RcRegistryResult result = vk_exec_buffer_transfer_prepare(h->op, a, vk_exec_resource_position(), &buffer_transfer);
            if (result == RC_REGISTRY_OK && h->op == OP_COPY_B2B)
                result = vk_exec_buffer_transfer_sources_prepare(ex, buffer_transfer);
            if (result != RC_REGISTRY_OK || vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) {
                vk_exec_buffer_transfer_release(buffer_transfer);
                if (!vk_sync_stopping() && !vk_device_is_lost() && !vk_exec_recording_failed()) {
                    rt_log("vk: ordered buffer transfer preparation registry status %d", (int)result);
                    vk_exec_required_graphics_unavailable(ex, h->op, "ordered buffer transfer preparation unavailable");
                }
                ex->resource_op = 0;
                return &walk_stopped;
            }
            ex->resource_op = 0;
        }
        VkExecBufferToTexture b2t __attribute__((cleanup(vk_exec_buffer_to_texture_cancel))) = {0};
        int exact_b2t = 0;
        if (h->op == OP_COPY_B2T) {
            if (h->size < sizeof *h + sizeof(p_copy_b2t) || (step && (step->ops != 1 || step->opcode != h->op)))
                rt_fatal("vk: malformed CopyBufferToTexture command payload");
            group_drain(ex, PW_OP);
            if (vk_exec_recording_failed()) return &walk_stopped;
            exact_b2t = prepare_ordered_buffer_to_texture(ex, a, &b2t);
            if (exact_b2t < 0) return &walk_stopped;
        }
        VkTextureTransferInput t2t_source __attribute__((cleanup(vk_texture_transfer_input_cancel))) = {0};
        VkExecTextureToTexture t2t = {0};
        int exact_t2t = 0;
        if (h->op == OP_COPY_T2T) {
            if (h->size < sizeof *h + sizeof(p_copy_t2t) || (step && (step->ops != 1 || step->opcode != h->op)))
                rt_fatal("vk: malformed CopyTextureToTexture command payload");
            group_drain(ex, PW_OP);
            if (vk_exec_recording_failed()) return &walk_stopped;
            exact_t2t = prepare_ordered_texture_to_texture(ex, a, &t2t, &t2t_source);
            if (exact_t2t < 0) return &walk_stopped;
        }
        VkExecProjectedTransfer projected __attribute__((cleanup(vk_exec_projected_transfer_cancel))) = {0};
        int projected_transfer = 0;
        if ((h->op == OP_COPY_B2T && !exact_b2t) || (h->op == OP_COPY_T2T && !exact_t2t)) {
            if (h->op == OP_COPY_T2T && t2t_source.texture) projected.image.input = &t2t_source;
            projected_transfer = prepare_ordered_projected_transfer(ex, h->op, a, &projected);
            if (projected_transfer < 0) return &walk_stopped;
        }
        const int compat_b2t = h->op == OP_COPY_B2T && !exact_b2t && !projected_transfer;
        if (compat_b2t && !prepare_compat_buffer_to_texture(ex, a, &b2t)) return &walk_stopped;
        VkExecRestoreCopy restored = {0};
        if (h->op == OP_COPY_T2B) {
            if (h->size < sizeof *h + sizeof(p_copy_b2t) || (step && (step->ops != 1 || step->opcode != h->op)))
                rt_fatal("vk: malformed CopyTextureToBuffer command payload");
            group_drain(ex, PW_OP);
            if (vk_exec_recording_failed() || prepare_ordered_texture_copy(ex, a, &restored) < 0)
                return &walk_stopped;
            ex->resource_op = ex->resource_next + 1; // The input prefix was sealed by texture preparation.
            int prepared_copy = vk_exec_restore_copy_prepare(ex, a, &restored);
            ex->resource_op = 0;
            if (prepared_copy < 0) {
                vk_exec_restore_copy_cancel(&restored);
                vk_exec_required_graphics_unavailable(ex, h->op, "texture copy source or destination could not be retained");
                return &walk_stopped;
            }
        }
        VkExecDispatch dispatch;
        VkExecDispatchResult exact_dispatch = VK_EXEC_DISPATCH_COMPAT;
        int compute = h->op == OP_DISPATCH || h->op == OP_DISPATCH_IND;
        if (compute) {
            if (Q.drained != Q.out && vk_gpa_write_scene_may_need_prefix(ex->fence)) group_drain(ex, PW_OP);
            if (vk_exec_recording_failed()) return &walk_stopped;
            dispatch = (VkExecDispatch){0};
            size_t size = h->op == OP_DISPATCH ? sizeof(p_dispatch) : sizeof(p_dispatch_ind);
            if (h->size < sizeof *h + size || (step && (step->opcode != h->op || step->ops != 1)))
                rt_fatal("vk: malformed ordered dispatch step");
            RcRegistryPosition next = {ex->fence, (ex->resource_next + 1) * 2};
            int retry_native = gpa_retry_dispatch.native_argument && !rc_registry_position_compare(next, gpa_attempted);
            exact_dispatch = prepare_ordered_dispatch(ex, h, &dispatch, &gpa_attempted,
                retry_native ? &gpa_retry_dispatch.argument_span : NULL,
                retry_native ? gpa_retry_dispatch.argument : NULL);
            if (gpa_retry_dispatch.shader && !rc_registry_position_compare(next, gpa_attempted) &&
                exact_dispatch != (gpa_retry_dispatch.selected ? VK_EXEC_DISPATCH_NATIVE : VK_EXEC_DISPATCH_COMPAT)) {
                rt_log("vk: ordered GPA retry changed the native dispatch selection");
                __atomic_store_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_RELEASE);
            }
            if (exact_dispatch == VK_EXEC_DISPATCH_STOPPED || vk_sync_stopping() || vk_device_is_lost() ||
                vk_exec_recording_failed()) {
                vk_exec_dispatch_cancel(&dispatch);
                return &walk_stopped;
            }
        }
        if (step && step->opcode == NVN_EXEC_STATE && RW.skip_to[depth])
            step = nvn_record_exec_slice(&plan, h, RW.skip_to[depth], UINT32_MAX, &exec_cursor);
        AH.at[depth] = h, AH.depth = depth;   // where vk_exec_ahead reads on from
        if (RR.on > 0) runs_op(h->op);
        if (step ? !(step->flags & NVN_EXEC_LINEAR) : !op_linear(h->op)) vk_exec_brk++;
        switch (h->op) {
        case OP_END: return NULL;
        case OP_JUMP: h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target; continue;
        case OP_CALL: {
            if (depth >= CALL_DEPTH) rt_fatal("vk: command buffer calls nested deeper than %d", CALL_DEPTH);
            const op_hdr *stop = walk(ex, nvn_op_first(((const p_jump *)a)->target), depth + 1, ((const p_jump *)a)->target);
            if (stop == &walk_stopped) return stop;
            break;
        }
        case OP_WAIT_EVENT: {
            int met = vk_exec_wait_event(a);
            vk_exec_evt(met ? 'm' : 'u', (const void *)((const p_wait_event *)a)->val, 0);
            if (!met && depth == 0 && evo_queue) {
                vk_exec_pass_end(ex);
                return h;
            }
            break;
        }
        default: {
            VkExecGraphicsDraw *prior_graphics = vk_exec_graphics_draw_enter(graphics);
            void *effect;
            VkExecDispatch *prior_dispatch = NULL;
            if (exact_dispatch == VK_EXEC_DISPATCH_NATIVE || exact_dispatch == VK_EXEC_DISPATCH_NOOP) {
                if (exact_dispatch == VK_EXEC_DISPATCH_NATIVE) {
                    uint64_t prior_op = ex->resource_op;
                    ex->resource_op = ex->resource_next + 1;
                    if (rc_registry_position_compare(vk_exec_resource_position(), dispatch.read_position))
                        rt_fatal("vk: final dispatch validation moved its read position");
                    vk_exec_dispatch_validate(ex, &dispatch, h->op, a);
                    ex->resource_op = prior_op;
                }
                resource_assign(ex, 1);
                effect = exact_dispatch == VK_EXEC_DISPATCH_NATIVE ? vk_exec_effects_declare_dispatch(ex, h->op, a,
                    dispatch.shader, &dispatch.selection, (RcRegistryPosition){ex->fence, ex->resource_op * 2 + 1},
                    &dispatch.scratch) : NULL;
                resource_seal(ex);
                if (exact_dispatch == VK_EXEC_DISPATCH_NATIVE) prior_dispatch = vk_exec_dispatch_enter(&dispatch);
            } else if (graphics_noop) {
                resource_declare_noop(ex);
                effect = NULL;
            } else if (is_buffer_transfer) {
                resource_assign(ex, 1);
                effect = vk_exec_effects_declare_buffer_transfer(ex, h->op, a, buffer_transfer,
                    (RcRegistryPosition){ex->fence, ex->resource_op * 2 + 1});
                resource_seal(ex);
            } else if (projected_transfer) {
                effect = resource_declare_projected_transfer(ex, &projected);
            } else if (exact_clear || exact_b2t || exact_t2t) {
                VkTextureWriteTarget *target = exact_clear ? clear.target : exact_b2t ? b2t.target : t2t.target;
                effect = resource_declare_texture_write(ex, h->op, target);
            } else if (h->op == OP_COPY_T2B) {
                resource_assign(ex, 1);
                effect = vk_exec_effects_declare_texture_buffer(ex, a, &restored,
                    (RcRegistryPosition){ex->fence, ex->resource_op * 2 + 1});
                resource_seal(ex);
            } else effect = resource_declare(ex, h, step ? step->ops : 1,
                compute && vk_exec_dispatch_has_compat_inputs(&dispatch) ? &dispatch : NULL);
            void *previous_effect = vk_exec_effects_enter(effect);
            RcRegistryPosition read = vk_exec_resource_position();
            const GpaDispatchBinding *retry = gpa_retry_dispatch.shader &&
                !rc_registry_position_compare(read, gpa_attempted) ? &gpa_retry_dispatch : NULL;
            GpaReadPreparation gpa_prepare = {.ex = ex, .effect = effect, .attempted = gpa_attempted,
                .compute = compute, .prior_dispatch = retry};
            GpaReadPreparation *previous_gpa = gpa_read_preparation;
            int may_park = !scratch_grant && (!step || (step->ops == 1 && step->opcode == h->op)) &&
                (compute || h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND);
            gpa_read_preparation = may_park ? &gpa_prepare : NULL;
            if (graphics_noop) {
                graphics_noop_record(ex, h);
            } else if (exact_dispatch == VK_EXEC_DISPATCH_NOOP) {
                RWT_CHAIN(work_operation, RWT_REGION_OP_COMPUTE);
                vk_exec_frame.dispatches++;
                split_draws++;
            } else if (is_buffer_transfer) {
                if (!vk_exec_recording_failed()) op_one(ex, h);
            } else if (clear_load.pass) {
                if (vk_exec_graphics_pass_activate(ex, clear_load.pass)) op_one(ex, h);
                else if (!vk_sync_stopping() && !vk_device_is_lost())
                    vk_exec_required_graphics_unavailable(ex, h->op, "partial clear LOAD receipt changed before issue");
                attachment_clear_load_cancel(&clear_load);
            } else if (exact_clear) {
                if (clear.target) vk_exec_clear_commit(ex, &clear);
            } else if (projected_transfer) {
                RWT_CHAIN(work_operation, RWT_REGION_OP_TRANSFER);
                vk_exec_frame.copies++;
                if (vk_dump_armed) vk_dump_copy(ex, h->op, a);
                vk_exec_projected_transfer_record(ex, &projected);
                vk_exec_projected_transfer_cancel(&projected);
                if (vk_sync_stopping() || vk_device_is_lost()) cancel_recording();
            } else if (exact_b2t || compat_b2t) {
                RWT_CHAIN(work_operation, RWT_REGION_OP_TRANSFER);
                vk_exec_frame.copies++;
                if (vk_dump_armed) vk_dump_copy(ex, h->op, a);
                if (compat_b2t) {
                    vk_exec_buffer_to_texture_commit_compat(ex, a, &b2t);
                    if (b2t.source_hold && vk_texture_deferred_take() && op_texture_turn(ex)) {
                        vk_exec_buffer_to_texture_commit_compat(ex, a, &b2t);
                        if (b2t.source_hold && vk_texture_deferred_take() && !vk_sync_stopping() && !vk_device_is_lost())
                            vk_exec_required_graphics_unavailable(ex, h->op, "retained buffer-to-texture destination stayed pending");
                    }
                } else if (b2t.target) vk_exec_buffer_to_texture_commit(ex, &b2t);
            } else if (exact_t2t) {
                RWT_CHAIN(work_operation, RWT_REGION_OP_TRANSFER);
                vk_exec_frame.copies++;
                if (vk_dump_armed) vk_dump_copy(ex, h->op, a);
                if (t2t.target) vk_exec_texture_to_texture_commit(ex, &t2t);
            } else if (h->op == OP_COPY_T2T) {
                RWT_CHAIN(work_operation, RWT_REGION_OP_TRANSFER);
                vk_exec_frame.copies++;
                if (vk_dump_armed) vk_dump_copy(ex, h->op, a);
                vk_exec_texture_to_texture_compat(ex, a, &t2t_source);
                if (vk_texture_deferred_take() && op_texture_turn(ex)) {
                    vk_exec_texture_to_texture_compat(ex, a, &t2t_source);
                    if (vk_texture_deferred_take() && !vk_sync_stopping() && !vk_device_is_lost())
                        vk_exec_required_graphics_unavailable(ex, h->op, "texture-to-texture destination stayed pending");
                }
            } else if (h->op == OP_COPY_T2B) {
                RWT_CHAIN(work_operation, RWT_REGION_OP_TRANSFER);
                vk_exec_frame.copies++;
                if (vk_dump_armed) vk_dump_copy(ex, h->op, a);
                vk_exec_restore_copy_commit(ex, a, &restored);
                vk_exec_restore_copy_cancel(&restored);
            } else if (h->op == OP_UPDATE_UB) {
                RWT_CHAIN(work_operation, RWT_REGION_OP_UPDATE);
                vk_exec_draw_update_commit(ex, a, scratch_grant);
            } else if (RW.n > 0) {   // the perf line's split of the replay outside groups
                if (Q.drained < Q.out && !pipe_op(h->op)) RW.pd_op[h->op]++, group_drain(ex, PW_OP);
                int k = ser_kind(h->op);
                uint64_t t = kern_now_ns();
                if (!vk_exec_recording_failed()) {
                    if (step) exec_step_run(ex, &plan, step);
                    else op_one(ex, h);
                }
                if (!retry && !gpa_prepare.request.guard.id && vk_texture_deferred_take() && !vk_exec_recording_failed())
                    op_again(ex, h);
                RW.ser_n[k] += step ? step->ops : 1, RW.ser_ns[k] += kern_now_ns() - t;
            } else if (step) exec_step_run(ex, &plan, step);
            else op_one(ex, h);
            gpa_read_preparation = previous_gpa;
            int retry_deferred = retry && vk_texture_deferred_take();
            if (retry && (!gpa_prepare.binding_checked || retry_deferred) && !vk_exec_recording_failed() &&
                !vk_sync_stopping() && !vk_device_is_lost()) {
                rt_log("vk: ordered GPA retry did not finish its native dispatch binding");
                __atomic_store_n(&recording_failed, RECORDING_RESOURCE_UNAVAILABLE, __ATOMIC_RELEASE);
            }
            if (gpa_prepare.request.guard.id && !vk_exec_effects_unrecorded(effect)) {
                rt_log("vk: an ordered GPA source request outlived its unrecorded operation");
                __atomic_store_n(&recording_failed, 1, __ATOMIC_RELEASE);
            }
            if (h->op == OP_DRAW && scratch_grant) vk_exec_draw_index_finish(ex, scratch_grant);
            if (plan.nexec) exec_stats_step(&stats, step);
            vk_exec_effects_recorded(effect,0);
            vk_exec_effects_leave(previous_effect);
            vk_exec_effects_release(effect);
            vk_exec_graphics_draw_enter(prior_graphics);
            if (exact_dispatch == VK_EXEC_DISPATCH_NATIVE) vk_exec_dispatch_enter(prior_dispatch);
            if (compute) vk_exec_dispatch_cancel(&dispatch);
            ex->resource_op = 0;
            if (gpa_prepare.request.guard.id) {
                vk_texture_deferred_take();
                if (!prepare_ordered_gpa_source(ex, &gpa_prepare)) {
                    gpa_dispatch_release(&gpa_prepare.dispatch);
                    vk_exec_graphics_draw_cancel(&graphics);
                    return &walk_stopped;
                }
                ordered_graphics_boundary = graphics ? GRAPHICS_BOUNDARY_ORDERED : GRAPHICS_BOUNDARY_NONE;
                gpa_attempted = gpa_prepare.request.read;
                gpa_dispatch_release(&gpa_retry_dispatch);
                gpa_retry_dispatch = gpa_prepare.dispatch;
                continue;
            }
            if (retry) gpa_dispatch_release(&gpa_retry_dispatch);
            vk_exec_graphics_draw_cancel(&graphics);
            break;
        }
        }
        if (vk_exec_recording_failed()) return &walk_stopped;
        if (vk_exec_signalled && depth == 0 && !evo_in && !evo_resume(ex)) return &walk_stopped;
        if (split_draws && split_wanted()) maybe_split(ex);
        h = step ? (const op_hdr *)(plan.commands + step->end) : nvn_op_next(h);
    }
}

// The replay records on the thread the game submitted from, and the kernel schedules that like any other guest
// thread: a preemption in the middle of a part leaves the submission late for the frame it belongs to (Kokiri
// Forest: 274 of 2,737 frames had a late submit). The recording thread therefore keeps the highest priority so the
// scheduler prefers it over the guest's other cores (NX2NAVI_RECORDER_PRIO=0 leaves the guest's priorities alone) and
// gets a core of its own (plat_cpu_reserve): every other thread of the process, the guest's cores, the audio, the
// shader and pipeline workers, is kept off it, so nothing of this process preempts the recording. That core is then
// lost to the rest, so NX2NAVI_RECORDER_CORE=auto reserves it only with 12 or more logical processors, and not while
// the process runs on the die with the largest L3 alone (perf.c cpu_placement): on a two-die X3D part the game's
// threads already have that die to themselves, and a core taken from them cost frames; 1 always, 0 never. Linux
// reserves none with fewer than four fast cores.
// 0 is the default: on an i9-14900KF (8 performance and 16 efficiency cores, 32 logical processors) with the replay on
// its own thread, the reserved core (logical processors 14 and 15) left the woken replay thread unscheduled for about
// 200 ms roughly once a second while the game stood in Kakariko: its queue held 40 to 50 items, its lock was free, the
// wakes had been sent and it used no CPU time, while the game waited on a fence of the work in that queue (27 frames a
// minute with a 150 ms or longer frame). With no core reserved the same minute had none.
// The recording thread is not the first to call vk_exec_record: the game's main thread submits first and goes on
// submitting a little at a time, while another guest thread does the work (Kokiri Forest: that thread spent 21.9 of
// the 22.5 s the replay took in a minute and a half, the first one 0.5 s). The first thread to have spent
// REC_CLAIM_NS in vk_exec_record is the one that keeps both.
#define REC_CLAIM_NS 500000000ull
static _Thread_local uint64_t rec_spent_ns;   // time this thread has spent in vk_exec_record

static void rec_claim(void) {
    static int done;
    if (done) return;
    done = 1;
    rt_log("vk: thread %u is the recording thread (the first to spend %.1f s in it)", plat_thread_id(),
           REC_CLAIM_NS / 1e9);
    const char *e = cfg_get("NX2NAVI_RECORDER_PRIO");
    if (!(e && !strcmp(e, "0"))) {
        plat_thread_priority(PLAT_PRIO_HIGHEST);
        rt_log("vk: the recording thread keeps the highest scheduling priority (NX2NAVI_RECORDER_PRIO=0 turns it off)");
    }
    const char *rc = cfg_value("NX2NAVI_RECORDER_CORE");
    unsigned ncpu = plat_cpu_usable();
    int cache_die = perf_cpu_on_cache_die();
    if (!strcmp(rc, "auto") && cache_die) rt_log("vk: no core is reserved for the recording thread: the process runs on "
                                                 "the die with the largest L3 cache (NX2NAVI_RECORDER_CORE=1 reserves one)");
    if (!strcmp(rc, "1") || (!strcmp(rc, "auto") && ncpu >= 12 && !cache_die)) {
        char which[96];
        int n = plat_cpu_reserve(which, sizeof which);
        if (n) rt_log("vk: the recording thread has a core of its own: logical processor%s %s (%d of %u; "
                      "NX2NAVI_RECORDER_CORE=0 turns it off)", n > 1 ? "s" : "", which, n, ncpu);
        else rt_log("vk: no core was reserved for the recording thread (too few fast cores, or the system refused it; "
                    "NX2NAVI_RECORDER_CORE=0 stops the attempt)");
    } else {
        rt_log("vk: the recording thread runs on any core (NX2NAVI_RECORDER_CORE=auto or 1 reserves one)");
    }
}

static VkResult snapshot_metadata_allocate(size_t bytes, int clear, RcBudgetTicket *ticket, void **out,
                                            VkReplayCaptureWait *wait) {
    VkResult result = vk_replay_metadata_allocate(bytes, clear, RC_BUDGET_SNAPSHOT, ticket, out);
    if (result != VK_NOT_READY) return result;
    wait->kind = VK_REPLAY_CAPTURE_WAIT_BUDGET;
    return vk_replay_metadata_peak_add(&wait->peak, bytes) ? VK_NOT_READY : VK_ERROR_OUT_OF_HOST_MEMORY;
}

// These allocations are refunded on refusal but will overlap the next attempt at this same allocation point.
static VkResult snapshot_metadata_retry(VkResult result, VkReplayCaptureWait *wait, const uint64_t *held, size_t count) {
    if (result == VK_NOT_READY && wait->kind == VK_REPLAY_CAPTURE_WAIT_BUDGET) {
        RcAdmissionVector peak = wait->peak;
        for (size_t i = 0; i < count; ++i)
            if (!vk_replay_metadata_peak_add(&peak, held[i])) return VK_ERROR_OUT_OF_HOST_MEMORY;
        wait->peak = peak;
    }
    return result;
}

static VkResult snapshot_preflight_live(const uint64_t *handles, int n, VkUsnapCapture **capture, VkReplayCaptureWait *wait) {
    VkUsnapInputs inputs = {.handles = handles, .nhandles = n};
    if (*capture || (!n && (!evo_pending || !held_sync()))) return vk_usnap_preflight(&inputs, capture, wait);
    enum { UBO_INPUTS = sizeof X.ubo / sizeof(p_bind_buf) };
    p_bind_buf current[UBO_INPUTS];
    if (x_ready) {
        memcpy(current, X.ubo, sizeof current);
        inputs.inherited = current;
        inputs.ninherited = UBO_INPUTS;
    }
    if (!evo_pending) return vk_usnap_preflight(&inputs, capture, wait);

    size_t handle_count = (size_t)n, inherited_count = inputs.ninherited;
    for (int s = 0; s < EVO_QUEUES; s++)
        for (Cont *c = evq[s].head; c; c = c->next) {
            if ((size_t)(c->n - c->i) > INT_MAX - handle_count) return VK_ERROR_OUT_OF_HOST_MEMORY;
            handle_count += (size_t)(c->n - c->i);
            if (c->has_st || c->parked_ubo) {
                if (inherited_count > SIZE_MAX / sizeof(p_bind_buf) - UBO_INPUTS) return VK_ERROR_OUT_OF_HOST_MEMORY;
                inherited_count += UBO_INPUTS;
            }
        }
    if (handle_count > SIZE_MAX / sizeof(uint64_t)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    RcBudgetTicket all_budget = {0}, inherited_budget = {0};
    size_t all_bytes = handle_count * sizeof(uint64_t), inherited_bytes = inherited_count * sizeof(p_bind_buf);
    void *all_storage = NULL, *inherited_storage = NULL;
    VkResult result = VK_SUCCESS;
    if (all_bytes)
        result = snapshot_metadata_allocate(all_bytes, 0, &all_budget, &all_storage, wait);
    if (result == VK_SUCCESS && inherited_bytes)
        result = snapshot_metadata_allocate(inherited_bytes, 0, &inherited_budget, &inherited_storage, wait);
    if (result != VK_SUCCESS) goto done;
    uint64_t *all = all_storage;
    p_bind_buf *inherited = inherited_storage;
    size_t k = (size_t)n, b = inputs.ninherited;
    if (n) memcpy(all, handles, k * sizeof *all);
    if (b) memcpy(inherited, current, sizeof current);
    for (int s = 0; s < EVO_QUEUES; s++)
        for (Cont *c = evq[s].head; c; c = c->next) {
            size_t remaining = (size_t)(c->n - c->i);
            if (remaining) memcpy(all + k, c->h + c->i, remaining * sizeof *all);
            k += remaining;
            const void *saved = c->has_st ? c->st + offsetof(VkExec, ubo) - ST_LO :
                c->parked_ubo ? (const void *)c->parked_ubo->bindings : NULL;
            if (saved) {
                memcpy(inherited + b, saved, sizeof current);
                b += UBO_INPUTS;
            }
        }
    inputs = (VkUsnapInputs){all, (int)handle_count, inherited, inherited_count};
    result = vk_usnap_preflight(&inputs, capture, wait);
done:;
    uint64_t held[] = {all_storage ? all_bytes : 0, inherited_storage ? inherited_bytes : 0};
    result = snapshot_metadata_retry(result, wait, held, sizeof held / sizeof *held);
    vk_replay_metadata_free(inherited_storage, &inherited_budget);
    vk_replay_metadata_free(all_storage, &all_budget);
    return result;
}

// Preflight follows only control and uniform-inventory operations. It never changes queues, registration
// horizons, replay event state, write receipts or buffers. Recording uses the same frozen event samples.
typedef struct {
    const volatile uint32_t *address;
    uint32_t value;
} ReadinessEvent;

struct VkExecReadiness {
    ReadinessEvent *events;
    ParkedUbo *parked_ubo;
    Cont *parked_commands;
    RcBudgetTicket header_budget, event_budget;
    size_t nevents, cevents;
    uint64_t sequence;
    int initial_resume, frozen;
};

static _Thread_local const VkExecReadiness *record_readiness;

typedef struct {
    const Cont *held;
    const op_hdr *at;
    uint64_t since;
    int index;
    int saved_state;
} ReadinessCursor;

typedef struct {
    VkExecReadiness *readiness;
    VkExecEventOrder order;
    ReadinessCursor slots[EVO_QUEUES];
    p_bind_buf *bindings;
    RcBudgetTicket binding_budget;
    VkReplayCaptureWait *wait;
    VkResult result;
    size_t nbindings, cbindings;
    int global_state;
    int signalled;
} ReadinessPlan;

static const ReadinessEvent *readiness_event(const VkExecReadiness *readiness,
                                            const volatile uint32_t *event) {
    for (size_t i = 0; i < readiness->nevents; ++i)
        if (readiness->events[i].address == event) return &readiness->events[i];
    return NULL;
}

int vk_exec_frozen_event(const volatile uint32_t *event, uint32_t *memory) {
    if (!record_readiness) return 0;
    const ReadinessEvent *sample = readiness_event(record_readiness, event);
    if (!sample) rt_fatal("vk: replayed an event outside its admitted uniform cohort");
    *memory = sample->value;
    return 1;
}

void vk_exec_readiness_discard(VkExecReadiness *readiness) {
    if (!readiness) return;
    parked_ubo_release(readiness->parked_ubo);
    cont_free(readiness->parked_commands);
    vk_replay_metadata_free(readiness->events, &readiness->event_budget);
    RcBudgetTicket header = readiness->header_budget;
    vk_replay_metadata_free(readiness, &header);
}

static void *readiness_grow(ReadinessPlan *plan, void *items, size_t *capacity, size_t used, size_t extra,
                            size_t item_size, RcBudgetTicket *ticket) {
    if (plan->result != VK_SUCCESS || extra <= *capacity - used) return items;
    size_t limit = SIZE_MAX / item_size;
    if (extra > limit - used) {
        plan->result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return items;
    }
    size_t new_capacity = *capacity <= limit / 2 ? *capacity * 2 : limit;
    if (new_capacity < 16) new_capacity = 16;
    if (new_capacity < used + extra) new_capacity = used + extra;
    RcBudgetTicket replacement = {0};
    void *data;
    plan->result = snapshot_metadata_allocate(new_capacity * item_size, 0, &replacement, &data, plan->wait);
    if (plan->result != VK_SUCCESS) return items;
    if (used) memcpy(data, items, used * item_size);
    vk_replay_metadata_free(items, ticket);
    *ticket = replacement;
    *capacity = new_capacity;
    return data;
}

static uint32_t readiness_sample(ReadinessPlan *plan, const volatile uint32_t *event) {
    VkExecReadiness *readiness = plan->readiness;
    const ReadinessEvent *sample = readiness_event(readiness, event);
    if (sample) return sample->value;
    readiness->events = readiness_grow(plan, readiness->events, &readiness->cevents, readiness->nevents,
                                        1, sizeof *readiness->events, &readiness->event_budget);
    if (plan->result != VK_SUCCESS) return 0;
    uint32_t value = __atomic_load_n(event, __ATOMIC_ACQUIRE);
    readiness->events[readiness->nevents++] = (ReadinessEvent){event, value};
    return value;
}

static void readiness_bindings(ReadinessPlan *plan, const void *bindings, size_t count) {
    plan->bindings = readiness_grow(plan, plan->bindings, &plan->cbindings, plan->nbindings,
                                   count, sizeof *plan->bindings, &plan->binding_budget);
    if (plan->result != VK_SUCCESS) return;
    memcpy(plan->bindings + plan->nbindings, bindings, count * sizeof *plan->bindings);
    plan->nbindings += count;
}

static void readiness_global_state(ReadinessPlan *plan) {
    if (plan->global_state) return;
    plan->global_state = 1;
    if (x_ready) readiness_bindings(plan, X.ubo, sizeof X.ubo / sizeof(p_bind_buf));
}

static int readiness_met(ReadinessPlan *plan, const p_wait_event *wait) {
    uint32_t value = readiness_sample(plan, wait->val);
    return plan->result == VK_SUCCESS && vk_exec_event_order_met(&plan->order, wait, value);
}

static int readiness_resume(ReadinessPlan *plan);

static const op_hdr *readiness_walk(ReadinessPlan *plan, const op_hdr *h, int depth, int may_park, int held) {
    while (plan->result == VK_SUCCESS) {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk: corrupt command record during uniform readiness at %p", (const void *)h);
        if (!held && h->op != OP_BEGIN && h->op != OP_END && h->op != OP_WAIT_EVENT &&
            h->op != OP_SIGNAL_EVENT && h->op != OP_CALL && h->op != OP_JUMP)
            readiness_global_state(plan);
        const void *payload = nvn_op_payload(h);
        switch (h->op) {
        case OP_END: return NULL;
        case OP_JUMP:
            h = (const op_hdr *)(uintptr_t)((const p_jump *)payload)->target;
            continue;
        case OP_CALL:
            if (depth >= CALL_DEPTH) rt_fatal("vk: command buffer calls nested deeper than %d", CALL_DEPTH);
            readiness_walk(plan, nvn_op_first(((const p_jump *)payload)->target), depth + 1, may_park, held);
            break;
        case OP_WAIT_EVENT:
            if (!readiness_met(plan, payload) && !depth && may_park) return h;
            break;
        case OP_SIGNAL_EVENT: {
            const p_signal_event *signal = payload;
            uint32_t value = readiness_sample(plan, signal->val);
            if (plan->result != VK_SUCCESS) return NULL;
            vk_exec_event_order_signal(&plan->order, signal, value);
            plan->signalled = 1;
            break;
        }
        case OP_BIND_UBO: readiness_bindings(plan, payload, 1); break;
        default: break;
        }
        if (plan->signalled && !depth && !held) readiness_resume(plan);
        h = nvn_op_next(h);
    }
    return NULL;
}

static void readiness_held_state(ReadinessPlan *plan, ReadinessCursor *cursor) {
    if (!cursor->saved_state) return;
    cursor->saved_state = 0;
    const Cont *c = cursor->held;
    if (c->has_st || c->parked_ubo) {
        const void *bindings = c->has_st ? c->st + offsetof(VkExec, ubo) - ST_LO :
            (const void *)c->parked_ubo->bindings;
        readiness_bindings(plan, bindings, sizeof X.ubo / sizeof(p_bind_buf));
    } else {
        readiness_global_state(plan);
    }
}

static int readiness_slot(void *context, int slot) {
    ReadinessPlan *plan = context;
    ReadinessCursor *cursor = &plan->slots[slot];
    const Cont *c = cursor->held;
    if (!c || (cursor->at && !readiness_met(plan, nvn_op_payload(cursor->at)) &&
               !cont_wait_timed_out(plan->readiness->sequence, cursor->since)))
        return plan->result == VK_SUCCESS ? 0 : -1;

    readiness_held_state(plan, cursor);
    if (plan->result != VK_SUCCESS) return -1;
    for (; c; c = cursor->held) {
        for (; cursor->index < c->n; ++cursor->index) {
            const op_hdr *from = cursor->at ? nvn_op_next(cursor->at) : nvn_op_first(c->h[cursor->index]);
            cursor->at = readiness_walk(plan, from, 0, evq[slot].q != 0, 1);
            if (plan->result != VK_SUCCESS) return -1;
            if (cursor->at) {
                cursor->since = plan->readiness->sequence;
                return 1;
            }
        }
        // FIFO followers start in the state left by their predecessor.
        *cursor = (ReadinessCursor){.held = c->next};
    }
    return 1;
}

static int readiness_resume(ReadinessPlan *plan) {
    plan->signalled = 0;
    return plan->result == VK_SUCCESS ? evo_scan(plan, readiness_slot) : -1;
}

static int snapshot_first_possible_park(const uint64_t *handles, int n) {
    for (int i = 0; i < n; ++i) {
        NvnRecordPlan commands;
        if (!nvn_record_plan(handles[i], &commands) || !commands.nexec) return i;
        for (size_t step = 0; step < commands.nexec; ++step) {
            unsigned op = commands.exec[step].opcode;
            // The index cannot prove the contents of a raw control-flow target.
            if (op == OP_WAIT_EVENT || op == OP_CALL || op == OP_JUMP) return i;
        }
    }
    return -1;
}

VkResult vk_exec_snapshot_preflight(const uint64_t *handles, int n, VkUsnapCapture **capture,
                                    VkExecReadiness **readiness, VkReplayCaptureWait *wait) {
    if (*readiness) rt_fatal("vk: snapshot preflight already owns a readiness plan");
    *wait = (VkReplayCaptureWait){0};
    // The live-wait diagnostic keeps its existing event timing, including a wait that closes after preflight.
    int live = *capture || !vk_exec_event_at_sync();
    int slot = evo_enabled() && gfx_submit_queue ? evq_find_slot(gfx_submit_queue) : -1;
    const int possible_wait = slot >= 0 ? snapshot_first_possible_park(handles, n) : -1;
    const int may_follow = slot >= 0 && n && evq[slot].head;
    const int may_park = possible_wait >= 0 || may_follow;
    int project = !live && (n || (evo_pending && held_sync())) && (evo_pending || may_park);
    if (!project && !(live && may_park))
        return snapshot_preflight_live(handles, n, capture, wait);

    RcBudgetTicket header = {0};
    void *allocation;
    VkResult result = snapshot_metadata_allocate(sizeof(VkExecReadiness), 1, &header, &allocation, wait);
    if (result != VK_SUCCESS) return result;
    VkExecReadiness *selected = allocation;
    selected->header_budget = header;
    selected->sequence = vk_exec_submit_seq + 1;
    selected->frozen = project;
    ReadinessPlan plan = {.readiness = selected, .signalled = vk_exec_signalled, .wait = wait};
    // A live wait can change after preflight. A blocked live queue might either follow or reach its own wait.
    int park_from = project ? -1 : may_follow ? 0 : possible_wait;
    int park_ubo = !project && possible_wait >= 0;
    if (project) {
        vk_exec_event_order_copy(&plan.order);
        for (int s = 0; s < EVO_QUEUES; ++s) {
            const Cont *c = evq[s].head;
            if (c) plan.slots[s] = (ReadinessCursor){c, c->at, c->since, c->i, 1};
        }
        selected->initial_resume = readiness_resume(&plan);
        if (n && slot >= 0 && plan.slots[slot].held) {
            park_from = 0;
        } else if (n) {
            for (int i = 0; plan.result == VK_SUCCESS && i < n; ++i) {
                if (!handles[i]) rt_fatal("vk: command handle %d of a submission is null", i);
                if (readiness_walk(&plan, nvn_op_first(handles[i]), 0, slot >= 0, 0)) {
                    park_from = i;
                    park_ubo = 1;
                    break;
                }
            }
        }
    }
    result = plan.result;
    if (result == VK_SUCCESS && park_from >= 0)
        result = cont_prepare(handles, n, park_from, &selected->parked_commands, wait);
    if (result == VK_SUCCESS && park_ubo) {
        RcBudgetTicket parked_budget = {0};
        result = snapshot_metadata_allocate(sizeof(ParkedUbo), 0, &parked_budget, &allocation, wait);
        if (result == VK_SUCCESS) {
            selected->parked_ubo = allocation;
            selected->parked_ubo->budget = parked_budget;
            selected->parked_ubo->refs = 1;
        }
    }
    if (result == VK_SUCCESS) {
        if (project) {
            // These are the actual bind payloads, including inherited state. USnap still validates the providers.
            VkUsnapInputs inputs = {.inherited = plan.bindings, .ninherited = plan.nbindings};
            result = vk_usnap_preflight(&inputs, capture, wait);
        } else result = snapshot_preflight_live(handles, n, capture, wait);
    }
    uint64_t held[] = {sizeof *selected, selected->cevents * sizeof *selected->events,
                       plan.cbindings * sizeof *plan.bindings, selected->parked_ubo ? sizeof(ParkedUbo) : 0,
                       selected->parked_commands ? selected->parked_commands->bytes : 0};
    result = snapshot_metadata_retry(result, wait, held, sizeof held / sizeof *held);
    vk_replay_metadata_free(plan.bindings, &plan.binding_budget);
    if (result == VK_SUCCESS) *readiness = selected;
    else vk_exec_readiness_discard(selected);
    return result;
}

static int settle_mesh_capacity(VkExec *ex) {
    if (vk_sync_stopping() || vk_device_is_lost()) goto stopping;
    int commands = vk_meshfb_capacity_copy(ex->cb, ex->fence);
    VkMeshFbCapacity status = {.scene = ex->fence, .error = VK_ERROR_INITIALIZATION_FAILED};
    VkMeshFbCapacityResult result = VK_MESHFB_CAPACITY_UNPROVEN;
    if (commands > 0) {
        ex->recorded += (uint32_t)commands;
        VkSyncProgressResult progress = vk_exec_graphics_pool_progress(ex, 0);
        if (progress == VK_SYNC_PROGRESS_STOPPED || progress == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT) goto stopping;
        if (progress == VK_SYNC_PROGRESS_COMPLETE)
            result = vk_meshfb_capacity_read(ex->fence, &status);
        else
            status.error = VK_NOT_READY;
    }
    if (result == VK_MESHFB_CAPACITY_STOPPED || vk_sync_stopping() || vk_device_is_lost()) goto stopping;
    if (!commands || result == VK_MESHFB_CAPACITY_READY) return 1;

    int active = 0;
    if (__atomic_compare_exchange_n(&recording_failed, &active, RECORDING_RESOURCE_UNAVAILABLE, 0,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        if (result == VK_MESHFB_CAPACITY_OVERFLOW)
            rt_log("vk: required mesh fallback capacity exceeded at scene %llu operation %llu "
                   "(limit %d, outer groups %u, mesh groups %llu, available %u, detail %u)",
                   (unsigned long long)status.scene, (unsigned long long)status.ordinal, (int)status.limit,
                   status.requested_outer, (unsigned long long)status.requested_mesh, status.available, status.detail);
        else
            rt_log("vk: mesh fallback scene %llu cannot signal success without checked capacity (%d)",
                   (unsigned long long)ex->fence, (int)status.error);
    }
    return 0;

stopping:
    cancel_recording();
    return 0;
}

int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence,
                   VkReplayCaptureBundle *bundle, const VkExecReadiness *readiness) {
    if (vk_device_is_lost()) return 0;   // the run is ending (vk_device.c vk_device_lost_check): nothing is recorded
    RP_SCOPE(rp_replay, RP_REPLAY);
    RWT_UNIT(work_replay, RWT_ROLE_OWNER, fence, gfx_rp_current(), 0, 0);
    RWT_CHAIN(work_service,RWT_REGION_REPLAY_SERVICE);
    RP_VALUE(RPK_FENCE, fence);
    RP_VALUE(RPK_HANDLES, n);
    RP_VALUE(RPK_HELD_BEFORE, evo_pending);
    RP_SCOPE(rp_setup, RP_SETUP);
    uint64_t in_ns = kern_now_ns();
    gfx_tl_note('R', 0, (uint64_t)n, fence);
    perf_lock_timing_here();   // NX2NAVI_LOCK_TIME
    if (prof_replay_tid != plat_thread_id()) perf_replay_thread();   // the profiler follows the thread replaying now
    perf_rec_waits = rec_wait_totals;
    VkExec *ex = &X;
    if (!x_ready) state_reset(ex), vk_dump_init(), x_ready = 1;
    if (RW.n < 0) rw_init();
    if (Q.drained != Q.out) rt_fatal("vk: a parallel group outlived its submission");
    // No recording worker can still consume a cached pool generation here. Pressure reclamation never waits
    // for a GPU fence and excludes the incoming recording, including continuations sharing its final fence.
    vk_exec_texture_publications_reclaim(fence);
    sweep_recording_pins(ex);
    ex->cb = cb;
    ex->fence = fence;
    VkTextureRegistrationScope *local_registration = n && !incoming_registration_scope ?
        vk_texture_registration_record_begin(NULL, 0) : NULL;
    ex->registration_scope = local_registration ? local_registration : incoming_registration_scope;
    ex->resource_next = ex->resource_op = 0;
    ex->recorded = 0;
    ex->in_pass = 0;
    ex->mark = NULL;
    ex->mark_passes = ex->mark_dispatches = 0;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->rt_resolved = 0;   // images may have changed since the last command buffer
    ex->nviews = 0;
    ex->barrier_at = -1;
    ex->nfbl = 0;   // the previous command buffer ended with a full barrier: every image is in GENERAL
    vk_exec_frame.submits++;
    vk_exec_submit_seq++;
    split_draws = part_start = 0;
    split_at = split_n < 0 ? 0 : split_n ? split_n : INT_MAX;   // 0 until maybe_split reads the settings
    part_t0 = kern_now_ns();
    x_cur = ex;
    __atomic_store_n(&recording_failed, 0, __ATOMIC_RELEASE);
    vk_exec_draw_update_credit_adopt(ex, &bundle->scratch);
    int evo = evo_enabled() && gfx_submit_queue;
    int slot = evo ? evq_slot(gfx_submit_queue) : -1;
    if (evo && slot < 0) evo = 0;
    vk_exec_evt('|', (const void *)gfx_submit_queue, (uint32_t)n);
    if (readiness && readiness->sequence != vk_exec_submit_seq)
        rt_fatal("vk: uniform readiness plan belongs to another submission");
    record_readiness = readiness && readiness->frozen ? readiness : NULL;
    if (!record_readiness) RP_CALL(RP_EVENT_SYNC, vk_exec_event_sync(1));
    RP_END(rp_setup);
    RP_VALUE(RPK_REPLAY_SEQ, vk_exec_submit_seq);
    int sync_held = evo_pending && held_sync();
    int synced = n || (sync_held && (record_readiness ? readiness->initial_resume : vk_exec_held_ready_locked()));
    if (synced) vk_mem_mirror_begin();
    // Even a scene with no snapshot must publish its own write fence. If held synchronization is disabled or
    // there is no eligible work yet, the caller cancels the unused reservation after recording.
    VkUsnapCapture *no_copy = NULL;
    VkResult snapshot = vk_usnap_begin(synced ? &bundle->snapshot : &no_copy, n, fence);
    int stopped = snapshot != VK_SUCCESS;
    if (stopped) {
        vk_device_lost_check(snapshot, "starting the uniform snapshot");
        if (vk_device_is_lost() || vk_sync_stopping()) cancel_recording();
        else
            vk_exec_required_graphics_unavailable(ex, OP_BEGIN, "uniform snapshot could not enter recording");
    }
    for (int i = 0; i < n; i++)
        if (!handles[i]) rt_fatal("vk: command handle %d of a submission is null", i);
    if (!stopped && evo_pending && (!sync_held || synced)) stopped = !evo_resume(ex);
    if (!stopped && evo && evq[slot].head) {   // the queue waits: this submission lines up behind it
        if (n) {
            Cont *follower = cont_new(readiness ? readiness->parked_commands : NULL, handles, n, 0, NULL);
            cont_registration_defer(follower);
            cont_push(slot, follower);
            evo_held++;
            vk_exec_evt('p', NULL, 0);
        }
    } else if (!stopped) {
        evo_queue = evo ? gfx_submit_queue : 0;
        AH.handles = handles, AH.n = n, AH.live = 1;
        RP_SCOPE(rp_walk, RP_COMMAND_WALK);
        for (int i = 0; i < n; i++) {
            AH.i = i;
            const op_hdr *stop = walk(ex, nvn_op_first(handles[i]), 0, handles[i]);
            if (stop == &walk_stopped) { stopped = 1; break; }
            if (!stop) continue;
            Cont *c = cont_new(readiness ? readiness->parked_commands : NULL, handles, n, i, stop);
            ParkedUbo *parked = readiness ? readiness->parked_ubo : NULL;
            if (!parked || parked->refs != 1) rt_fatal("vk: first held wait has no admitted uniform storage");
            memcpy(parked->bindings, ex->ubo, sizeof parked->bindings);
            parked->refs++; // The continuation keeps its own reference after readiness cleanup.
            c->parked_ubo = parked;
            c->parked_rt = ex->rt;
            c->parked_tex_pool = ex->tex_pool;
            c->parked_tex_pool_count = ex->tex_pool_count;
            cont_push(slot, c);
            evo_held++;
            vk_exec_evt('h', NULL, 0);
            break;
        }
        AH.live = 0;
        evo_queue = 0;
    }
    RP_CALL(RP_GROUP_DRAIN, group_drain(ex, PW_END));
    stopped |= vk_exec_recording_failed();
    if (!stopped && __atomic_load_n(&probe_pass_record, __ATOMIC_ACQUIRE)) {
        void *context = probe_pass_context;
        void (*record)(VkExec *, void *) = __atomic_exchange_n(&probe_pass_record, NULL, __ATOMIC_ACQ_REL);
        record(ex, context);
        stopped |= vk_exec_recording_failed();
    }
    RP_SCOPE(rp_teardown, RP_TEARDOWN);
    if (!stopped) {
        vk_exec_pass_end(ex);
        stopped = vk_exec_recording_failed() || !settle_mesh_capacity(ex);
    }
    if (stopped) resource_targets_forget(ex);
    if (RR.on > 0) runs_break(RB_SUBMIT);
    vk_exec_draw_prepared_stats_flush(ex);
    vk_exec_draw_update_history_end(ex->fence);
    x_cur = NULL;
    ex->registration_scope = NULL;
    if (local_registration) {
        vk_texture_registration_record_end(local_registration, 0);
        vk_texture_registration_scope_end(local_registration);
    }
    record_readiness = NULL;
    if (!readiness) vk_exec_event_sync(0);
    if (!stopped && ex->recorded) vk_exec_barrier(ex), vk_exec_pool_marks(ex);   // later submissions see this one's writes
    // NX2NAVI_WB_SPLIT: the write-backs in a part of their own, after one that signals the executed timeline
    // (vk_sync.c exec_tl): the wait after a frame's first submission then ends when the submission's passes are
    // through, not its write-backs. Off by default: vk_sync.c says what else that wait then leaves out.
    if (!stopped && synced && ex->recorded && vk_sync_wb_split() && vk_mem_mirror_wb_pending() && vk_exec_pool_split(ex->fence)) {
        ex->cb = vk_sync_split_exec();
        ex->dirty = VK_EXEC_DIRTY_ALL;
        vk_exec_draw_new_cb();
    }
    if (!stopped && synced) {
        const int writebacks = vk_mem_mirror_end(ex->cb, fence);
        if (writebacks < 0) {
            vk_exec_required_graphics_unavailable(ex, 0, "ordered mesh word writeback lost its exact tail");
            stopped = 1;
        } else ex->recorded += writebacks;
    }
    vk_usnap_end(fence);
    views_release(ex, fence);
    ex->cb = VK_NULL_HANDLE;
    if (stopped) return VK_EXEC_RECORD_STOPPED;
    if (n) n_subs++;
    if ((rt_trace & RT_TRACE_PERF) && n_subs >= 240) {
        rt_log("perf: replay command buffer parts per submission %.2f (%.2f first parts sent while the GPU was idle, %.2f "
               "splits inside a pass); event order: held %llu, resumed %llu (%llu without their signal), switches %llu, "
               "held now %d; waits a CPU signal met only after the submission's copies were taken: %llu",
               (double)(n_subs + n_parts) / (double)n_subs, (double)n_idle_parts / (double)n_subs,
               (double)n_pass_splits / (double)n_subs, (unsigned long long)evo_held, (unsigned long long)evo_resumed,
               (unsigned long long)evo_forced, (unsigned long long)evo_switches, evo_pending,
               (unsigned long long)vk_exec_event_late_take());
        uint64_t xw, xe, xl, xm, le, lc, lo;
        vk_sync_exec_stats(&xw, &xe, &xl, &xm);
        vk_exec_late_stats(&le, &lc, &lo);
        if (xw)   // NX2NAVI_WB_SPLIT: what the wait on the executed timeline left behind
            rt_log("perf: waits after a submit on the executed timeline: %llu, %llu of them back before the completion "
                   "thread had run the side effects of the submissions up to theirs (through %.3f ms later on average, "
                   "%.3f at most); written into guest memory in that time: %llu event values, %llu counter reports, "
                   "%llu occlusion counts",
                   (unsigned long long)xw, (unsigned long long)xe, xe ? xl / 1e6 / (double)xe : 0.0, xm / 1e6,
                   (unsigned long long)le, (unsigned long long)lc, (unsigned long long)lo);
        char cbs[1600];
        vk_sync_cb_stats(cbs, sizeof cbs);
        rt_log("perf: completion thread since the last line, by callback (distance from vk_completion_add):%s", cbs);
        for (unsigned i = 0; i < EXEC_PATHS; i++) {
            static const char *const names[] = {"main", "workers", "front"};
            ExecPlanStats *s = &exec_totals[i];
            uint64_t blocks = __atomic_exchange_n(&s->blocks, 0, __ATOMIC_RELAXED);
            uint64_t state_ops = __atomic_exchange_n(&s->state_ops, 0, __ATOMIC_RELAXED);
            uint64_t direct = __atomic_exchange_n(&s->direct, 0, __ATOMIC_RELAXED);
            uint64_t reference = __atomic_exchange_n(&s->reference, 0, __ATOMIC_RELAXED);
            uint64_t front_draws = __atomic_exchange_n(&s->front_draws, 0, __ATOMIC_RELAXED);
            if (blocks || direct || reference)
                rt_log("execution plans: %s: %llu state blocks cover %llu original operations; %llu typed operations; "
                       "%llu reference operations; %llu draws covered by front summaries", names[i],
                       (unsigned long long)blocks, (unsigned long long)state_ops, (unsigned long long)direct,
                       (unsigned long long)reference, (unsigned long long)front_draws);
        }
        if (RW.n > 0) {
            uint64_t turns = __atomic_exchange_n(&RW.turns, 0, __ATOMIC_RELAXED);
            uint64_t turn_ns = __atomic_exchange_n(&RW.turn_ns, 0, __ATOMIC_RELAXED);
            uint64_t redos = __atomic_exchange_n(&RW.redos, 0, __ATOMIC_RELAXED);
            rt_log("perf: replay workers: per submission %.1f groups of %.0f draws in %.1f chunks, %.2f ms each; %.0f draws "
                   "in smaller groups; waits for an earlier chunk %.1f (%.3f ms), ops replayed again in turn %.2f",
                   (double)RW.groups / (double)n_subs, RW.groups ? (double)RW.group_draws / (double)RW.groups : 0.0,
                   RW.groups ? (double)RW.chunks / (double)RW.groups : 0.0,
                   RW.groups ? (double)RW.wait_ns / 1e6 / (double)RW.groups : 0.0, (double)RW.serial_draws / (double)n_subs,
                   (double)turns / (double)n_subs, (double)turn_ns / 1e6 / (double)n_subs, (double)redos / (double)n_subs);
            if (RW.planned_groups || RW.plan_fallbacks)
                rt_log("perf: command plans: per submission %.1f group searches covering %.0f draws, %.1f fallback walks",
                       (double)RW.planned_groups / (double)n_subs, (double)RW.planned_draws / (double)n_subs,
                       (double)RW.plan_fallbacks / (double)n_subs);
            if (RW.front_planned_draws)
                rt_log("perf: command front pass: %.0f draws per submission consumed through structural runs",
                       (double)RW.front_planned_draws / (double)n_subs);
            char e[512];
            int o = 0;
            for (int k = 0; k < GE_N; k++)
                o += snprintf(e + o, sizeof e - (size_t)o, "%s %s %.0f", k ? "," : "", ge_name[k],
                              (double)RW.end_draws[k] / (double)n_subs);
            o += snprintf(e + o, sizeof e - (size_t)o, "; of them in groups too small to split:");
            for (int k = 0; k < GE_N; k++)
                o += snprintf(e + o, sizeof e - (size_t)o, "%s %s %.0f", k ? "," : "", ge_name[k],
                              (double)RW.small_end[k] / (double)n_subs);
            rt_log("perf: replay workers: draws per submission by what ended their group:%s", e);
            o = 0;
            for (int k = 0; k < 6; k++) {   // the ops outside groups that ended the most, by their groups' draws
                int best = 0;
                for (int op = 1; op < OP_COUNT_; op++)
                    if (RW.run_end[op] > RW.run_end[best]) best = op;
                if (!RW.run_end[best]) break;
                o += snprintf(e + o, sizeof e - (size_t)o, "%s %s %.0f", k ? "," : "", nvn_op_name((uint32_t)best),
                              (double)RW.run_end[best] / (double)n_subs);
                RW.run_end[best] = 0;
            }
            memset(RW.run_end, 0, sizeof RW.run_end);
            if (o) rt_log("perf: replay workers: draws per submission of groups a run end ended, by the op:%s", e);
            memset(RW.end_draws, 0, sizeof RW.end_draws), memset(RW.small_end, 0, sizeof RW.small_end);
            double ns = 1e6 * (double)n_subs;
            rt_log("perf: replay workers: the submitting thread per submission: outside groups draws %.0f (%.3f ms), mesh "
                   "draws %.0f (%.3f ms), dispatches %.0f (%.3f ms), other ops %.0f (%.3f ms); in groups the front pass "
                   "%.3f ms, its own chunks %.3f ms, waiting for the others %.3f ms",
                   (double)RW.ser_n[SER_DRAW] / (double)n_subs, (double)RW.ser_ns[SER_DRAW] / ns,
                   (double)RW.ser_n[SER_MESH] / (double)n_subs, (double)RW.ser_ns[SER_MESH] / ns,
                   (double)RW.ser_n[SER_DISPATCH] / (double)n_subs, (double)RW.ser_ns[SER_DISPATCH] / ns,
                   (double)RW.ser_n[SER_OTHER] / (double)n_subs, (double)RW.ser_ns[SER_OTHER] / ns,
                   (double)RW.front_ns / ns, (double)RW.own_ns / ns, (double)RW.spin_ns / ns);
            memset(RW.ser_n, 0, sizeof RW.ser_n), memset(RW.ser_ns, 0, sizeof RW.ser_ns);
            RW.front_ns = RW.own_ns = RW.spin_ns = 0;
            rt_log("perf: replay workers: the chunks took %.3f ms per submission, every thread's",
                   (double)__atomic_exchange_n(&RW.chunk_ns, 0, __ATOMIC_RELAXED) / ns);
            if (RW.pipe) {
                o = 0;
                for (int k = 0; k < PW_N; k++)
                    o += snprintf(e + o, sizeof e - (size_t)o, "%s %s %.2f (%.3f ms)", k ? "," : "", pw_name[k],
                                  (double)RW.pw_n[k] / (double)n_subs, (double)RW.pw_ns[k] / ns);
                rt_log("perf: replay workers: past a group the submitting thread went on %.3f ms per submission until "
                       "it was done; waits for it per submission:%s", (double)RW.behind_ns / ns, e);
                o = 0;
                for (int k = 0; k < 5; k++) {   // the ops that waited the most often
                    int best = 0;
                    for (int op = 1; op < OP_COUNT_; op++)
                        if (RW.pd_op[op] > RW.pd_op[best]) best = op;
                    if (!RW.pd_op[best]) break;
                    o += snprintf(e + o, sizeof e - (size_t)o, "%s %s %.2f", k ? "," : "", nvn_op_name((uint32_t)best),
                                  (double)RW.pd_op[best] / (double)n_subs);
                    RW.pd_op[best] = 0;
                }
                if (o) rt_log("perf: replay workers: ops that waited for a group, per submission:%s", e);
                uint64_t td[3];
                vk_texture_deferred_stats(td);
                rt_log("perf: replay workers: texture lookups put off until their turn, per submission: no entry "
                       "%.2f, a retired one %.2f, a released one %.2f", (double)td[0] / (double)n_subs,
                       (double)td[1] / (double)n_subs, (double)td[2] / (double)n_subs);
                memset(RW.pw_n, 0, sizeof RW.pw_n), memset(RW.pw_ns, 0, sizeof RW.pw_ns);
                memset(RW.pd_op, 0, sizeof RW.pd_op);
                RW.behind_ns = 0;
            }
        }
        RW.groups = RW.group_draws = RW.chunks = RW.serial_draws = RW.wait_ns = 0;
        RW.planned_groups = RW.planned_draws = RW.plan_fallbacks = 0;
        RW.front_planned_draws = 0;
        evo_held = evo_resumed = evo_forced = evo_switches = 0;
        n_subs = n_parts = n_idle_parts = n_pass_splits = 0;
    }
    rec_spent_ns += kern_now_ns() - in_ns;
    if (rec_spent_ns >= REC_CLAIM_NS) rec_claim();
    RP_END(rp_teardown);
    RP_VALUE(RPK_HELD_AFTER, evo_pending);
    RP_VALUE(RPK_RECORDED_LAST, ex->recorded - part_start);
    gfx_tl_note('r', 0, (uint64_t)(ex->recorded - part_start), 0);
    return ex->recorded - part_start;   // what the last part holds
}

void vk_exec_resource_handoff_before(uint64_t fence, uint64_t keep_open_ordinal) {
    RcRegistry *registry = rc_registry_runtime();
    if (X.fence != fence || !X.resource_op || X.resource_next != X.resource_op || Q.drained != Q.out ||
        keep_open_ordinal < 2 || keep_open_ordinal != X.resource_op * 2 ||
        !rc_registry_position_open(registry, (RcRegistryPosition){fence, keep_open_ordinal + 1}) ||
        rc_registry_handoff_through(registry, (RcRegistryPosition){fence, keep_open_ordinal - 1}) != RC_REGISTRY_OK)
        rt_fatal("vk: a native phase could not preserve its guest operation boundary");
}

VkResult vk_exec_pointer_phase_prepare(VkExecPointerReadback **out) {
    if (*out) return VK_SUCCESS;
    VkExecPointerReadback *readback = calloc(1, sizeof *readback);
    if (!readback) return VK_ERROR_OUT_OF_HOST_MEMORY;
    readback->owners = 1;
    RcAdmissionVector peak;
    VkResult staged = vk_staging_prepare_plan(sizeof(uint64_t), &readback->stage, &peak);
    if (staged == VK_SUCCESS) staged = vk_staging_prepare_allocate(&readback->stage, NULL, 1);
    if (staged != VK_SUCCESS) device_pointer_readback_drop(readback);
    else *out = readback;
    return staged;
}

void vk_exec_pointer_phase_cancel(VkExecPointerReadback **readback) {
    if (!*readback) return;
    device_pointer_readback_drop(*readback);
    *readback = NULL;
}

// Only the in-order phase split below can move this unopened, post-barrier receipt to a new recorder.
static int graphics_pass_adopt_task_tail(VkExec *ex, VkExecGraphicsPass *pass, VkCommandBuffer prefix,
                                         VkCommandBuffer tail) {
    if (!tail || tail == prefix || ex->cb != tail || pass->barrier_cb != prefix ||
        pass->hook_cb || pass->shape.in_pass || ex->in_pass) return 0;
    pass->barrier_cb = tail;
    if (vk_exec_graphics_pass_current(ex, pass)) return 1;
    pass->barrier_cb = prefix;
    return 0;
}

VkSyncProgressResult vk_exec_read_device_pointer_phase(VkExec *ex, VkExecPointerReadback **staging,
    const VkMemSpan *span, VkMemSpanCapture *capture, const RcInterval *input, uint32_t input_count,
    const RcRegistryBufferPhaseExtent *task, uint32_t count, RcRegistryPosition read, uint64_t *value,
    VkExecGraphicsPass *pass) {
    if (vk_sync_stopping() || vk_device_is_lost()) return VK_SYNC_PROGRESS_STOPPED;
    if (!ex || x_cur != ex || rw_chunk || !staging || !*staging || !span || !capture || !span->alloc ||
        !input || !input_count || !task || !count || !value || span->cls != VK_MCLS_DEVICE || (span->offset & 3) ||
        !ex->resource_op || ex->resource_next != ex->resource_op || ex->in_pass || ex->recorded == part_start)
        return VK_SYNC_PROGRESS_UNAVAILABLE;
    group_drain(ex, PW_SPLIT);
    if (vk_sync_stopping() || vk_device_is_lost() || vk_exec_recording_failed()) return VK_SYNC_PROGRESS_STOPPED;
    const VkCommandBuffer prefix = ex->cb;
    if (!pass || !prefix || pass->barrier_cb != prefix || pass->hook_cb || pass->shape.in_pass ||
        !vk_exec_graphics_pass_current(ex, pass) || !device_pointer_current(span, capture, input, input_count, read) ||
        !vk_mem_span_record_input(capture, span)) return VK_SYNC_PROGRESS_UNAVAILABLE;
    VkExecPointerReadback *readback = *staging;
    *staging = NULL;
    device_pointer_record_copy(ex, span, capture, readback);
    VkCommandBuffer tail = VK_NULL_HANDLE;
    VkSyncProgressResult result = vk_sync_progress_split_before(ex, &tail, read.ordinal);
    if (tail) progress_adopt_tail(ex, tail);
    if (result == VK_SYNC_PROGRESS_COMPLETE || result == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT) {
        RcRegistryBufferPhaseOutcome outcome = result == VK_SYNC_PROGRESS_COMPLETE ?
            RC_REGISTRY_BUFFER_PHASE_COMPLETED : RC_REGISTRY_BUFFER_PHASE_POSSIBLY_ISSUED;
        if (rc_registry_buffer_phase_note(rc_registry_runtime(), task, count, read, outcome) != RC_REGISTRY_OK)
            rt_fatal("vk: the submitted task phase lost its exact buffer writer");
    }
    if (result == VK_SYNC_PROGRESS_COMPLETE) {
        VkResult visible = vmaInvalidateAllocation(vk_ctx()->vma, readback->stage.allocation, 0, sizeof *value);
        if (visible != VK_SUCCESS) vk_device_lost_check(visible, "invalidating completed task pointer readback");
        // Keep the exact target loans/feedback shape. Activation records fresh attachment barriers and LOAD
        // on this tail; no pass was started there as part of this pointer copy.
        if (visible == VK_SUCCESS && device_pointer_current(span, capture, input, input_count, read) &&
            rc_registry_read_device_buffer_phase(rc_registry_runtime(), input, input_count, task, count, read) == RC_REGISTRY_OK &&
            graphics_pass_adopt_task_tail(ex, pass, prefix, tail))
            memcpy(value, readback->stage.map, sizeof *value);
        else result = VK_SYNC_PROGRESS_UNAVAILABLE;
        // The private semaphore is the staging lifetime proof even if the operation's tail cannot issue.
        vk_staging_prepare_cancel(&readback->stage);
    }
    device_pointer_readback_drop(readback);
    return result;
}

void vk_exec_resource_handoff(uint64_t fence) {
    if (X.fence != fence || !X.resource_next) return; // non-replay owned command buffer
    if (Q.drained != Q.out) rt_fatal("vk: resource handoff preceded recording workers");
    if (rc_registry_handoff_through(rc_registry_runtime(),
        (RcRegistryPosition){fence, X.resource_next * 2 + 1}) != RC_REGISTRY_OK)
        rt_fatal("vk: resource recording prefix could not be closed before GPU handoff");
}

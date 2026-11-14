// Internal header of the NVN front end, shared by nvn_api_core.c and the other
// slices (textures, samplers, programs, windows, state objects, command buffers).
// Contains code Copyright (c) 2026 jhaizhou (MIT; see THIRD_PARTY_NOTICES.md).
#pragma once
#include <stdint.h>

#include "gfx.h"
#include "host.h"
#include "nvn_obj.h"
#include "nvn_exec_plan.h"

// NVN_API(nvnX) { body }: the strong nvnp_nvnX(Cpu *c); counts the call in gfx_counters, NX2NAVI_TRACE=gfx logs the first.
#define NVN_API(n)                                                                                                   \
    static void nvn_impl_##n(Cpu *c);                                                                                \
    void nvnp_##n(Cpu *c) {                                                                                          \
        static uint8_t seen;                                                                                         \
        nvn_called(&seen, #n);                                                                                       \
        nvn_impl_##n(c);                                                                                             \
    }                                                                                                                \
    static void nvn_impl_##n(Cpu *c)
void nvn_called(uint8_t *seen, const char *name);          // the counting prologue of NVN_API
#define NVN_LOG(...) do { if (rt_trace & RT_TRACE_GFX) rt_log("nvn: " __VA_ARGS__); } while (0)   // NX2NAVI_TRACE=gfx
// A wait for the GPU, the display or another thread gives the virtual core up until the end of the enclosing scope.
#define NVN_BLOCKING() __attribute__((cleanup(kern_core_back))) int nvn_away_ = kern_core_away()
static inline int nvn_arg_bool(const Cpu *c, int i) { return (int)(arg_x(c, i) & 0xff); }   // NVNboolean: low 8 bits only
// 1 when argument 0, the object of a getter, is NULL: the call is answered with 0 instead of reading the object at
// address 0 plus a field's offset. The game's scene-loading thread makes a few such calls while an area or Kakariko's
// time of day changes (a memory pool's flags, a texture builder's storage size). The first calls are logged with
// their caller (nvn_api_core.c).
int nvn_no_object(Cpu *c, const char *what);

#define NVN_POOL_MAGIC MAGIC('P', 'O', 'O', 'L')           // pool_t.magic of an initialized pool
#define NVN_VPOOL_PAGE 0x10000ull                          // NVN virtual mapping granularity (64 KiB)
static inline int nvn_gpu_only(uint64_t a) { return a - RT_GPU_VA_BASE < RT_GPU_VA_SIZE; }   // address in the GPU window
// A CPU-invisible physical or virtual pool lives in the GPU window: mem is its GPU address (never mapped on the host),
// game_mem the game's storage (never touched), span the reserved window bytes, host NULL. Every other pool: mem is the
// game's storage, the GPU address equals the guest address.

typedef struct { NVNdevice *dev; pool_t *pool; int64_t off; uint64_t size; } buf_builder_t;   FITS(buf_builder_t, NVNbufferBuilder);
typedef struct { uint32_t magic; pool_t *pool; int64_t off; uint64_t size; } buf_t;           FITS(buf_t, NVNbuffer);
typedef struct {                                           // NVNmappingRequest as main passes it (0x28 bytes)
    const NVNmemoryPool *physical_pool;                    // NULL unmaps
    int64_t physical_offset, virtual_offset;
    uint64_t size;
    int32_t storage_class;
    uint32_t padding;
} nvn_mapping_t;
_Static_assert(sizeof(nvn_mapping_t) == 0x28, "NVNmappingRequest stride");

const pool_t *nvn_pool_find(uint64_t addr, uint64_t len);  // GPU-visible pool holding [addr, addr+len], NULL if none
uint64_t nvn_prog_made_ns(uint64_t serial);                    // NX2NAVI_PROG_LEAD: when the program was made, 0 unknown
uint64_t nvn_new_serial(void);                             // next tex_t / prog_t serial, from 1
// NX2NAVI_TRACE=perf: a guest thread enters (enter 1) or leaves (0) QueueSubmitCommands (present 0) or
// QueuePresentTexture (1) on queue q: the first call of each thread, kind and queue gets a line, and so does a call
// that enters while another of these calls runs (nvn_api_core.c nvn_call_note).
void nvn_call_note(int present, uintptr_t q, int enter);
uint64_t nvn_fence_last(void);                             // fence value of the newest submit (0 before the first)
int nvn_depth_minus_w(void);                               // 1 after DeviceSetDepthMode(NEAR_IS_MINUS_W), else 0
extern int nvn_submits_in_frame;                           // nvnQueueSubmitCommands calls since the last present (NX2NAVI_SUBMIT_FINISH)
void nvn_guest_written(uint64_t addr, uint64_t size);      // the host or the game wrote pool storage: back end re-reads
// NX2NAVI_TRACE=perf: MemoryPoolFlushMappedRange [0] and BufferFlushMappedRange [1] calls [0] and bytes [1] by the pool's
// CPU class (0 CPU_CACHED, 1 CPU_UNCACHED, 2 other) since the previous call.
void nvn_flush_stats(uint64_t out[2][3][2]);
void nvn_submit_prepare(const uint64_t *handles, int n);   // weak no-op; the command-buffer slice reads deferred binds
// Asynchronous replay: as nvn_submit_prepare, writing the handles to replay to out (a clone where a still queued
// submission uses the same recording with other deferred binds)
void nvn_submit_prepare_async(const uint64_t *handles, uint64_t *out, int n);
int gfx_replay_async(void);                                // 1: the back end replays submissions on its own thread
uint64_t gfx_replayed_submits(void);                       // newest submit number (nvn_submit_count) replayed
void gfx_event_signalling(const void *value, uint32_t to); // nvnEventSignal is about to write `to` into the event's value
#define NVN_HANDLE_TEX(h) ((int)((h) & 0xfffff))           // combined texture handle = texture id | sampler id << 20
#define NVN_HANDLE_SAMP(h) ((int)(((h) >> 20) & 0xfff))
extern GfxStats gfx_counters;                              // behind gfx_stats(); add with __atomic_fetch_add. frames:
                                                           // the back end's present; nvn_calls/nvn_distinct: NVN_API

// ---- nvn_api_obj.c: textures, samplers, programs, windows
#define NVN_TEX_MAGIC MAGIC('T', 'E', 'X', 0)              // tex_t.magic of an initialized texture
#define NVN_SAMP_MAGIC MAGIC('S', 'M', 'P', 0)             // samp_t.magic
#define NVN_PROG_MAGIC MAGIC('P', 'R', 'O', 'G')           // prog_t.magic
#define NVN_WIN_MAGIC MAGIC('W', 'I', 'N', 'D')            // win_t.magic
// The live texture whose storage starts at mem, copied into *out; 0 if none. The game re-initializes transient textures
// at the same pool memory without registering them again, so a registered serial that was finalized since resolves here.
int nvn_tex_live_at(const void *mem, tex_t *out);
// Byte offset of a view's first level and layer inside the texture storage (nvnTextureGetViewOffset); v may be NULL.
uint64_t nvn_tex_view_offset(const tex_t *t, const view_t *v);

// ---- nvn_api_cmd.c, nvn_ext.c: command buffers and their replay contract
// A command handle (what QueueSubmitCommands receives) points at a host copy of one recording: a flat sequence of
// 8-byte-aligned records {op_hdr, payload}, OP_BEGIN first and OP_END last. Called recordings are inlined, so a copy
// holds no OP_CALL or OP_JUMP. Texture pointers in payloads (p_set_rt, p_copy_b2t, p_copy_t2t, p_clear_tex) are
// immutable host copies ("pins"), one per tex_t.serial, that live while a recording copy names them and a while after
// (nvn_pins_sweep). Deferred binds (p_bind_handles.src != 0) hold the
// handles read at submit (nvn_submit_prepare). Payloads of the ops nvn_obj.h does not define:
typedef struct { volatile uint32_t *val; uint32_t mode, value; } p_signal_event;   // OP_SIGNAL_EVENT; mode: NVNeventSignalMode
typedef struct { volatile uint32_t *val; uint32_t mode, value; } p_wait_event;     // OP_WAIT_EVENT; mode: NVNeventWaitMode
typedef struct { uint64_t type, addr; } p_report_counter;  // OP_REPORT_COUNTER: NVNcounterType, NVNcounterData address
typedef struct { uint64_t type; } p_reset_counter;         // OP_RESET_COUNTER
typedef struct { uint64_t src, dst, size; } p_copy_b2b;    // OP_COPY_B2B: GPU addresses
typedef struct { uint64_t dst, size; uint8_t data[]; } p_update_ub;   // OP_UPDATE_UB: size bytes written at dst
typedef struct { int32_t bits, pad; } p_barrier;           // OP_BARRIER: NVNbarrierBits
typedef struct { uint64_t target; } p_jump;                // OP_JUMP / OP_CALL: next record / called recording
// OP_DRAW_IDX_IND: DrawElementsIndirect; ind = GPU address of {count, instances, first index, base vertex, base
// instance} (VkDrawIndexedIndirectCommand's layout), read by the GPU when the draw runs
typedef struct { int32_t prim, itype; uint64_t ibuf, ind; } p_draw_idx_ind;
typedef struct { uint64_t ind, caller; } p_dispatch_ind;   // OP_DISPATCH_IND: ind = GPU address of the 3 group counts
// OP_CLEAR_BUF: size bytes at addr filled with the 32-bit value in GPU order (ClearBuffer; ReportValue is a 4-byte one)
typedef struct { uint64_t addr, size; uint32_t value, pad; } p_clear_buf;
_Static_assert(sizeof(p_signal_event) == 16 && sizeof(p_report_counter) == 16 && sizeof(p_copy_b2b) == 24 &&
               sizeof(p_update_ub) == 16, "op payload layouts of hle_nvn.c");
static inline const op_hdr *nvn_op_first(uint64_t handle) { return (const op_hdr *)(uintptr_t)handle; }   // OP_BEGIN
static inline const op_hdr *nvn_op_next(const op_hdr *h) { return (const op_hdr *)(const void *)((const uint8_t *)h + h->size); }
static inline const void *nvn_op_payload(const op_hdr *h) { return h + 1; }
int nvn_handle_live(uint64_t handle);                      // 1 while the handle's copy exists (its control memory was
                                                           // not added again)
uint64_t nvn_handle_bytes(uint64_t handle);                // bytes of the copy, OP_BEGIN to OP_END; 0 if not live
// Prepared copies, including queued retired handles and submission clones: the OP_BIND_UBO offsets generated at
// EndRecording. Returned storage has the command copy's lifetime; the caller must already hold its submit/replay
// lifetime. Returns 0 for unindexed or raw backend-probe handles, whose commands still need walking.
int nvn_record_uniform_offsets(uint64_t handle, const size_t **offsets, size_t *count);
// A recording's structural plan contains no resolved GPU objects or captured memory contents. Offsets survive
// submission cloning; the arrays share the command copy's retirement lifetime. A RUN span has only ordinary state
// operations and draws. Target, clear, mesh and update boundaries remain explicit for the replay's dynamic rules.
enum { NVN_PLAN_SERIAL, NVN_PLAN_RUN, NVN_PLAN_TARGET, NVN_PLAN_CLEAR, NVN_PLAN_MESH, NVN_PLAN_UPDATE };
typedef struct {
    size_t begin, end;
    uint32_t ops, draws, kind, draw_stride;   // nonzero: only ordinary draw records, each this many bytes
} NvnRecordSpan;
typedef struct {
    const uint8_t *commands;
    size_t bytes;
    const size_t *capture_offsets;
    size_t ncapture;
    const NvnRecordSpan *spans;
    size_t nspans;
    const NvnExecStep *exec;
    size_t nexec;
    const NvnFrontStep *front;
    size_t nfront;
    const uint8_t *state_data;
    size_t state_bytes;
} NvnRecordPlan;
int nvn_record_plan(uint64_t handle, NvnRecordPlan *out);
// Return the exact instruction beginning at h. An interior continuation cannot apply a complete state block; it
// follows the reference path until the next instruction boundary. A monotone cursor avoids repeated binary search.
static inline const NvnExecStep *nvn_record_exec_at(const NvnRecordPlan *plan, const op_hdr *h, size_t *cursor) {
    if (!plan || !plan->nexec) return NULL;
    uintptr_t base = (uintptr_t)plan->commands, address = (uintptr_t)h;
    if (address < base || address - base >= plan->bytes) return NULL;
    size_t offset = (size_t)(address - base), i = *cursor;
    if (i == SIZE_MAX || i >= plan->nexec || plan->exec[i].begin > offset) {
        size_t lo = 0, hi = plan->nexec;
        while (lo < hi) {
            size_t m = lo + (hi - lo) / 2;
            if (plan->exec[m].end <= offset) lo = m + 1;
            else hi = m;
        }
        i = lo;
    } else while (i < plan->nexec && plan->exec[i].end <= offset) i++;
    *cursor = i;
    return i < plan->nexec && plan->exec[i].begin == offset ? &plan->exec[i] : NULL;
}
static inline const NvnExecStep *nvn_record_exec_slice(const NvnRecordPlan *plan, const op_hdr *h,
                                                     const op_hdr *limit, uint32_t ops, size_t *cursor) {
    const NvnExecStep *step = nvn_record_exec_at(plan, h, cursor);
    if (!step || step->ops > ops) return NULL;
    if (limit) {
        uintptr_t base = (uintptr_t)plan->commands, stop = (uintptr_t)limit;
        if (stop < base || step->end > stop - base) return NULL;
    }
    return step;
}
static inline const NvnFrontStep *nvn_record_front_at(const NvnRecordPlan *plan, const op_hdr *h, size_t *cursor) {
    if (!plan || !plan->nfront) return NULL;
    uintptr_t base = (uintptr_t)plan->commands, address = (uintptr_t)h;
    if (address < base || address - base >= plan->bytes) return NULL;
    size_t offset = (size_t)(address - base), i = *cursor;
    if (i == SIZE_MAX || i >= plan->nfront || plan->front[i].begin > offset) {
        size_t lo = 0, hi = plan->nfront;
        while (lo < hi) {
            size_t m = lo + (hi - lo) / 2;
            if (plan->front[m].end <= offset) lo = m + 1;
            else hi = m;
        }
        i = lo;
    } else while (i < plan->nfront && plan->front[i].end <= offset) i++;
    *cursor = i;
    return i < plan->nfront && plan->front[i].begin == offset ? &plan->front[i] : NULL;
}
static inline const NvnFrontStep *nvn_record_front_slice(const NvnRecordPlan *plan, const op_hdr *h,
                                                       const op_hdr *limit, uint32_t draws, size_t *cursor) {
    const NvnFrontStep *step = nvn_record_front_at(plan, h, cursor);
    if (!step || step->draws > draws) return NULL;
    uintptr_t base = (uintptr_t)plan->commands, stop = (uintptr_t)limit;
    return stop >= base && step->end <= stop - base ? step : NULL;
}
// A monotone front-pass cursor over an immutable plan. SIZE_MAX initializes it at the first queried address.
// Returns only a homogeneous ordinary-draw run, clipped to limit; the recording worker still executes each draw.
static inline uint32_t nvn_record_draw_run(const NvnRecordPlan *plan, const op_hdr *h, const op_hdr *limit,
                                         size_t *cursor, uint32_t *stride) {
    if (!plan || !plan->nspans) return 0;
    uintptr_t base = (uintptr_t)plan->commands, address = (uintptr_t)h, stop = (uintptr_t)limit;
    if (address < base || address - base >= plan->bytes || stop < address || stop - base > plan->bytes) return 0;
    size_t offset = (size_t)(address - base), i = *cursor;
    if (i == SIZE_MAX) {
        size_t lo = 0, hi = plan->nspans;
        while (lo < hi) {
            size_t m = lo + (hi - lo) / 2;
            if (plan->spans[m].end <= offset) lo = m + 1;
            else hi = m;
        }
        i = lo;
    } else while (i < plan->nspans && plan->spans[i].end <= offset) i++;
    *cursor = i;
    if (i == plan->nspans) return 0;
    const NvnRecordSpan *s = &plan->spans[i];
    if (s->kind != NVN_PLAN_RUN || s->draw_stride < sizeof(op_hdr) || s->begin > offset ||
        s->end <= s->begin || s->end > plan->bytes || s->draws != s->ops ||
        (s->end - s->begin) / s->draw_stride != s->ops || (s->end - s->begin) % s->draw_stride ||
        (offset - s->begin) % s->draw_stride) return 0;
    size_t end = (size_t)(stop - base);
    if (s->end < end) end = s->end;
    *stride = s->draw_stride;
    return (uint32_t)((end - offset) / s->draw_stride);
}
// The producer and consumer use one classification. An unfamiliar operation is serial until its ordering contract
// is described here; a plan never makes an operation eligible merely because its payload is immutable.
static inline uint32_t nvn_record_op_class(uint32_t op) {
    switch (op) {
    case OP_SET_RT: return NVN_PLAN_TARGET;
    case OP_CLEAR_COLOR: case OP_CLEAR_DS: return NVN_PLAN_CLEAR;
    case OP_DRAW_MESH: case OP_DRAW_MESH_IND: return NVN_PLAN_MESH;
    case OP_UPDATE_UB: return NVN_PLAN_UPDATE;
    case OP_BEGIN: case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_STENCIL: case OP_POLY_OFFSET:
    case OP_BLEND_COLOR: case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD: case OP_PRIM_RESTART: case OP_COPY_STRIDE:
    case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR:
    case OP_BIND_VSTREAM: case OP_BIND_VBUF: case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP:
    case OP_BIND_IMG: case OP_PUSH_CC: case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND:
    case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_ENABLE: case OP_RENDER_COND:
        return NVN_PLAN_RUN;
    default: return NVN_PLAN_SERIAL;
    }
}
static inline uint32_t nvn_record_exec_flags(uint32_t op) {
    if (op == NVN_EXEC_STATE) return NVN_EXEC_LINEAR | NVN_EXEC_RUN;
    uint32_t kind = nvn_record_op_class(op), flags = kind != NVN_PLAN_SERIAL ? NVN_EXEC_RUN : 0;
    if (kind == NVN_PLAN_RUN || kind == NVN_PLAN_MESH || op == OP_JUMP) flags |= NVN_EXEC_LINEAR;
    if (op == OP_DRAW || op == OP_DRAW_IDX || op == OP_DRAW_IDX_IND) flags |= NVN_EXEC_DRAW;
    if (kind == NVN_PLAN_MESH) flags |= NVN_EXEC_MESH;
    if (op == OP_DISPATCH || op == OP_DISPATCH_IND) flags |= NVN_EXEC_DISPATCH;
    if (kind == NVN_PLAN_RUN && !(flags & NVN_EXEC_DRAW) && op != OP_BEGIN &&
        op != OP_SET_TEX_POOL && op != OP_SET_SAMP_POOL && op != OP_REPORT_COUNTER &&
        op != OP_RESET_COUNTER && op != OP_RENDER_ENABLE && op != OP_RENDER_COND)
        flags |= NVN_EXEC_STATE_INPUT;
    return flags;
}
uint64_t nvn_submit_count(void);                           // QueueSubmitCommands calls so far
// Submits a texture pin outlives its last tex_pin and the release of the last recording copy naming it: about 20 s at
// one submit a frame.
#define NVN_PIN_AGE 1200
// Frees the texture pins no recording copy has named for NVN_PIN_AGE submits (nvn_api_cmd.c, "texture pins"), except keep[0..nkeep):
// the replay's current render targets, which carry over to the next submission. Called between submissions by the
// thread that replays them; now = nvn_submit_count().
void nvn_pins_sweep(uint64_t now, const tex_t *const *keep, int nkeep);
void nvn_pins_stats(uint64_t *live, uint64_t *freed);      // pins that exist, pins freed so far
const char *nvn_op_name(uint32_t op);                      // "DRAW_IDX", ...; "?" past OP_COUNT_
// MemoryPoolInvalidateMappedRange: the CPU is about to read pool storage the GPU wrote. Weak no-op in nvn_ext.c; a back
// end that renders into host-side copies defines it and writes them back.
void nvn_pool_invalidate(const pool_t *p, uint64_t off, uint64_t size);
// MemoryPoolFlushMappedRange: the CPU wrote [mem + off, mem + off + size). Weak no-op in nvn_ext.c; a back end that keeps
// device-side copies of pool memory (vk_memory.c's VRAM mirror) defines it and copies the range again.
void nvn_pool_flushed(const pool_t *p, uint64_t off, uint64_t size);

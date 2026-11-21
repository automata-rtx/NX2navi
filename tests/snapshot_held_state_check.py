"""Exercise synchronous uniform-binding inheritance at the held replay caller boundary."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from bind_recipe_check import function


def section(source, first, following):
    start = source.index(first)
    return source[start:source.index(following, start + len(first))]


def main():
    replay = (ROOT / 'runtime/vk_exec.c').read_text()
    draws = (ROOT / 'runtime/vk_exec_draw.c').read_text()
    declarations = section(replay, '#define CONT_MAX', '// ---- event order trace')
    held_storage = section(replay, 'static int evo_enabled(void)', '// ---- the gate:')
    held_recording = section(replay, 'int vk_exec_held_ready_locked(void)', '// ---- looking ahead for pipelines')
    preflight = section(replay, 'static VkResult snapshot_metadata_allocate(size_t bytes, int clear, RcBudgetTicket *ticket, void **out,\n                                            VkReplayCaptureWait *wait) {',
                        'static int settle_mesh_capacity(')
    journal = section(draws, '#define UPD_N 4096', 'static VkExecUniformUpdateRead upd_match(')
    usnap = (ROOT / 'runtime/vk_usnap.c').read_text()
    usnap = usnap[usnap.index('static RcAdmissionVector arena_host_cost(uint64_t size) {'):]
    metadata = '\n'.join(function(usnap, name) for name in (
        'arena_host_cost', 'peak_sum', 'vk_replay_metadata_peak_add', 'vk_replay_metadata_free',
        'arena_admission_failure', 'metadata_touch_pages', 'replay_metadata_allocate', 'vk_replay_metadata_allocate',
    ))
    transfers = (ROOT / 'runtime/vk_exec_xfer.c').read_text()
    event_order = section(transfers, 'static uint32_t *ev_slot(', '// ---- events the CPU signals')
    event_memory = function(transfers, 'ev_mem')
    event_met = function(transfers, 'ev_met')
    event_check = function(transfers, 'vk_exec_event_met')
    signal = function(transfers, 'signal_event')
    failed = function(replay, 'vk_exec_recording_failed')
    park = section(replay, '    vk_exec_draw_update_credit_adopt(ex, &bundle->scratch);',
                   '    RP_CALL(RP_GROUP_DRAIN')
    bind_ubo = function(draws, 'op_bind_ubo')

    fixture = r'''
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"

#define RP_SCOPE(...)
#define RP_VALUE(...)
#define RP_END(...)
#define RP_CALL(id, expression) (expression)
#define CALL_DEPTH 8

static VkExec X;
static tex_t targets[4];
static unsigned seen_rt_pins;
static int x_ready = 1, recording_failed;
static int event_at_sync = 1;
static volatile uint32_t *change_after_inventory;
static uint32_t change_value;
static uint64_t indexed_record;
static const NvnExecStep *indexed_steps;
static size_t indexed_count;
static size_t indexed_bytes;
static struct { const uint64_t *handles; int live, n, i; } AH;
static struct { int on; } RR;
enum { RB_HELD };
enum { VK_BR_GROUP_BUFFER };
typedef struct DrawCtx {
    uint64_t prep_dirty[1], prep_ubo[8][18], bind_gen;
} DrawCtx;
static DrawCtx draw_context;
uint64_t vk_exec_submit_seq;
uint32_t vk_exec_brk;
int vk_exec_signalled;
_Thread_local uintptr_t gfx_submit_queue;

static unsigned failures;
static void check(int condition, const char *message) {
    if (condition) return;
    fprintf(stderr, "%s\n", message);
    failures++;
}
static void check_address(uint64_t actual, uint64_t expected, const char *message) {
    if (actual == expected) return;
    fprintf(stderr, "%s (expected 0x%llx, observed 0x%llx)\n", message,
            (unsigned long long)expected, (unsigned long long)actual);
    failures++;
}
void rt_log(const char *fmt, ...) { (void)fmt; }
_Noreturn void rt_fatal(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    abort();
}
const char *cfg_get(const char *name) { (void)name; return NULL; }
int rt_knob(const char *name, const char *def) {
    assert(strcmp(name, "HELD_SYNC") == 0 && strcmp(def, "1") == 0);
    return 1;
}
int gfx_replay_async(void) { return 0; }
int nvn_record_plan(uint64_t handle, NvnRecordPlan *plan) {
    if (handle != indexed_record) return 0;
    *plan = (NvnRecordPlan){.exec = indexed_steps, .nexec = indexed_count};
    return 1;
}
uint64_t nvn_submission_handle_bytes(uint64_t handle) {
    return handle == indexed_record ? indexed_bytes : 0;
}
void vk_exec_pass_end(VkExec *ex) { assert(ex == &X); }
static void resource_targets_forget(VkExec *ex) { assert(ex == &X); }
void vk_exec_barrier(VkExec *ex) { assert(ex == &X); }
void vk_exec_draw_state_adopted(VkExec *ex) { assert(ex == &X); }
void vk_exec_draw_new_cb(void) {}
void vk_exec_event_sync(int begin) { assert(begin); }
int vk_exec_event_at_sync(void) { return event_at_sync; }
static void cancel_recording(void) { abort(); }
void vk_exec_required_graphics_unavailable(VkExec *ex, uint32_t opcode, const char *reason) {
    (void)ex; (void)opcode; (void)reason; abort();
}
void vk_exec_draw_update_credit_adopt(VkExec *ex, VkDrawScratchCredit **credit) {
    assert(ex == &X && credit && !*credit);
}
uint32_t vk_exec_upd_top(uint32_t all) { return all; }
void vk_uniform_payload_reap(void) {}
RcRegistry *rc_registry_runtime(void) { abort(); }
RcRegistryResult rc_registry_produced_scene_end(RcRegistry *registry, uint64_t scene) {
    (void)registry; (void)scene; abort();
}
void vk_mem_mirror_begin(void) {}
static void runs_break(int why) { (void)why; abort(); }

enum { PW_SPLIT };
static int split_draws, part_start, n_parts;
static uint64_t part_t0;
static void group_drain(VkExec *ex, int why) { assert(ex == &X && why == PW_SPLIT); }
int vk_exec_pool_registration_needs_prefix(uint64_t fence) {
    assert(fence == vk_exec_submit_seq);
    return 0;
}
void vk_exec_pool_registration_prefix_complete(uint64_t fence) { (void)fence; abort(); }
VkCommandBuffer vk_sync_split(void) { abort(); }
VkResult vk_device_wait_idle(void) { abort(); }
void vk_device_lost_check(VkResult result, const char *where) { (void)result; (void)where; abort(); }
uint64_t kern_now_ns(void) { return 0; }
uint64_t nvn_submit_count(void) { return vk_exec_submit_seq; }
void nvn_pins_sweep(uint64_t now, const tex_t *const *keep, int count) {
    assert(now == vk_exec_submit_seq && count >= 9 && !(count % 9));
    seen_rt_pins = 0;
    for (int k = 0; k < count; ++k)
        for (unsigned i = 0; i < sizeof targets / sizeof targets[0]; i++)
            if (keep[k] == &targets[i]) seen_rt_pins |= 1u << i;
}

struct VkTextureRegistrationScope { unsigned refs, publication; };
static unsigned registration_publication = 1;
static VkTextureRegistrationScope scopes[8];
VkTextureRegistrationScope *vk_texture_registration_scope_begin(void) {
    for (unsigned i = 0; i < sizeof scopes / sizeof scopes[0]; i++) {
        if (!scopes[i].refs) {
            scopes[i].refs = 1;
            scopes[i].publication = registration_publication;
            return &scopes[i];
        }
    }
    abort();
}
static VkTextureRegistrationScope *recording_continuation;
VkTextureRegistrationScope *vk_texture_registration_record_begin(VkTextureRegistrationScope *scope,int resumed) {
    assert(resumed && !recording_continuation);
    recording_continuation=scope?scope:vk_texture_registration_scope_begin();
    return recording_continuation;
}
void vk_texture_registration_record_end(VkTextureRegistrationScope *scope,int resumed) {
    assert(resumed && scope==recording_continuation);
    recording_continuation=NULL;
}
VkTextureRegistrationScope *vk_texture_registration_scope_retain(VkTextureRegistrationScope *scope) {
    if (scope) {
        assert(scope->refs);
        scope->refs++;
    }
    return scope;
}
void vk_texture_registration_scope_end(VkTextureRegistrationScope *scope) {
    if (scope) {
        assert(scope->refs);
        scope->refs--;
    }
}
VkTextureRegistrationScope *vk_exec_texture_registration_scope(void) { return X.registration_scope; }

static unsigned parks, followers, reparks;
void vk_exec_evt(char what, const void *event, uint32_t value) {
    (void)event;
    (void)value;
    parks += what == 'h';
    followers += what == 'p';
    reparks += what == 'w';
}
static VkExecEventOrder ev_seen;
typedef int PlatMutex;
static PlatMutex cpu_ev_lock;
static int cpu_ev_n;
static uint64_t ev_sync_seq = UINT64_MAX;
#define CPU_EV_HIST 8
static struct { const void *val; uint64_t seq[CPU_EV_HIST]; uint32_t before[CPU_EV_HIST]; unsigned n; } cpu_ev[1];
static void plat_mutex_lock(PlatMutex *lock) { (void)lock; }
static void plat_mutex_unlock(PlatMutex *lock) { (void)lock; }
static int ev_open_n;
static struct { volatile uint32_t *val; uint32_t mode, value; uint64_t sub; } ev_open[1];
static uint64_t ev_late, ev_late_subs;
typedef struct { int target; volatile uint32_t *val; uint32_t mode, value; } EventWrite;
static int cpu_result_prepare(uintptr_t val, size_t size, uint64_t fence) {
    assert(val && size == 4 && fence);
    return 0;
}
static void event_cb(void *arg) { free(arg); }
static void event_cancel(void *arg, int may_have_executed) { assert(!may_have_executed); free(arg); }
static unsigned signal_effects;
void vk_completion_add_result_owned(uint64_t fence, void (*complete)(void *),
                                           void (*cancel)(void *, int), void *arg) {
    assert(fence && complete && cancel);
    signal_effects++;
    cancel(arg, 0);
}
void resource_effects_buffers_commit(void) {}
''' + event_order + event_memory + event_met + event_check + signal + r'''


enum { UBO_STAGE = 1, UBO_SLOT = 4, UBO_SIZE = 256, A = 0x1000, B = 0x2000, C = 0x3000, D = 0x4000 };
struct VkUsnapCapture { int reserved; };
static VkUsnapCapture reservation;
static unsigned seen_bindings, seen_commands, refuse_bindings, active_bindings;
static unsigned address_bit(uint64_t addr) {
    assert(addr >= A && addr <= D && addr % A == 0);
    return 1u << (addr / A);
}
static void capture_binding(const p_bind_buf *binding) {
    if (binding->stage == UBO_STAGE && binding->index == UBO_SLOT && binding->addr) {
        assert(binding->size == UBO_SIZE);
        seen_bindings |= address_bit(binding->addr);
    }
}
VkResult vk_usnap_preflight(const VkUsnapInputs *inputs, VkUsnapCapture **capture, VkReplayCaptureWait *wait) {
    assert(!*capture && !reservation.reserved);
    *wait = (VkReplayCaptureWait){0};
    seen_commands = (unsigned)inputs->nhandles;
    seen_bindings = 0;
    for (size_t i = 0; i < inputs->ninherited; i++) capture_binding(&inputs->inherited[i]);
    // The raw path collects the whole recording, so an unreachable binding can cause a real refusal here.
    for (int i = 0; i < inputs->nhandles; i++)
        for (const op_hdr *h = nvn_op_first(inputs->handles[i]); h->op != OP_END; h = nvn_op_next(h))
            if (h->op == OP_BIND_UBO) capture_binding(nvn_op_payload(h));
    if (seen_bindings & refuse_bindings) {
        wait->kind = VK_REPLAY_CAPTURE_WAIT_BUDGET;
        return VK_NOT_READY;
    }
    if (inputs->nhandles || seen_bindings) {
        reservation.reserved = 1;
        *capture = &reservation;
    }
    return VK_SUCCESS;
}
VkResult vk_usnap_begin(VkUsnapCapture **capture, int n, uint64_t fence) {
    (void)n;
    assert(fence == vk_exec_submit_seq);
    active_bindings = 0;
    if (*capture) {
        assert(*capture == &reservation && reservation.reserved);
        reservation.reserved = 0;
        *capture = NULL;
        active_bindings = seen_bindings;
    }
    return VK_SUCCESS;
}

static struct { uint64_t handle; unsigned refs; } records_with_pins[3];
static unsigned pin_retains, pin_releases;
void *nvn_record_retain_pins(uint64_t handle) {
    assert(handle);
    for (unsigned i = 0; i < sizeof records_with_pins / sizeof records_with_pins[0]; i++)
        if (records_with_pins[i].handle == handle) {
            records_with_pins[i].refs++;
            pin_retains++;
            return &records_with_pins[i];
        }
    return NULL;
}
void nvn_record_release_pins(void *lease) {
    if (!lease) return;
    for (unsigned i = 0; i < sizeof records_with_pins / sizeof records_with_pins[0]; i++)
        if (lease == &records_with_pins[i]) {
            assert(records_with_pins[i].refs);
            records_with_pins[i].refs--;
            pin_releases++;
            return;
        }
    abort();
}

static const void *watched_allocation;
static unsigned watched_releases;
static void watch_parked_ubo(const void *allocation, const p_bind_buf *saved, uint64_t addr) {
    assert(allocation && saved && !watched_allocation);
    const p_bind_buf *binding = &saved[UBO_STAGE * 18 + UBO_SLOT];
    assert(binding->addr == addr && binding->size == UBO_SIZE);
    watched_allocation = allocation;
    watched_releases = 0;
}
static void watch_free(void *allocation) {
    if (allocation && allocation == watched_allocation) {
        watched_releases++;
        watched_allocation = NULL;
    }
    free(allocation);
}
static size_t fail_malloc_bytes;
static void *watch_malloc(size_t bytes) {
    if (fail_malloc_bytes && bytes == fail_malloc_bytes) return NULL;
    return malloc(bytes);
}
static RcAdmission metadata_ledger;
static RcAdmissionRecord metadata_records[32];
static size_t metadata_refuse;

RcAdmissionStatus resource_core_budget_reserve(RcBudgetProducer p, int required, const RcAdmissionVector *cost,
                                              RcBudgetTicket *ticket) {
    assert(required && p == RC_BUDGET_SNAPSHOT && cost->amount[RC_BUDGET_RAM] == cost->amount[RC_BUDGET_COMMIT]);
    if (metadata_refuse && cost->amount[RC_BUDGET_RAM] == metadata_refuse) return RC_ADMISSION_REFUSED;
    ticket->producer = p;
    return rc_admission_reserve(&metadata_ledger, p, cost, &ticket->token);
}
RcAdmissionStatus resource_core_budget_split(RcBudgetTicket *t, const RcAdmissionVector *cost, RcBudgetProducer p,
                                             RcBudgetTicket *part) {
    (void)t; (void)cost; (void)p; (void)part; abort();
}
RcAdmissionStatus resource_core_budget_materialize(RcBudgetTicket *t, const RcAdmissionVector *cost) {
    return rc_admission_materialize(&metadata_ledger, t->token, t->producer, cost, &t->token);
}
RcAdmissionStatus resource_core_budget_cancel(RcBudgetTicket *t) {
    RcAdmissionStatus result = rc_admission_cancel(&metadata_ledger, t->token, t->producer);
    if (result == RC_ADMISSION_OK) *t = (RcBudgetTicket){0};
    return result;
}
RcAdmissionStatus resource_core_budget_retire(RcBudgetTicket *t) {
    return rc_admission_retire(&metadata_ledger, t->token, t->producer, &t->token);
}
RcAdmissionStatus resource_core_budget_destroyed(RcBudgetTicket *t) {
    RcAdmissionStatus result = rc_admission_destroyed(&metadata_ledger, t->token, t->producer);
    if (result == RC_ADMISSION_OK) *t = (RcBudgetTicket){0};
    return result;
}
''' + failed + declarations + bind_ubo + '\n#define free watch_free\n#define malloc watch_malloc\n' + metadata + '\n' + journal + held_storage + held_recording + preflight + '\n#undef malloc\n#undef free\n' + r'''

static uint64_t draw_binding[3];
static const tex_t *draw_target[3];
static unsigned draw_registration[3];
static unsigned draws_seen;
static uint64_t prefix_binding[4];
static unsigned prefix_registration[4], prefix_draws;
static const op_hdr *walk(VkExec *ex, const op_hdr *h, int depth, uint64_t owner) {
    assert(ex == &X && depth == 0 && owner && evo_queue);
    for (; h->op != OP_END; h = nvn_op_next(h)) {
        const void *payload = nvn_op_payload(h);
        switch (h->op) {
        case OP_BEGIN:
            break;
        case OP_BIND_UBO:
            op_bind_ubo(ex, payload);
            break;
        case OP_SET_RT:
            ex->rt = *(const p_set_rt *)payload;
            break;
        case OP_WAIT_EVENT:
            if (!vk_exec_event_met(payload)) return h;
            break;
        case OP_SIGNAL_EVENT:
            signal_event(ex, payload);
            break;
        case OP_DRAW: {
            const p_draw *draw = payload;
            if (draw->first >= 3) {
                assert((unsigned)draw->first == 3 + prefix_draws && prefix_draws < 4);
                uint64_t addr = ex->ubo[UBO_STAGE][UBO_SLOT].addr;
                assert(active_bindings & address_bit(addr));
                prefix_binding[prefix_draws] = addr;
                prefix_registration[prefix_draws++] = ex->registration_scope->publication;
                break;
            }
            assert(evo_in && draw->first >= 0 && (unsigned)draw->first == draws_seen && draws_seen < 3);
            assert(records_with_pins[draw->first == 2].refs == 1);
            assert(ex->registration_scope);
            draw_registration[draws_seen] = ex->registration_scope->publication;
            draw_target[draws_seen] = ex->rt.color[0];
            draw_binding[draws_seen++] = ex->ubo[UBO_STAGE][UBO_SLOT].addr;
            break;
        }
        default:
            abort();
        }
        if (vk_exec_signalled && !evo_in && !evo_resume(ex)) return &walk_stopped;
    }
    return NULL;
}

static void record(uintptr_t queue, const uint64_t *handles, int n) {
    VkExec *ex = &X;
    sweep_recording_pins(ex);
    gfx_submit_queue = queue;
    VkTextureRegistrationScope *scope = vk_texture_registration_scope_begin();
    ex->registration_scope = scope;
    uint64_t fence = vk_exec_submit_seq + 1;
    ex->fence = fence;
    VkReplayCaptureBundle capture = {0}, *bundle = &capture;
    VkReplayCaptureWait wait;
    VkExecReadiness *selected = NULL;
    assert(vk_exec_snapshot_preflight(handles, n, &bundle->snapshot, &selected, &wait) == VK_SUCCESS);
    if (change_after_inventory) {
        __atomic_store_n(change_after_inventory, change_value, __ATOMIC_RELEASE);
        change_after_inventory = NULL;
    }
    const VkExecReadiness *readiness = selected;
    ++vk_exec_submit_seq;
''' + park + r'''
    record_readiness = NULL;
    vk_exec_readiness_discard(selected);
    assert(!stopped && !bundle->snapshot && !bundle->scratch && !reservation.reserved);
    assert(!evo_in && !evo_queue);
    assert(ex->registration_scope == scope);
    vk_texture_registration_scope_end(scope);
    ex->registration_scope = NULL;
}

typedef struct { op_hdr h; p_bind_buf bind; } Bind;
typedef struct { op_hdr h; p_set_rt target; } Target;
typedef struct { op_hdr h; p_wait_event wait; } Wait;
typedef struct { op_hdr h; p_signal_event signal; } Signal;
typedef struct { op_hdr h; p_draw draw; uint32_t pad; } Draw;
typedef struct { op_hdr begin; Bind binding; Target target; op_hdr end; } BindCommands;

#define BEGIN {OP_BEGIN, sizeof(op_hdr)}
#define END {OP_END, sizeof(op_hdr)}
#define BIND(addr) {{OP_BIND_UBO, sizeof(Bind)}, {UBO_STAGE, UBO_SLOT, (addr), UBO_SIZE}}
#define TARGET(i) {{OP_SET_RT, sizeof(Target)}, {.n=1, .color[0]=&targets[i]}}
#define WAIT(signal) {{OP_WAIT_EVENT, sizeof(Wait)}, {.val = &(signal), .value = 1}}
#define DRAW(id) {{OP_DRAW, sizeof(Draw)}, {.first = (id)}, 0}
#define WAIT_VALUE(event, count) {{OP_WAIT_EVENT, sizeof(Wait)}, {.val = &(event), .value = (count)}}
#define ADD(event, count) {{OP_SIGNAL_EVENT, sizeof(Signal)},                             {.val = &(event), .mode = NVN_EVENT_SIGNAL_MODE_ADD, .value = (count)}}

static void check_budget_empty(void) {
    for (unsigned d = RC_BUDGET_RAM; d <= RC_BUDGET_COMMIT; d++)
        assert(!metadata_ledger.reserved.amount[d] && !metadata_ledger.retained.amount[d] &&
               !metadata_ledger.pending.amount[d]);
}

static uint64_t cont_size(size_t commands, size_t count) {
    return sizeof(Cont) + CONT_STATE_BYTES + sizeof(uint64_t) * (count + 1) + sizeof(void *) * count + commands;
}

static void check_update_borrow(int evict) {
    typedef struct { op_hdr h; uint64_t dst, size; uint8_t data[8]; } Update;
    struct { op_hdr begin; Update a, b; op_hdr end; } commands = {
        BEGIN, {{OP_UPDATE_UB, sizeof(Update)}, 0x1000, 8, {11,12,13,14,15,16,17,18}},
        {{OP_UPDATE_UB, sizeof(Update)}, 0x2000, 8, {21,22,23,24,25,26,27,28}}, END
    };
    const uint64_t handle = (uintptr_t)&commands;
    const uint64_t bytes = cont_size(sizeof commands, 1);
    Cont *c = NULL;
    VkReplayCaptureWait wait = {0};
    assert(cont_prepare(&handle, 1, 0, &c, &wait) == VK_SUCCESS && c->bytes == bytes && !c->n);
    assert(cont_new(c, &handle, 1, 0, NULL) == c && c->n == 1 && !(c->h[0] & 7) && c->h[0] != handle);
    cont_free(c); // The readiness plan drops its reference to the parked commands.
    const op_hdr *first = nvn_op_next(nvn_op_first(c->h[0]));
    const p_update_ub *a = nvn_op_payload(first), *b = nvn_op_payload(nvn_op_next(first));
    memset(commands.a.data, 0, sizeof commands.a.data);
    memset(commands.b.data, 0, sizeof commands.b.data);
    const uint8_t expected_a[8] = {11,12,13,14,15,16,17,18}, expected_b[8] = {21,22,23,24,25,26,27,28};
    assert(!memcmp(a->data, expected_a, sizeof expected_a) && !memcmp(b->data, expected_b, sizeof expected_b));

    X.fence = 1;
    upd_begin(&X);
    VkExecUpdateDestination destination = {0};
    upd_note(&X, a, (VkMemSpan){0}, &destination);
    upd_note(&X, b, (VkMemSpan){0}, &destination);
    watched_allocation = c;
    watched_releases = 0;
    cont_free(c); // The held queue completes while this recorder still reads the exact copied OP7 bytes.
    assert(watched_allocation && !watched_releases && UPD.u[0].command_owner == UPD.u[1].command_owner);
    assert(UPD.u[0].command_owner->refs == 2 && UPD.u[0].command_owner->budget.token.ledger);
    assert(!memcmp(UPD.u[0].data, expected_a, sizeof expected_a) && !memcmp(UPD.u[1].data, expected_b, sizeof expected_b));
    assert(metadata_ledger.retained.amount[RC_BUDGET_RAM] == bytes && metadata_ledger.retained.amount[RC_BUDGET_COMMIT] == bytes);
    if (evict) {
        const p_update_ub *other = nvn_op_payload(&commands.a.h);
        for (unsigned i = UPD.n; i <= UPD_N; ++i) upd_note(&X, other, (VkMemSpan){0}, &destination);
        assert(UPD.u[1].command_owner->refs == 1 && watched_allocation && !watched_releases);
        upd_note(&X, other, (VkMemSpan){0}, &destination);
        assert(!watched_allocation && watched_releases == 1);
        check_budget_empty();
    }
    vk_exec_draw_update_history_end(X.fence);
    assert(!watched_allocation && watched_releases == 1);
    check_budget_empty();
}

static void check_readiness_resize(void) {
    VkReplayCaptureWait wait = {0};
    ReadinessPlan plan = {.wait = &wait};
    RcBudgetTicket retained = {0};
    size_t capacity = 0, bytes = 16 * sizeof(ReadinessEvent);
    ReadinessEvent *events = readiness_grow(&plan, NULL, &capacity, 0, 16, sizeof *events, &retained);
    assert(events && capacity == 16 && metadata_ledger.retained.amount[RC_BUDGET_RAM] == bytes);
    events[0].value = 73;
    assert(rc_admission_set_limit(&metadata_ledger, RC_BUDGET_RAM, 3 * bytes - 1) == RC_ADMISSION_OK);
    assert(readiness_grow(&plan, events, &capacity, 16, 1, sizeof *events, &retained) == events);
    assert(plan.result == VK_NOT_READY && capacity == 16 && events[0].value == 73);
    const uint64_t held[] = {bytes};
    assert(snapshot_metadata_retry(plan.result, &wait, held, 1) == VK_NOT_READY);
    assert(wait.peak.amount[RC_BUDGET_RAM] == 3 * bytes && wait.peak.amount[RC_BUDGET_COMMIT] == 3 * bytes);
    wait.peak.amount[RC_BUDGET_COMMIT] = UINT64_MAX - bytes;
    const RcAdmissionVector prior = wait.peak;
    const uint64_t overflow[] = {bytes, 1};
    assert(snapshot_metadata_retry(plan.result, &wait, overflow, 2) == VK_ERROR_OUT_OF_HOST_MEMORY);
    assert(!memcmp(&prior, &wait.peak, sizeof prior));
    assert(rc_admission_set_limit(&metadata_ledger, RC_BUDGET_RAM, UINT64_MAX) == RC_ADMISSION_OK);
    plan.result = VK_SUCCESS;
    events = readiness_grow(&plan, events, &capacity, 16, 1, sizeof *events, &retained);
    assert(events && capacity == 32 && events[0].value == 73 &&
           metadata_ledger.retained.amount[RC_BUDGET_RAM] == 2 * bytes);
    vk_replay_metadata_free(events, &retained);
    assert(!metadata_ledger.retained.amount[RC_BUDGET_RAM] && !metadata_ledger.pending.amount[RC_BUDGET_COMMIT]);
}

int main(void) {
    const RcAdmissionDomain domains[] = {
        {RC_ADMISSION_RAM, 0, UINT64_MAX}, {RC_ADMISSION_COMMIT, 0, UINT64_MAX},
    };
    assert(rc_admission_init(&metadata_ledger, 1, domains, sizeof domains / sizeof *domains, metadata_records,
                             sizeof metadata_records / sizeof *metadata_records) == RC_ADMISSION_OK);
    check_readiness_resize();
    check_update_borrow(0);
    check_update_borrow(1);
    enum { HELD_QUEUE = 1, OTHER_QUEUE };
    X.dc = &draw_context;
    volatile uint32_t first_signal = 0, second_signal = 0;
    const BindCommands initial = {BEGIN, BIND(A), TARGET(0), END};
    const BindCommands intervening = {BEGIN, BIND(B), TARGET(1), END};
    const BindCommands later = {BEGIN, BIND(D), TARGET(3), END};
    const struct {
        op_hdr begin;
        Wait first;
        Draw read_inherited;
        Bind change;
        Target change_target;
        Wait second;
        Draw read_repeated;
        op_hdr end;
    } head = {BEGIN, WAIT(first_signal), DRAW(0), BIND(C), TARGET(2), WAIT(second_signal), DRAW(1), END};
    const struct { op_hdr begin; Draw read_follower; op_hdr end; } follower = {BEGIN, DRAW(2), END};

    uint64_t handle = (uintptr_t)&initial;
    const NvnExecStep no_wait[] = {{.opcode = OP_BEGIN}, {.opcode = NVN_EXEC_STATE}, {.opcode = OP_END}};
    indexed_record = handle;
    indexed_steps = no_wait;
    indexed_count = sizeof no_wait / sizeof *no_wait;
    indexed_bytes = sizeof initial;
    // A wait-free indexed recording needs no continuation reservation even in live mode.
    event_at_sync = 0;
    VkUsnapCapture *open_capture = NULL;
    VkExecReadiness *open_plan = NULL;
    VkReplayCaptureWait open_wait;
    gfx_submit_queue = HELD_QUEUE;
    assert(vk_exec_snapshot_preflight(&handle, 1, &open_capture, &open_plan, &open_wait) == VK_SUCCESS);
    assert(!open_plan && open_capture && !vk_exec_held());
    reservation.reserved = 0;
    check_budget_empty();
    event_at_sync = 1;
    record(HELD_QUEUE, &handle, 1);
    assert(X.ubo[UBO_STAGE][UBO_SLOT].addr == A && !vk_exec_held() && seen_commands == 1);

    handle = (uintptr_t)&head;
    records_with_pins[0].handle = handle;
    const NvnExecStep first_wait[] = {
        {.opcode = OP_BEGIN}, {.opcode = OP_WAIT_EVENT}, {.opcode = OP_DRAW}, {.opcode = NVN_EXEC_STATE},
        {.opcode = OP_WAIT_EVENT}, {.opcode = OP_DRAW}, {.opcode = OP_END},
    };
    indexed_record = handle;
    indexed_steps = first_wait;
    indexed_count = sizeof first_wait / sizeof *first_wait;
    indexed_bytes = sizeof head;
    VkUsnapCapture *park_capture = NULL;
    VkExecReadiness *park_plan = NULL;
    VkReplayCaptureWait park_wait;
    const uint64_t head_size = cont_size(sizeof head, 1);
    const uint64_t plan_size = sizeof(VkExecReadiness) + 16 * sizeof(ReadinessEvent);
    assert(rc_admission_set_limit(&metadata_ledger, RC_BUDGET_COMMIT, plan_size + head_size - 1) == RC_ADMISSION_OK);
    assert(vk_exec_snapshot_preflight(&handle, 1, &park_capture, &park_plan, &park_wait) == VK_NOT_READY);
    assert(!park_capture && !park_plan && !vk_exec_held() && !records_with_pins[0].refs);
    assert(park_wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && park_wait.peak.amount[RC_BUDGET_RAM] == plan_size + head_size &&
           park_wait.peak.amount[RC_BUDGET_COMMIT] == plan_size + head_size);
    check_budget_empty();
    assert(rc_admission_set_limit(&metadata_ledger, RC_BUDGET_COMMIT, UINT64_MAX) == RC_ADMISSION_OK);
    fail_malloc_bytes = head_size;
    assert(vk_exec_snapshot_preflight(&handle, 1, &park_capture, &park_plan, &park_wait) == VK_ERROR_OUT_OF_HOST_MEMORY);
    assert(!park_capture && !park_plan && !vk_exec_held() && !records_with_pins[0].refs);
    check_budget_empty();
    fail_malloc_bytes = 0;
    metadata_refuse = sizeof(ParkedUbo);
    assert(vk_exec_snapshot_preflight(&handle, 1, &park_capture, &park_plan, &park_wait) == VK_NOT_READY);
    assert(!park_capture && !park_plan && !vk_exec_held() && !metadata_ledger.retained.amount[RC_BUDGET_RAM]);
    uint64_t parked_peak = sizeof(ParkedUbo) + plan_size + head_size;
    assert(park_wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && park_wait.peak.amount[RC_BUDGET_RAM] == parked_peak &&
           park_wait.peak.amount[RC_BUDGET_COMMIT] == parked_peak);
    metadata_refuse = 0;
    refuse_bindings = address_bit(A) | address_bit(C);
    change_after_inventory = &first_signal;
    change_value = 1;
    record(HELD_QUEUE, &handle, 1);
    assert(!seen_commands && !seen_bindings && first_signal == 1);
    assert(vk_exec_held() == 1 && parks == 1 && !draws_seen);
    first_signal = 0;
    refuse_bindings = 0;
    indexed_record = 0;
    assert(evq[0].head->h[0] != handle && records_with_pins[0].refs == 1);
    assert(evq[0].head->parked_ubo && evq[0].head->parked_ubo->refs == 1 && evq[0].head->refs == 1 &&
           metadata_ledger.retained.amount[RC_BUDGET_RAM] == sizeof(ParkedUbo) + head_size &&
           metadata_ledger.retained.amount[RC_BUDGET_COMMIT] == sizeof(ParkedUbo) + head_size);
    watch_parked_ubo(evq[0].head->parked_ubo, evq[0].head->parked_ubo->bindings, A);

    handle = (uintptr_t)&intervening;
    record(OTHER_QUEUE, &handle, 1);
    assert(X.ubo[UBO_STAGE][UBO_SLOT].addr == B && vk_exec_held() == 1);
    assert(!seen_commands && seen_bindings == (address_bit(A) | address_bit(B)));

    handle = (uintptr_t)&follower;
    records_with_pins[1].handle = handle;
    const uint64_t follower_size = cont_size(sizeof follower, 1), older = sizeof(ParkedUbo) + head_size;
    gfx_submit_queue = HELD_QUEUE;
    assert(rc_admission_set_limit(&metadata_ledger, RC_BUDGET_RAM, older + plan_size + follower_size - 1) == RC_ADMISSION_OK);
    assert(vk_exec_snapshot_preflight(&handle, 1, &park_capture, &park_plan, &park_wait) == VK_NOT_READY);
    assert(!park_plan && !park_capture && vk_exec_held() == 1 && records_with_pins[0].refs == 1 && !records_with_pins[1].refs);
    assert(park_wait.peak.amount[RC_BUDGET_RAM] == plan_size + follower_size &&
           park_wait.peak.amount[RC_BUDGET_COMMIT] == plan_size + follower_size);
    assert(metadata_ledger.retained.amount[RC_BUDGET_RAM] == older && metadata_ledger.retained.amount[RC_BUDGET_COMMIT] == older);
    assert(rc_admission_set_limit(&metadata_ledger, RC_BUDGET_RAM, UINT64_MAX) == RC_ADMISSION_OK);
    record(HELD_QUEUE, &handle, 1);
    assert(vk_exec_held() == 2 && followers == 1 && !draws_seen);
    assert(seen_rt_pins == 0x3); // active B and parked initial A both participate in the sweep
    assert(!evq[0].head->next->parked_ubo && records_with_pins[1].refs == 1);
    assert(metadata_ledger.retained.amount[RC_BUDGET_RAM] == older + follower_size);
    assert(!evq[0].head->next->registration_scope && evq[0].head->next->registration_deferred);
    assert(!seen_commands && !seen_bindings);

    first_signal = 1;
    change_after_inventory = &first_signal;
    change_value = 0;
    record(OTHER_QUEUE, NULL, 0);
    assert(vk_exec_held() == 2 && reparks == 1 && draws_seen == 1 && !seen_commands);
    assert(watched_releases == 1 && !watched_allocation && !evq[0].head->parked_ubo);
    assert(evq[0].head->has_st && metadata_ledger.retained.amount[RC_BUDGET_COMMIT] == head_size + follower_size);
    assert(pin_retains == 2 && !pin_releases && records_with_pins[0].refs && records_with_pins[1].refs);
    check((seen_bindings & address_bit(A)) != 0, "Resume preflight omitted the binding from the first wait.");
    check(seen_bindings == (address_bit(A) | address_bit(C)), "Resume preflight captured unrelated running state.");
    check_address(draw_binding[0], A, "First resumed draw selected a different address from the binding at the wait");
    check_address(X.ubo[UBO_STAGE][UBO_SLOT].addr, B, "First resume failed to restore the caller's binding");
    assert(draw_target[0] == &targets[0] && X.rt.color[0] == &targets[1]);

    handle = (uintptr_t)&later;
    record(OTHER_QUEUE, &handle, 1);
    assert(X.ubo[UBO_STAGE][UBO_SLOT].addr == D && vk_exec_held() == 2);
    assert(seen_rt_pins == 0x6); // active B and twice parked C
    assert(!seen_commands && seen_bindings == (address_bit(B) | address_bit(D)));

    registration_publication = 7;
    second_signal = 1;
    record(OTHER_QUEUE, NULL, 0);
    assert(!vk_exec_held() && !evq[0].head && !evq[0].tail && draws_seen == 3 && !seen_commands);
    assert(pin_releases == 2 && !records_with_pins[0].refs && !records_with_pins[1].refs);
    check_budget_empty();
    check(seen_bindings == address_bit(C), "Repeated resume must carry the exact saved binding.");
    check_address(draw_binding[1], C, "Repeated wait lost the binding recorded before it");
    check_address(draw_binding[2], C, "Follower did not inherit the binding left by completed held work");
    assert(draw_registration[0] == 1 && draw_registration[1] == 7 && draw_registration[2] == 7);
    check_address(X.ubo[UBO_STAGE][UBO_SLOT].addr, D, "Second resume failed to restore the caller's binding");
    assert(draw_target[1] == &targets[2] && draw_target[2] == &targets[2] && X.rt.color[0] == &targets[3]);

    assert(!n_parts && !split_draws && !part_start && !part_t0);
    assert(evo_held == 2 && evo_resumed == 3 && evo_switches == 2 && !evo_forced);

    // Canceling before a signal frees the saved initial bindings and leaves the running stream alone.
    volatile uint32_t unsignaled = 0;
    const struct { op_hdr begin; Wait wait; op_hdr end; } canceled = {BEGIN, WAIT(unsignaled), END};
    handle = (uintptr_t)&canceled;
    records_with_pins[2].handle = handle;
    record(HELD_QUEUE, &handle, 1);
    assert(vk_exec_held() == 1 && parks == 2 && !vk_exec_held_ready_locked());
    assert(evq[0].head->parked_ubo && evq[0].head->parked_ubo->refs == 1 &&
           metadata_ledger.retained.amount[RC_BUDGET_RAM] == sizeof(ParkedUbo) + cont_size(sizeof canceled, 1));
    watch_parked_ubo(evq[0].head->parked_ubo, evq[0].head->parked_ubo->bindings, D);

    handle = (uintptr_t)&initial;
    record(OTHER_QUEUE, &handle, 1);
    assert(vk_exec_held() == 1 && !watched_releases && watched_allocation && records_with_pins[2].refs == 1);
    vk_exec_discard_held_locked();
    assert(!vk_exec_held() && !evq[0].head && !evq[0].tail && draws_seen == 3);
    assert(watched_releases == 1 && !watched_allocation);
    assert(pin_retains == 3 && pin_releases == 3 && !records_with_pins[2].refs);
    check_budget_empty();
    check_address(X.ubo[UBO_STAGE][UBO_SLOT].addr, A, "Canceling held work changed the running binding");

    // A denied ready cohort leaves no queue or event mutation. Closing the gate makes the
    // same refusal irrelevant to the next attempt. Reopening it permits a same-record cascade.
    volatile uint32_t cpu = 0, gpu = 10, dormant = 0;
    const struct { op_hdr begin; Wait wait; Bind bind; Signal signal; op_hdr end; } producer =
        {BEGIN, WAIT(cpu), BIND(C), ADD(gpu, 1), END};
    const struct { op_hdr begin; Wait wait; Bind bind; op_hdr end; } consumer =
        {BEGIN, WAIT_VALUE(gpu, 11), BIND(B), END};
    const struct { op_hdr begin; Wait wait; Bind bind; op_hdr end; } unrelated =
        {BEGIN, WAIT(dormant), BIND(D), END};
    const uint64_t producer_handle = (uintptr_t)&producer;
    const uint64_t consumer_handle = (uintptr_t)&consumer;
    const uint64_t dormant_handle = (uintptr_t)&unrelated;
    record(HELD_QUEUE, &producer_handle, 1);
    record(OTHER_QUEUE, &consumer_handle, 1);
    record(3, &dormant_handle, 1);
    assert(vk_exec_held() == 3);
    refuse_bindings = address_bit(C);
    cpu = 1;
    VkExecReadiness *refused_plan = NULL;
    VkUsnapCapture *refused_capture = NULL;
    VkReplayCaptureWait refused_wait;
    assert(vk_exec_snapshot_preflight(NULL, 0, &refused_capture, &refused_plan, &refused_wait) == VK_NOT_READY);
    assert(!refused_plan && !refused_capture && !signal_effects && !ev_seen.count && vk_exec_held() == 3);
    assert(refused_wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET);

    cpu = 0;
    change_after_inventory = &cpu;
    change_value = 1;
    handle = (uintptr_t)&initial;
    record(4, &handle, 1);
    assert(!seen_commands && seen_bindings == address_bit(A) && !signal_effects && vk_exec_held() == 3);
    assert(cpu == 1);

    refuse_bindings = 0;
    change_after_inventory = &gpu;
    change_value = 20;
    record(4, NULL, 0);
    assert(!seen_commands && seen_bindings == (address_bit(A) | address_bit(B) | address_bit(C)));
    assert(vk_exec_held() == 1 && ev_seen.count == 1 && ev_seen.slots[0].value == 11 && signal_effects == 1);
    assert(gpu == 20 && !ev_late && !ev_late_subs);
    event_at_sync = 0;
    record(4, &handle, 1);
    assert(seen_commands == 2 && vk_exec_held() == 1 && signal_effects == 1);
    assert(!reservation.reserved);
    vk_exec_discard_held_locked();

    // No earlier held submission exists. The first command buffer still records its runnable prefix
    // even though both later cohorts would be refused. Late CPU signals are observed by new preflights.
    event_at_sync = 1;
    volatile uint32_t midstream_open = 0, repeated_open = 0;
    const struct {
        op_hdr begin; Bind a; Draw before; Wait first; Bind d; Draw after; Wait second;
        Bind b; Draw repeated; op_hdr end;
    } sliced = {BEGIN, BIND(A), DRAW(3), WAIT(midstream_open), BIND(D), DRAW(4), WAIT(repeated_open),
                BIND(B), DRAW(5), END};
    const uint64_t sliced_handles[] = {(uintptr_t)&initial, (uintptr_t)&sliced, (uintptr_t)&intervening};
    refuse_bindings = address_bit(B) | address_bit(D);
    change_after_inventory = &midstream_open;
    change_value = 1;
    record(HELD_QUEUE, sliced_handles, 3);
    assert(evq[0].head->first == 1 && evq[0].head->n == 2 &&
           evq[0].head->bytes == cont_size(sizeof sliced + sizeof intervening, 2));
    assert(prefix_draws == 1 && prefix_binding[0] == A && seen_bindings == address_bit(A) && !seen_commands);
    assert(vk_exec_held() == 1 && vk_exec_held_ready_locked());
    assert(vk_exec_snapshot_preflight(NULL, 0, &refused_capture, &refused_plan, &refused_wait) == VK_NOT_READY);
    assert(refused_wait.kind == VK_REPLAY_CAPTURE_WAIT_BUDGET && !refused_capture && !refused_plan);
    assert((seen_bindings & address_bit(D)) && !(seen_bindings & address_bit(B)) && prefix_draws == 1);

    registration_publication = 8;
    refuse_bindings = address_bit(B);
    record(OTHER_QUEUE, NULL, 0);
    assert(prefix_draws == 2 && prefix_binding[1] == D && prefix_registration[1] == 8);
    assert(vk_exec_held() == 1 && !vk_exec_held_ready_locked() && !(seen_bindings & address_bit(B)));
    repeated_open = 1;
    registration_publication = 9;
    refuse_bindings = 0;
    record(OTHER_QUEUE, NULL, 0);
    assert(prefix_draws == 3 && prefix_binding[2] == B && prefix_registration[2] == 9 && !vk_exec_held());

    // A wait-only predecessor supplies the parked bindings to an unstarted follower.
    volatile uint32_t follower_open = 0;
    const struct { op_hdr begin; Wait wait; op_hdr end; } wait_only = {BEGIN, WAIT(follower_open), END};
    const struct { op_hdr begin; Draw read; op_hdr end; } wait_follower = {BEGIN, DRAW(6), END};
    handle = (uintptr_t)&wait_only;
    record(HELD_QUEUE, &handle, 1);
    assert(!seen_bindings && X.ubo[UBO_STAGE][UBO_SLOT].addr == A);
    handle = (uintptr_t)&wait_follower;
    indexed_record = handle;
    indexed_steps = no_wait;
    indexed_count = sizeof no_wait / sizeof *no_wait;
    indexed_bytes = sizeof wait_follower;
    event_at_sync = 0;
    record(HELD_QUEUE, &handle, 1);
    assert(evq[0].head->next->n == 1 && !evq[0].head->next->parked_ubo);
    event_at_sync = 1;
    handle = (uintptr_t)&intervening;
    record(OTHER_QUEUE, &handle, 1);
    assert(X.ubo[UBO_STAGE][UBO_SLOT].addr == B && vk_exec_held() == 2);
    follower_open = 1;
    record(OTHER_QUEUE, NULL, 0);
    assert(prefix_draws == 4 && prefix_binding[3] == A && seen_bindings == address_bit(A) && !vk_exec_held());

    for (unsigned i = 0; i < sizeof scopes / sizeof scopes[0]; i++) assert(!scopes[i].refs);
    assert(!metadata_ledger.retained.amount[RC_BUDGET_RAM] && !metadata_ledger.retained.amount[RC_BUDGET_COMMIT]);
    if (failures) return 1;
    puts("snapshot held state: command admission, RAM and commit pressure, allocator rollback, exact OP7 custody, FIFO resume and cancellation passed");
    return 0;
}
'''

    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise SystemExit('Clang is required (set CC)')
    toolchain = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get('VMA_INCLUDE', toolchain.parent.parent / 'include' / 'vma'))
    env = dict(os.environ, PATH=str(toolchain.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='snapshot-held-state-') as directory:
        source = Path(directory) / 'check.c'
        executable = Path(directory) / ('check.exe' if os.name == 'nt' else 'check')
        source.write_text(fixture)
        subprocess.run([str(toolchain), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(ROOT / 'runtime'), '-I' + str(vma), str(source), '-o', str(executable)],
                       env=env, check=True)
        return subprocess.run([str(executable)], env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())

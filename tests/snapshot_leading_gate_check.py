"""Exercise the production snapshot gate against retained command streams without Vulkan."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]

STUBS = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_api.h"

_Thread_local uintptr_t gfx_submit_queue = 1;
unsigned rt_trace;
static volatile uint32_t cpu_event;
static unsigned pauses, open_after_pauses, stop_after_pauses;
static int stopping;
static uint64_t now;

typedef struct { uint64_t handle, bytes; } RetainedRecording;
static RetainedRecording retained[16];
static unsigned retained_count;

static uint64_t retain_recording(const void *stream, size_t bytes) {
    uint64_t handle = (uintptr_t)stream;
    assert(!(handle & 7) && retained_count < sizeof retained / sizeof retained[0]);
    retained[retained_count++] = (RetainedRecording){handle, bytes};
    return handle;
}

uint64_t nvn_submission_handle_bytes(uint64_t handle) {
    for (unsigned i = 0; i < retained_count; i++)
        if (retained[i].handle == handle) return retained[i].bytes;
    return 0;
}

int gfx_replay_async(void) { return 1; }
static int evo_enabled(void) { return 0; }
static int vk_sync_stopping(void) { return stopping; }
static int vk_device_is_lost(void) { return 0; }

static int vk_exec_event_holds_now(const void *payload) {
    const p_wait_event *wait = payload;
    assert(wait->val == &cpu_event && wait->mode == NVN_EVENT_WAIT_MODE_EQUAL);
    return *wait->val == wait->value;
}

static int vk_exec_event_cpu_known(const void *value) { return value == (const void *)&cpu_event; }

int cfg_int(const char *name) {
    assert(!strcmp(name, "NX2NAVI_GATE_WAIT_MS"));
    return 5;
}

uint64_t kern_now_ns(void) { return now += 1000000; }
uint64_t nvn_submit_count(void) { return 1; }
void rt_log(const char *format, ...) { (void)format; }

static void plat_pause(void) {
    pauses++;
    if (open_after_pauses && !--open_after_pauses) cpu_event = 1;
    if (stop_after_pauses && !--stop_after_pauses) stopping = 1;
}

static void plat_yield(void) { plat_pause(); }
'''

CASES = r'''
static int failures;

static void check_gate(const char *label, const uint64_t *handles, int count, int initially_open,
                       unsigned opening_delay, int expected_ready) {
    cpu_event = (uint32_t)initially_open;
    open_after_pauses = opening_delay;
    int ready = !!vk_exec_gate_wait(handles, count);
    if (ready != expected_ready || (opening_delay && !cpu_event)) {
        fprintf(stderr, "%s: expected %s, observed %s (CPU event %s)\n", label,
                expected_ready ? "ready" : "blocked", ready ? "ready" : "blocked",
                cpu_event ? "open" : "closed");
        failures++;
    } else {
        printf("%s: passed\n", label);
    }
}

int main(void) {
    typedef struct { op_hdr header; p_wait_event payload; } WaitRecord;
    typedef struct { op_hdr header; p_scissor payload; } StateRecord;
    typedef struct { op_hdr header; p_copy_b2b payload; } ActionRecord;
    op_hdr begin = {OP_BEGIN, sizeof(op_hdr)}, end = {OP_END, sizeof(op_hdr)};
    WaitRecord wait = {{OP_WAIT_EVENT, sizeof(WaitRecord)}, {&cpu_event, NVN_EVENT_WAIT_MODE_EQUAL, 1}};
    StateRecord state = {{OP_SCISSOR, sizeof(StateRecord)}, {0, 0, 8, 8}};
    ActionRecord action = {{OP_COPY_B2B, sizeof(ActionRecord)}, {0x1000, 0x2000, 32}};

    struct { _Alignas(8) op_hdr begin; WaitRecord wait; op_hdr end; } gated = {begin, wait, end};
    struct { _Alignas(8) op_hdr begin, end; } empty = {begin, end};
    struct { _Alignas(8) op_hdr begin; StateRecord state; op_hdr end; } states = {begin, state, end};
    struct {
        _Alignas(8) op_hdr begin;
        StateRecord states[260];
        WaitRecord wait;
        op_hdr end;
    } long_prefix = {.begin = begin, .wait = wait, .end = end};
    for (size_t i = 0; i < sizeof long_prefix.states / sizeof long_prefix.states[0]; i++)
        long_prefix.states[i] = state;
    struct {
        _Alignas(8) op_hdr begin;
        ActionRecord action;
        WaitRecord wait;
        op_hdr end;
    } nonleading = {begin, action, wait, end};
    struct { _Alignas(8) op_hdr begin; WaitRecord waits[2]; op_hdr end; } consecutive = {begin, {wait, wait}, end};
    consecutive.waits[1].payload.value = 2;
    struct { _Alignas(8) op_hdr begin, wait, end; } short_wait = {begin, {OP_WAIT_EVENT, sizeof(op_hdr)}, end};
    struct { _Alignas(8) op_hdr begin, state; int32_t first, n; op_hdr end; } short_state = {
        begin, {OP_BIND_VATTR, sizeof(op_hdr) + 2 * sizeof(int32_t)}, 0, 1, end
    };
    struct { _Alignas(8) op_hdr begin, jump; p_jump target; op_hdr end; } opaque = {
        begin, {OP_JUMP, sizeof(op_hdr) + sizeof(p_jump)}, {8}, end
    };
    struct { _Alignas(8) op_hdr begin; WaitRecord wait; op_hdr end; } truncated = {begin, wait, end};

    uint64_t gated_handle = retain_recording(&gated, sizeof gated);
    uint64_t empty_then_wait[] = {retain_recording(&empty, sizeof empty), gated_handle};
    uint64_t state_then_wait[] = {retain_recording(&states, sizeof states), gated_handle};
    uint64_t long_handle = retain_recording(&long_prefix, sizeof long_prefix);
    uint64_t action_handle = retain_recording(&nonleading, sizeof nonleading);
    uint64_t consecutive_handle = retain_recording(&consecutive, sizeof consecutive);
    uint64_t short_wait_handle = retain_recording(&short_wait, sizeof short_wait);
    uint64_t short_state_handle = retain_recording(&short_state, sizeof short_state);
    uint64_t opaque_handle = retain_recording(&opaque, sizeof opaque);
    uint64_t truncated_handle = retain_recording(&truncated, sizeof truncated - sizeof(op_hdr));
    uint64_t unknown_handle = 8;

    check_gate("first handle has a closed leading wait", &gated_handle, 1, 0, 0, 0);
    check_gate("first leading wait opens while polling", &gated_handle, 1, 0, 3, 1);
    check_gate("empty first handle precedes a closed leading wait", empty_then_wait, 2, 0, 0, 0);
    check_gate("empty first handle precedes an open leading wait", empty_then_wait, 2, 1, 0, 1);
    check_gate("state-only first handle precedes a closed leading wait", state_then_wait, 2, 0, 0, 0);
    check_gate("state-only first handle precedes an open leading wait", state_then_wait, 2, 1, 0, 1);
    check_gate("long pure-state prefix cannot admit a closed wait", &long_handle, 1, 0, 0, 0);
    check_gate("long pure-state prefix reaches a wait that opens", &long_handle, 1, 0, 3, 1);
    check_gate("action before a closed wait is nonleading", &action_handle, 1, 0, 0, 1);
    check_gate("a later leading wait remains closed after an open wait", &consecutive_handle, 1, 1, 0, 0);
    check_gate("a later leading wait remains closed after polling opened the first", &consecutive_handle, 1, 0, 3, 0);
    check_gate("wait payload must fit in its record", &short_wait_handle, 1, 0, 0, 0);
    check_gate("a counted state payload cannot extend beyond its record", &short_state_handle, 1, 0, 0, 0);
    check_gate("opaque jumps cannot certify a leading prefix", &opaque_handle, 1, 0, 0, 0);
    check_gate("retained extent must include the trailing end marker", &truncated_handle, 1, 0, 0, 0);
    check_gate("unretained handles remain unresolved", &unknown_handle, 1, 0, 0, 0);

    unsigned before_stop = pauses;
    stop_after_pauses = 1;
    check_gate("shutdown interrupts a closed wait", &gated_handle, 1, 0, 0, 0);
    assert(stopping && pauses - before_stop <= 16);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
'''


def main():
    replay = (ROOT / "runtime" / "vk_exec.c").read_text()
    begin = replay.index("// ---- the gate:")
    end = replay.index("int vk_exec_held_ready_locked(void)", begin)
    compiler = os.environ.get("CC", "clang")
    cc = shutil.which(compiler)
    if cc is None:
        raise SystemExit(f"Clang not found. Set CC to the project compiler. Requested {compiler!r}.")
    env = dict(os.environ, PATH=str(Path(cc).parent) + os.pathsep + os.environ.get("PATH", ""))
    work = ROOT / "build" / "test"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="snapshot-leading-gate-", dir=work) as directory:
        source = Path(directory) / "check.c"
        executable = Path(directory) / "check.exe"
        source.write_text(STUBS + replay[begin:end] + CASES)
        subprocess.run([cc, "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined",
                        "-I" + str(ROOT / "runtime"), str(source), "-o", str(executable)], env=env, check=True)
        return subprocess.run([str(executable)], env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

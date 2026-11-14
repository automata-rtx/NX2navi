// Immutable execution metadata built from validated command copies. Source offsets always refer to the current
// copy, including submission clones whose deferred values differ. Resolved resources never belong to this plan.
#pragma once
#include <stddef.h>
#include <stdint.h>

struct VkExec;
typedef void (*NvnExecFn)(struct VkExec *ex, const void *payload);
#define NVN_EXEC_STATE UINT32_MAX
enum { NVN_EXEC_LINEAR = 1, NVN_EXEC_RUN = 2, NVN_EXEC_DRAW = 4, NVN_EXEC_MESH = 8,
       NVN_EXEC_DISPATCH = 16, NVN_EXEC_STATE_INPUT = 32 };
typedef struct {
    size_t begin, end;
    NvnExecFn run;
    uint32_t ops, opcode;
    uint32_t state_offset, state_bytes, flags, reserved;
} NvnExecStep;

typedef void (*NvnExecStateEmit)(void *user, const void *bytes, size_t n);
typedef struct { size_t end; uint32_t ops, reserved; } NvnExecStateSpan;
typedef struct { size_t end; uint32_t ops, draws; } NvnExecFrontSpan;
typedef struct { size_t end, end_step; uint32_t ops, draws; } NvnExecFrontComposeSpan;
typedef struct {
    size_t begin, end;
    uint32_t ops, draws, state_offset, state_bytes;
} NvnFrontStep;
// Compile a bounded uninterrupted pure-state prefix beginning exactly at begin. Return zero without emission when
// unsupported. Success emits one immutable, eight-byte-aligned blob and describes its complete source interval.
// The consumer owns the blob format; embedded source references are offsets, never pointers into commands.
int nvn_exec_state_compile(const uint8_t *commands, size_t begin, size_t limit,
                           NvnExecStateEmit emit, void *user, NvnExecStateSpan *out);
// The planning pass observes state but never executes ordinary draws. This summary folds state writes across at
// most sixteen such draws, stopping at every operation that can observe state or alter query/ordering conditions.
int nvn_exec_front_compile(const uint8_t *commands, size_t begin, size_t limit,
                           NvnExecStateEmit emit, void *user, NvnExecFrontSpan *out);
// Compose complete primary steps into a planning summary without decoding their compiled state blocks again.
// end_step is exclusive, and end is the original command offset after it. Failure emits nothing and leaves out
// untouched. The helper finishes reading primary_state_data before emit, which may reallocate its backing storage.
int nvn_exec_front_compose(const uint8_t *commands, size_t command_bytes,
                           const NvnExecStep *steps, size_t nsteps,
                           const uint8_t *primary_state_data, size_t primary_state_bytes, size_t begin_step,
                           NvnExecStateEmit emit, void *user, NvnExecFrontComposeSpan *out);
NvnExecFn gfx_exec_prepared_handler(uint32_t opcode);

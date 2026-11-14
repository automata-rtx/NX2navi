// Immutable state deltas shared by recording construction and replay.
#pragma once
#include <stdint.h>
#include <stddef.h>

#define EXEC_STATE_WRITES 192u
#define EXEC_STATE_OPS 1024u
enum { ES_COPY, ES_ZERO, ES_BOOL };
enum { ES_NO_NOTIFY, ES_UBO, ES_SSBO, ES_TEXTURE, ES_IMAGE };
enum { ES_HAS_VIEWPORT = 1, ES_HAS_SCISSOR = 2 };

typedef struct {
    size_t source;
    uint32_t destination;
    uint16_t bytes;
    uint8_t operation, notify;
} ExecStateWrite;

typedef struct {
    uint32_t count, dirty, bind_increments, texture_increments, flags, reserved;
    ExecStateWrite writes[EXEC_STATE_WRITES];
} ExecStateBuild;

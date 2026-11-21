// Classify physical-address writes from the loaded module, including old shader caches whose table reflection says
// read-only. Pointer storage class belongs to its result type, so Phi/Select/function parameters need no guess at
// their runtime address. Mesh output counts, invocation control and annotations do not write guest-addressable
// buffers. Classify those emitted extension instructions by their effects, not by their numeric opcode range.
#pragma once
#include <stdint.h>
#include <stdlib.h>

static inline int rc_shader_extension_no_guest_write(uint32_t op, uint32_t length) {
    switch (op) {
    case 5294: return length == 4 || length == 5; // EmitMeshTasksEXT: dispatch counts and optional task payload.
    case 5295: return length == 3; // SetMeshOutputsEXT: vertex/primitive counts, no pointer destination.
    case 5296: return length == 4; // GroupNonUniformPartitionEXT: subgroup value result.
    case 5299: return length == 3; // WritePackedPrimitiveIndices4x8NV: mesh output, not physical storage.
    case 5380: return length == 1; // DemoteToHelperInvocation: invocation control.
    case 5381: return length == 3; // IsHelperInvocationEXT: boolean result.
    case 5632: return length >= 3; // DecorateString: metadata, no execution effect.
    case 5633: return length >= 4; // MemberDecorateString: metadata, no execution effect.
    default: return 0;
    }
}

static inline int rc_shader_physical_writes(const uint32_t *w, uint64_t n) {
    if (!w || n < 5 || w[0] != 0x07230203 || !w[3] || w[3] > (1u << 20)) return 1;
    uint32_t bound = w[3];
    uint8_t *types = calloc(bound, 1), *pointers = calloc(bound, 1);
    int writes = 1, physical = 0;
    if (!types || !pointers) goto done;
    // 0 unknown, 1 a nonphysical pointer, 2 a physical pointer. Gather types before values, including forward uses.
    for (uint64_t i = 5; i < n;) {
        uint32_t len = w[i] >> 16, op = w[i] & 0xffff;
        if (!len || len > n - i) goto done;
        if (op == 32) {
            if (len != 4 || w[i+1] >= bound) goto done;
            types[w[i+1]] = w[i+2] == 5349 ? 2 : 1;
            physical |= w[i+2] == 5349;
        }
        i += len;
    }
    if (!physical) { writes = 0; goto done; }
    for (uint64_t i = 5; i < n;) {
        uint32_t len = w[i] >> 16, op = w[i] & 0xffff;
        int typed = op == 1 || op == 46 || op == 55 || op == 57 || (op >= 59 && op <= 61) ||
            (op >= 65 && op <= 67) || op == 70 || op == 83 || (op >= 120 && op <= 124) ||
            op == 169 || op == 245;
        if (typed) {
            if (len < 3 || w[i+1] >= bound || w[i+2] >= bound) goto done;
            pointers[w[i+2]] = types[w[i+1]];
        }
        i += len;
    }
    for (uint64_t i = 5; i < n;) {
        uint32_t len = w[i] >> 16, op = w[i] & 0xffff, at = 0;
        // Stores/copies and atomic stores have no result type. Read-only atomic loads do not mutate the source.
        if ((op >= 62 && op <= 64) || op == 228 || op == 319) at = 1;
        else if ((op >= 229 && op <= 242) || op == 318 || op == 5614 || op == 5615 || op == 6035) at = 3;
        else if (op >= 4096 && !rc_shader_extension_no_guest_write(op, len)) goto done;
        // Unknown extension effects remain conservative. Known floating-point atomics above still inspect their
        // actual destination pointer; a benign mesh/control/annotation opcode never hides a later physical store.
        if (at && (len <= at || w[i+at] >= bound || pointers[w[i+at]] != 1)) goto done;
        i += len;
    }
    writes = 0;
done:
    free(types);
    free(pointers);
    return writes;
}

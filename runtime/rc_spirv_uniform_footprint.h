#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { uint32_t offset, bytes; } RcSpirvUniformRange;
typedef struct { uint32_t variable, declared_bytes, range_count; } RcSpirvUniformProof;
typedef enum { RC_SPIRV_UNIFORM_NONE, RC_SPIRV_UNIFORM_PROVEN, RC_SPIRV_UNIFORM_CAPACITY } RcSpirvUniformResult;

// Recognize exact read offsets in a bounded subset of validated final Vertex SPIR-V.
// The selected binding is a single-member Block at offset zero with a literal array
// of 1..4096 vec4<f32> at stride 16. Only literal scalar loads qualify. This proves
// byte demand and leaves descriptor range, original provider, and binding to the caller.
// NONE clears proof. CAPACITY reports the complete count and writes no ranges.
RcSpirvUniformResult rc_spirv_uniform_vertex_footprint(const uint32_t *words, size_t word_count,
    uint32_t set, uint32_t binding, RcSpirvUniformProof *proof, RcSpirvUniformRange *ranges, size_t capacity);

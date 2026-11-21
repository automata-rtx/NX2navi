"""Exercise the production planner for an aligned read-only SSBO handle domain."""

import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = r'''
#include "gfx.h"
#include "shader_pool_indices.h"
#include <assert.h>
#include <stdio.h>

#define I(op, ...) ((1u + sizeof((uint32_t[]){__VA_ARGS__}) / sizeof(uint32_t)) << 16 | (op)), __VA_ARGS__
// The texture variable has ID 4. OpTypeVector also uses the literal 4, which is not an ID use.
enum { U32=1, F32, VOID, POOL, EXTENSION, ZERO, MAXIMUM, MASK, COUNT, RUNTIME,
       RECORD, PTR_STORAGE, PTR_WORD, TABLE, PTR_LOCAL, LOCAL, IMAGE, ARRAY,
       PTR_POOL, PTR_IMAGE, FUNCTION_TYPE, ENTRY, BLOCK, DYNAMIC, RAW_PTR,
       C0_ROWS, C0_RECORD, C0, C0_POINTER, C0_WORD, C0_ROW, C0_OFFSET, C0_VALUE, STEP, SCALED, ADDRESS,
       RAW_VALUE, STORED, MASKED, CLAMPED, SELECTED, UNUSED, SAMPLER, SAMPLERS, PTR_SAMPLERS, SAMPLER_POOL,
       PTR_SAMPLER, SAMPLER_COUNT, SAMPLER_MAX, SHIFT20, SAMPLER_BITS, SAMPLER_CLAMPED, SELECTED_SAMPLER,
       IMAGE_VALUE, SAMPLER_VALUE, SAMPLED_IMAGE, COMBINED, FLOAT_ZERO, COORD_TYPE, COORD, COMPARED, F4_TYPE };

static const uint32_t module[] = {
    0x07230203, 0x00010300, 0, 64, 0,
    I(11, EXTENSION, 0x4c534c47, 0x6474732e, 0x3035342e, 0),
    I(15, 4, ENTRY, 0x6e69616d, 0),
    I(71, RUNTIME, 6, 4), I(71, C0_ROWS, 6, 16), I(71, C0, 33, 3),
    I(72, C0_RECORD, 0, 35, 0),
    I(71, TABLE, 33, 7), I(71, POOL, 33, 9), I(71, SAMPLER_POOL, 33, 10),
    I(71, UNUSED, 0),
    I(72, RECORD, 0, 35, 0),
    I(19, VOID), I(21, U32, 32, 0), I(22, F32, 32),
    I(43, U32, ZERO, 0), I(43, U32, MAXIMUM, 7),
    I(43, U32, MASK, 0xfffff), I(43, U32, COUNT, 8),
    I(43, U32, SAMPLER_COUNT, 4096), I(43, U32, SAMPLER_MAX, 4095), I(43, U32, SHIFT20, 20),
    I(29, RUNTIME, U32), I(30, RECORD, RUNTIME),
    I(43, U32, C0_ROW, 5), I(43, U32, STEP, 8),
    I(28, C0_ROWS, U32, COUNT), I(30, C0_RECORD, C0_ROWS),
    I(32, C0_POINTER, 2, C0_RECORD), I(32, C0_WORD, 2, U32), I(59, C0_POINTER, C0, 2),
    I(32, PTR_STORAGE, 12, RECORD), I(32, PTR_WORD, 12, U32),
    I(59, PTR_STORAGE, TABLE, 12), I(32, PTR_LOCAL, 7, U32),
    I(25, IMAGE, F32, 1, 1, 0, 0, 1, 0), I(28, ARRAY, IMAGE, COUNT),
    I(32, PTR_POOL, 0, ARRAY), I(32, PTR_IMAGE, 0, IMAGE), I(59, PTR_POOL, POOL, 0),
    I(26, SAMPLER), I(28, SAMPLERS, SAMPLER, SAMPLER_COUNT), I(32, PTR_SAMPLERS, 0, SAMPLERS),
    I(32, PTR_SAMPLER, 0, SAMPLER), I(59, PTR_SAMPLERS, SAMPLER_POOL, 0),
    I(27, SAMPLED_IMAGE, IMAGE), I(43, F32, FLOAT_ZERO, 0), I(23, COORD_TYPE, F32, 2),
    I(23, F4_TYPE, F32, 4), I(44, COORD_TYPE, COORD, FLOAT_ZERO, FLOAT_ZERO),
    I(33, FUNCTION_TYPE, VOID), I(54, VOID, ENTRY, 0, FUNCTION_TYPE), I(248, BLOCK),
    I(59, PTR_LOCAL, LOCAL, 7, ZERO), I(1, U32, DYNAMIC), I(128, U32, UNUSED, ZERO, ZERO),
    I(65, C0_WORD, C0_OFFSET, C0, ZERO, C0_ROW), I(61, U32, C0_VALUE, C0_OFFSET),
    I(132, U32, SCALED, DYNAMIC, STEP), I(128, U32, ADDRESS, SCALED, C0_VALUE),
    I(65, PTR_WORD, RAW_PTR, TABLE, ZERO, ADDRESS), I(61, U32, RAW_VALUE, RAW_PTR),
    I(62, LOCAL, RAW_VALUE), I(61, U32, STORED, LOCAL),
    I(199, U32, MASKED, STORED, MASK), I(12, U32, CLAMPED, EXTENSION, 38, MASKED, MAXIMUM),
    I(65, PTR_IMAGE, SELECTED, POOL, CLAMPED), I(61, IMAGE, IMAGE_VALUE, SELECTED),
    I(194, U32, SAMPLER_BITS, STORED, SHIFT20),
    I(12, U32, SAMPLER_CLAMPED, EXTENSION, 38, SAMPLER_BITS, SAMPLER_MAX),
    I(65, PTR_SAMPLER, SELECTED_SAMPLER, SAMPLER_POOL, SAMPLER_CLAMPED),
    I(61, SAMPLER, SAMPLER_VALUE, SELECTED_SAMPLER), I(86, SAMPLED_IMAGE, COMBINED, IMAGE_VALUE, SAMPLER_VALUE),
    I(89, F32, COMPARED, COMBINED, COORD, FLOAT_ZERO), 1u << 16 | 253, 1u << 16 | 56,
};

static GfxShaderRes c0 = {.binding=3, .count=1, .slot=0};
static GfxShaderRes buffer = {.binding=7, .count=1, .slot=1};
static const GfxShaderRes textures[] = {
    {.binding=9, .count=8, .type=0x403, .pool=1, .separate=1},
    {.binding=10, .count=4096, .pool=2},
};
static const GfxShaderRes *const resources[] = {&c0, &buffer, textures, NULL};
static const int counts[] = {1, 1, 2, 0};
static const uint32_t handles[] = {0x40100004, 9, 1, 1};
static int reader_status = 1;

static int read_domain(void *context, const GfxShaderRes *source, const uint8_t **data, uint32_t *size) {
    assert(context == handles && source == &buffer);
    *data = (const uint8_t *)handles;
    *size = sizeof handles;
    return reader_status;
}

static int c0_status = 1;
static unsigned c0_reads;
static uint32_t c0_value = 2;
static int read_c0(void *context, int kind, const GfxShaderRes *resource, uint32_t offset, uint32_t *out) {
    assert(context == handles && kind == GFX_RES_UBO && resource == &c0 && offset == 5*16);
    ++c0_reads;
    *out = c0_value;
    return c0_status;
}

static uint32_t *instruction(uint32_t *code, unsigned opcode, unsigned result) {
    for (size_t i = 5; i < sizeof module / sizeof module[0]; i += pi_len(code + i))
        if (pi_op(code + i) == opcode && (!result || code[i + pi_result_operand(opcode)] == result)) return code + i;
    abort();
}

static GfxShaderRes guard_buffers[] = {{.binding=7,.count=1,.slot=1}, {.binding=8,.count=1,.slot=2}};
typedef struct { uint32_t count; int status, reads; } GateWord;

static int read_gate(void *opaque, int kind, const GfxShaderRes *resource, uint32_t offset, uint32_t *out) {
    GateWord *gate = opaque;
    assert(kind == GFX_RES_SSBO && resource == &guard_buffers[1] && offset == 0);
    gate->reads++;
    *out = gate->count;
    return gate->status;
}

static void check_inactive_pool(void) {
    enum { GUARD_BUFFER=64, GUARD_BOOL, GUARD_PTR, GUARD_COUNT, GUARD_TEST, LIVE, MERGE, BYPASS, BOUND };
    const uint32_t deco[] = {I(71, GUARD_BUFFER, 33, 8)};
    const uint32_t vars[] = {I(59, PTR_STORAGE, GUARD_BUFFER, 12)};
    const uint32_t type[] = {I(20, GUARD_BOOL)};
    const uint32_t before_gate[] = {I(65, PTR_IMAGE, BYPASS, POOL, CLAMPED)};
    const uint32_t branch[] = {I(65, PTR_WORD, GUARD_PTR, GUARD_BUFFER, ZERO, ZERO),
        I(61, U32, GUARD_COUNT, GUARD_PTR), I(173, GUARD_BOOL, GUARD_TEST, GUARD_COUNT, ZERO),
        I(247, MERGE, 0), I(250, GUARD_TEST, LIVE, MERGE), I(248, LIVE)};
    const uint32_t merge[] = {I(249, MERGE), I(248, MERGE)};
    const GfxShaderRes *const reflected[] = {&c0, guard_buffers, textures, NULL};
    const int sizes[] = {1, 2, 2, 0};
    uint32_t code[sizeof module / sizeof module[0] + 48];
    for (unsigned mode = 0; mode < 4; ++mode) {
        size_t used = 5;
        memcpy(code, module, used * sizeof *code);
        code[3] = BOUND;
        for (size_t pos = 5; pos < sizeof module / sizeof module[0]; pos += pi_len(module + pos)) {
            const uint32_t *i = module + pos, op = pi_op(i), length = pi_len(i);
#define APPEND(words) do { memcpy(code+used, (words), sizeof (words)); used += sizeof (words)/sizeof *(words); } while (0)
            if (op == 19) { APPEND(deco); APPEND(type); }
            if (op == PX_VAR && i[2] == TABLE) APPEND(vars);
            if (op == PX_CHAIN && i[2] == SELECTED) {
                if (mode == 1) APPEND(before_gate);
                const size_t edge = used;
                APPEND(branch);
                if (mode == 2) code[edge+3] = TABLE;
            }
            if (op == 253) APPEND(merge);
#undef APPEND
            memcpy(code+used, i, length*sizeof *code);
            used += length;
        }
        guard_buffers[1].flags = mode == 3;
        ShaderPoolIndices *plan = shader_pool_indices_build(code, used, reflected, sizes);
        assert(plan && plan->domain_source == &guard_buffers[0]);
        if (mode == 0) {
            uint32_t remapped[sizeof code / sizeof *code];
            memcpy(remapped, code, used * sizeof *code);
            assert(shader_pool_inactive_source(code, remapped, used, 0, 0));
            uint32_t *branch = NULL;
            for (size_t at = 5; at < used; at += pi_len(remapped + at)) {
                uint32_t *i = remapped + at;
                if (pi_op(i) == PX_DECORATE && pi_len(i) == 4 && i[2] == PI_DESCRIPTOR_BINDING)
                    i[3] += 4;
                if (pi_op(i) == 250) branch = i;
            }
            assert(branch && shader_pool_inactive_source(code, remapped, used, 0, 4));
            assert(!shader_pool_inactive_source(code, remapped, used, 0, 3));
            branch[3] = branch[2]; // Same length, final false edge would execute the pooled accesses.
            assert(!shader_pool_inactive_source(code, remapped, used, 0, 4));
        }
        GateWord word = {.status=1};
        assert(shader_pool_inactive_eval(plan, read_gate, &word) == (mode == 0));
        if (mode == 0) {
            assert(word.reads == 1);
            word.count = 1;
            assert(!shader_pool_inactive_eval(plan, read_gate, &word));
            word.count = UINT32_MAX;
            assert(shader_pool_inactive_eval(plan, read_gate, &word) == 1);
            word.status = 2;
            assert(shader_pool_inactive_eval(plan, read_gate, &word) == 2);
            word.status = 0;
            assert(!shader_pool_inactive_eval(plan, read_gate, &word));
        } else assert(!word.reads);
        free(plan);
    }
}

int main(void) {
    const size_t length = sizeof module / sizeof module[0];
    ShaderPoolIndices *plan = shader_pool_indices_build(module, length, resources, counts);
    assert(plan && plan->domain_source == &buffer && plan->domain_maximum == 7 && (plan->sampler_modes & GFX_POOL_SAMPLER_COMPARE));
    GfxShaderPoolIndex out[8];
    uint32_t count;
    int comparison_zero;
    assert(shader_pool_domain_eval(plan, read_domain, NULL, (void *)handles, out, 8, &count, &comparison_zero) == 1 && count == 4);
    assert(!comparison_zero);
    assert(out[0].index == 4 && out[1].index == 7 && out[2].index == 1 && out[3].index == 0);
    for (uint32_t i = 0; i < count; ++i) assert(out[i].type == 0x403);
    GfxShaderPoolSamplers samplers;
    assert(shader_pool_domain_evaluate(plan, read_domain, NULL, (void *)handles, out, 8, &count, NULL, &samplers) == 1);
    assert(samplers.compare[0] == 1 && samplers.compare[16] == 2); // Robust zero and separate index 1025.
    assert(shader_pool_domain_evaluate(plan, read_domain, NULL, (void *)handles, out, 3, &count, NULL, &samplers) == 0);
    assert(!count && !samplers.compare[0] && !samplers.compare[16]);
    assert(!shader_pool_domain_eval(plan, read_domain, NULL, (void *)handles, out, 3, &count, NULL) && !count);
    assert(shader_pool_domain_eval(plan, read_domain, read_c0, (void *)handles, out, 8, &count, &comparison_zero) == 1 && count == 2);
    assert(comparison_zero);
    assert(c0_reads == 1 && out[0].index == 1 && out[1].index == 0);
    c0_value = 7;
    assert(shader_pool_domain_eval(plan, read_domain, read_c0, (void *)handles, out, 8, &count, &comparison_zero) == 1 && count == 1);
    assert(out[0].index == 0 && comparison_zero);
    c0_status = 0;
    assert(shader_pool_domain_eval(plan, read_domain, read_c0, (void *)handles, out, 8, &count, &comparison_zero) == 1 && count == 4);
    assert(!comparison_zero);
    assert(out[0].index == 4 && out[1].index == 7 && out[2].index == 1 && out[3].index == 0);
    reader_status = 2;
    assert(shader_pool_domain_eval(plan, read_domain, NULL, (void *)handles, out, 8, &count, &comparison_zero) == 2 && !count);
    assert(!comparison_zero);
    assert(!samplers.ordinary[0]);
    free(plan);

    uint32_t changed[sizeof module / sizeof module[0]];
    reader_status = c0_status = 1;

    // A scalar index may name a completely different retained source from its image index.
    memcpy(changed, module, sizeof changed);
    instruction(changed, PX_CHAIN, SELECTED)[4] = ZERO;
    instruction(changed, PX_CHAIN, SELECTED_SAMPLER)[4] = C0_VALUE;
    plan = shader_pool_indices_build(changed, length, resources, counts);
    assert(plan && !plan->domain_source && (plan->sampler_modes & GFX_POOL_SAMPLER_COMPARE) && plan->sampler_count == 1);
    c0_reads = 0;
    c0_value = 4095;
    assert(shader_pool_indices_eval(plan, read_c0, (void *)handles, out, 8, &count) == 1);
    assert(!c0_reads && count == 1 && out[0].index == 0);
    assert(shader_pool_indices_evaluate(plan, read_c0, (void *)handles, out, 8, &count, &samplers) == 1);
    assert(c0_reads == 1 && count == 1 && out[0].index == 0 && samplers.compare[63] == UINT64_C(1) << 63);
    c0_status = 2;
    assert(shader_pool_indices_evaluate(plan, read_c0, (void *)handles, out, 8, &count, &samplers) == 2);
    assert(!count && !samplers.compare[63]);
    c0_status = 1;
    c0_value = 4096;
    assert(!shader_pool_indices_evaluate(plan, read_c0, (void *)handles, out, 8, &count, &samplers));
    free(plan);
    instruction(changed, PX_TYPE_IMAGE, IMAGE)[4] = 0;
    uint32_t *ordinary_op = instruction(changed, 89, 0);
    ordinary_op[0] = 5u << 16 | 87u;
    ordinary_op[1] = F4_TYPE;
    ordinary_op[5] = 1u << 16; // Remaining word becomes OpNop.
    GfxShaderRes ordinary_resources[2] = {textures[0], textures[1]};
    ordinary_resources[0].type = 3;
    const GfxShaderRes *const ordinary[] = {&c0, &buffer, ordinary_resources, NULL};
    plan = shader_pool_indices_build(changed, length, ordinary, counts);
    assert(plan && !plan->domain_source && plan->sampler_modes == GFX_POOL_SAMPLER_ORDINARY);
    c0_value = 17;
    assert(shader_pool_indices_evaluate(plan, read_c0, (void *)handles, out, 8, &count, &samplers) == 1);
    assert(!samplers.compare[0] && samplers.ordinary[0] == UINT64_C(1) << 17);
    free(plan);
    c0_value = 2;
    const uint32_t inexact[] = {PI_RELAXED_PRECISION, PI_NO_SIGNED_WRAP, PI_NO_UNSIGNED_WRAP};
    for (unsigned i = 0; i < sizeof inexact / sizeof inexact[0]; ++i) {
        memcpy(changed, module, sizeof changed);
        for (size_t at = 5; at < length; at += pi_len(changed + at)) {
            if (pi_op(changed + at) != PX_DECORATE || changed[at + 1] != UNUSED) continue;
            changed[at + 1] = SCALED;
            changed[at + 2] = inexact[i];
            break;
        }
        plan = shader_pool_indices_build(changed, length, resources, counts);
        assert(plan && !plan->domain_address);
        assert(shader_pool_domain_eval(plan, read_domain, read_c0, (void *)handles, out, 8, &count, &comparison_zero) == 1 && count == 4);
        assert(!comparison_zero);
        free(plan);
    }
    memcpy(changed, module, sizeof changed);
    instruction(changed, 43, SHIFT20)[3] = 19;
    plan = shader_pool_indices_build(changed, length, resources, counts);
    assert(plan && plan->domain_address && !(plan->sampler_modes & GFX_POOL_SAMPLER_COMPARE));
    free(plan);
    memcpy(changed, module, sizeof changed);
    instruction(changed, 89, 0)[0] = 6u << 16 | 87;
    plan = shader_pool_indices_build(changed, length, resources, counts);
    assert(plan && plan->domain_address && !(plan->sampler_modes & GFX_POOL_SAMPLER_COMPARE));
    free(plan);
    memcpy(changed, module, sizeof changed);
    instruction(changed, 43, MASK)[3] = 0xffff;
    assert(!shader_pool_indices_build(changed, length, resources, counts));
    memcpy(changed, module, sizeof changed);
    instruction(changed, 71, 0)[3] = 8;
    assert(!shader_pool_indices_build(changed, length, resources, counts));
    memcpy(changed, module, sizeof changed);
    instruction(changed, 65, RAW_PTR)[0] = 6u << 16 | 66;
    assert(!shader_pool_indices_build(changed, length, resources, counts));
    memcpy(changed, module, sizeof changed);
    instruction(changed, 62, 0)[1] = RAW_PTR;
    assert(!shader_pool_indices_build(changed, length, resources, counts));
    memcpy(changed, module, sizeof changed);
    instruction(changed, 25, IMAGE)[4] = 0;
    assert(!shader_pool_indices_build(changed, length, resources, counts));
    buffer.flags = 1;
    assert(!shader_pool_indices_build(module, length, resources, counts));
    check_inactive_pool();
    puts("Pool domains: dynamic table, C0 residue, unproved C0 widening, robust zero, exact physical type and unsafe shape refusals passed");
}
'''


def main():
    compiler = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="shader-pool-domain-") as temp:
        source = Path(temp) / "check.c"
        source.write_text(FIXTURE, encoding="utf-8")
        exe = source.with_suffix(".exe" if os.name == "nt" else "")
        subprocess.run([str(compiler), "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "runtime"),
                        str(source), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

// Mirror the guest fragment's final opacity into a second attachment. Doing this in the original
// draw shares its sampled fonts and textures, shader discard, scissor and primitive coverage.
#include <stdlib.h>
#include <string.h>
#include "vk_dlaa.h"
#include "platform/platform.h"

enum Coverage { SOURCE_OVER, REPLACE, MULTIPLY, MULTIPLY_PLUS_ALPHA, ADD_ALPHA };

enum { OpExtInstImport = 11, OpExtInst = 12, OpEntryPoint = 15, OpTypeFloat = 22, OpTypeVector = 23,
       OpTypePointer = 32, OpConstant = 43, OpFunction = 54, OpVariable = 59, OpLoad = 61, OpStore = 62,
       OpDecorate = 71, OpCompositeConstruct = 80, OpCompositeExtract = 81, OpFAdd = 129, OpFSub = 131,
       OpFMul = 133, OpReturn = 253, Output = 3, Location = 30, FMin = 37, FMax = 40, FClamp = 43 };

// Only translated fragment entry points with one float4 color export are eligible. Read it at normal
// entry-point returns, after helper calls and component stores. A discarded fragment writes neither target.
static uint32_t *coverage_code(const uint32_t *w, size_t n, enum Coverage mode, size_t *words) {
    if (!w || n < 5 || w[0] != 0x07230203) return NULL;
    uint32_t scalar = 0, vec = 0, ptr = 0, color = 0, entry = 0, glsl = 0;
    size_t fn = 0, decoration = 0, returns = 0;
    for (size_t i = 5; i < n; i += w[i] >> 16) {
        const uint32_t *p = w + i;
        unsigned len = p[0] >> 16;
        if (!len || len > n - i) return NULL;
        switch (p[0] & 65535) {
        case OpExtInstImport:
            if (len >= 6 && !memcmp(p + 2, "GLSL.std.450", 13)) glsl = p[1];
            break;
        case OpEntryPoint:
            if (len < 4 || entry || p[1] != 4) return NULL;
            entry = p[2];
            break;
        case OpTypeFloat: if (len == 3 && p[2] == 32) scalar = p[1]; break;
        case OpTypeVector: if (len == 4 && scalar && p[2] == scalar && p[3] == 4) vec = p[1]; break;
        case OpTypePointer: if (len == 4 && vec && p[2] == Output && p[3] == vec) ptr = p[1]; break;
        case OpVariable:
            if (len >= 4 && ptr && p[1] == ptr && p[3] == Output) {
                for (size_t j = 5; j < i; j += w[j] >> 16)
                    if (w[j] == (4u << 16 | OpDecorate) && w[j + 1] == p[2] && w[j + 2] == Location && w[j + 3] == 0)
                        color = p[2], decoration = j;
            }
            break;
        case OpFunction:
            if (len != 5) return NULL;
            if (!fn) fn = i;
            break;
        case OpReturn: returns++; break;
        }
    }
    if (!fn || !color || !entry || !returns || (!glsl && mode >= MULTIPLY)) return NULL;
    uint32_t *code = malloc((n + 17 + 128 * returns) * sizeof *code);
    if (!code) return NULL;
    uint32_t id = w[3], out = id++, zero = id++, one = id++, current = 0;
    size_t k = 5;
    memcpy(code, w, 5 * sizeof *w);
#define EMIT(op, ...) do { const uint32_t operands[] = {__VA_ARGS__}; \
        code[k++] = ((1 + sizeof operands / sizeof *operands) << 16) | (op); \
        memcpy(code + k, operands, sizeof operands); k += sizeof operands / sizeof *operands; } while (0)
    for (size_t i = 5; i < n; i += w[i] >> 16) {
        unsigned op = w[i] & 65535, len = w[i] >> 16;
        if (i == decoration) EMIT(OpDecorate, out, Location, 1);
        if (i == fn) {
            EMIT(OpConstant, scalar, zero, 0);
            EMIT(OpConstant, scalar, one, 0x3f800000);
            EMIT(OpVariable, ptr, out, Output);
        }
        if (op == OpFunction) current = w[i + 2];
        if (op == OpReturn && current == entry) {
            uint32_t rgba = id++, alpha = id++, coverage = alpha;
            EMIT(OpLoad, vec, rgba, color);
            EMIT(OpCompositeExtract, scalar, alpha, rgba, 3);
            if (mode == REPLACE) coverage = one;
            if (mode >= MULTIPLY) {
                uint32_t rgb[3];
                for (unsigned c = 0; c < 3; c++) {
                    rgb[c] = id++;
                    EMIT(OpCompositeExtract, scalar, rgb[c], rgba, c);
                }
                uint32_t pair = id++, brightest = id++;
                EMIT(OpExtInst, scalar, pair, glsl, FMax, rgb[0], rgb[1]);
                EMIT(OpExtInst, scalar, brightest, glsl, FMax, pair, rgb[2]);
                if (mode == ADD_ALPHA) {
                    // Additive color has no unique source-over opacity. Weight the authored alpha by
                    // the light actually added so faint glows fade smoothly into the untouched scene.
                    coverage = id++;
                    EMIT(OpFMul, scalar, coverage, alpha, brightest);
                } else {
                    // A gray multiplier attenuates the background by exactly 1 - transmission.
                    // For a colored tint or gain use the channel changed most from identity.
                    pair = id++;
                    uint32_t darkest = id++;
                    EMIT(OpExtInst, scalar, pair, glsl, FMin, rgb[0], rgb[1]);
                    EMIT(OpExtInst, scalar, darkest, glsl, FMin, pair, rgb[2]);
                    if (mode == MULTIPLY_PLUS_ALPHA) {
                        uint32_t lo = id++, hi = id++;
                        EMIT(OpFAdd, scalar, lo, darkest, alpha);
                        EMIT(OpFAdd, scalar, hi, brightest, alpha);
                        darkest = lo;
                        brightest = hi;
                    }
                    uint32_t attenuation = id++, gain = id++;
                    EMIT(OpFSub, scalar, attenuation, one, darkest);
                    EMIT(OpFSub, scalar, gain, brightest, one);
                    coverage = id++;
                    EMIT(OpExtInst, scalar, coverage, glsl, FMax, attenuation, gain);
                }
                uint32_t clamped = id++;
                EMIT(OpExtInst, scalar, clamped, glsl, FClamp, coverage, zero, one);
                coverage = clamped;
            }
            uint32_t result = id++;
            EMIT(OpCompositeConstruct, vec, result, coverage, coverage, coverage, coverage);
            EMIT(OpStore, out, result);
        }
        memcpy(code + k, w + i, len * sizeof *w);
        if (op == OpEntryPoint) code[k] += 1u << 16;
        k += len;
        if (op == OpEntryPoint) code[k++] = out;
    }
#undef EMIT
    code[3] = id;
    *words = k;
    return code;
}

uint32_t *vk_hud_overlay_code(const uint32_t *spirv, size_t n, size_t *words) {
    return coverage_code(spirv, n, SOURCE_OVER, words);
}

static const GfxShader *coverage_shader(const GfxShader *fs, enum Coverage mode) {
    if (gfx_shader_outputs(fs, NULL) != 1) return NULL;
    uint64_t n;
    const uint32_t *w = gfx_shader_spirv(fs, &n);
    size_t words;
    uint32_t *code = coverage_code(w, n, mode, &words);
    if (!code) return NULL;
    const GfxShader *variant = gfx_shader_variant(fs, code, words);
    free(code);
    return variant;
}

static const GfxShader *variant_for(const GfxShader *fs, enum Coverage mode) {
    typedef struct Variant {
        const GfxShader *fs, *variant;
        enum Coverage mode;
        struct Variant *next;
    } Variant;
    static Variant *head;
    static PlatMutex lock = PLAT_MUTEX_INIT;
    plat_mutex_lock(&lock);
    Variant *v = head;
    while (v && (v->fs != fs || v->mode != mode)) v = v->next;
    if (!v && (v = malloc(sizeof *v))) {
        *v = (Variant){fs, coverage_shader(fs, mode), mode, head};
        head = v;
    }
    plat_mutex_unlock(&lock);
    return v ? v->variant : NULL;
}

int vk_hud_alpha_pipeline(VkPipeState *s) {
    unsigned rgb = vk_pipeline_target_wmask(s, 0) & 7;
    s->color_fmt[1] = NVN_FORMAT_R8;
    s->chmask.mask &= ~0xf0u;
    if (!rgb || !s->fs) return 1;
    if (rgb != 7 || s->color.logic_op != NVN_LOGIC_OP_COPY) return 0;
    enum Coverage mode;
    blend_st b = s->blend[0];
    if (!(s->color.blend_en & 1)) mode = REPLACE;
    else if (b.eq_rgb != NVN_BLEND_EQUATION_ADD) return 0;
    else if (b.dst_rgb == NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA &&
             (b.src_rgb == NVN_BLEND_FUNC_SRC_ALPHA || b.src_rgb == NVN_BLEND_FUNC_ONE)) mode = SOURCE_OVER;
    else if (b.src_rgb == NVN_BLEND_FUNC_ONE && b.dst_rgb == NVN_BLEND_FUNC_ZERO) mode = REPLACE;
    else if (b.src_rgb == NVN_BLEND_FUNC_ZERO && b.dst_rgb == NVN_BLEND_FUNC_SRC_COLOR) mode = MULTIPLY;
    else if (b.src_rgb == NVN_BLEND_FUNC_DST_COLOR && b.dst_rgb == NVN_BLEND_FUNC_SRC_ALPHA) mode = MULTIPLY_PLUS_ALPHA;
    else if (b.src_rgb == NVN_BLEND_FUNC_SRC_ALPHA && b.dst_rgb == NVN_BLEND_FUNC_ONE) mode = ADD_ALPHA;
    else return 0;
    const GfxShader *variant = variant_for(s->fs, mode);
    if (!variant) return 0;
    s->fs = variant;
    s->chmask.mask |= 0x10;
    s->color.blend_en |= 2;
    s->blend[1] = (blend_st){1, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA,
                             NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA,
                             NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    return 1;
}

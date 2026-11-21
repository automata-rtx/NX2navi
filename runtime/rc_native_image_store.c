#include "rc_native_image_store.h"
#include "rc_spirv_reconvergence.h"

#include <stdlib.h>
#include <string.h>

enum {
    SpvNop = 0, SpvName = 5, SpvMemberName = 6, SpvString = 7, SpvLine = 8,
    SpvExtension = 10, SpvExtInstImport = 11, SpvExtInst = 12,
    SpvMemoryModel = 14, SpvEntryPoint = 15, SpvExecutionMode = 16,
    SpvCapability = 17, SpvTypeVoid = 19, SpvTypeBool = 20, SpvTypeInt = 21, SpvTypeFloat = 22, SpvTypeVector = 23,
    SpvTypeImage = 25, SpvTypeSampler = 26, SpvTypeSampledImage = 27, SpvTypeArray = 28, SpvTypeRuntimeArray = 29,
    SpvTypeStruct = 30, SpvTypePointer = 32, SpvTypeFunction = 33,
    SpvConstantTrue = 41, SpvConstantFalse = 42, SpvConstant = 43, SpvConstantComposite = 44, SpvConstantNull = 46,
    SpvFunction = 54, SpvFunctionParameter = 55, SpvFunctionEnd = 56, SpvFunctionCall = 57,
    SpvVariable = 59, SpvImageTexelPointer = 60, SpvLoad = 61, SpvStore = 62, SpvAccessChain = 65,
    SpvInBoundsAccessChain = 66, SpvDecorate = 71, SpvMemberDecorate = 72, SpvDecorationGroup = 73,
    SpvGroupDecorate = 74, SpvGroupMemberDecorate = 75, SpvVectorShuffle = 79, SpvCompositeConstruct = 80,
    SpvCompositeExtract = 81, SpvCopyObject = 83, SpvImageSampleExplicitLod = 88, SpvImageSampleDrefExplicitLod = 90,
    SpvImageFetch = 95, SpvImageGather = 96, SpvImageRead = 98, SpvImageWrite = 99, SpvImage = 100, SpvImageQuerySizeLod = 103,
    SpvConvertFToU = 109, SpvConvertFToS = 110, SpvConvertSToF = 111, SpvConvertUToF = 112, SpvBitcast = 124,
    SpvSNegate = 126, SpvFNegate = 127, SpvIAdd = 128, SpvFAdd = 129, SpvISub = 130, SpvFSub = 131,
    SpvIMul = 132, SpvFMul = 133, SpvFDiv = 136, SpvIsNan = 156, SpvLogicalNotEqual = 165, SpvLogicalOr = 166, SpvLogicalAnd = 167, SpvLogicalNot = 168,
    SpvSelect = 169, SpvIEqual = 170, SpvINotEqual = 171, SpvUGreaterThan = 172, SpvSGreaterThan = 173,
    SpvUGreaterThanEqual = 174, SpvSGreaterThanEqual = 175, SpvULessThan = 176, SpvSLessThan = 177, SpvULessThanEqual = 178,
    SpvSLessThanEqual = 179, SpvFOrdEqual = 180, SpvFOrdNotEqual = 182, SpvFOrdLessThan = 184, SpvFOrdLessThanEqual = 188,
    SpvFOrdGreaterThan = 186, SpvFOrdGreaterThanEqual = 190, SpvShiftRightLogical = 194, SpvShiftRightArithmetic = 195,
    SpvShiftLeftLogical = 196, SpvBitwiseOr = 197, SpvBitwiseXor = 198, SpvBitwiseAnd = 199, SpvNot = 200, SpvBitFieldSExtract = 202, SpvBitFieldUExtract = 203,
    SpvBitReverse = 204, SpvBitCount = 205, SpvAtomicOr = 241,
    SpvPhi = 245, SpvLoopMerge = 246, SpvSelectionMerge = 247, SpvLabel = 248, SpvBranch = 249, SpvBranchConditional = 250,
    SpvReturn = 253, SpvReturnValue = 254, SpvNoLine = 317, SpvExecutionModeId = 331,
    SpvImageSparseRead = 320, SpvDecorateId = 332,
    SpvUniformConstant = 0, SpvInput = 1, SpvUniform = 2, SpvFunctionStorage = 7,
    SpvImageStorage = 11, SpvStorageBuffer = 12,
    SpvWorkgroupSize = 25, SpvWorkgroupId = 26, SpvLocalInvocationId = 27, SpvGlobalInvocationId = 28,
    SpvBuiltIn = 11, SpvNonWritable = 24, SpvNonReadable = 25, SpvBinding = 33, SpvDescriptorSet = 34, SpvLocalSize = 17,
    SpvRelaxedPrecision = 0, SpvBlock = 2, SpvImageOperandLod = 2,
    SpvImageOperandConstOffset = 8, SpvImageOperandConstOffsets = 32, SpvImageGatherExtended = 25,
    SpvR32ui = 33, SpvScopeDevice = 1, SpvNoSignedWrap = 4469, SpvNoUnsignedWrap = 4470,
    SpvGLCompute = 5, SpvDim2D = 1, SpvDim3D = 2, GlslRoundEven = 2, GlslTrunc = 3, GlslFAbs = 4, GlslFloor = 8, GlslFract = 9,
    GlslSin = 13, GlslCos = 14, GlslExp2 = 29, GlslLog2 = 30, GlslSqrt = 31, GlslInverseSqrt = 32, GlslUMin = 38, GlslSMin = 39, GlslFMax = 40, GlslUMax = 41,
    GlslSMax = 42, GlslFClamp = 43, GlslUClamp = 44, GlslSClamp = 45, GlslFma = 50, GlslFindILsb = 73, GlslFindUMsb = 75, GlslNMin = 79, GlslNMax = 80, GlslNClamp = 81,
    SpvShader = 1, SpvStorageImageReadWithoutFormat = 55, SpvStorageImageWriteWithoutFormat = 56,
};

typedef struct {
    const uint32_t *code;
    uint32_t builtin, set, binding; // encoded as value + 1
    uint32_t stored, source;       // last local store, and its captured SSA source at a load
    uint8_t denied_access, block, atomic_used;
} ImageId;

typedef struct {
    ImageId *ids;
    uint32_t bound;
    RcNativeImageStore proof;
    RcNativeImageSeedBindings *seed;
    uint32_t seed_descriptors[64];
    int formatless_writes, formatless_reads, rmw, gather_extended;
} ImageModule;

static uint32_t op(const uint32_t *p) { return *p & UINT16_MAX; }
static uint32_t len(const uint32_t *p) { return *p >> 16; }

static const uint32_t *definition(const ImageModule *m, uint32_t id) {
    return id < m->bound ? m->ids[id].code : NULL;
}

static int int32(const ImageModule *m, uint32_t id) {
    const uint32_t *p = definition(m, id);
    return p && op(p) == SpvTypeInt && len(p) == 4 && p[2] == 32;
}

static int int_vector(const ImageModule *m, uint32_t id, uint32_t width) {
    const uint32_t *p = definition(m, id);
    return p && op(p) == SpvTypeVector && len(p) == 4 && int32(m, p[2]) && p[3] == width;
}

static int literal(const ImageModule *m, uint32_t id, uint32_t *value) {
    const uint32_t *p = definition(m, id);
    if (!p || !((op(p) == SpvConstant && len(p) == 4) ||
                (op(p) == SpvConstantNull && len(p) == 3)) || !int32(m, p[1])) return 0;
    *value = op(p) == SpvConstant ? p[3] : 0;
    return 1;
}

static int variable(const uint32_t *p, uint32_t storage) {
    return p && op(p) == SpvVariable && len(p) >= 4 && p[3] == storage;
}

// Coefficients in SPIR-V's modulo-2^32 integer arithmetic, including bitcasts
// between int and uint. Dispatch then checks that the matched result is in
// nonnegative signed coordinate range.
typedef struct { uint32_t workgroup[3], invocation[3], constant; } Affine;

static int builtin(const ImageModule *m, uint32_t id, uint32_t axis, Affine *out) {
    const uint32_t *v = definition(m, id);
    if (axis >= 3 || !variable(v, SpvInput)) return 0;
    const uint32_t *ptr = definition(m, v[1]);
    if (!ptr || op(ptr) != SpvTypePointer || len(ptr) != 4 || ptr[2] != SpvInput ||
        !int_vector(m, ptr[3], 3)) return 0;
    switch (m->ids[id].builtin) {
    case SpvWorkgroupId + 1: out->workgroup[axis] = 1; return 1;
    case SpvLocalInvocationId + 1: out->invocation[axis] = 1; return 1;
    case SpvGlobalInvocationId + 1:
        out->workgroup[axis] = m->proof.local[axis];
        out->invocation[axis] = 1;
        return 1;
    default: return 0;
    }
}

static int affine(const ImageModule *m, uint32_t id, Affine *out, unsigned *fuel) {
    const uint32_t *p = definition(m, id);
    if (!p || !*fuel || len(p) < 3 || !int32(m, p[1])) return 0;
    --*fuel;
    *out = (Affine){0};
    if (literal(m, id, &out->constant)) return 1;
    uint32_t axis, factor;
    const uint32_t *q;
    Affine right;
    switch (op(p)) {
    case SpvLoad:
        if (len(p) != 4) return 0;
        if (m->ids[id].source) return affine(m, m->ids[id].source, out, fuel);
        q = definition(m, p[3]);
        return q && (op(q) == SpvAccessChain || op(q) == SpvInBoundsAccessChain) && len(q) == 5 &&
               literal(m, q[4], &axis) && builtin(m, q[3], axis, out);
    case SpvCompositeExtract:
        if (len(p) != 5) return 0;
        q = definition(m, p[3]);
        if (!q || op(q) != SpvLoad || len(q) != 4 || !int_vector(m, q[1], 3)) return 0;
        return builtin(m, q[3], p[4], out);
    case SpvBitcast:
    case SpvCopyObject:
        return len(p) == 4 && affine(m, p[3], out, fuel);
    case SpvIAdd:
    case SpvISub:
        if (len(p) != 5 || !affine(m, p[3], out, fuel) || !affine(m, p[4], &right, fuel)) return 0;
        factor = op(p) == SpvIAdd ? 1 : UINT32_MAX;
        for (uint32_t i = 0; i < 3; i++) {
            out->workgroup[i] += right.workgroup[i] * factor;
            out->invocation[i] += right.invocation[i] * factor;
        }
        out->constant += right.constant * factor;
        return 1;
    case SpvIMul:
    case SpvShiftLeftLogical:
        if (len(p) != 5 || !literal(m, p[4], &factor) || !affine(m, p[3], out, fuel)) return 0;
        if (op(p) == SpvShiftLeftLogical) {
            if (factor >= 32) return 0;
            factor = UINT32_C(1) << factor;
        }
        for (uint32_t i = 0; i < 3; i++) {
            out->workgroup[i] *= factor;
            out->invocation[i] *= factor;
        }
        out->constant *= factor;
        return 1;
    default: return 0;
    }
}

static int global_axis(const ImageModule *m, uint32_t id, uint32_t axis) {
    Affine coordinate;
    unsigned fuel = 128;
    if (!affine(m, id, &coordinate, &fuel) || coordinate.constant) return 0;
    for (uint32_t i = 0; i < 3; i++)
        if (coordinate.workgroup[i] != (i == axis ? m->proof.local[i] : 0) ||
            coordinate.invocation[i] != (i == axis)) return 0;
    return 1;
}

static RcNativeImageScalar image_scalar(const ImageModule *m, uint32_t id) {
    const uint32_t *type = definition(m, id);
    if (type && op(type) == SpvTypeFloat && len(type) == 3 && type[2] == 32) return RC_NATIVE_IMAGE_FLOAT32;
    if (int32(m, id) && type[3] <= 1) return type[3] ? RC_NATIVE_IMAGE_SINT32 : RC_NATIVE_IMAGE_UINT32;
    return RC_NATIVE_IMAGE_SCALAR_NONE;
}

static RcNativeImageScalar format_scalar(uint32_t spirv_format) {
    if (spirv_format >= 1 && spirv_format <= 20) return RC_NATIVE_IMAGE_FLOAT32;
    if (spirv_format >= 21 && spirv_format <= 29) return RC_NATIVE_IMAGE_SINT32;
    if (spirv_format >= 30 && spirv_format <= 39) return RC_NATIVE_IMAGE_UINT32;
    return RC_NATIVE_IMAGE_SCALAR_NONE;
}

static uint32_t seed_value_type(const ImageModule *m, uint32_t id) {
    const uint32_t *value = definition(m, id);
    if (!value || len(value) < 3 || value[2] != id) return 0;
    const uint32_t opcode = op(value);
    return (opcode >= SpvTypeVoid && opcode <= SpvTypeFunction) || opcode == SpvFunction ||
        opcode == SpvString || opcode == SpvExtInstImport ? 0 : value[1];
}

static int seed_int_vector(const ImageModule *m, uint32_t type, uint32_t width) {
    if (!int_vector(m, type, width)) return 0;
    const RcNativeImageScalar scalar = image_scalar(m, definition(m, type)[2]);
    return scalar == RC_NATIVE_IMAGE_SINT32 || scalar == RC_NATIVE_IMAGE_UINT32;
}

static int seed_descriptor(ImageModule *m, uint32_t root) {
    if (!root || root >= m->bound) return 0;
    const ImageId *descriptor = &m->ids[root];
    if (descriptor->set != 1 || !descriptor->binding || descriptor->binding > 64) return 0;
    uint32_t *recorded = &m->seed_descriptors[descriptor->binding - 1];
    if (*recorded && *recorded != root) return 0;
    *recorded = root;
    return 1;
}

static int seed_access(ImageModule *m, uint32_t root, uint32_t image_format, RcNativeImageScalar scalar, uint8_t access,
                       uint8_t dimension) {
    const ImageId *descriptor = &m->ids[root];
    if ((descriptor->denied_access & access) || !seed_descriptor(m, root)) return 0;
    const uint8_t binding = descriptor->binding - 1;
    RcNativeImageSeedBindings *seed = m->seed;
    for (uint32_t i = 0; i < seed->count; i++) {
        RcNativeImageSeedBinding *entry = &seed->entries[i];
        if (entry->binding != binding) continue;
        if (entry->scalar != scalar || entry->image_format != image_format || entry->dimension != dimension) return 0;
        entry->access |= access;
        return 1;
    }
    if (seed->count == RC_NATIVE_IMAGE_SEED_MAX_BINDINGS) return 0;
    seed->entries[seed->count++] = (RcNativeImageSeedBinding){binding, (uint8_t)image_format, (uint8_t)scalar, access, dimension};
    return 1;
}

static int storage_binding(ImageModule *m, uint32_t image, uint32_t coordinate, uint32_t texel_type, int read) {
    const uint32_t *load = definition(m, image);
    if (!load || op(load) != SpvLoad || len(load) != 4) return 0;
    const uint32_t *v = definition(m, load[3]);
    const uint32_t *type = definition(m, load[1]);
    if (!variable(v, SpvUniformConstant) || !type || op(type) != SpvTypeImage ||
        len(type) != 9 || (type[3] != SpvDim2D && (!m->seed || type[3] != SpvDim3D)) ||
        type[4] || type[5] || type[6] || type[7] != 2) return 0;
    const uint32_t *pointer = definition(m, v[1]);
    if (!pointer || op(pointer) != SpvTypePointer || len(pointer) != 4 ||
        pointer[2] != SpvUniformConstant || pointer[3] != load[1]) return 0;
    RcNativeImageScalar scalar = image_scalar(m, type[2]);
    if (!scalar || (!type[8] && !(read ? m->formatless_reads : m->formatless_writes)) ||
        (type[8] && format_scalar(type[8]) != scalar)) return 0;
    const uint32_t *value = definition(m, texel_type);
    if (!value || op(value) != SpvTypeVector || len(value) != 4 || value[2] != type[2] || value[3] != 4) return 0;
    const ImageId *descriptor = &m->ids[load[3]];
    if (!descriptor->set || !descriptor->binding) return 0;
    if (m->seed) {
        const int volume = type[3] == SpvDim3D;
        const uint32_t *coordinates = definition(m, coordinate);
        if (len(v) != 4 || !coordinates || len(coordinates) < 3 || (volume
            ? !seed_int_vector(m, seed_value_type(m, coordinate), 3) : !int_vector(m, coordinates[1], 2))) return 0;
        const uint8_t access = read ? RC_NATIVE_IMAGE_SEED_READ : RC_NATIVE_IMAGE_SEED_WRITE;
        return seed_access(m, load[3], type[8], scalar, access, volume ? RC_NATIVE_IMAGE_SEED_3D : RC_NATIVE_IMAGE_SEED_2D);
    }
    if (m->proof.scalar && (m->proof.set != descriptor->set - 1 || m->proof.binding != descriptor->binding - 1 ||
                            m->proof.image_format != type[8] || m->proof.scalar != scalar)) return 0;
    m->proof.set = descriptor->set - 1;
    m->proof.binding = descriptor->binding - 1;
    m->proof.image_format = type[8];
    m->proof.scalar = scalar;
    m->proof.reads_storage |= read;
    return 1;
}

static int storage_write_binding(ImageModule *m, const uint32_t *p) {
    // No image operands that can change the selected texel, lod or memory semantics.
    if (len(p) != 4 && (len(p) != 5 || p[4])) return 0;
    const uint32_t *texel = definition(m, p[3]);
    return texel && len(texel) >= 3 && storage_binding(m, p[1], p[2], texel[1], 0);
}

static int storage_write(ImageModule *m, const uint32_t *p) {
    if (!storage_write_binding(m, p)) return 0;
    const uint32_t *xy = definition(m, p[2]);
    return xy && op(xy) == SpvCompositeConstruct && len(xy) == 5 && int_vector(m, xy[1], 2) &&
        global_axis(m, xy[3], 0) && global_axis(m, xy[4], 1);
}

static int record_id(ImageModule *m, const uint32_t *p, unsigned slot) {
    if (len(p) <= slot || !p[slot] || p[slot] >= m->bound || m->ids[p[slot]].code) return 0;
    m->ids[p[slot]].code = p;
    return 1;
}

static int decoration(ImageModule *m, const uint32_t *p, int coordinates) {
    if (len(p) < 3 || !p[1] || p[1] >= m->bound) return 0;
    // These allow an intermediate to differ from modular int32 arithmetic even
    // when the resulting canonical coefficients appear to be global XY.
    if (p[2] == SpvNoSignedWrap || p[2] == SpvNoUnsignedWrap || (coordinates && p[2] == SpvRelaxedPrecision)) return 0;
    ImageId *id = &m->ids[p[1]];
    uint32_t *v = p[2] == SpvBuiltIn ? &id->builtin :
                  p[2] == SpvDescriptorSet ? &id->set : p[2] == SpvBinding ? &id->binding : NULL;
    if (!v) return 1;
    if (len(p) != 4 || *v || p[3] == UINT32_MAX) return 0;
    // In Vulkan a WorkgroupSize built-in can supersede the entry point's LocalSize.
    if (coordinates && p[2] == SpvBuiltIn && p[3] == SpvWorkgroupSize) return 0;
    *v = p[3] + 1;
    return 1;
}

static int glsl_std450(const ImageModule *m, const uint32_t *p) {
    if (len(p) < 5) return 0;
    const uint32_t *import = definition(m, p[3]);
    static const char name[] = "GLSL.std.450";
    return import && op(import) == SpvExtInstImport && len(import) == 6 &&
           !memcmp(import + 2, name, sizeof name);
}

static int body_value(const ImageModule *m, const uint32_t *p) {
    // Value instructions whose results may feed the texel expression. Unknown
    // instructions, including control flow, calls, side effects and image
    // loads/atomics, make the whole certificate unavailable.
    switch (op(p)) {
    case SpvVariable: return (len(p) == 4 || len(p) == 5) && p[3] == SpvFunctionStorage;
    case SpvLoad: return len(p) == 4;
    case SpvAccessChain: case SpvInBoundsAccessChain: return len(p) >= 5;
    case SpvCompositeConstruct: case SpvCompositeExtract: return len(p) >= 4;
    case SpvCopyObject: case SpvBitcast: case SpvConvertUToF: case SpvConvertSToF:
    case SpvFNegate: case SpvIsNan: return len(p) == 4;
    case SpvIAdd: case SpvFAdd: case SpvISub: case SpvFSub: case SpvIMul: case SpvFMul:
    case SpvShiftLeftLogical: case SpvBitwiseAnd: case SpvUGreaterThan: return len(p) == 5;
    case SpvImageSampleExplicitLod: return len(p) >= 6;
    case SpvSelect: return len(p) == 6;
    case SpvExtInst: return len(p) == 8 && p[4] == GlslFma && glsl_std450(m, p);
    default: return 0;
    }
}

static int scan_module(ImageModule *m, const uint32_t *words, size_t count) {
    int shader = 0, has_reconvergence_extension = 0, reconverges = 0;
    uint32_t entry = 0;
    int logical = 0, label = 0, written = 0, returned = 0, function = 0, complete = 0;
    for (size_t i = 5; i < count;) {
        const uint32_t *p = words + i;
        uint32_t n = len(p), opcode = op(p);
        if (!n || n > count - i || complete) return 0;
        i += n;
        if (opcode == SpvNop || opcode == SpvLine || opcode == SpvNoLine) continue;
        if (function) {
            if (opcode == SpvLabel) {
                if (n != 2 || label++) return 0;
            } else if (opcode == SpvFunctionEnd) {
                if (n != 1 || !returned) return 0;
                complete = 1;
            } else if (!label || returned) {
                return 0;
            } else if (opcode == SpvReturn) {
                if (n != 1 || !written) return 0;
                returned = 1;
            } else if (opcode == SpvImageWrite) {
                if (written++ || !storage_write(m, p)) return 0;
            } else if (opcode == SpvStore) {
                if (n != 3 || !variable(definition(m, p[1]), SpvFunctionStorage)) return 0;
                m->ids[p[1]].stored = p[2];
            } else {
                if (!body_value(m, p) || !record_id(m, p, 2)) return 0;
                if (opcode == SpvVariable && n == 5) m->ids[p[2]].stored = p[4];
                if (opcode == SpvLoad && variable(definition(m, p[3]), SpvFunctionStorage))
                    m->ids[p[2]].source = m->ids[p[3]].stored;
            }
            continue;
        }
        switch (opcode) {
        case SpvCapability:
            if (n != 2) return 0;
            if (p[1] == SpvShader) shader = 1;
            if (p[1] == SpvStorageImageWriteWithoutFormat) m->formatless_writes = 1;
            break;
        case SpvExtension:
            if (rc_spirv_maximal_reconvergence_extension(p, n)) has_reconvergence_extension = 1;
            break;
        case SpvMemoryModel:
            if (n != 3 || p[1] || (p[2] != 1 && p[2] != 3)) return 0; // Logical, GLSL450 or Vulkan
            logical = 1;
            break;
        case SpvEntryPoint:
            if (entry || n < 5 || p[1] != SpvGLCompute || p[3] != 0x6e69616d || p[4]) return 0;
            entry = p[2];
            break;
        case SpvExecutionMode:
            if (n < 3 || !entry || p[1] != entry) return 0;
            if (p[2] == SpvLocalSize) {
                if (n != 6 || m->proof.local[0] || !p[3] || !p[4] || !p[5]) return 0;
                memcpy(m->proof.local, p + 3, sizeof m->proof.local);
            } else if (p[2] == RC_SPV_MAXIMALLY_RECONVERGES_KHR) {
                // The existing body check still requires a single unconditional block.
                if (n != 3 || !shader || !has_reconvergence_extension || reconverges) return 0;
                reconverges = 1;
            } else if (p[2] < 4459 || p[2] > 4463) return 0; // floating point controls
            break;
        case SpvFunction:
            if (!logical || !entry || !m->proof.local[0] || n != 5 || p[2] != entry) return 0;
            function = 1;
            break;
        case SpvDecorate:
            if (!decoration(m, p, 1)) return 0;
            break;
        case SpvExecutionModeId: case SpvDecorateId: case SpvDecorationGroup:
        case SpvGroupDecorate: case SpvGroupMemberDecorate:
            return 0;
        case SpvExtInstImport: case SpvTypeInt: case SpvTypeFloat: case SpvTypeVector: case SpvTypeImage: case SpvTypePointer:
            if (!record_id(m, p, 1)) return 0;
            break;
        case SpvVariable: case SpvConstant: case SpvConstantComposite: case SpvConstantNull:
            if (!record_id(m, p, 2)) return 0;
            break;
        default:
            break;
        }
    }
    return complete;
}

enum { SEED_MAX_HELPERS = 4, SEED_MAX_PARAMS = 4, SEED_MAX_UNIFORM_INDICES = 4 };

typedef struct {
    uint32_t id, result;
    uint32_t params[SEED_MAX_PARAMS];
    uint8_t count, defined;
} SeedHelper;

typedef struct {
    SeedHelper helpers[SEED_MAX_HELPERS];
    unsigned count;
    const uint32_t *entry_start, *start;
    const SeedHelper *active;
    unsigned parameter, returned;
} SeedFunctions;

static int seed_scalar_type(const ImageModule *m, uint32_t id) {
    const uint32_t *type = definition(m, id);
    return image_scalar(m, id) || (type && op(type) == SpvTypeBool && len(type) == 2);
}

static uint32_t seed_pointer_type(const ImageModule *m, uint32_t type, uint32_t storage) {
    const uint32_t *pointer = definition(m, type);
    return pointer && op(pointer) == SpvTypePointer && len(pointer) == 4 && pointer[2] == storage ? pointer[3] : 0;
}

static int seed_local_pointer(const ImageModule *m, uint32_t id, const uint32_t *start, int parameter) {
    const uint32_t *value = definition(m, id);
    return value && value > start && (variable(value, SpvFunctionStorage) ||
        (parameter && op(value) == SpvFunctionParameter)) &&
        seed_scalar_type(m, seed_pointer_type(m, value[1], SpvFunctionStorage));
}

static int seed_helper_call(ImageModule *m, SeedFunctions *f, const uint32_t *p, uint32_t entry) {
    const uint32_t n = len(p);
    if (f->active || n < 4 || n > 4 + SEED_MAX_PARAMS || !p[3] || p[3] == entry || p[3] >= m->bound ||
        !seed_scalar_type(m, p[1])) return 0;
    SeedHelper *helper = f->helpers;
    while (helper < f->helpers + f->count && helper->id != p[3]) ++helper;
    const int fresh = helper == f->helpers + f->count;
    if (fresh && f->count == SEED_MAX_HELPERS) return 0;
    if (!fresh && (helper->result != p[1] || helper->count != n - 4)) return 0;
    for (uint32_t j = 4; j < n; ++j) {
        if (!seed_local_pointer(m, p[j], f->start, 0)) return 0;
        const uint32_t type = seed_value_type(m, p[j]);
        if (!fresh && helper->params[j - 4] != type) return 0;
        helper->params[j - 4] = type;
    }
    if (fresh) {
        helper->id = p[3];
        helper->result = p[1];
        helper->count = n - 4;
        f->count++;
    }
    return record_id(m, p, 2);
}

static int seed_function_begin(ImageModule *m, SeedFunctions *f, const uint32_t *p, int main) {
    const uint32_t *signature = definition(m, p[4]);
    if (!signature || op(signature) != SpvTypeFunction || len(signature) < 3 || signature[2] != p[1]) return 0;
    f->start = p;
    f->active = NULL;
    f->parameter = f->returned = 0;
    if (main) {
        f->entry_start = p;
        const uint32_t *result = definition(m, p[1]);
        return len(signature) == 3 && result && op(result) == SpvTypeVoid && len(result) == 2;
    }
    for (unsigned i = 0; i < f->count; ++i) {
        SeedHelper *helper = &f->helpers[i];
        if (helper->id != p[2]) continue;
        if (helper->defined || helper->result != p[1] || len(signature) != 3u + helper->count ||
            memcmp(signature + 3, helper->params, helper->count * sizeof *signature)) return 0;
        helper->defined = 1;
        f->active = helper;
        return 1;
    }
    return 0;
}

static uint32_t seed_helper_scalar(const ImageModule *m, const SeedFunctions *f, uint32_t id) {
    const uint32_t *value = definition(m, id);
    const uint32_t type = seed_value_type(m, id);
    if (!value || !seed_scalar_type(m, type)) return 0;
    if (value > f->start) return type;
    if (value >= f->entry_start) return 0;
    switch (op(value)) {
    case SpvConstant: case SpvConstantNull: case SpvConstantTrue: case SpvConstantFalse:
        return type;
    default:
        return 0;
    }
}

// Helpers read exact scalar parameters, locals, or a typed member of a set 0 uniform block.
static int seed_uniform_scalar(ImageModule *m, const SeedFunctions *f, const uint32_t *p) {
    const uint32_t n = len(p);
    if (n < 5 || n > 4 + SEED_MAX_UNIFORM_INDICES) return 0;
    const uint32_t *root = definition(m, p[3]);
    if (!variable(root, SpvUniform) || len(root) != 4) return 0;
    uint32_t current = seed_pointer_type(m, root[1], SpvUniform);
    const uint32_t *block = definition(m, current);
    if (!block || op(block) != SpvTypeStruct || !m->ids[current].block || !seed_descriptor(m, p[3])) return 0;
    for (uint32_t j = 4; j < n; ++j) {
        const uint32_t *type = definition(m, current);
        uint32_t index;
        if (!type || !int32(m, seed_helper_scalar(m, f, p[j]))) return 0;
        if (op(type) == SpvTypeStruct) {
            if (len(type) < 3 || !literal(m, p[j], &index) || index >= len(type) - 2) return 0;
            current = type[2 + index];
        } else if ((op(type) == SpvTypeArray || op(type) == SpvTypeVector) && len(type) == 4) {
            current = type[2];
        } else return 0;
    }
    return seed_scalar_type(m, current) && seed_pointer_type(m, p[1], SpvUniform) == current;
}

static int seed_helper_memory(ImageModule *m, const SeedFunctions *f, const uint32_t *p) {
    switch (op(p)) {
    case SpvVariable: {
        const uint32_t type = seed_pointer_type(m, p[1], SpvFunctionStorage);
        return seed_scalar_type(m, type) && (len(p) == 4 || seed_helper_scalar(m, f, p[4]) == type);
    }
    case SpvAccessChain: case SpvInBoundsAccessChain:
        return seed_uniform_scalar(m, f, p);
    case SpvLoad: {
        if (len(p) != 4) return 0;
        const uint32_t *pointer = definition(m, p[3]);
        return pointer && pointer > f->start && (seed_local_pointer(m, p[3], f->start, 1)
            ? seed_pointer_type(m, pointer[1], SpvFunctionStorage) == p[1]
            : (op(pointer) == SpvAccessChain || op(pointer) == SpvInBoundsAccessChain) &&
              seed_pointer_type(m, pointer[1], SpvUniform) == p[1]);
    }
    case SpvStore:
        return len(p) == 3 && seed_local_pointer(m, p[1], f->start, 0) &&
            seed_pointer_type(m, seed_value_type(m, p[1]), SpvFunctionStorage) == seed_helper_scalar(m, f, p[2]);
    default:
        return 1;
    }
}


// The image texel pointer is a direct typed descriptor reference. It may only
// be consumed by an attested atomic instruction in the entry point.
static int seed_rmw_texel_pointer(ImageModule *m, const uint32_t *p) {
    if (len(p) != 6) return 0;
    const uint32_t *root = definition(m, p[3]);
    if (!variable(root, SpvUniformConstant) || len(root) != 4) return 0;
    const uint32_t type = seed_pointer_type(m, root[1], SpvUniformConstant);
    const uint32_t *image = definition(m, type);
    if (!image || op(image) != SpvTypeImage || len(image) != 9 || image[3] != SpvDim2D ||
        image[4] || image[5] || image[6] || image[7] != 2 || image[8] != SpvR32ui ||
        image_scalar(m, image[2]) != RC_NATIVE_IMAGE_UINT32 ||
        seed_pointer_type(m, p[1], SpvImageStorage) != image[2] ||
        !seed_int_vector(m, seed_value_type(m, p[4]), 2)) return 0;
    uint32_t sample;
    return literal(m, p[5], &sample) && !sample && seed_descriptor(m, p[3]);
}

static int seed_rmw_atomic_or(ImageModule *m, const uint32_t *p) {
    if (len(p) != 7) return 0;
    const uint32_t *pointer = definition(m, p[3]);
    uint32_t scope, semantics;
    if (!pointer || op(pointer) != SpvImageTexelPointer ||
        image_scalar(m, p[1]) != RC_NATIVE_IMAGE_UINT32 ||
        seed_pointer_type(m, pointer[1], SpvImageStorage) != p[1] || seed_value_type(m, p[6]) != p[1] ||
        !literal(m, p[4], &scope) || scope != SpvScopeDevice ||
        !literal(m, p[5], &semantics) || semantics) return 0;
    if (!seed_access(m, pointer[3], SpvR32ui, RC_NATIVE_IMAGE_UINT32,
        RC_NATIVE_IMAGE_SEED_READ | RC_NATIVE_IMAGE_SEED_WRITE | RC_NATIVE_IMAGE_SEED_ATOMIC,
        RC_NATIVE_IMAGE_SEED_2D)) return 0;
    m->ids[p[3]].atomic_used = 1;
    return 1;
}

// Validate the pointer-producing and memory instructions permitted alongside
// the atomic. A texel pointer cannot enter ordinary memory or be copied,
// selected, extracted or passed to a helper.
static int seed_rmw_memory(ImageModule *m, const uint32_t *p) {
    const uint32_t n = len(p), opcode = op(p);
    const uint32_t *type = n >= 3 ? definition(m, p[1]) : NULL;
    const int pointer = type && op(type) == SpvTypePointer && len(type) == 4;
    if (opcode == SpvVariable) {
        if (!pointer || n < 4 || p[3] != type[2]) return 0;
        if (p[3] == SpvUniformConstant || p[3] == SpvUniform || p[3] == SpvStorageBuffer)
            return n == 4 && seed_descriptor(m, p[2]);
        return p[3] == SpvFunctionStorage || (p[3] == SpvInput && n == 4);
    }
    if (opcode == SpvAccessChain || opcode == SpvInBoundsAccessChain) {
        if (!pointer || n < 5 || (type[2] != SpvFunctionStorage && type[2] != SpvInput &&
            type[2] != SpvUniform && type[2] != SpvStorageBuffer)) return 0;
        const uint32_t *base = definition(m, p[3]);
        if (!variable(base, type[2])) return 0;
        uint32_t pointee = seed_pointer_type(m, base[1], type[2]);
        for (uint32_t i = 4; i < n; ++i) {
            const uint32_t *aggregate = definition(m, pointee);
            if (!aggregate || !int32(m, seed_value_type(m, p[i]))) return 0;
            if (op(aggregate) == SpvTypeStruct) {
                uint32_t member;
                if (!literal(m, p[i], &member) || member >= len(aggregate) - 2) return 0;
                pointee = aggregate[2 + member];
            } else if ((op(aggregate) == SpvTypeArray && len(aggregate) == 4) ||
                       (op(aggregate) == SpvTypeRuntimeArray && len(aggregate) == 3) ||
                       (op(aggregate) == SpvTypeVector && len(aggregate) == 4)) {
                pointee = aggregate[2];
            } else return 0;
        }
        return pointee == type[3];
    }
    if (opcode == SpvLoad || opcode == SpvStore) {
        if (n != (opcode == SpvLoad ? 4u : 3u)) return 0;
        const uint32_t target = opcode == SpvLoad ? p[3] : p[1];
        const uint32_t *address = definition(m, target);
        const uint32_t *address_type = definition(m, seed_value_type(m, target));
        if (!address || !address_type || op(address_type) != SpvTypePointer || len(address_type) != 4 ||
            address_type[2] == SpvImageStorage) return 0;
        if (opcode == SpvStore)
            return address_type[2] == SpvFunctionStorage &&
                (variable(address, SpvFunctionStorage) || op(address) == SpvAccessChain || op(address) == SpvInBoundsAccessChain) &&
                seed_value_type(m, p[2]) == address_type[3];
        return !pointer && address_type[3] == p[1] && (address_type[2] != SpvUniformConstant ||
            variable(address, SpvUniformConstant));
    }
    return !pointer && (!type || (op(type) != SpvTypeImage && op(type) != SpvTypeSampledImage &&
        op(type) != SpvTypeSampler));
}

static int seed_constant_vec2(const ImageModule *m, uint32_t id) {
    const uint32_t *p = definition(m, id);
    uint32_t literal_value;
    if (!p || len(p) < 3 || !seed_int_vector(m, p[1], 2)) return 0;
    return (op(p) == SpvConstantNull && len(p) == 3) ||
        (op(p) == SpvConstantComposite && len(p) == 5 &&
         seed_value_type(m, p[3]) == definition(m, p[1])[2] &&
         seed_value_type(m, p[4]) == definition(m, p[1])[2] &&
         literal(m, p[3], &literal_value) && literal(m, p[4], &literal_value));
}

static int seed_constant_offsets(const ImageModule *m, uint32_t id) {
    const uint32_t *p = definition(m, id);
    const uint32_t *array = p && len(p) >= 3 ? definition(m, p[1]) : NULL;
    uint32_t size;
    if (!array || op(array) != SpvTypeArray || len(array) != 4 ||
        !literal(m, array[3], &size) || size != 4 || !seed_int_vector(m, array[2], 2)) return 0;
    if (op(p) == SpvConstantNull && len(p) == 3) return 1;
    if (op(p) != SpvConstantComposite || len(p) != 7) return 0;
    for (unsigned i = 3; i < 7; ++i)
        if (seed_value_type(m, p[i]) != array[2] || !seed_constant_vec2(m, p[i])) return 0;
    return 1;
}

static const uint32_t *seed_sampled_type(ImageModule *m, uint32_t id, uint32_t depth) {
    const uint32_t *load = definition(m, id);
    if (!load || op(load) != SpvLoad || len(load) != 4) return NULL;
    const uint32_t *root = definition(m, load[3]);
    if (!variable(root, SpvUniformConstant) || len(root) != 4 ||
        seed_pointer_type(m, root[1], SpvUniformConstant) != load[1]) return NULL;
    const uint32_t *sampled = definition(m, load[1]);
    if (!sampled || op(sampled) != SpvTypeSampledImage || len(sampled) != 3) return NULL;
    const uint32_t *image = definition(m, sampled[2]);
    if (!image || op(image) != SpvTypeImage || len(image) != 9 || depth > 1 ||
        (image[3] != SpvDim2D && (image[3] != SpvDim3D || image[5])) || image[4] != depth || image[5] > 1 ||
        image[6] || image[7] != 1 || image[8] || !image_scalar(m, image[2]) ||
        (depth && (image[3] != SpvDim2D || image[5] != 1 || image_scalar(m, image[2]) != RC_NATIVE_IMAGE_FLOAT32)) ||
        !seed_descriptor(m, load[3])) return NULL;
    return image;
}

static int seed_sampled_texel_type(const ImageModule *m, uint32_t id, const uint32_t *image) {
    const uint32_t *result = definition(m, id);
    return result && op(result) == SpvTypeVector && len(result) == 4 && result[2] == image[2] && result[3] == 4;
}

// Only an ordinary direct sampled 2D descriptor, with a known component and no image operands.
static int seed_sampled_gather(ImageModule *m, const uint32_t *p) {
    const uint32_t n = len(p);
    if (n != 6 && (n != 7 || p[6]) && (!m->rmw || !m->gather_extended ||
        n != 8 || p[6] != SpvImageOperandConstOffsets || !seed_constant_offsets(m, p[7]))) return 0;
    const uint32_t *image = seed_sampled_type(m, p[3], 0);
    if (!image || image[3] != SpvDim2D || image[5] || !seed_sampled_texel_type(m, p[1], image)) return 0;
    const uint32_t *coordinate = definition(m, seed_value_type(m, p[4]));
    if (!coordinate || op(coordinate) != SpvTypeVector || len(coordinate) != 4 || coordinate[3] != 2 ||
        image_scalar(m, coordinate[2]) != RC_NATIVE_IMAGE_FLOAT32) return 0;
    const RcNativeImageScalar scalar = image_scalar(m, seed_value_type(m, p[5]));
    uint32_t component;
    return (scalar == RC_NATIVE_IMAGE_SINT32 || scalar == RC_NATIVE_IMAGE_UINT32) &&
        literal(m, p[5], &component) && component < 4;
}

// The supported comparison samples are direct 2D array views with explicit LOD zero.
static int seed_sampled_comparison(ImageModule *m, const uint32_t *p) {
    if (len(p) != 8 || p[6] != SpvImageOperandLod) return 0;
    const uint32_t *image = seed_sampled_type(m, p[3], 1);
    if (!image || p[1] != image[2]) return 0;
    const uint32_t *coordinate = definition(m, seed_value_type(m, p[4]));
    const uint32_t *lod = definition(m, p[7]);
    return coordinate && op(coordinate) == SpvTypeVector && len(coordinate) == 4 && coordinate[3] == 3 &&
        image_scalar(m, coordinate[2]) == RC_NATIVE_IMAGE_FLOAT32 &&
        image_scalar(m, seed_value_type(m, p[5])) == RC_NATIVE_IMAGE_FLOAT32 &&
        image_scalar(m, seed_value_type(m, p[7])) == RC_NATIVE_IMAGE_FLOAT32 && lod &&
        ((op(lod) == SpvConstant && len(lod) == 4 && !lod[3]) || (op(lod) == SpvConstantNull && len(lod) == 3));
}

static int seed_sampled_access(ImageModule *m, const uint32_t *p) {
    const uint32_t opcode = op(p), n = len(p);
    if (opcode == SpvImage) {
        const uint32_t *image = n == 4 ? definition(m, p[1]) : NULL;
        return image && op(image) == SpvTypeImage && len(image) == 9 &&
            seed_sampled_type(m, p[3], image[4]) == image;
    }
    const int offset = m->rmw && opcode == SpvImageFetch && n == 8 &&
        p[5] == (SpvImageOperandLod | SpvImageOperandConstOffset) && seed_constant_vec2(m, p[7]);
    if (!offset && n != (opcode == SpvImageQuerySizeLod ? 5u : 7u)) return 0;
    const uint32_t *value = definition(m, p[3]);
    if (!value || op(value) != SpvImage || len(value) != 4) return 0;
    // The extraction was already checked as a direct sampled descriptor.
    const uint32_t *image = definition(m, value[1]);
    const int volume = image[3] == SpvDim3D;
    const uint32_t width = 2 + image[5];
    if (image[4]) {
        const uint32_t lod_id = opcode == SpvImageQuerySizeLod ? p[4] : p[6];
        const RcNativeImageScalar lod_type = image_scalar(m, seed_value_type(m, lod_id));
        uint32_t lod;
        if ((lod_type != RC_NATIVE_IMAGE_SINT32 && lod_type != RC_NATIVE_IMAGE_UINT32) ||
            !literal(m, lod_id, &lod) || lod) return 0;
        if (opcode == SpvImageQuerySizeLod) return seed_int_vector(m, p[1], 3);
        return p[5] == SpvImageOperandLod && seed_int_vector(m, seed_value_type(m, p[4]), 3) &&
            seed_sampled_texel_type(m, p[1], image);
    }
    if (opcode == SpvImageQuerySizeLod)
        return !volume && int_vector(m, p[1], width) && int32(m, seed_value_type(m, p[4]));
    const uint32_t coordinate = seed_value_type(m, p[4]);
    return (p[5] == SpvImageOperandLod || (offset && !volume && !image[5])) &&
        int32(m, seed_value_type(m, p[6])) && (volume ? seed_int_vector(m, coordinate, 3) : int_vector(m, coordinate, width)) &&
        seed_sampled_texel_type(m, p[1], image);
}

static int seed_shuffle(const ImageModule *m, const uint32_t *p) {
    const uint32_t n = len(p);
    if (n < 7 || n > 9) return 0;
    const uint32_t *result = definition(m, p[1]);
    const uint32_t *first = definition(m, seed_value_type(m, p[3]));
    const uint32_t *second = definition(m, seed_value_type(m, p[4]));
    if (!result || !first || !second || op(result) != SpvTypeVector || op(first) != SpvTypeVector ||
        op(second) != SpvTypeVector || len(result) != 4 || len(first) != 4 || len(second) != 4 ||
        result[2] != first[2] || result[2] != second[2] || result[3] != n - 5 ||
        first[3] < 2 || first[3] > 4 || second[3] < 2 || second[3] > 4) return 0;
    for (uint32_t j = 5; j < n; ++j)
        if (p[j] != UINT32_MAX && p[j] >= first[3] + second[3]) return 0;
    return 1;
}

// A leaf cannot load opaque resources or route pointers through arithmetic or calls.
static int seed_helper_value(const ImageModule *m, const SeedFunctions *f, const uint32_t *p) {
    switch (op(p)) {
    case SpvVariable: case SpvAccessChain: case SpvInBoundsAccessChain: case SpvLoad:
        return 1; // Checked by seed_helper_memory.
    case SpvConvertFToS: case SpvConvertSToF: case SpvIAdd: case SpvFMul: case SpvFDiv: case SpvFOrdEqual:
    case SpvExtInst:
        if (!seed_scalar_type(m, p[1])) return 0;
        for (uint32_t j = op(p) == SpvExtInst ? 5 : 3; j < len(p); ++j)
            if (!seed_helper_scalar(m, f, p[j])) return 0;
        return 1;
    default:
        return 0;
    }
}

static int seed_helper_branches(const ImageModule *m, const SeedFunctions *f, const uint32_t *end) {
    for (const uint32_t *p = f->start + len(f->start); p < end; p += len(p)) {
        const uint32_t opcode = op(p);
        if (opcode != SpvBranch && opcode != SpvBranchConditional && opcode != SpvSelectionMerge &&
            opcode != SpvLoopMerge) continue;
        uint32_t first = 1, count = opcode == SpvLoopMerge ? 2 : 1;
        if (opcode == SpvBranchConditional) {
            const uint32_t *type = definition(m, seed_helper_scalar(m, f, p[1]));
            if (!type || op(type) != SpvTypeBool) return 0;
            first = 2;
            count = 2;
        }
        for (uint32_t j = first; j < first + count; ++j) {
            const uint32_t *target = definition(m, p[j]);
            if (!target || op(target) != SpvLabel || target <= f->start || target >= end) return 0;
        }
    }
    return 1;
}

enum { SEED_MAX_PHI_BLOCKS = 8 };

typedef struct {
    uint32_t block, merges[SEED_MAX_PHI_BLOCKS];
    uint32_t count;
    int leading;
} SeedPhiBlocks;

// Exact forward joins only: each incoming scalar was defined in its own terminating predecessor.
static int seed_phi_predecessor(const ImageModule *m, const SeedFunctions *f, const uint32_t *join,
    uint32_t block, uint32_t value, uint32_t parent) {
    const uint32_t *label = definition(m, parent);
    const uint32_t *defined = definition(m, value);
    if (!label || op(label) != SpvLabel || label <= f->start || label >= join ||
        !defined || defined <= label || defined >= join) return 0;
    for (const uint32_t *p = label + len(label); p < join; p += len(p)) {
        switch (op(p)) {
        case SpvBranch: return len(p) == 2 && p[1] == block && defined < p;
        case SpvLabel: case SpvBranchConditional: case SpvReturn: case SpvReturnValue:
            return 0;
        default: break;
        }
    }
    return 0;
}

static int seed_phi(ImageModule *m, const SeedFunctions *f, SeedPhiBlocks *blocks, const uint32_t *p) {
    if (!blocks->leading || len(p) != 7 || !image_scalar(m, p[1]) || p[4] == p[6]) return 0;
    const uint32_t *join = definition(m, blocks->block);
    if (!join || seed_value_type(m, p[3]) != p[1] || seed_value_type(m, p[5]) != p[1] ||
        !seed_phi_predecessor(m, f, join, blocks->block, p[3], p[4]) ||
        !seed_phi_predecessor(m, f, join, blocks->block, p[5], p[6])) return 0;
    if (!blocks->count || blocks->merges[blocks->count - 1] != blocks->block) {
        if (blocks->count == SEED_MAX_PHI_BLOCKS) return 0;
        blocks->merges[blocks->count++] = blocks->block;
    }
    return record_id(m, p, 2);
}

// Count every incoming edge after the entry scan, including any later backedge into a join.
static int seed_phi_complete(const SeedFunctions *f, const SeedPhiBlocks *blocks, const uint32_t *end) {
    if (!blocks->count) return 1;
    uint8_t incoming[SEED_MAX_PHI_BLOCKS] = {0};
    for (const uint32_t *p = f->start + len(f->start); p < end; p += len(p)) {
        const uint32_t opcode = op(p);
        if (opcode != SpvBranch && opcode != SpvBranchConditional) continue;
        const uint32_t first = opcode == SpvBranch ? p[1] : p[2];
        const uint32_t second = opcode == SpvBranch ? first : p[3];
        for (uint32_t j = 0; j < blocks->count; ++j)
            if ((blocks->merges[j] == first || blocks->merges[j] == second) && ++incoming[j] > 2) return 0;
    }
    for (uint32_t j = 0; j < blocks->count; ++j)
        if (incoming[j] != 2) return 0;
    return 1;
}

static int seed_glsl_min_max_clamp(const ImageModule *m, const uint32_t *p) {
    const uint32_t n = len(p), opcode = p[4], type = p[1];
    const uint32_t expected = opcode == GlslFClamp || opcode == GlslSClamp ? 8 : 7;
    if (n != expected) return 0;
    if (opcode == GlslSClamp) {
        const uint32_t *vector = definition(m, type);
        if (!vector || op(vector) != SpvTypeVector || len(vector) != 4 || vector[3] != 2 ||
            image_scalar(m, vector[2]) != RC_NATIVE_IMAGE_SINT32) return 0;
    } else {
        const RcNativeImageScalar scalar = opcode == GlslUMin || opcode == GlslUMax ? RC_NATIVE_IMAGE_UINT32 : RC_NATIVE_IMAGE_FLOAT32;
        if (image_scalar(m, type) != scalar) return 0;
    }
    for (uint32_t j = 5; j < n; ++j)
        if (seed_value_type(m, p[j]) != type) return 0;
    return 1;
}

// Seeded bindings use a closed instruction subset. Known control flow and
// arithmetic can affect which texels change within an already defined plane.
static int seed_extra_value(const ImageModule *m, const uint32_t *p) {
    switch (op(p)) {
    case SpvSNegate: case SpvLogicalNot: case SpvConvertFToU: case SpvConvertFToS: case SpvNot:
        return len(p) == 4;
    case SpvFDiv: case SpvLogicalOr: case SpvLogicalAnd: case SpvIEqual: case SpvINotEqual:
    case SpvSGreaterThan: case SpvUGreaterThanEqual: case SpvSGreaterThanEqual: case SpvSLessThan:
    case SpvULessThan: case SpvULessThanEqual: case SpvFOrdEqual: case SpvFOrdNotEqual: case SpvFOrdLessThan:
    case SpvFOrdGreaterThan: case SpvFOrdLessThanEqual: case SpvFOrdGreaterThanEqual: case SpvBitwiseOr: case SpvBitwiseXor:
        return len(p) == 5;
    case SpvLogicalNotEqual: {
        if (len(p) != 5) return 0;
        const uint32_t *result = definition(m, p[1]);
        return result && op(result) == SpvTypeBool && len(result) == 2 &&
            seed_value_type(m, p[3]) == p[1] && seed_value_type(m, p[4]) == p[1];
    }
    case SpvSLessThanEqual: {
        if (len(p) != 5) return 0;
        const uint32_t *result = definition(m, p[1]);
        const uint32_t type = seed_value_type(m, p[3]);
        return result && op(result) == SpvTypeBool && len(result) == 2 &&
            image_scalar(m, type) == RC_NATIVE_IMAGE_SINT32 && seed_value_type(m, p[4]) == type;
    }
    case SpvBitFieldSExtract:
        return m->rmw && len(p) == 6 && image_scalar(m, p[1]) == RC_NATIVE_IMAGE_SINT32 &&
            seed_value_type(m, p[3]) == p[1] && int32(m, seed_value_type(m, p[4])) && int32(m, seed_value_type(m, p[5]));
    case SpvBitReverse: case SpvBitCount:
        return m->rmw && len(p) == 4 && int32(m, p[1]) && seed_value_type(m, p[3]) == p[1];
    case SpvBitFieldUExtract:
        return len(p) == 6 && image_scalar(m, p[1]) == RC_NATIVE_IMAGE_UINT32 &&
            seed_value_type(m, p[3]) == p[1] && seed_value_type(m, p[4]) == p[1] && seed_value_type(m, p[5]) == p[1];
    case SpvVectorShuffle:
        return seed_shuffle(m, p);
    case SpvExtInst:
        if (!glsl_std450(m, p)) return 0;
        switch (p[4]) {
        case GlslRoundEven: case GlslTrunc: case GlslFAbs: case GlslFloor: case GlslFract: case GlslSin: case GlslCos: case GlslExp2:
        case GlslLog2: case GlslSqrt: case GlslInverseSqrt: return len(p) == 6;
        case GlslFindILsb: case GlslFindUMsb:
            return m->rmw && len(p) == 6 && int32(m, p[1]) && int32(m, seed_value_type(m, p[5]));
        case GlslSMin: case GlslSMax: case GlslNMin: case GlslNMax: return len(p) == 7;
        case GlslUClamp: case GlslNClamp: return len(p) == 8;
        case GlslUMin: case GlslUMax: case GlslFMax: case GlslFClamp: case GlslSClamp:
            return seed_glsl_min_max_clamp(m, p);
        default: return 0;
        }
    default:
        return 0;
    }
}

static int seed_non_value(ImageModule *m, const uint32_t *p, uint32_t function) {
    const uint32_t n = len(p);
    switch (op(p)) {
    case SpvNop: case SpvNoLine: return n == 1;
    case SpvLine: return n == 4;
    case SpvName: return !function && n >= 3;
    case SpvMemberName: case SpvMemberDecorate: return !function && n >= 4;
    case SpvString: return !function && n >= 3 && record_id(m, p, 1);
    case SpvExtInstImport: return !function && n >= 3 && record_id(m, p, 1);
    case SpvTypeVoid: case SpvTypeBool: case SpvTypeSampler:
        return !function && n == 2 && record_id(m, p, 1);
    case SpvTypeSampledImage: case SpvTypeRuntimeArray:
        return !function && n == 3 && record_id(m, p, 1);
    case SpvTypeArray:
        return !function && n == 4 && record_id(m, p, 1);
    case SpvTypeStruct: case SpvTypeFunction:
        return !function && n >= (op(p) == SpvTypeStruct ? 2u : 3u) && record_id(m, p, 1);
    case SpvConstantTrue: case SpvConstantFalse:
        return !function && n == 3 && record_id(m, p, 2);
    case SpvStore: return function && n == 3;
    case SpvLoopMerge: return function && n == 4;
    case SpvSelectionMerge: return function && n == 3;
    case SpvLabel: return function && n == 2 && record_id(m, p, 1);
    case SpvBranch: return function && n == 2;
    case SpvBranchConditional: return function && (n == 4 || n == 6);
    case SpvReturn: return function && n == 1;
    default: return 0;
    }
}

// Binding proof establishes descriptor compatibility without proving XY
// coverage. In seeded mode each statically present storage access contributes
// to the inventory of direct descriptors.

static int scan_binding(ImageModule *m, const uint32_t *words, size_t count) {
    SeedFunctions functions = {0};
    SeedPhiBlocks phi_blocks = {0};
    uint32_t entry = 0, function = 0;
    int logical = 0, entry_complete = 0, wrote = 0, shader = 0;
    int in_block = 0, saw_block = 0, local_size = 0, reconverges = 0, has_reconvergence_extension = 0, atomic = 0;
    for (size_t i = 5; i < count;) {
        const uint32_t *p = words + i;
        uint32_t n = len(p), opcode = op(p);
        if (!n || n > count - i) return 0;
        i += n;
        if (m->rmw && opcode == SpvImageSampleExplicitLod) return 0;
        if (m->seed && (opcode == SpvVariable && (n < 4 || n > 5 ||
            (function ? p[3] != SpvFunctionStorage : p[3] == SpvFunctionStorage)))) return 0;
        if (m->seed && function) {
            if (opcode == SpvLabel) {
                if (in_block || (functions.active && functions.parameter != functions.active->count)) return 0;
                if (n != 2) return 0;
                phi_blocks.block = p[1];
                phi_blocks.leading = 1;
                in_block = saw_block = 1;
            } else if (opcode == SpvFunctionEnd) {
                if (in_block || !saw_block || (functions.active &&
                    (!functions.returned || !seed_helper_branches(m, &functions, p)))) return 0;
            } else if (opcode != SpvFunctionParameter && opcode != SpvLine && opcode != SpvNoLine && opcode != SpvNop) {
                if (!in_block) return 0;
                if (opcode != SpvPhi) phi_blocks.leading = 0;
                if (opcode == SpvBranch || opcode == SpvBranchConditional || opcode == SpvReturn ||
                    opcode == SpvReturnValue) in_block = 0;
            } else if (opcode == SpvNop) phi_blocks.leading = 0;
            if (functions.active && !seed_helper_memory(m, &functions, p)) return 0;
        }
        switch (opcode) {
        case SpvMemoryModel:
            if ((m->seed && function) || logical || n != 3 || p[1] || (p[2] != 1 && p[2] != 3)) return 0;
            logical = 1;
            break;
        case SpvEntryPoint:
            if ((m->seed && function) || entry || n < 5 || p[1] != SpvGLCompute || !p[2] || p[3] != 0x6e69616d || p[4]) return 0;
            entry = p[2];
            break;
        case SpvExtension:
            if (m->seed) {
                if (function || n < 2) return 0;
                if (rc_spirv_maximal_reconvergence_extension(p, n)) has_reconvergence_extension = 1;
            }
            break;
        case SpvExecutionMode:
            if (!m->seed) break;
            if (function || !entry || n < 3 || p[1] != entry) return 0;
            if (p[2] == SpvLocalSize) {
                if (n != 6 || !p[3] || !p[4] || !p[5] || local_size) return 0;
                local_size = 1;
            } else if (p[2] == RC_SPV_MAXIMALLY_RECONVERGES_KHR) {
                if (n != 3 || !shader || !has_reconvergence_extension || reconverges) return 0;
                reconverges = 1;
            } else if (p[2] < 4459 || p[2] > 4463 || n != 4 || (p[3] != 16 && p[3] != 32 && p[3] != 64)) return 0;
            break;
        case SpvFunction:
            if (function || n != 5 || !p[2]) return 0;
            if (m->seed && (!logical || !entry || (p[2] == entry ? entry_complete : !entry_complete) ||
                !seed_function_begin(m, &functions, p, p[2] == entry) || !record_id(m, p, 2))) return 0;
            function = p[2];
            saw_block = 0;
            break;
        case SpvFunctionParameter:
            if (!m->seed) break;
            if (!function || !functions.active || saw_block || n != 3 || functions.parameter == functions.active->count ||
                p[1] != functions.active->params[functions.parameter] || !record_id(m, p, 2)) return 0;
            functions.parameter++;
            break;
        case SpvFunctionCall:
            if (!m->seed) break;
            if (!function || function != entry || !seed_helper_call(m, &functions, p, entry)) return 0;
            break;
        case SpvReturnValue:
            if (!m->seed) break;
            if (!function || !functions.active || n != 2 || seed_helper_scalar(m, &functions, p[1]) != functions.active->result) return 0;
            functions.returned = 1;
            break;
        case SpvFunctionEnd:
            if (n != 1 || !function || (m->seed && function == entry && !seed_phi_complete(&functions, &phi_blocks, p)))
                return 0;
            entry_complete |= function == entry;
            function = 0;
            break;
        case SpvCapability:
            if ((m->seed && function) || n != 2) return 0;
            if (p[1] == SpvShader) shader = 1;
            if (p[1] == SpvImageGatherExtended) m->gather_extended = 1;
            if (p[1] == SpvStorageImageReadWithoutFormat) m->formatless_reads = 1;
            if (p[1] == SpvStorageImageWriteWithoutFormat) m->formatless_writes = 1;
            break;
        case SpvDecorate:
            if (!decoration(m, p, 0)) return 0;
            if (m->seed && function) return 0;
            if (m->seed && p[2] == SpvBlock) {
                if (n != 3 || m->ids[p[1]].block) return 0;
                m->ids[p[1]].block = 1;
            }
            if (m->seed && (p[2] == SpvNonWritable || p[2] == SpvNonReadable)) {
                const uint8_t access = p[2] == SpvNonWritable ? RC_NATIVE_IMAGE_SEED_WRITE : RC_NATIVE_IMAGE_SEED_READ;
                if (n != 3 || (m->ids[p[1]].denied_access & access)) return 0;
                m->ids[p[1]].denied_access |= access;
            }
            break;
        case SpvTypeInt: case SpvTypeFloat: case SpvTypeVector: case SpvTypeImage: case SpvTypePointer:
            if ((m->seed && function) || !record_id(m, p, 1)) return 0;
            break;
        case SpvPhi:
            if (!m->seed) break;
            if (!function || function != entry || !seed_phi(m, &functions, &phi_blocks, p)) return 0;
            break;
        case SpvImageSampleDrefExplicitLod:
            if (!m->seed) break;
            if (!function || functions.active || !seed_sampled_comparison(m, p) || !record_id(m, p, 2)) return 0;
            break;
        case SpvImageGather:
            if (!m->seed) break;
            if (!function || functions.active || !seed_sampled_gather(m, p) || !record_id(m, p, 2)) return 0;
            break;
        case SpvImage: case SpvImageQuerySizeLod: case SpvImageFetch:
            if (!m->seed) break;
            if (!function || functions.active || !seed_sampled_access(m, p) || !record_id(m, p, 2)) return 0;
            break;
        case SpvImageRead:
            if (!function || (m->seed && functions.active) || (n != 5 && (n != 6 || p[5])) ||
                !storage_binding(m, p[3], p[4], p[1], 1) || !record_id(m, p, 2))
                return 0;
            break;
        case SpvImageWrite:
            if (!function || (m->seed && functions.active) || !storage_write_binding(m, p)) return 0;
            wrote = 1;
            break;
        case SpvShiftLeftLogical: case SpvShiftRightLogical: case SpvShiftRightArithmetic: {
            uint32_t shift;
            if ((m->seed && function && functions.active) || n != 5 ||
                (m->rmw && !seed_rmw_memory(m, p)) || (int32(m, p[1]) && literal(m, p[4], &shift) && shift >= 32) ||
                !record_id(m, p, 2)) return 0;
            break;
        }
        case SpvImageTexelPointer:
            if (!m->rmw || !function || function != entry || !seed_rmw_texel_pointer(m, p) || !record_id(m, p, 2)) return 0;
            break;
        case SpvAtomicOr:
            if (!m->rmw) {
                if (m->seed) return 0;
                break;
            }
            if (!function || function != entry || !seed_rmw_atomic_or(m, p) || !record_id(m, p, 2)) return 0;
            atomic = wrote = 1;
            break;
        case SpvDecorateId: case SpvDecorationGroup: case SpvGroupDecorate: case SpvGroupMemberDecorate:
        case SpvImageSparseRead:
            return 0;
        default:
            if (m->rmw && opcode == SpvStore && !seed_rmw_memory(m, p)) return 0;
            if (body_value(m, p) || opcode == SpvVariable || opcode == SpvConstant ||
                opcode == SpvConstantComposite || opcode == SpvConstantNull || (m->seed && seed_extra_value(m, p))) {
                if ((m->seed && functions.active && function && !seed_helper_value(m, &functions, p)) ||
                    (m->rmw && !seed_rmw_memory(m, p)) || !record_id(m, p, 2)) return 0;
            } else if (m->seed && (!seed_non_value(m, p, function) || (functions.active && function && opcode == SpvReturn))) {
                return 0;
            }
            break;
        }
    }
    for (unsigned i = 0; i < functions.count; ++i)
        if (!functions.helpers[i].defined) return 0;
    if (m->rmw) {
        if (!atomic) return 0;
        for (uint32_t i = 1; i < m->bound; ++i)
            if (m->ids[i].code && op(m->ids[i].code) == SpvImageTexelPointer && !m->ids[i].atomic_used) return 0;
    }
    return logical && entry_complete && !function && wrote && (!m->seed || (shader && local_size));
}

static int image_module_header(const uint32_t *words, size_t count) {
    return words && count >= 5 && words[0] == 0x07230203 && words[3] && words[3] <= count && words[3] <= 65536 &&
        !words[4];
}

RcNativeImageStoreResult rc_native_image_store(const uint32_t *words, size_t count, RcNativeImageStore *out) {
    *out = (RcNativeImageStore){0};
    if (!image_module_header(words, count)) return RC_NATIVE_IMAGE_STORE_NONE;
    ImageModule module = {.bound = words[3]};
    module.ids = calloc(module.bound, sizeof *module.ids);
    if (!module.ids) return RC_NATIVE_IMAGE_STORE_NONE;
    RcNativeImageStoreResult result = RC_NATIVE_IMAGE_STORE_XY;
    if (!scan_module(&module, words, count)) {
        memset(module.ids, 0, module.bound * sizeof *module.ids);
        module.proof = (RcNativeImageStore){0};
        module.formatless_reads = module.formatless_writes = 0;
        result = scan_binding(&module, words, count) ? RC_NATIVE_IMAGE_STORE_BINDING : RC_NATIVE_IMAGE_STORE_NONE;
    }
    free(module.ids);
    if (result != RC_NATIVE_IMAGE_STORE_NONE) *out = module.proof;
    return result;
}

int rc_native_image_store_dispatch_safe(const RcNativeImageStore *proof, const uint32_t groups[3],
                                       uint32_t width, uint32_t height, int robust_image_access) {
    if (!proof->scalar || !width || !height || !groups[0] || !groups[1] || !groups[2]) return 0;
    if (!proof->local[0]) return robust_image_access;
    if (groups[2] != 1 || proof->local[2] != 1) return 0;
    const uint64_t x = (uint64_t)groups[0] * proof->local[0];
    const uint64_t y = (uint64_t)groups[1] * proof->local[1];
    if (x > (uint64_t)INT32_MAX + 1 || y > (uint64_t)INT32_MAX + 1) return 0;
    return robust_image_access || (x <= width && y <= height);
}

int rc_native_image_store_covers(const RcNativeImageStore *proof, const uint32_t groups[3],
                                uint32_t width, uint32_t height, int robust_image_access) {
    return proof->local[0] && rc_native_image_store_dispatch_safe(proof, groups, width, height, robust_image_access) &&
        (uint64_t)groups[0] * proof->local[0] >= width && (uint64_t)groups[1] * proof->local[1] >= height;
}

int rc_native_image_store_format(const RcNativeImageStore *proof, uint32_t compatible_format,
                                 int formatless_writes, int formatless_reads) {
    return proof->scalar && proof->scalar == format_scalar(compatible_format) &&
        (proof->image_format ? proof->image_format == compatible_format :
         formatless_writes && (!proof->reads_storage || formatless_reads));
}

static int image_seed_bindings(const uint32_t *words, size_t count, RcNativeImageSeedBindings *out, int rmw) {
    if (!out) return 0;
    *out = (RcNativeImageSeedBindings){0};
    if (!image_module_header(words, count)) return 0;
    RcNativeImageSeedBindings seed = {0};
    ImageModule module = {.bound = words[3], .seed = &seed, .rmw = rmw};
    module.ids = calloc(module.bound, sizeof *module.ids);
    if (!module.ids) return 0;
    const int accepted = scan_binding(&module, words, count);
    free(module.ids);
    if (accepted) *out = seed;
    return accepted;
}

int rc_native_image_seed_bindings(const uint32_t *words, size_t count, RcNativeImageSeedBindings *out) {
    return image_seed_bindings(words, count, out, 0);
}

int rc_native_image_seed_rmw_bindings(const uint32_t *words, size_t count, RcNativeImageSeedBindings *out) {
    return image_seed_bindings(words, count, out, 1);
}

static int seed_binding_valid(const RcNativeImageSeedBinding *binding) {
    return binding && binding->binding < 64 &&
        (binding->dimension == RC_NATIVE_IMAGE_SEED_2D || binding->dimension == RC_NATIVE_IMAGE_SEED_3D) &&
        binding->scalar >= RC_NATIVE_IMAGE_FLOAT32 &&
        binding->scalar <= RC_NATIVE_IMAGE_UINT32 && binding->access &&
        !(binding->access & ~(RC_NATIVE_IMAGE_SEED_READ | RC_NATIVE_IMAGE_SEED_WRITE | RC_NATIVE_IMAGE_SEED_ATOMIC)) &&
        (!(binding->access & RC_NATIVE_IMAGE_SEED_ATOMIC) ||
         (binding->access == (RC_NATIVE_IMAGE_SEED_READ | RC_NATIVE_IMAGE_SEED_WRITE | RC_NATIVE_IMAGE_SEED_ATOMIC) &&
          binding->dimension == RC_NATIVE_IMAGE_SEED_2D && binding->image_format == SpvR32ui)) &&
        (!binding->image_format || binding->scalar == format_scalar(binding->image_format));
}

int rc_native_image_seed_store(const RcNativeImageSeedBinding *binding, RcNativeImageStore *out) {
    if (!out) return 0;
    *out = (RcNativeImageStore){0};
    if (!seed_binding_valid(binding) || !(binding->access & RC_NATIVE_IMAGE_SEED_WRITE) ||
        (binding->access & RC_NATIVE_IMAGE_SEED_ATOMIC)) return 0;
    *out = (RcNativeImageStore){.binding = binding->binding, .image_format = binding->image_format,
        .scalar = (RcNativeImageScalar)binding->scalar, .reads_storage = !!(binding->access & RC_NATIVE_IMAGE_SEED_READ)};
    return 1;
}

int rc_native_image_seed_binding_format(const RcNativeImageSeedBinding *binding, uint32_t compatible_format,
                                        int formatless_writes, int formatless_reads) {
    return seed_binding_valid(binding) && binding->scalar == format_scalar(compatible_format) &&
        (binding->image_format ? binding->image_format == compatible_format :
         (!(binding->access & RC_NATIVE_IMAGE_SEED_WRITE) || formatless_writes) &&
         (!(binding->access & RC_NATIVE_IMAGE_SEED_READ) || formatless_reads));
}

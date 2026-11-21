#include "rc_spirv_uniform_footprint.h"
#include "rc_spirv_reconvergence.h"
#include <stdlib.h>

/* Only instructions used in this deliberately restricted final vertex form. */
enum {
    OpName = 5, OpMemberName = 6, OpExtension = 10, OpExtInstImport = 11, OpExtInst = 12,
    OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16, OpCapability = 17,
    OpTypeVoid = 19, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
    OpTypeArray = 28, OpTypeStruct = 30, OpTypePointer = 32, OpTypeFunction = 33,
    OpConstant = 43, OpConstantNull = 46, OpFunction = 54, OpFunctionEnd = 56, OpVariable = 59,
    OpLoad = 61, OpStore = 62, OpAccessChain = 65, OpInBoundsAccessChain = 66,
    OpDecorate = 71, OpMemberDecorate = 72, OpConvertSToF = 111, OpSNegate = 126,
    OpIAdd = 128, OpFAdd = 129, OpFMul = 133, OpIsNan = 156, OpLogicalOr = 166,
    OpLogicalNot = 168, OpSelect = 169, OpFOrdLessThan = 184, OpFOrdGreaterThan = 186,
    OpFOrdGreaterThanEqual = 190, OpSelectionMerge = 247, OpLabel = 248, OpBranch = 249,
    OpBranchConditional = 250, OpReturn = 253,
    ScInput = 1, ScUniform = 2, ScOutput = 3, ScFunction = 7,
    Block = 2, ArrayStride = 6, BuiltIn = 11, Invariant = 18, Location = 30,
    Binding = 33, DescriptorSet = 34, Offset = 35, NoContraction = 42,
    MAX_VEC4S = 4096, MAX_WORDS = 65536, MAX_IDS = 16384,
};

typedef struct {
    const uint32_t *def;
    uint32_t set, binding, at;
    uint8_t has_set, has_binding, selected_pointer, reached;
} UniformId;

typedef struct {
    const uint32_t *words, *entry;
    size_t count;
    uint32_t bound, root, block, array, vector, scalar, elements;
    UniformId *ids;
    uint64_t lanes[MAX_VEC4S * 4 / 64];
} UniformModule;

static uint32_t op(const uint32_t *p) { return p[0] & UINT16_MAX; }
static uint32_t len(const uint32_t *p) { return p[0] >> 16; }
static int id_valid(const UniformModule *m, uint32_t id) { return id && id < m->bound; }

static const uint32_t *find(const UniformModule *m, uint32_t id, uint32_t opcode, uint32_t size) {
    if (!id_valid(m, id)) return NULL;
    const uint32_t *p = m->ids[id].def;
    return p && op(p) == opcode && (!size || len(p) == size) ? p : NULL;
}

static int add(UniformModule *m, uint32_t id, const uint32_t *p) {
    if (!id_valid(m, id) || m->ids[id].def) return 0;
    m->ids[id].def = p;
    return 1;
}

static int string_ok(const uint32_t *p, uint32_t start, uint32_t n) {
    for (uint32_t i = start; i < n; ++i)
        for (unsigned j = 0; j < 32; j += 8)
            if (!((p[i] >> j) & 255u)) return i + 1 == n && !(p[i] >> j);
    return 0;
}

static int known_capability(uint32_t capability) {
    switch (capability) {
    case 1: case 10: case 25: case 46: case 47: case 50:
    case 62: case 64: case 65: case 4427:
        return 1;
    default:
        return 0;
    }
}

static int decoration(UniformModule *m, const uint32_t *p, uint32_t n) {
    if (n < 3 || !id_valid(m, p[1])) return 0;
    UniformId *id = &m->ids[p[1]];
    switch (p[2]) {
    case Binding:
        if (n != 4 || id->has_binding) return 0;
        id->binding = p[3];
        id->has_binding = 1;
        return 1;
    case DescriptorSet:
        if (n != 4 || id->has_set) return 0;
        id->set = p[3];
        id->has_set = 1;
        return 1;
    case Block: case NoContraction:
        return n == 3;
    case ArrayStride: case Location:
        return n == 4;
    default:
        return 0;
    }
}

static int inventory(UniformModule *m) {
    int seen_main = 0, in_main = 0, model = 0, shader = 0, ext = 0, mode = 0, import = 0;
    for (size_t i = 5; i < m->count;) {
        const uint32_t *p = m->words + i;
        const uint32_t code = op(p), n = len(p);
        if (!n || n > m->count - i) return 0;
        i += n;
        if (seen_main && !in_main) return 0;
        if (in_main) {
            switch (code) {
            case OpFunctionEnd:
                if (n != 1) return 0;
                in_main = 0;
                break;
            case OpLabel:
                if (n != 2 || !add(m, p[1], p)) return 0;
                break;
            case OpVariable:
                if ((n != 4 && n != 5) || !add(m, p[2], p)) return 0;
                break;
            case OpLoad:
                if (n != 4 || !add(m, p[2], p)) return 0;
                break;
            case OpAccessChain: case OpInBoundsAccessChain:
                if (n < 5 || n > 8 || !add(m, p[2], p)) return 0;
                break;
            case OpExtInst:
                if (n != 8 || !add(m, p[2], p)) return 0;
                break;
            case OpConvertSToF: case OpSNegate: case OpLogicalNot: case OpIsNan:
                if (n != 4 || !add(m, p[2], p)) return 0;
                break;
            case OpIAdd: case OpFAdd: case OpFMul: case OpLogicalOr:
            case OpFOrdLessThan: case OpFOrdGreaterThan: case OpFOrdGreaterThanEqual:
                if (n != 5 || !add(m, p[2], p)) return 0;
                break;
            case OpSelect:
                if (n != 6 || !add(m, p[2], p)) return 0;
                break;
            case OpStore: case OpSelectionMerge:
                if (n != 3) return 0;
                break;
            case OpBranch:
                if (n != 2) return 0;
                break;
            case OpBranchConditional:
                if (n != 4) return 0;
                break;
            case OpReturn:
                if (n != 1) return 0;
                break;
            default:
                return 0;
            }
            continue;
        }
        switch (code) {
        case OpCapability:
            if (n != 2 || !known_capability(p[1])) return 0;
            if (p[1] == 1 && shader++) return 0;
            break;
        case OpExtension:
            if (ext++ || !rc_spirv_maximal_reconvergence_extension(p, n)) return 0;
            break;
        case OpExtInstImport:
            if (n != 6 || import++ || p[2] != UINT32_C(0x4c534c47) || p[3] != UINT32_C(0x6474732e) ||
                p[4] != UINT32_C(0x3035342e) || p[5] || !add(m, p[1], p)) return 0;
            break;
        case OpMemoryModel:
            if (n != 3 || model++ || p[1] || p[2] != 1) return 0;
            break;
        case OpEntryPoint:
            if (!model || m->entry || n < 5 || p[1] || !id_valid(m, p[2]) ||
                p[3] != UINT32_C(0x6e69616d) || p[4]) return 0;
            m->entry = p;
            break;
        case OpExecutionMode:
            if (n != 3 || !m->entry || p[1] != m->entry[2] || !ext || mode++ ||
                p[2] != RC_SPV_MAXIMALLY_RECONVERGES_KHR) return 0;
            break;
        case OpName: case OpMemberName:
            if (n < (code == OpName ? 3u : 4u) || !id_valid(m, p[1]) ||
                !string_ok(p, code == OpName ? 2 : 3, n)) return 0;
            break;
        case OpDecorate:
            if (!decoration(m, p, n)) return 0;
            break;
        case OpMemberDecorate:
            if (n < 4 || !id_valid(m, p[1]) ||
                !((n == 4 && p[3] == Invariant) || (n == 5 && (p[3] == Offset || p[3] == BuiltIn)))) return 0;
            break;
        case OpTypeVoid: case OpTypeBool:
            if (n != 2 || !add(m, p[1], p)) return 0;
            break;
        case OpTypeFloat: case OpTypeFunction:
            if (n != 3 || !add(m, p[1], p)) return 0;
            break;
        case OpTypeInt: case OpTypeVector: case OpTypeArray: case OpTypePointer:
            if (n != 4 || !add(m, p[1], p)) return 0;
            break;
        case OpTypeStruct:
            if (n < 3 || !add(m, p[1], p)) return 0;
            break;
        case OpConstant: case OpVariable:
            if (n != 4 || !add(m, p[2], p)) return 0;
            break;
        case OpConstantNull:
            if (n != 3 || !add(m, p[2], p)) return 0;
            break;
        case OpFunction: {
            const uint32_t *type = n == 5 ? find(m, p[4], OpTypeFunction, 3) : NULL;
            if (!m->entry || !type || p[2] != m->entry[2] || p[3] || p[1] != type[2] ||
                !find(m, p[1], OpTypeVoid, 2) || !add(m, p[2], p)) return 0;
            seen_main = in_main = 1;
            break;
        }
        default:
            return 0;
        }
    }
    return shader && model && seen_main && !in_main && ext == mode;
}

static int int32_type(const UniformModule *m, uint32_t type) {
    const uint32_t *p = find(m, type, OpTypeInt, 4);
    return p && p[2] == 32 && p[3] <= 1;
}

static int float32_type(const UniformModule *m, uint32_t type) {
    const uint32_t *p = find(m, type, OpTypeFloat, 3);
    return p && p[2] == 32;
}

static const uint32_t *int_constant(const UniformModule *m, uint32_t id) {
    const uint32_t *p = find(m, id, OpConstant, 4);
    return p && int32_type(m, p[1]) ? p : NULL;
}

static int data_type_before(const UniformModule *m, uint32_t id, const uint32_t *before) {
    if (!id_valid(m, id) || !m->ids[id].def || m->ids[id].def >= before) return 0;
    switch (op(m->ids[id].def)) {
    case OpTypeBool: case OpTypeInt: case OpTypeFloat: case OpTypeVector: case OpTypeArray: case OpTypeStruct:
        return 1;
    default:
        return 0;
    }
}

static int declarations_valid(const UniformModule *m) {
    const uint32_t *main = m->ids[m->entry[2]].def;
    for (uint32_t i = 5; i < len(m->entry); ++i) {
        const uint32_t *v = find(m, m->entry[i], OpVariable, 4);
        if (!v || v >= main || (v[3] != ScInput && v[3] != ScOutput)) return 0;
        for (uint32_t j = 5; j < i; ++j)
            if (m->entry[i] == m->entry[j]) return 0;
    }
    for (const uint32_t *p = m->words + 5; p < main; p += len(p)) {
        switch (op(p)) {
        case OpTypeInt:
            if (!int32_type(m, p[1])) return 0;
            break;
        case OpTypeFloat:
            if (!float32_type(m, p[1])) return 0;
            break;
        case OpTypeVector:
            if (!data_type_before(m, p[2], p) || p[3] < 2 || p[3] > 4 ||
                (!int32_type(m, p[2]) && !float32_type(m, p[2]) && !find(m, p[2], OpTypeBool, 2))) return 0;
            break;
        case OpTypeArray: {
            const uint32_t *n = int_constant(m, p[3]);
            if (!data_type_before(m, p[2], p) || !n || n >= p || !n[3] || n[3] > MAX_VEC4S) return 0;
            break;
        }
        case OpTypeStruct:
            for (uint32_t j = 2; j < len(p); ++j)
                if (!data_type_before(m, p[j], p)) return 0;
            break;
        case OpTypePointer:
            if ((p[2] != ScInput && p[2] != ScOutput && p[2] != ScUniform && p[2] != ScFunction) ||
                !data_type_before(m, p[3], p)) return 0;
            break;
        case OpTypeFunction: {
            const uint32_t *ret = find(m, p[2], OpTypeVoid, 2);
            if (!ret || ret >= p) return 0;
            break;
        }
        case OpConstant:
            if (!data_type_before(m, p[1], p) || (!int32_type(m, p[1]) && !float32_type(m, p[1]))) return 0;
            break;
        case OpConstantNull:
            if (!data_type_before(m, p[1], p)) return 0;
            break;
        case OpVariable: {
            const uint32_t *ptr = find(m, p[1], OpTypePointer, 4);
            const UniformId *id = &m->ids[p[2]];
            if (!ptr || ptr >= p || p[3] != ptr[2] || p[3] == ScFunction ||
                id->has_set != id->has_binding || (p[3] == ScUniform) != !!id->has_set) return 0;
            break;
        }
        case OpMemberName: case OpMemberDecorate: {
            const uint32_t *type = find(m, p[1], OpTypeStruct, 0);
            if (!type || p[2] >= len(type) - 2) return 0;
            break;
        }
        default:
            break;
        }
    }
    return 1;
}

static int select_root(UniformModule *m, uint32_t set, uint32_t binding) {
    for (uint32_t i = 1; i < m->bound; ++i) {
        const UniformId *id = &m->ids[i];
        if (!id->has_binding || !id->has_set || id->set != set || id->binding != binding) continue;
        if (m->root) return 0;
        m->root = i;
    }
    const uint32_t *root = find(m, m->root, OpVariable, 4);
    const uint32_t *ptr = root ? find(m, root[1], OpTypePointer, 4) : NULL;
    if (!ptr || root[3] != ScUniform || ptr[2] != ScUniform || root > m->ids[m->entry[2]].def) return 0;
    const uint32_t *block = find(m, ptr[3], OpTypeStruct, 3);
    const uint32_t *array = block ? find(m, block[2], OpTypeArray, 4) : NULL;
    const uint32_t *vector = array ? find(m, array[2], OpTypeVector, 4) : NULL;
    const uint32_t *elements = array ? int_constant(m, array[3]) : NULL;
    if (!vector || vector[3] != 4 || !float32_type(m, vector[2]) || !elements || !elements[3] ||
        elements[3] > MAX_VEC4S) return 0;
    m->block = block[1];
    m->array = array[1];
    m->vector = vector[1];
    m->scalar = vector[2];
    m->elements = elements[3];
    int block_dec = 0, offset_dec = 0, stride_dec = 0;
    for (size_t i = 5; i < m->count; i += len(m->words + i)) {
        const uint32_t *p = m->words + i;
        if (op(p) == OpDecorate) {
            if (p[1] == m->block && (p[2] != Block || block_dec++)) return 0;
            if (p[1] == m->array && (p[2] != ArrayStride || p[3] != 16 || stride_dec++)) return 0;
            if (p[1] == m->root && p[2] != Binding && p[2] != DescriptorSet) return 0;
            if (p[1] == m->vector || p[1] == m->scalar || p[1] == ptr[1]) return 0;
        } else if (op(p) == OpMemberDecorate && p[1] == m->block &&
                   (p[2] || p[3] != Offset || p[4] || offset_dec++)) return 0;
    }
    return block_dec && offset_dec && stride_dec;
}

static uint32_t value_type(const UniformModule *m, uint32_t id, const uint32_t *before) {
    if (!id_valid(m, id)) return 0;
    const uint32_t *p = m->ids[id].def;
    if (!p || p >= before) return 0;
    switch (op(p)) {
    case OpConstant: case OpConstantNull: case OpVariable: case OpLoad: case OpAccessChain: case OpInBoundsAccessChain:
    case OpConvertSToF: case OpSNegate: case OpIAdd: case OpFAdd: case OpFMul: case OpIsNan: case OpLogicalOr:
    case OpLogicalNot: case OpSelect: case OpFOrdLessThan: case OpFOrdGreaterThan: case OpFOrdGreaterThanEqual: case OpExtInst:
        return p[1];
    default:
        return 0;
    }
}

static const uint32_t *value_ptr(const UniformModule *m, uint32_t id, const uint32_t *before) {
    return find(m, value_type(m, id, before), OpTypePointer, 4);
}

static int chain(UniformModule *m, const uint32_t *p, uint32_t n) {
    const uint32_t *base = value_ptr(m, p[3], p), *result = find(m, p[1], OpTypePointer, 4);
    if (!base || !result || base[2] != result[2]) return 0;
    uint32_t type = base[3];
    for (uint32_t j = 4; j < n; ++j) {
        if (!int32_type(m, value_type(m, p[j], p))) return 0;
        const uint32_t *index = int_constant(m, p[j]);
        const uint32_t *def = find(m, type, OpTypeStruct, 0);
        if (def) {
            if (!index || index[3] >= len(def) - 2) return 0;
            type = def[2 + index[3]];
        } else if ((def = find(m, type, OpTypeArray, 4))) {
            const uint32_t *count = int_constant(m, def[3]);
            if (!count || !count[3] || (index && index[3] >= count[3])) return 0;
            type = def[2];
        } else if ((def = find(m, type, OpTypeVector, 4))) {
            if (index && index[3] >= def[3]) return 0;
            type = def[2];
        } else return 0;
    }
    if (type != result[3]) return 0;
    if (p[3] != m->root) return !m->ids[p[3]].selected_pointer;
    if (n != 7 || type != m->scalar) return 0;
    const uint32_t *element = int_constant(m, p[5]), *lane = int_constant(m, p[6]);
    if (!element || !lane || element[3] >= m->elements || lane[3] >= 4) return 0;
    m->ids[p[2]].selected_pointer = 1;
    m->ids[p[2]].at = element[3] * 4 + lane[3];
    return 1;
}

static int bool_type(const UniformModule *m, uint32_t id) { return find(m, id, OpTypeBool, 2) != NULL; }

static int value_instruction(const UniformModule *m, const uint32_t *p) {
    const uint32_t code = op(p), type = p[1];
    const uint32_t a = value_type(m, p[3], p);
    const uint32_t b = len(p) > 4 ? value_type(m, p[4], p) : 0;
    switch (code) {
    case OpExtInst:
        return p[4] == 50 && find(m, p[3], OpExtInstImport, 6) && float32_type(m, type) &&
            value_type(m, p[5], p) == type && value_type(m, p[6], p) == type && value_type(m, p[7], p) == type;
    case OpConvertSToF:
        return float32_type(m, type) && int32_type(m, a) && m->ids[a].def[3];
    case OpSNegate:
        // Signed negate accepts 32-bit integer values, including the translator's u32 result type.
        return int32_type(m, type) && a == type;
    case OpIAdd:
        return int32_type(m, type) && a == type && b == type;
    case OpFAdd: case OpFMul:
        return float32_type(m, type) && a == type && b == type;
    case OpIsNan:
        return bool_type(m, type) && float32_type(m, a);
    case OpLogicalOr:
        return bool_type(m, type) && a == type && b == type;
    case OpLogicalNot:
        return bool_type(m, type) && a == type;
    case OpFOrdLessThan: case OpFOrdGreaterThan: case OpFOrdGreaterThanEqual:
        return bool_type(m, type) && float32_type(m, a) && a == b;
    case OpSelect:
        return (int32_type(m, type) || float32_type(m, type) || bool_type(m, type)) &&
            bool_type(m, a) && b == type && value_type(m, p[5], p) == type;
    default:
        return 0;
    }
}

static int branch_target(UniformModule *m, uint32_t id, const uint32_t *p) {
    const uint32_t *target = find(m, id, OpLabel, 2);
    if (!target || target <= p) return 0;
    m->ids[id].reached = 1;
    return 1;
}

static int collect(UniformModule *m) {
    const uint32_t *p = m->ids[m->entry[2]].def + 5;
    int first = 1, open = 0, locals = 0;
    uint32_t merge = 0;
    for (; op(p) != OpFunctionEnd; p += len(p)) {
        const uint32_t code = op(p), n = len(p);
        if (code == OpLabel) {
            if (open || (!first && !m->ids[p[1]].reached)) return 0;
            open = 1;
            locals = first;
            first = 0;
            continue;
        }
        if (!open || (merge && code != OpBranchConditional)) return 0;
        if (code != OpVariable) locals = 0;
        switch (code) {
        case OpVariable: {
            const uint32_t *type = find(m, p[1], OpTypePointer, 4);
            if (!locals || !type || type[2] != ScFunction || p[3] != ScFunction) return 0;
            if (n == 5 && (value_type(m, p[4], p) != type[3] ||
                (!find(m, p[4], OpConstant, 4) && !find(m, p[4], OpConstantNull, 3)))) return 0;
            break;
        }
        case OpAccessChain: case OpInBoundsAccessChain:
            if (!chain(m, p, n)) return 0;
            break;
        case OpLoad: {
            const uint32_t *ptr = value_ptr(m, p[3], p);
            if (!ptr || ptr[3] != p[1] || find(m, p[1], OpTypePointer, 4)) return 0;
            if (m->ids[p[3]].selected_pointer) {
                const uint32_t at = m->ids[p[3]].at;
                m->lanes[at / 64] |= UINT64_C(1) << (at % 64);
            }
            break;
        }
        case OpStore: {
            const uint32_t *ptr = value_ptr(m, p[1], p);
            if (!ptr || (ptr[2] != ScFunction && ptr[2] != ScOutput) || value_type(m, p[2], p) != ptr[3]) return 0;
            break;
        }
        case OpSelectionMerge: {
            const uint32_t *target = find(m, p[1], OpLabel, 2);
            if (!target || target <= p || p[2]) return 0;
            merge = p[1];
            break;
        }
        case OpBranch:
            if (!branch_target(m, p[1], p)) return 0;
            open = 0;
            break;
        case OpBranchConditional:
            if (!merge || p[2] == p[3] || !bool_type(m, value_type(m, p[1], p)) ||
                !branch_target(m, p[2], p) || !branch_target(m, p[3], p)) return 0;
            merge = 0;
            open = 0;
            break;
        case OpReturn:
            open = 0;
            break;
        default:
            if (!value_instruction(m, p)) return 0;
            break;
        }
    }
    return !first && !open;
}

/* Enumerate literal operand positions in the accepted grammar. */
static int operand_is_id(uint32_t code, uint32_t j) {
    switch (code) {
    case OpCapability: case OpExtension: case OpMemoryModel:
        return 0;
    case OpName: case OpMemberName: case OpDecorate: case OpMemberDecorate: case OpExtInstImport:
    case OpExecutionMode: case OpSelectionMerge: case OpTypeInt: case OpTypeFloat:
        return j == 1;
    case OpEntryPoint:
        return j == 2 || j >= 5;
    case OpTypeVector: case OpConstant:
        return j <= 2;
    case OpVariable:
        return j != 3;
    case OpTypePointer:
        return j != 2;
    case OpFunction:
        return j != 3;
    case OpExtInst:
        return j != 4;
    default:
        return 1;
    }
}

/* Reject every use of the binding or a derived pointer except a typed load. */
static int all_uses_known(const UniformModule *m) {
    for (size_t i = 5; i < m->count; i += len(m->words + i)) {
        const uint32_t *p = m->words + i;
        const uint32_t code = op(p), n = len(p);
        for (uint32_t j = 1; j < n; ++j) {
            if (!operand_is_id(code, j)) continue;
            const uint32_t id = p[j];
            if (!id_valid(m, id) || !m->ids[id].def) return 0;
            if (id == m->root) {
                if ((code == OpName || code == OpDecorate) && j == 1) continue;
                if (code == OpVariable && j == 2) continue;
                if ((code == OpAccessChain || code == OpInBoundsAccessChain) && j == 3) continue;
                return 0;
            }
            if (!m->ids[id].selected_pointer) continue;
            if ((code == OpAccessChain || code == OpInBoundsAccessChain) && j == 2) continue;
            if ((code == OpLoad && j == 3) || (code == OpName && j == 1)) continue;
            return 0;
        }
    }
    return 1;
}

static int lane_used(const UniformModule *m, uint32_t i) {
    return (m->lanes[i / 64] >> (i % 64)) & 1u;
}

static uint32_t coalesce(const UniformModule *m, RcSpirvUniformRange *ranges) {
    uint32_t count = 0;
    for (uint32_t i = 0, end = m->elements * 4; i < end;) {
        if (!lane_used(m, i)) { ++i; continue; }
        uint32_t start = i++;
        while (i < end && lane_used(m, i)) ++i;
        if (ranges) ranges[count] = (RcSpirvUniformRange){start * 4, (i - start) * 4};
        ++count;
    }
    return count;
}

RcSpirvUniformResult rc_spirv_uniform_vertex_footprint(const uint32_t *words, size_t count,
    uint32_t set, uint32_t binding, RcSpirvUniformProof *proof, RcSpirvUniformRange *ranges, size_t capacity) {
    if (!proof) return RC_SPIRV_UNIFORM_NONE;
    *proof = (RcSpirvUniformProof){0};
    if (!words || count < 6 || count > MAX_WORDS || words[0] != UINT32_C(0x07230203) ||
        words[1] < UINT32_C(0x00010000) || words[1] > UINT32_C(0x00010300) || (words[1] & 255u) ||
        !words[3] || words[3] > MAX_IDS || words[4] || (!ranges && capacity)) return RC_SPIRV_UNIFORM_NONE;
    UniformModule m = {.words = words, .count = count, .bound = words[3]};
    m.ids = calloc(m.bound, sizeof *m.ids);
    if (!m.ids) return RC_SPIRV_UNIFORM_NONE;
    RcSpirvUniformResult result = RC_SPIRV_UNIFORM_NONE;
    if (inventory(&m) && declarations_valid(&m) && select_root(&m, set, binding) && collect(&m) && all_uses_known(&m)) {
        const uint32_t needed = coalesce(&m, NULL);
        if (needed) {
            *proof = (RcSpirvUniformProof){m.root, m.elements * 16, needed};
            result = RC_SPIRV_UNIFORM_CAPACITY;
            if (capacity >= needed) {
                coalesce(&m, ranges);
                result = RC_SPIRV_UNIFORM_PROVEN;
            }
        }
    }
    free(m.ids);
    return result;
}

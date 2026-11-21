#include "rc_sampled_image_query.h"
#include "rc_spirv_reconvergence.h"

#include <stdlib.h>
#include <string.h>

enum {
    SpvName = 5, SpvMemberName = 6, SpvExtension = 10, SpvEntryPoint = 15,
    SpvExecutionMode = 16, SpvCapability = 17,
    SpvTypeInt = 21, SpvTypeFloat = 22, SpvTypeVector = 23, SpvTypeImage = 25,
    SpvTypeSampledImage = 27, SpvTypeArray = 28, SpvTypeRuntimeArray = 29,
    SpvTypePointer = 32, SpvConstant = 43,
    SpvFunction = 54, SpvFunctionEnd = 56, SpvVariable = 59, SpvLoad = 61,
    SpvAccessChain = 65, SpvInBoundsAccessChain = 66,
    SpvDecorate = 71, SpvMemberDecorate = 72, SpvDecorationGroup = 73,
    SpvGroupDecorate = 74, SpvGroupMemberDecorate = 75, SpvSampledImage = 86,
    SpvImage = 100, SpvImageQuerySizeLod = 103, SpvDecorateId = 332,
    SpvUniformConstant = 0, SpvDim2D = 1,
    SpvArrayStride = 6, SpvBuiltIn = 11, SpvBinding = 33, SpvDescriptorSet = 34, SpvNonUniform = 5300,
    SpvLocalSize = 17, SpvLocalSizeHint = 18,
};

typedef enum {
    QUERY_OTHER, QUERY_VARIABLE, QUERY_MEMBER, QUERY_OTHER_MEMBER, QUERY_OTHER_IMAGE,
    QUERY_SAMPLED_IMAGE, QUERY_IMAGE, QUERY_SITE,
} QueryKind;

typedef struct {
    const uint32_t *declaration;
    uint32_t set, binding;
    uint8_t has_set, has_binding, kind, used;
} QueryId;

typedef struct {
    const uint32_t *words;
    size_t count;
    QueryId *ids;
    uint32_t bound;
    uint32_t variable, image_type, sampled_type;
    RcSampledImageQueryTarget target;
    size_t sites;
} QueryModule;

static uint32_t opcode(const uint32_t *p) { return p[0] & UINT16_MAX; }
static uint32_t length(const uint32_t *p) { return p[0] >> 16; }

static int valid_id(const QueryModule *m, uint32_t id) { return id && id < m->bound; }

static const uint32_t *declaration(const QueryModule *m, uint32_t id, uint32_t op, uint32_t n) {
    if (!valid_id(m, id)) return NULL;
    const uint32_t *p = m->ids[id].declaration;
    return p && opcode(p) == op && length(p) == n ? p : NULL;
}

static int record_id(QueryModule *m, uint32_t id, const uint32_t *p) {
    if (!valid_id(m, id) || m->ids[id].declaration) return 0;
    m->ids[id].declaration = p;
    return 1;
}

static int literal_string(const uint32_t *p, uint32_t first, uint32_t n) {
    for (uint32_t j = first; j < n; ++j)
        for (unsigned byte = 0; byte < 32; byte += 8)
            if (!((p[j] >> byte) & 255u))
                return j + 1 == n && !(p[j] >> byte);
    return 0;
}

static int descriptor_indexing_extension(const uint32_t *p, uint32_t n) {
    // The little-endian words spell SPV_EXT_descriptor_indexing and its terminator.
    static const uint32_t name[] = {
        UINT32_C(0x5f565053), UINT32_C(0x5f545845), UINT32_C(0x63736564),
        UINT32_C(0x74706972), UINT32_C(0x695f726f), UINT32_C(0x7865646e), UINT32_C(0x00676e69),
    };
    return n == 1 + sizeof name / sizeof name[0] && !memcmp(p + 1, name, sizeof name);
}

static int inventory(QueryModule *m) {
    uint32_t entry = 0;
    int seen_function = 0, in_function = 0;
    int reconvergence_extension = 0, reconverges = 0;
    for (size_t i = 5; i < m->count;) {
        const uint32_t *p = m->words + i;
        const uint32_t op = opcode(p), n = length(p);
        if (!n || n > m->count - i) return 0;
        i += n;
        switch (op) {
        case SpvEntryPoint:
            // Single main for the requested execution model, followed by interface IDs.
            if (seen_function || entry || n < 5 || p[1] != (uint32_t)m->target.stage || !valid_id(m, p[2]) ||
                p[3] != UINT32_C(0x6e69616d) || p[4]) return 0;
            for (uint32_t j = 5; j < n; ++j)
                if (!valid_id(m, p[j])) return 0;
            entry = p[2];
            break;
        case SpvName: case SpvMemberName:
            if (seen_function || n < (op == SpvName ? 3u : 4u) || !valid_id(m, p[1]) ||
                !literal_string(p, op == SpvName ? 2 : 3, n)) return 0;
            break;
        case SpvDecorate: {
            if (seen_function || n < 3 || !valid_id(m, p[1])) return 0;
            QueryId *id = &m->ids[p[1]];
            if (p[2] == SpvDescriptorSet) {
                if (n != 4 || id->has_set) return 0;
                id->set = p[3];
                id->has_set = 1;
            } else if (p[2] == SpvBinding) {
                if (n != 4 || id->has_binding) return 0;
                id->binding = p[3];
                id->has_binding = 1;
            }
            break;
        }
        case SpvMemberDecorate:
            if (seen_function || n < 4 || !valid_id(m, p[1]) || (p[3] == SpvBuiltIn && n != 5)) return 0;
            break;
        case SpvExtension:
            if (seen_function) return 0;
            if (rc_spirv_maximal_reconvergence_extension(p, n)) {
                if (reconvergence_extension) return 0;
                reconvergence_extension = 1;
            } else if (!m->target.table_count || !descriptor_indexing_extension(p, n)) return 0;
            break;
        case SpvExecutionMode:
            if (seen_function || n < 3 || !entry || p[1] != entry) return 0;
            if (p[2] == RC_SPV_MAXIMALLY_RECONVERGES_KHR) {
                if (n != 3 || !reconvergence_extension || reconverges) return 0;
                reconverges = 1;
            }
            break;
        case SpvDecorationGroup: case SpvGroupDecorate: case SpvGroupMemberDecorate: case SpvDecorateId:
            return 0;
        case SpvTypeInt: case SpvTypeFloat: case SpvTypeVector: case SpvTypeImage: case SpvTypeSampledImage:
        case SpvTypeArray: case SpvTypeRuntimeArray: case SpvTypePointer:
            if (seen_function || n < 2 || !record_id(m, p[1], p)) return 0;
            break;
        case SpvConstant:
            if (seen_function || n < 4 || !record_id(m, p[2], p)) return 0;
            break;
        case SpvVariable:
            if (n < 4 || (seen_function && !in_function) || (in_function && p[3] == SpvUniformConstant) ||
                !record_id(m, p[2], p)) return 0;
            break;
        case SpvFunction:
            if (in_function || n != 5 || !record_id(m, p[2], p)) return 0;
            seen_function = in_function = 1;
            break;
        case SpvFunctionEnd:
            if (!in_function || n != 1) return 0;
            in_function = 0;
            break;
        case SpvAccessChain: case SpvInBoundsAccessChain: case SpvLoad: case SpvSampledImage:
        case SpvImage: case SpvImageQuerySizeLod:
            if (!in_function || n < 4 || !record_id(m, p[2], p)) return 0;
            break;
        default:
            break;
        }
    }
    return !in_function && declaration(m, entry, SpvFunction, 5);
}

static int int32(const QueryModule *m, uint32_t id) {
    const uint32_t *p = declaration(m, id, SpvTypeInt, 4);
    return p && p[2] == 32 && p[3] <= 1;
}

static const uint32_t *constant_int32(const QueryModule *m, uint32_t id) {
    const uint32_t *constant = declaration(m, id, SpvConstant, 4);
    return constant && int32(m, constant[1]) ? constant : NULL;
}

static int select_binding(QueryModule *m) {
    for (uint32_t i = 1; i < m->bound; ++i) {
        const QueryId *id = &m->ids[i];
        if (!id->has_set || !id->has_binding || id->set != m->target.set || id->binding != m->target.binding) continue;
        if (m->variable) return 0;
        m->variable = i;
    }
    const uint32_t *v = declaration(m, m->variable, SpvVariable, 4);
    if (!v || v[3] != SpvUniformConstant) return 0;
    const uint32_t *ptr = declaration(m, v[1], SpvTypePointer, 4);
    if (!ptr || ptr[2] != SpvUniformConstant) return 0;
    if (m->target.table_count) {
        const uint32_t *array = declaration(m, ptr[3], SpvTypeArray, 4);
        if (array) {
            const uint32_t *count = constant_int32(m, array[3]);
            if (!count || count[3] != m->target.table_count) return 0;
        } else {
            array = declaration(m, ptr[3], SpvTypeRuntimeArray, 3);
            if (!array) return 0;
        }
        m->image_type = array[2];
    } else {
        const uint32_t *sampled = declaration(m, ptr[3], SpvTypeSampledImage, 3);
        if (!sampled) return 0;
        m->sampled_type = ptr[3];
        m->image_type = sampled[2];
    }
    const uint32_t *image = declaration(m, m->image_type, SpvTypeImage, 9);
    if (!image || image[3] != SpvDim2D || image[4] || image[5] || image[6] ||
        image[7] != 1 || image[8]) return 0;
    const uint32_t *scalar = declaration(m, image[2], SpvTypeFloat, 3);
    if (!scalar || scalar[2] != 32) return 0;
    m->ids[m->variable].kind = QUERY_VARIABLE;
    return 1;
}

static QueryKind table_member_kind(const QueryModule *m, const uint32_t *p, uint32_t n) {
    if (n != 5 || !valid_id(m, p[4])) return QUERY_OTHER;
    const uint32_t *ptr = declaration(m, p[1], SpvTypePointer, 4);
    if (!ptr || ptr[2] != SpvUniformConstant || ptr[3] != m->image_type) return QUERY_OTHER;
    const uint32_t *index = constant_int32(m, p[4]);
    if (index && index[3] >= m->target.table_count) return QUERY_OTHER;
    if (m->target.member == RC_SAMPLED_IMAGE_QUERY_ALL_MEMBERS) return QUERY_MEMBER;
    if (!index) return QUERY_OTHER;
    return index[3] == m->target.member ? QUERY_MEMBER : QUERY_OTHER_MEMBER;
}

static int collect_sites(QueryModule *m) {
    for (size_t i = 5; i < m->count; i += length(m->words + i)) {
        const uint32_t *p = m->words + i;
        const uint32_t op = opcode(p), n = length(p);
        if (op != SpvAccessChain && op != SpvInBoundsAccessChain && op != SpvLoad &&
            op != SpvSampledImage && op != SpvImage && op != SpvImageQuerySizeLod) continue;
        if (!valid_id(m, p[3])) continue;
        QueryId *input = &m->ids[p[3]], *result = &m->ids[p[2]];
        if ((op == SpvAccessChain || op == SpvInBoundsAccessChain) && input->kind == QUERY_VARIABLE &&
            m->target.table_count) {
            result->kind = table_member_kind(m, p, n);
            if (result->kind == QUERY_OTHER) return 0;
        } else if (op == SpvLoad && input->kind == QUERY_VARIABLE && !m->target.table_count) {
            if (n != 4 || p[1] != m->sampled_type) return 0;
            result->kind = QUERY_SAMPLED_IMAGE;
        } else if (op == SpvLoad && (input->kind == QUERY_MEMBER || input->kind == QUERY_OTHER_MEMBER)) {
            if (n != 4 || p[1] != m->image_type) return 0;
            // Check the pointer uses of sibling members, but their loaded image cannot select this member.
            result->kind = input->kind == QUERY_MEMBER ? QUERY_IMAGE : QUERY_OTHER_IMAGE;
        } else if (op == SpvSampledImage && input->kind == QUERY_IMAGE && m->target.table_count) {
            const uint32_t *type = declaration(m, p[1], SpvTypeSampledImage, 3);
            if (n != 5 || !type || type[2] != m->image_type || !valid_id(m, p[4])) return 0;
            result->kind = QUERY_SAMPLED_IMAGE;
        } else if (op == SpvImage && input->kind == QUERY_SAMPLED_IMAGE) {
            if (n != 4 || p[1] != m->image_type) return 0;
            result->kind = QUERY_IMAGE;
        } else if (op == SpvImageQuerySizeLod && input->kind == QUERY_IMAGE) {
            if (n != 5) return 0;
            const uint32_t *lod = constant_int32(m, p[4]);
            const uint32_t *type = declaration(m, p[1], SpvTypeVector, 4);
            if (!lod || lod[3] || !type || !int32(m, type[2]) || type[3] != 2) return 0;
            result->kind = QUERY_SITE;
            m->sites++;
        } else {
            continue;
        }
        input->used = 1;
    }
    return m->sites != 0;
}

static int allowed_word(const QueryModule *m, const uint32_t *p, uint32_t j, const QueryId *id) {
    const uint32_t op = opcode(p);
    // These positions were validated during inventory. Other literal collisions
    // are deliberately refused because we do not have an operand grammar for them.
    if (op == SpvName || op == SpvMemberName) return j >= 2 || op == SpvName;
    if (op == SpvEntryPoint) return j == 1 || j == 3 || j == 4;
    // Inventory admits only exact extension strings and an operand-free reconvergence mode.
    if (op == SpvExtension) return 1;
    if (op == SpvExecutionMode && length(p) == 3 && p[2] == RC_SPV_MAXIMALLY_RECONVERGES_KHR) return j == 2;
    if (m->target.table_count || m->target.stage != RC_SAMPLED_IMAGE_QUERY_VERTEX) {
        if (op == SpvCapability) return length(p) == 2 && j == 1;
        if (op == SpvTypeVector) return length(p) == 4 && j == 3;
        if (op == SpvDecorate && length(p) == 4 && j == 3 &&
            (p[2] == SpvBuiltIn || p[2] == SpvArrayStride)) return 1;
        if (op == SpvExecutionMode) return j == 2 || (j > 2 && length(p) == 6 &&
            (p[2] == SpvLocalSize || p[2] == SpvLocalSizeHint));
    }
    if (op == SpvMemberDecorate) return j == 2 || j == 3 || (p[3] == SpvBuiltIn && j == 4);
    if (op == SpvDecorate) {
        const int selected_literal = p[2] == SpvDescriptorSet || p[2] == SpvBinding;
        return j == 2 || (selected_literal && (j == 3 || id->kind == QUERY_VARIABLE)) ||
            (j == 1 && m->target.table_count && p[2] == SpvNonUniform && length(p) == 3 &&
             id->kind != QUERY_VARIABLE);
    }
    if (j == 2 && id->declaration == p) return 1;
    if (j != 3) return 0;
    switch (op) {
    case SpvAccessChain: case SpvInBoundsAccessChain:
        return m->ids[p[2]].kind == QUERY_MEMBER || m->ids[p[2]].kind == QUERY_OTHER_MEMBER;
    case SpvLoad:
        return m->ids[p[2]].kind == QUERY_SAMPLED_IMAGE || m->ids[p[2]].kind == QUERY_IMAGE ||
               m->ids[p[2]].kind == QUERY_OTHER_IMAGE;
    case SpvSampledImage: return m->ids[p[2]].kind == QUERY_SAMPLED_IMAGE;
    case SpvImage: return m->ids[p[2]].kind == QUERY_IMAGE;
    case SpvImageQuerySizeLod: return m->ids[p[2]].kind == QUERY_SITE;
    default: return 0;
    }
}

static int carries_selected_object(uint32_t kind) {
    return kind != QUERY_OTHER && kind != QUERY_SITE && kind != QUERY_OTHER_IMAGE;
}

static int all_uses_known(const QueryModule *m) {
    for (uint32_t i = 1; i < m->bound; ++i)
        if (carries_selected_object(m->ids[i].kind) && !m->ids[i].used) return 0;
    for (size_t i = 5; i < m->count; i += length(m->words + i)) {
        const uint32_t *p = m->words + i;
        for (uint32_t j = 1; j < length(p); ++j) {
            if (!valid_id(m, p[j])) continue;
            const QueryId *id = &m->ids[p[j]];
            if (carries_selected_object(id->kind) && !allowed_word(m, p, j, id)) return 0;
        }
    }
    return 1;
}

static void write_sites(const QueryModule *m, RcSampledImageQuerySite *sites) {
    for (size_t i = 5; i < m->count; i += length(m->words + i)) {
        const uint32_t *p = m->words + i;
        if (opcode(p) != SpvImageQuerySizeLod || m->ids[p[2]].kind != QUERY_SITE) continue;
        const uint32_t *image = m->ids[p[3]].declaration;
        const uint32_t *load = image;
        if (opcode(image) == SpvImage) {
            load = m->ids[image[3]].declaration;
            while (opcode(load) == SpvSampledImage || opcode(load) == SpvImage)
                load = m->ids[load[3]].declaration;
        }
        *sites++ = (RcSampledImageQuerySite){(size_t)(load - m->words), (size_t)(image - m->words), i, p[1]};
    }
}

RcSampledImageQueryResult rc_sampled_image_query_size_lod0_for(const uint32_t *words, size_t word_count,
    RcSampledImageQueryTarget target, RcSampledImageQueryProof *proof,
    RcSampledImageQuerySite *sites, size_t site_capacity) {
    if (!proof) return RC_SAMPLED_IMAGE_QUERY_NONE;
    *proof = (RcSampledImageQueryProof){0};
    if (target.stage != RC_SAMPLED_IMAGE_QUERY_VERTEX && target.stage != RC_SAMPLED_IMAGE_QUERY_FRAGMENT &&
        target.stage != RC_SAMPLED_IMAGE_QUERY_COMPUTE) return RC_SAMPLED_IMAGE_QUERY_NONE;
    if (target.table_count > INT32_MAX || (!target.table_count && target.member) ||
        (target.table_count && target.member >= target.table_count &&
        target.member != RC_SAMPLED_IMAGE_QUERY_ALL_MEMBERS)) return RC_SAMPLED_IMAGE_QUERY_NONE;
    if ((!sites && site_capacity) || !words || word_count < 5 || words[0] != UINT32_C(0x07230203) ||
        words[1] < UINT32_C(0x00010000) || words[1] > UINT32_C(0x00010300) || (words[1] & 255u) ||
        !words[3] || words[3] > word_count || words[3] > 65536 || words[4]) return RC_SAMPLED_IMAGE_QUERY_NONE;
    QueryModule m = {.words = words, .count = word_count, .bound = words[3], .target = target};
    m.ids = calloc(m.bound, sizeof *m.ids);
    if (!m.ids) return RC_SAMPLED_IMAGE_QUERY_NONE;
    RcSampledImageQueryResult result = RC_SAMPLED_IMAGE_QUERY_NONE;
    if (inventory(&m) && select_binding(&m) && collect_sites(&m) && all_uses_known(&m)) {
        *proof = (RcSampledImageQueryProof){m.variable, m.sites};
        result = RC_SAMPLED_IMAGE_QUERY_PROVEN;
        if (sites) {
            if (site_capacity < m.sites) result = RC_SAMPLED_IMAGE_QUERY_CAPACITY;
            else write_sites(&m, sites);
        }
    }
    free(m.ids);
    return result;
}

RcSampledImageQueryResult rc_sampled_image_query_size_lod0(const uint32_t *words, size_t word_count,
    uint32_t set, uint32_t binding, RcSampledImageQueryProof *proof,
    RcSampledImageQuerySite *sites, size_t site_capacity) {
    const RcSampledImageQueryTarget target = {.stage = RC_SAMPLED_IMAGE_QUERY_VERTEX, .set = set, .binding = binding};
    return rc_sampled_image_query_size_lod0_for(words, word_count, target, proof, sites, site_capacity);
}

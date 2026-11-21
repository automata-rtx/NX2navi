// Internal accompanying proof for a readonly branch that guards every pooled descriptor access.
#pragma once

enum { PI_GUARD_SGT = 173, PI_GLSL_SMIN = 39, PI_GLSL_SMAX = 42,
       PI_GUARD_RANGE = 450, PI_GUARD_STORAGE = 451, PI_GUARD_VALUES = 16 };

static const PoolSpvId *pi_guard_value(PoolIndexBuilder *p, uint32_t id) {
    for (uint32_t step = 0; step < PI_RECURSION_LIMIT; ++step) {
        const PoolSpvId *value = pi_id(p, id);
        if (!value || !value->insn || value->inexact) return NULL;
        const uint32_t *i = value->insn;
        const uint32_t op = pi_op(i);
        if ((op == PX_COPY || op == PX_BITCAST) && pi_len(i) == 4) {
            const PoolSpvId *input = pi_id(p, i[3]);
            if (!input || !pi_scalar(p, input->type) ||
                pi_scalar(p, value->type) != pi_scalar(p, input->type)) return NULL;
            id = i[3];
            continue;
        }
        if (op == PX_LOAD && pi_len(i) == 4) {
            const PoolSpvId *ptr = pi_id(p, i[3]);
            if (ptr && ptr->insn && pi_op(ptr->insn) == PX_VAR && pi_len(ptr->insn) >= 4 &&
                ptr->insn[3] == PI_STORAGE_FUNCTION) {
                if (!pi_store_dominates(p, ptr, value)) return NULL;
                id = ptr->store[2];
                continue;
            }
        }
        return value;
    }
    return NULL;
}

static int pi_guard_glsl(PoolIndexBuilder *p, const uint32_t *i, uint32_t opcode) {
    static const char name[] = "GLSL.std.450";
    if (!i || pi_op(i) != PX_EXT || pi_len(i) != 7 || i[4] != opcode) return 0;
    const uint32_t *set = pi_insn(p, i[3]);
    return set && pi_op(set) == PX_EXT_IMPORT && (pi_len(set) - 2) * 4 >= sizeof name &&
        !memcmp(set + 2, name, sizeof name);
}

static uint16_t pi_guard_address(PoolIndexBuilder *p, uint32_t id, unsigned depth) {
    if (depth >= PI_RECURSION_LIMIT) return 0;
    const PoolSpvId *value = pi_guard_value(p, id);
    if (!value || pi_scalar(p, value->type) != 32) return 0;
    const uint32_t *op = value->insn;
    switch (pi_op(op)) {
    case PX_CONST:
        return pi_len(op) == 4 ? pi_literal(p, op[3]) : 0;
    case PX_CONST_NULL:
        return pi_len(op) == 3 ? pi_literal(p, 0) : 0;
    case PX_LOAD: {
        if (pi_len(op) != 4) return 0;
        const uint32_t *chain = pi_insn(p, op[3]);
        if (!chain || (pi_op(chain) != PX_CHAIN && pi_op(chain) != PX_IN_CHAIN) || pi_len(chain) < 5) return 0;
        const uint16_t node = pi_buffer_load(p, chain, depth + 1);
        // The existing scalar selector checks the exact declared UBO and its byte address.
        return node && p->nodes[node].resource_kind == GFX_RES_UBO ? node : 0;
    }
    case PX_EXT: {
        if (!pi_guard_glsl(p, op, PI_GLSL_SMIN)) return 0;
        const PoolSpvId *lower = pi_guard_value(p, op[5]);
        if (!lower || !pi_guard_glsl(p, lower->insn, PI_GLSL_SMAX) ||
            !pi_domain_constant(p, lower->insn[5], 0)) return 0;
        const uint16_t maximum = pi_guard_address(p, op[6], depth + 1);
        return maximum ? pi_node(p, (PoolIndexNode){.op = PI_GUARD_RANGE, .a = maximum}) : 0;
    }
    case PX_ADD: case PX_SUB: case PX_MUL: case PX_AND: case PX_RSHIFT: case PX_LSHIFT: {
        if (pi_len(op) != 5) return 0;
        const uint16_t a = pi_guard_address(p, op[3], depth + 1);
        const uint16_t b = pi_guard_address(p, op[4], depth + 1);
        return a && b ? pi_node(p, (PoolIndexNode){.op = (uint16_t)pi_op(op), .a = a, .b = b}) : 0;
    }
    default: return 0;
    }
}

static int pi_guard_excludes_access(PoolIndexBuilder *p, const uint32_t *guard) {
    if (!p->walk) {
        p->walk = malloc((size_t)p->bound * sizeof *p->walk);
        if (!p->walk) return 0;
    }
    if (!++p->visits) return 0;
    uint32_t pending = 0;
    if (!pi_visit_block(p, p->first, 0, &pending)) return 0;
    while (pending) {
        const PoolSpvId *block = pi_id(p, p->walk[--pending]);
        if (block->pool_access || !block->terminator) return 0;
        const uint32_t *end = block->terminator;
        const uint32_t count = pi_len(end);
        switch (pi_op(end)) {
        case 249:
            if (count != 2 || !pi_visit_block(p, end[1], 0, &pending)) return 0;
            break;
        case 250:
            if ((count != 4 && count != 6) || !pi_visit_block(p, end[3], 0, &pending) ||
                (end != guard && !pi_visit_block(p, end[2], 0, &pending))) return 0;
            break;
        case 252: case 253: case 254: case 255: case 4416:
            break;
        default: return 0;
        }
    }
    return 1;
}

static uint16_t pi_guard_build(PoolIndexBuilder *p) {
    if (p->execution_model != 4 || p->domain_inexact || !p->sampler_modes) return 0;
    const uint16_t start = p->node_count;
    uint32_t accesses = 0;
    for (uint32_t id = 1; id < p->bound; ++id) {
        const PoolSpvId *access = pi_id(p, id);
        const uint32_t *i = access->insn;
        if (!i || (pi_op(i) != PX_CHAIN && pi_op(i) != PX_IN_CHAIN) || pi_len(i) != 5) continue;
        const PoolSpvId *base = pi_id(p, i[3]);
        if (!base || (!base->pool && base->sampler != PI_SAMPLER_POOL)) continue;
        PoolSpvId *block = pi_id(p, access->block);
        if (!block || block->function != p->entry) return 0;
        block->pool_access = 1;
        ++accesses;
    }
    if (!accesses) return 0;
    for (uint32_t id = 1; id < p->bound; ++id) {
        const PoolSpvId *label = pi_id(p, id);
        const uint32_t *branch = label->terminator;
        if (!branch || pi_op(branch) != 250 || (pi_len(branch) != 4 && pi_len(branch) != 6) ||
            label->function != p->entry) continue;
        const PoolSpvId *value = pi_guard_value(p, branch[1]);
        const uint32_t *test = value ? value->insn : NULL;
        if (!test || pi_op(test) != PI_GUARD_SGT || pi_len(test) != 5 || !pi_domain_constant(p, test[4], 0)) continue;
        const PoolSpvId *length = pi_guard_value(p, test[3]);
        if (!length || pi_scalar(p, length->type) != 32) continue;
        uint32_t address, member;
        const GfxShaderRes *source = pi_domain_storage_source(p, length->insn, &address, &member);
        if (!source || source == p->domain_source || !pi_guard_excludes_access(p, branch)) continue;
        uint16_t index = pi_guard_address(p, address, 0);
        uint16_t base = pi_literal(p, member * 4);
        if (index && base) index = pi_node(p, (PoolIndexNode){.op = PX_IN_CHAIN, .a = index, .constant = 4});
        if (index) index = pi_node(p, (PoolIndexNode){.op = PX_CHAIN, .a = index, .b = base});
        if (index) return pi_node(p, (PoolIndexNode){.op = PI_GUARD_STORAGE, .a = index, .resource = source});
        // pi_buffer_load may have cached scalar subexpressions. Refrain from trying a
        // different guard if this particular address did not qualify.
        p->node_count = start;
        return 0;
    }
    return 0;
}

static inline int shader_pool_inactive_source(const uint32_t *original, const uint32_t *selected, uint64_t words,
                                       uint32_t set, uint32_t binding_offset) {
    if (!original || !selected || words < 5 || memcmp(original, selected, 5 * sizeof *original)) return 0;
    int in_function = 0;
    for (uint64_t at = 5; at < words; ) {
        const uint32_t *a = original + at, *b = selected + at;
        const uint32_t count = pi_len(a);
        if (!count || count > words - at || a[0] != b[0]) return 0;
        if (pi_op(a) == PX_FN) in_function = 1;
        if (!in_function && pi_op(a) == PX_DECORATE && count == 4 &&
            (a[2] == PI_DESCRIPTOR_SET || a[2] == PI_DESCRIPTOR_BINDING)) {
            if (a[1] != b[1] || a[2] != b[2]) return 0;
            uint32_t expected = a[3];
            if (a[2] == PI_DESCRIPTOR_SET && set) expected = set;
            if (a[2] == PI_DESCRIPTOR_BINDING && expected) {
                if (binding_offset > UINT32_MAX - expected) return 0;
                expected += binding_offset;
            }
            if (b[3] != expected) return 0;
        } else if (memcmp(a, b, count * sizeof *a)) return 0;
        at += count;
    }
    return 1;
}

typedef struct { uint32_t value[PI_GUARD_VALUES]; uint8_t count; } PoolGuardValues;

static int pi_guard_put(PoolGuardValues *out, uint32_t word) {
    for (unsigned i = 0; i < out->count; ++i) if (out->value[i] == word) return 1;
    if (out->count == PI_GUARD_VALUES) return 0;
    out->value[out->count++] = word;
    return 1;
}

static int pi_guard_math(const PoolIndexNode *node, uint32_t a, uint32_t b, uint32_t *out) {
    switch (node->op) {
    case PX_ADD: *out = a + b; return 1;
    case PX_SUB: *out = a - b; return 1;
    case PX_MUL: *out = a * b; return 1;
    case PX_AND: *out = a & b; return 1;
    case PX_RSHIFT: if (b >= 32) return 0; *out = a >> b; return 1;
    case PX_LSHIFT: if (b >= 32) return 0; *out = a << b; return 1;
    case PX_CHAIN: if (b > UINT32_MAX - a) return 0; *out = a + b; return 1;
    case PX_IN_CHAIN: if (a > UINT32_MAX / node->constant) return 0; *out = a * node->constant; return 1;
    case PX_COMPOSITE_EXTRACT: if (b >= node->constant) return 0; *out = a; return 1;
    default: return 0;
    }
}

// Ready if every possible signed count is zero or negative. An unready word requests
// ordered capture; unsupported data, shape, or excessive branching leaves ordinary selection.
static inline int shader_pool_inactive_eval(const ShaderPoolIndices *plan, GfxShaderPoolReadWord read,
                                     void *context) {
    if (!plan || !plan->inactive_guard || !read) return 0;
    const uint16_t gate = plan->inactive_guard;
    uint8_t needed[PI_NODE_LIMIT] = {0};
    needed[gate] = 1;
    for (uint16_t i = gate; i > 0; --i) if (needed[i]) {
        needed[plan->nodes[i].a] = 1;
        needed[plan->nodes[i].b] = 1;
    }
    PoolGuardValues values[PI_NODE_LIMIT] = {{.count = 1}};
    int pending = 0;
    for (uint16_t i = 1; i <= gate; ++i) {
        if (!needed[i]) continue;
        const PoolIndexNode *node = &plan->nodes[i];
        const PoolGuardValues *a = &values[node->a], *b = &values[node->b];
        PoolGuardValues *out = &values[i];
        if (node->op == PI_VALUE) {
            if (!pi_guard_put(out, node->constant)) return 0;
            continue;
        }
        int local_pending = 0;
        for (unsigned x = 0; x < a->count; ++x) {
            const uint32_t word = a->value[x];
            if (node->op == PI_GUARD_RANGE) {
                if (word >= PI_GUARD_VALUES) return 0;
                for (uint32_t each = 0; each <= word; ++each)
                    if (!pi_guard_put(out, each)) return 0;
            } else if (node->op == PI_U32 || node->op == PI_GUARD_STORAGE) {
                uint32_t value;
                const int result = read(context, node->op == PI_U32 ? GFX_RES_UBO : GFX_RES_SSBO,
                                        node->resource, word, &value);
                if (!result) return 0;
                if (result == 2) local_pending = 1;
                else if (!pi_guard_put(out, value)) return 0;
            } else {
                for (unsigned y = 0; y < b->count; ++y) {
                    uint32_t value;
                    if (!pi_guard_math(node, word, b->value[y], &value) || !pi_guard_put(out, value)) return 0;
                }
            }
        }
        if (local_pending) {
            pending = 1;
            out->count = 0;
        }
    }
    if (pending) return 2;
    const PoolGuardValues *result = &values[gate];
    for (unsigned i = 0; i < result->count; ++i)
        if (result->value[i] && !(result->value[i] >> 31)) return 0;
    return result->count != 0;
}

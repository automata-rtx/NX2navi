#ifndef VK_POOL_SAMPLER_SPIRV_H
#define VK_POOL_SAMPLER_SPIRV_H

enum {
    POOL_SAMPLER_LOAD = 61, POOL_SAMPLER_CHAIN = 65, POOL_SAMPLER_IN_BOUNDS = 66,
    POOL_SAMPLER_GROUP_DECORATE = 74, POOL_SAMPLER_COPY = 83, POOL_SAMPLER_IMAGE = 86,
    POOL_SAMPLER_NON_UNIFORM = 5300
};

typedef struct {
    const uint32_t *insn;
    unsigned non_uniform;
} PoolSamplerSpirvId;

typedef struct {
    const uint32_t *chain, *load, *image, *compare;
    unsigned decoration_mask;
} PoolSamplerSpirvUse;

static int pool_sampler_spirv_dref(uint32_t op) {
    return op == 89 || op == 90 || op == 93 || op == 94 || op == 97 ||
        op == 307 || op == 308 || op == 311 || op == 312 || op == 315;
}

static const uint32_t *pool_sampler_spirv_source(const PoolSamplerSpirvId *ids, uint32_t bound, uint32_t id) {
    for (unsigned i = 0; id && id < bound && i < 32; ++i) {
        const uint32_t *insn = ids[id].insn;
        if (!insn || (insn[0] & 0xffff) != POOL_SAMPLER_COPY) return insn;
        id = insn[3];
    }
    return NULL;
}

static int pool_sampler_spirv_chain(const uint32_t *insn) {
    return insn && ((insn[0] & 0xffff) == POOL_SAMPLER_CHAIN || (insn[0] & 0xffff) == POOL_SAMPLER_IN_BOUNDS);
}

static int pool_sampler_spirv_use(const PoolSamplerSpirvId *ids, uint32_t bound,
    uint32_t ordinary, const uint32_t *compare, PoolSamplerSpirvUse *use) {
    if ((compare[0] >> 16) < 6) return -1;
    const uint32_t *image = pool_sampler_spirv_source(ids, bound, compare[3]);
    if (!image) return -1;
    if ((image[0] & 0xffff) == POOL_SAMPLER_LOAD) return 0; // A combined descriptor load.
    if ((image[0] & 0xffff) != POOL_SAMPLER_IMAGE) return -1;
    const uint32_t *load = pool_sampler_spirv_source(ids, bound, image[4]);
    if (!load || (load[0] & 0xffff) != POOL_SAMPLER_LOAD) return -1;
    const uint32_t *chain = pool_sampler_spirv_source(ids, bound, load[3]);
    if (!chain) return -1;
    if ((chain[0] & 0xffff) == SPV_OP_VARIABLE) return chain[2] == ordinary ? -1 : 0;
    if (!pool_sampler_spirv_chain(chain)) return -1;
    const uint32_t *base = pool_sampler_spirv_source(ids, bound, chain[3]);
    if (!base || (base[0] & 0xffff) != SPV_OP_VARIABLE) return -1;
    if (base[2] != ordinary) return 0;
    if ((chain[0] >> 16) != 5) return -1;
    *use = (PoolSamplerSpirvUse){chain, load, image, compare,
        (ids[chain[2]].non_uniform || ids[load[3]].non_uniform ? 1u : 0u) |
        (ids[load[2]].non_uniform || ids[image[4]].non_uniform ? 2u : 0u) |
        (ids[image[2]].non_uniform || ids[compare[3]].non_uniform ? 4u : 0u)};
    return 1;
}

static uint32_t *pool_sampler_spirv_decoration(uint32_t *out, uint32_t id, uint32_t decoration, uint32_t value) {
    const unsigned literal = decoration != POOL_SAMPLER_NON_UNIFORM;
    *out++ = ((3 + literal) << 16) | SPV_OP_DECORATE;
    *out++ = id;
    *out++ = decoration;
    if (literal) *out++ = value;
    return out;
}

static int pool_sampler_spirv_interface(const uint32_t *entry, uint32_t id) {
    const uint32_t len = entry[0] >> 16;
    if (len <= 3) return -1;
    const char *const name = (const char *)(entry + 3);
    const char *const end = memchr(name, 0, (len - 3) * sizeof(uint32_t));
    if (!end) return -1;
    for (uint32_t i = 4 + (uint32_t)(end - name) / 4; i < len; ++i)
        if (entry[i] == id) return 1;
    return 0;
}

// Route each actual table comparison through the same runtime sampler index in the compare binding.
// Inserting the new access at the sampling instruction leaves shared ordinary operands and their
// control flow intact. The caller has already verified and rebound the typed pool variable.
static int pool_sampler_spirv_split(const uint32_t *words, uint64_t count, uint32_t ordinary,
    uint32_t **rewritten, uint64_t *rewritten_count) {
    *rewritten = NULL;
    *rewritten_count = 0;
    if (count < 5 || !ordinary || ordinary >= words[3] || words[3] >= 1u << 22) return -1;
    const uint32_t bound = words[3];
    const int interface = words[1] >= 0x00010400u;
    uint64_t drefs = 0;
    for (uint64_t at = 5; at < count;) {
        const uint32_t len = words[at] >> 16;
        if (!len || len > count - at) return -1;
        drefs += pool_sampler_spirv_dref(words[at] & 0xffff);
        at += len;
    }
    if (!drefs) return 0;
    if (drefs > ((1u << 22) - 1u - bound) / 3u) return -1;
    PoolSamplerSpirvId *ids = calloc(bound, sizeof *ids);
    PoolSamplerSpirvUse *uses = calloc((size_t)drefs, sizeof *uses);
    uint32_t *out = NULL;
    int result = -1;
    uint64_t found = 0, annotations = 0, first_function = 0, additional = 12;
    unsigned interfaces = 0;
    if (!ids || !uses) goto done;
    for (uint64_t at = 5; at < count;) {
        const uint32_t *insn = words + at;
        const uint32_t op = insn[0] & 0xffff, len = insn[0] >> 16;
        if (!first_function && op == SPV_OP_FUNCTION) first_function = at;
        if (op == SPV_OP_DECORATE && len >= 3 && insn[1] < bound) {
            if (insn[2] == POOL_SAMPLER_NON_UNIFORM) ids[insn[1]].non_uniform = 1;
            if (!annotations && insn[1] == ordinary && insn[2] == SPV_DEC_DESCRIPTOR_SET) annotations = at;
        }
        if (op == POOL_SAMPLER_GROUP_DECORATE) goto done;
        if (op == SPV_OP_ENTRY_POINT && interface) {
            const int present = pool_sampler_spirv_interface(insn, ordinary);
            if (present < 0 || (present && len == UINT16_MAX)) goto done;
            interfaces += (unsigned)present;
        }
        if (op == SPV_OP_VARIABLE || op == POOL_SAMPLER_LOAD || op == POOL_SAMPLER_COPY ||
            op == POOL_SAMPLER_IMAGE || pool_sampler_spirv_chain(insn)) {
            const uint32_t minimum = op == POOL_SAMPLER_IMAGE || pool_sampler_spirv_chain(insn) ? 5u : 4u;
            if (len < minimum || !insn[2] || insn[2] >= bound || ids[insn[2]].insn) goto done;
            if (op != SPV_OP_VARIABLE && (!insn[3] || insn[3] >= bound)) goto done;
            if (op == POOL_SAMPLER_IMAGE && (!insn[4] || insn[4] >= bound)) goto done;
            ids[insn[2]].insn = insn;
        }
        at += len;
    }
    for (uint64_t at = 5; at < count;) {
        const uint32_t *insn = words + at;
        if (pool_sampler_spirv_dref(insn[0] & 0xffff)) {
            const int compare = pool_sampler_spirv_use(ids, bound, ordinary, insn, &uses[found]);
            if (compare < 0) goto done;
            if (compare) {
                const PoolSamplerSpirvUse *use = &uses[found++];
                const unsigned mask = use->decoration_mask;
                additional += 10 + (use->load[0] >> 16) + 3 * ((mask & 1) + (mask >> 1 & 1) + (mask >> 2 & 1));
            }
        }
        at += insn[0] >> 16;
    }
    if (!found) {
        result = 0;
        goto done;
    }
    const uint32_t *variable = ids[ordinary].insn;
    additional += interfaces;
    if (!annotations || !first_function || !variable || (variable[0] >> 16) != 4 ||
        (variable[0] & 0xffff) != SPV_OP_VARIABLE || (interface && !interfaces) ||
        count > SIZE_MAX / sizeof *out || additional > SIZE_MAX / sizeof *out - count) goto done;
    out = malloc((size_t)(count + additional) * sizeof *out);
    if (!out) goto done;
    memcpy(out, words, 5 * sizeof *out);
    const uint32_t comparison = bound;
    out[3] = bound + 1 + 3 * (uint32_t)found;
    uint32_t *next = out + 5;
    uint64_t selected = 0;
    for (uint64_t at = 5; at < count;) {
        const uint32_t *insn = words + at;
        const uint32_t op = insn[0] & 0xffff, len = insn[0] >> 16;
        if (at == annotations) {
            next = pool_sampler_spirv_decoration(next, comparison, SPV_DEC_DESCRIPTOR_SET, 2);
            next = pool_sampler_spirv_decoration(next, comparison, SPV_DEC_BINDING, VK_POOL_COMPARISON_BINDING);
            for (uint32_t use = 0; use < found; ++use)
                for (unsigned part = 0; part < 3; ++part)
                    if (uses[use].decoration_mask >> part & 1)
                        next = pool_sampler_spirv_decoration(next, comparison + 1 + 3 * use + part,
                                                            POOL_SAMPLER_NON_UNIFORM, 0);
        }
        if (at == first_function) {
            memcpy(next, variable, 4 * sizeof *next);
            next[2] = comparison;
            next += 4;
        }
        const PoolSamplerSpirvUse *use = selected < found ? &uses[selected] : NULL;
        if (use && use->compare == insn) {
            const uint32_t new_chain = comparison + 1 + 3 * (uint32_t)selected++;
            const uint32_t load_len = use->load[0] >> 16;
            memcpy(next, use->chain, 5 * sizeof *next);
            next[2] = new_chain;
            next[3] = comparison;
            next += 5;
            memcpy(next, use->load, load_len * sizeof *next);
            next[2] = new_chain + 1;
            next[3] = new_chain;
            next += load_len;
            memcpy(next, use->image, 5 * sizeof *next);
            next[2] = new_chain + 2;
            next[4] = new_chain + 1;
            next += 5;
            memcpy(next, insn, len * sizeof *next);
            next[3] = new_chain + 2;
        } else {
            memcpy(next, insn, len * sizeof *next);
            if (op == SPV_OP_ENTRY_POINT && interface && pool_sampler_spirv_interface(insn, ordinary) > 0) {
                next[0] = ((len + 1) << 16) | op;
                next[len] = comparison;
                ++next;
            }
        }
        next += len;
        at += len;
    }
    if ((uint64_t)(next - out) != count + additional) goto done;
    *rewritten = out;
    *rewritten_count = count + additional;
    out = NULL;
    result = 1;
done:
    free(out);
    free(uses);
    free(ids);
    return result;
}

#endif

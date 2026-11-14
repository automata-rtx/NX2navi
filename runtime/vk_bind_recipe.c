#include "vk_bind_recipe.h"
#include <stddef.h>
#include <stdlib.h>

static int stage_of(int stage, int override) {
    if (stage == GFX_STAGE_CS && override) return override;
    return stage == GFX_STAGE_VS ? 0 : stage == GFX_STAGE_FS ? 1 : stage == GFX_STAGE_TS ? 6 :
           stage == GFX_STAGE_MS ? 7 : 5;
}

static uint32_t buffer_source(int st, int slot, int uniform) {
    return (uint32_t)((uniform ? offsetof(VkExec, ubo) : offsetof(VkExec, ssbo)) +
                     st * (uniform ? sizeof(((VkExec *)0)->ubo[0]) : sizeof(((VkExec *)0)->ssbo[0])) +
                     slot * sizeof(p_bind_buf));
}

static void input_init(VkBindRecipeInput *in, const VkBindRecipeOp *op, uint32_t k) {
    const GfxShaderRes *r = op->binding->res;
    int st = op->stage;
    *in = (VkBindRecipeInput){.source = VK_BR_NO_SOURCE, .reuse = VK_BR_ALWAYS};
    if (op->kind == VK_BR_SUPPORT) return;
    in->slot = r->slot + (int)k;
    if (op->kind == VK_BR_UBO) {
        if (in->slot >= 3 && in->slot < 21) {
            in->source = buffer_source(st, in->slot - 3, 1);
            in->reuse = VK_BR_UNIFORM;
        } else if (in->slot == 1) in->reuse = VK_BR_CONSTANT;
        else if (in->slot != 0) in->reuse = VK_BR_ZERO;
    } else if (op->kind == VK_BR_SSBO) {
        // Indirections, written ranges, the global address table and mesh scratch keep their ordered resolver.
        if ((r->flags & 1) || r->ptr || r->cbuf_slot != 0) return;
        int off = r->handle * 4;
        int base = st == 0 ? 0x110 : st == 1 ? 0x510 : st == 5 ? 0x310 : st == 6 ? 0x670 : 0x770;
        int uniform = st == 5 && off >= 0x210 && off < 0x310;
        if (uniform) base = 0x210;
        if (off < base || off >= base + 0x100 || (off - base) % 16) return;
        in->source = buffer_source(st, (off - base) / 16, uniform);
        in->reuse = VK_BR_STORAGE;
    } else if (op->kind == VK_BR_TEXTURE || op->kind == VK_BR_IMAGE) {
        int image = op->kind == VK_BR_IMAGE, cap = image ? 32 : 128;
        if (in->slot >= 0 && in->slot < cap)
            in->source = (uint32_t)((image ? offsetof(VkExec, img) : offsetof(VkExec, tex)) +
                         st * (image ? sizeof(((VkExec *)0)->img[0]) : sizeof(((VkExec *)0)->tex[0])) +
                         in->slot * sizeof(uint64_t));
        if (!image) in->reuse = VK_BR_SAMPLED;
    } else if (op->kind == VK_BR_TEXEL) in->reuse = VK_BR_ZERO;
}

VkBindRecipe *vk_bind_recipe_create(const VkPipeInfo *info, int stage_override) {
    if (!info || info->n_bindings > VK_BR_MAX_BINDINGS || info->n_sets > 2 ||
        (info->bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS && info->bind_point != VK_PIPELINE_BIND_POINT_COMPUTE) ||
        (info->n_bindings && !info->bindings) || (stage_override && stage_override != 6 && stage_override != 7))
        return NULL;
    uint32_t count = 0;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        uint32_t n = info->bindings[i].count ? info->bindings[i].count : 1;
        if (n > VK_BR_MAX_DESC - count) return NULL;
        count += n;
    }
    VkBindRecipe *p = calloc(1, sizeof *p + count * sizeof *p->input);
    if (!p) return NULL;
    p->info = info, p->n_ops = info->n_bindings, p->n_sets = info->n_sets ? info->n_sets : 1;
    p->stage_override = stage_override;
    uint32_t next = 0;
    for (uint32_t s = 0; s < p->n_sets; s++) {
        VkBindRecipeSet *set = &p->sets[s];
        set->lo = next, set->first = p->n_desc;
        while (next < info->n_bindings && (p->n_sets == 1 || info->bindings[next].set == s)) {
            const VkPipeBinding *b = &info->bindings[next];
            if (b->stage != GFX_STAGE_VS && b->stage != GFX_STAGE_FS && b->stage != GFX_STAGE_CS &&
                b->stage != GFX_STAGE_TS && b->stage != GFX_STAGE_MS) goto invalid;
            if (next > set->lo && b->binding <= info->bindings[next - 1].binding) goto invalid;
            VkBindRecipeOp *op = &p->op[next];
            op->binding = b, op->first = p->n_desc, op->count = b->count ? b->count : 1;
            op->stage = (uint8_t)stage_of(b->stage, stage_override), op->set = (uint8_t)s;
            if (b->kind == VK_PIPE_SUPPORT) {
                if (op->count != 1 || b->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) goto invalid;
                op->kind = VK_BR_SUPPORT;
            } else {
                if (!b->res) goto invalid;
                if (b->kind == GFX_RES_UBO && b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) op->kind = VK_BR_UBO;
                else if (b->kind == GFX_RES_SSBO && b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) op->kind = VK_BR_SSBO;
                else if (b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) {
                    int image = b->kind == GFX_RES_IMG;
                    if (image ? b->type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE &&
                                b->type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                              : b->type != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
                                b->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                                b->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) goto invalid;
                    op->kind = b->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
                               b->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER ? VK_BR_TEXEL :
                               b->kind == GFX_RES_IMG ? VK_BR_IMAGE : VK_BR_TEXTURE;
                } else goto invalid;
            }
            for (uint32_t k = 0; k < op->count; k++) input_init(&p->input[p->n_desc + k], op, k);
            op->stable = op->kind != VK_BR_SUPPORT && op->kind != VK_BR_IMAGE &&
                         (op->kind != VK_BR_TEXTURE || !op->binding->res->exact);
            for (uint32_t k = 0; k < op->count; k++)
                if (p->input[p->n_desc + k].reuse == VK_BR_ALWAYS) op->stable = 0;
            if (op->stable) set->stable_groups |= 1u << vk_bind_recipe_group(op);
#if NX2NAVI_PREP_COMPILED_RESOLVE
            unsigned group = vk_bind_recipe_group(op);
            op->group_bit = (uint8_t)(1u << (s * VK_BR_GROUPS + group));
            VkBindRecipeMembers *members = &p->members[s][group];
            if (op->stable) members->ops[next / 64] |= 1ull << (next % 64);
            for (uint32_t k = 0; k < op->count; k++) {
                uint32_t j = p->n_desc + k;
                p->input[j].owner_op = (uint16_t)next;
                if (op->stable) {
                    members->descriptors[j / 64] |= 1ull << (j % 64);
                    members->words |= (uint16_t)(1u << (j / 64));
                }
            }
#endif
            p->n_desc += op->count;
            next++;
        }
        set->hi = next, set->count = p->n_desc - set->first;
        if (next != set->lo && !info->tpl[s]) p->need_w = 1;
    }
    if (next != info->n_bindings) goto invalid;
    return p;
invalid:
    free(p);
    return NULL;
}

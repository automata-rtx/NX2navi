// Membership is compared with the ordinary ordered recipe traversal, including both mask boundaries.
#include "vk_bind_recipe.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t random_state = 0x9e3779b9;
static uint32_t random_next(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    return random_state ^= random_state << 5;
}

static void verify(const VkBindRecipe *p) {
#if NX2NAVI_PREP_COMPILED_RESOLVE
    uint64_t refreshed[16];
    for (unsigned round = 0; round < 5; round++) {
        for (unsigned w = 0; w < 16; w++) refreshed[w] = round == 0 ? 0 : round == 1 ? UINT64_MAX :
            ((uint64_t)random_next() << 32) | random_next();
        for (unsigned set = 0; set < 2; set++) for (unsigned group = 0; group < VK_BR_GROUPS; group++) {
            const VkBindRecipeMembers *m = &p->members[set][group];
            uint64_t ops[2] = {m->ops[0], m->ops[1]};
            uint64_t desc[16] = {0};
            uint32_t words = 0;
            VkBindRecipeRefreshWalk walk = {.members = m, .refreshed = refreshed, .words = m->words};
            for (unsigned i = p->sets[set].lo; i < p->sets[set].hi; i++) {
                const VkBindRecipeOp *op = &p->op[i];
                assert(op->group_bit == 1u << (set * VK_BR_GROUPS + vk_bind_recipe_group(op)));
                for (unsigned k = 0; k < op->count; k++) assert(p->input[op->first + k].owner_op == i);
                if (!op->stable || vk_bind_recipe_group(op) != group) continue;
                assert(vk_bind_recipe_next_op(ops) == i);
                for (unsigned k = 0; k < op->count; k++) {
                    uint32_t j = op->first + k;
                    desc[j / 64] |= 1ull << (j % 64);
                    words |= 1u << (j / 64);
                    if (refreshed[j / 64] & (1ull << (j % 64))) assert(vk_bind_recipe_next_refreshed(&walk) == j);
                }
            }
            assert(vk_bind_recipe_next_op(ops) == VK_BR_MAX_BINDINGS);
            assert(vk_bind_recipe_next_refreshed(&walk) == VK_BR_MAX_DESC);
            assert(words == m->words);
            for (unsigned w = 0; w < 16; w++) assert(desc[w] == m->descriptors[w]);
        }
    }
#else
    (void)p;
#endif
}

int main(void) {
    unsigned max_ops = 0, max_desc = 0, mixed = 0;
    for (unsigned trial = 0; trial < 1000; trial++) {
        GfxShaderRes res[128] = {0};
        VkPipeBinding binding[128] = {0};
        unsigned n = trial == 0 ? 128 : random_next() % 129;
        unsigned split = trial == 0 ? 64 : random_next() % (n + 1), total = 0;
        for (unsigned i = 0; i < n; i++) {
            unsigned count = trial == 0 ? 8 : 1 + random_next() % 15;
            if (count > 1024 - total - (n - i - 1)) count = 1024 - total - (n - i - 1);
            total += count;
            binding[i] = (VkPipeBinding){.binding = i, .set = i >= split,
                .stage = i >= split ? GFX_STAGE_FS : GFX_STAGE_VS, .count = count, .res = &res[i]};
            unsigned kind = random_next() % 6;
            if (kind < 3) {
                binding[i].kind = GFX_RES_UBO;
                binding[i].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                res[i].slot = kind == 0 ? 0 : kind == 1 ? 1 : 3;
                if (kind == 1 && count >= 3) mixed++;
            } else if (kind < 5) {
                binding[i].kind = GFX_RES_TEX;
                binding[i].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                res[i].slot = random_next() % 128;
                res[i].exact = kind == 4;
            } else {
                binding[i].kind = GFX_RES_IMG;
                binding[i].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                res[i].slot = random_next() % 32;
            }
        }
        VkPipeInfo info = {.n_sets = 2, .n_bindings = n, .bindings = binding};
        VkBindRecipe *p = vk_bind_recipe_create(&info, 0);
        assert(p && p->n_ops == n && p->n_desc == total);
        if (n > max_ops) max_ops = n;
        if (total > max_desc) max_desc = total;
        verify(p);
        free(p);
    }
    assert(max_ops == 128 && max_desc == 1024 && mixed);
    printf("compiled membership: 1000 recipes, 128 operations, 1024 descriptors, mixed inputs and sparse ordered visits passed (mode %d)\n", NX2NAVI_PREP_COMPILED_RESOLVE);
    return 0;
}

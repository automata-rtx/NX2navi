// CPU-only coverage of compiled resource sources and ordered fallbacks. No game inputs or GPU are needed.
#include "vk_bind_recipe.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static VkPipeBinding binding(uint32_t index, int stage, int kind, GfxShaderRes *res, uint32_t count) {
    VkDescriptorType type = kind == GFX_RES_UBO || kind == VK_PIPE_SUPPORT ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER :
        kind == GFX_RES_SSBO ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : kind == GFX_RES_IMG ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    return (VkPipeBinding){.binding = index, .stage = stage, .kind = kind, .res = res, .count = count, .type = type};
}

int main(void) {
    VkExec state = {0};
    int stages[] = {GFX_STAGE_VS, GFX_STAGE_FS, GFX_STAGE_CS, GFX_STAGE_TS, GFX_STAGE_MS};
    int nvn[] = {0, 1, 5, 6, 7};
    for (unsigned st = 0; st < sizeof stages / sizeof *stages; st++) {
        GfxShaderRes u = {.slot = 2}, t = {.slot = 127}, image = {.slot = 31};
        int base[] = {0x110, 0x510, 0x310, 0x670, 0x770};
        GfxShaderRes s = {.cbuf_slot = 0, .handle = (base[st] + 16 * 4) / 4};
        GfxShaderRes written = s; written.flags = 1;
        GfxShaderRes indirect = {.cbuf_slot = 5, .handle = 8};
        VkPipeBinding b[] = {binding(0, stages[st], VK_PIPE_SUPPORT, NULL, 1),
            binding(1, stages[st], GFX_RES_UBO, &u, 3), binding(2, stages[st], GFX_RES_SSBO, &s, 2),
            binding(3, stages[st], GFX_RES_SSBO, &written, 1), binding(4, stages[st], GFX_RES_SSBO, &indirect, 1),
            binding(5, stages[st], GFX_RES_TEX, &t, 2), binding(6, stages[st], GFX_RES_IMG, &image, 2)};
        VkPipeInfo info = {.n_bindings = 7, .bindings = b, .n_sets = 2, .pool = 1};
        b[5].set = b[6].set = 1;
        VkBindRecipe *p = vk_bind_recipe_create(&info, 0);
        assert(p && p->n_desc == 12 && p->sets[0].count == 8 && p->sets[1].first == 8);
        assert(p->op[0].stage == nvn[st] && p->op[6].kind == VK_BR_IMAGE);
        assert(p->op[0].set == 0 && p->op[5].set == 1 && p->op[6].set == 1);
        assert(p->input[1].reuse == VK_BR_ZERO);
        assert(p->input[2].reuse == VK_BR_UNIFORM && p->input[3].reuse == VK_BR_UNIFORM);
        assert((const uint8_t *)&state + p->input[2].source == (const uint8_t *)&state.ubo[nvn[st]][0]);
        assert((const uint8_t *)&state + p->input[4].source == (const uint8_t *)&state.ssbo[nvn[st]][4]);
        assert(p->input[4].source == p->input[5].source); // storage arrays repeat the same underlying range
        assert(p->input[6].reuse == VK_BR_ALWAYS && p->input[7].reuse == VK_BR_ALWAYS);
        assert((const uint8_t *)&state + p->input[8].source == (const uint8_t *)&state.tex[nvn[st]][127]);
        assert(p->input[9].source == VK_BR_NO_SOURCE && p->input[11].source == VK_BR_NO_SOURCE);
        assert(p->input[10].reuse == VK_BR_ALWAYS); // storage images keep their write marks
        assert(!p->op[0].stable && p->op[1].stable && p->op[2].stable);
        assert(!p->op[3].stable && !p->op[4].stable && p->op[5].stable && !p->op[6].stable);
        free(p);
    }
    GfxShaderRes u = {.slot = 3};
    VkPipeBinding b = binding(0, GFX_STAGE_CS, GFX_RES_UBO, &u, 1);
    VkPipeInfo info = {.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE, .n_bindings = 1, .bindings = &b, .n_sets = 2};
    for (int stage = 6; stage <= 7; stage++) {
        VkBindRecipe *p = vk_bind_recipe_create(&info, stage);
        assert(p && p->op[0].stage == stage && p->sets[1].first == 1 && p->sets[1].count == 0);
        assert((const uint8_t *)&state + p->input[0].source == (const uint8_t *)&state.ubo[stage][0]);
        free(p);
    }
    GfxShaderRes s = {.cbuf_slot = 0, .handle = (0x210 + 16 * 7) / 4};
    b = binding(0, GFX_STAGE_CS, GFX_RES_SSBO, &s, 1);
    VkBindRecipe *p = vk_bind_recipe_create(&info, 0);
    assert(p && (const uint8_t *)&state + p->input[0].source == (const uint8_t *)&state.ubo[5][7]);
    free(p);
    s.ptr = 1;
    p = vk_bind_recipe_create(&info, 0);
    assert(p && p->input[0].reuse == VK_BR_ALWAYS);
    free(p);
    b.count = VK_BR_MAX_DESC + 1;
    assert(!vk_bind_recipe_create(&info, 0));
    b.count = 1;
    info.n_sets = 3;
    assert(!vk_bind_recipe_create(&info, 0));
    info.n_sets = 1;
    b.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    assert(!vk_bind_recipe_create(&info, 0)); // runtime dynamic offsets require a separate binding contract
    b = binding(0, GFX_STAGE_CS, GFX_RES_TEX, &u, 1);
    b.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    assert(!vk_bind_recipe_create(&info, 0));
    b.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    u.exact = 1;
    p = vk_bind_recipe_create(&info, 0);
    assert(p && !p->op[0].stable); // exact-level promotion keeps its ordered resolver
    free(p);
    u.exact = 0;
    b = binding(0, GFX_STAGE_VS, GFX_RES_UBO, &u, 4);
    u.slot = 0;
    p = vk_bind_recipe_create(&info, 0);
    assert(p && !p->op[0].stable); // mixed c0/constant/zero/uniform arrays never skip the dynamic c0 contribution
    free(p);
    puts("bind_recipe: graphics, compute, mesh, stage remapping, arrays, pools and ordered resources passed");
    return 0;
}

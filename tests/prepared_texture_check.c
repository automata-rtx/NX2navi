// Exercise the production sampled-image loop and token validation with controlled table mutations.
#define main binding_fixture_main
#include "prepared_bindings_check.c"
#undef main

static uint64_t texture_tokens[3], texture_views[3];
static int texture_locked, texture_mutate_lookup, texture_mutate_lock, texture_refresh_defer, texture_refresh_mutate;
static int texture_full_defer;
#define texture_defer test_deferred
static unsigned texture_mismatches, texture_lookups;

static uint64_t bt_now(void) { return 0; }
static uint64_t token_of(uint64_t h) { assert(h < 3); return texture_tokens[h]; }
void vk_texture_lock(void) {
    assert(!texture_locked); texture_locked = 1;
    if (texture_mutate_lock) {
        texture_mutate_lock = 0;
        test_textures++; texture_tokens[1]++; texture_views[1]++;
    }
}
void vk_texture_unlock(void) { assert(texture_locked); texture_locked = 0; }
static void actual_tex_info_lookup(VkExec *, const GfxShaderRes *, uint64_t, int, int, VkDescriptorImageInfo *, VkImage *);
static void tex_info(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                     VkDescriptorImageInfo *out, VkImage *image) {
    actual_tex_info_lookup(ex,r,h,storage,separate,out,image);
}
const VkTexImage *vk_texture_by_id_locked(int id, view_t *v, int *have) {
    (void)v; (void)have;
    assert(texture_locked && id>0 && id<3);
    if (texture_refresh_defer) texture_defer=1;
    if (texture_refresh_mutate) {
        texture_refresh_mutate=0;
        test_textures++; texture_tokens[id]++; texture_views[id]++;
    }
    return NULL;
}
static int tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                         VkDescriptorImageInfo *out, VkImage *image) {
    (void)ex; (void)r; (void)storage; (void)separate;
    assert(texture_locked && h > 0 && h < 3); texture_lookups++;
    if (texture_full_defer) texture_defer=1;
    if (texture_mutate_lookup && h == 1) {
        texture_mutate_lookup = 0;
        test_textures++; texture_tokens[2]++; texture_views[2]++;
    }
    *out = (VkDescriptorImageInfo){(VkSampler)(uintptr_t)4,
        (VkImageView)(uintptr_t)(texture_defer ? 99 : texture_views[h]), VK_IMAGE_LAYOUT_GENERAL};
    *image = texture_defer ? VK_NULL_HANDLE : (VkImage)(uintptr_t)(1000 + h);
    return 0;
}
static void texture_check(VkExec *ex, const VkBindRecipeOp *op, const PreparedBindings *p, uint32_t j,
                           int hit, const PreparedSlot *prior, const VkDescSlot *before, const VkDescSlot *after) {
    (void)ex; (void)op; (void)p; (void)j; (void)prior;
    if (hit && memcmp(before, after, sizeof *before)) texture_mismatches++;
}
#include "prepared_images_under_test.h"

static void reset_textures(PreparedBindings *p, VkExec *ex) {
    texture_locked = texture_mutate_lookup = texture_mutate_lock = texture_defer = 0;
    texture_refresh_defer=texture_refresh_mutate=texture_full_defer=0;
    memset(TD,0,sizeof TD);
    texture_lookups = texture_mismatches = 0;
    for (unsigned j = 0; j < 2; j++) {
        unsigned h = j + 1;
        texture_tokens[h] = h + 100; texture_views[h] = h + 200;
        ex->tex[0][3+j] = h;
        p->slot[j].texture = (TexMemo){.valid = 1, .h = h, .tok = texture_tokens[h],
            .ep = test_texture_epoch, .sg = test_samplers, .gen = test_textures,
            .image = (VkImage)(uintptr_t)(1000+h),
            .out = {(VkSampler)(uintptr_t)4, (VkImageView)(uintptr_t)texture_views[h], VK_IMAGE_LAYOUT_GENERAL}};
        p->desc[j] = (VkDescSlot){0}; p->desc[j].image = p->slot[j].texture.out;
    }
}

int main(void) {
    DrawCtx *cx = context_new(); VkExec ex = {.dc=cx, .fence=90};
    GfxShaderRes texture = {.slot=3};
    VkPipeBinding binding = {.kind=GFX_RES_TEX, .stage=GFX_STAGE_VS, .res=&texture, .count=2,
                            .type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
    VkPipeInfo info = {.n_bindings=1, .bindings=&binding};
    VkBindRecipe *recipe=vk_bind_recipe_create(&info,0); assert(recipe && recipe->n_desc==2);
    PreparedBindings *p=prepared_instance(&ex,recipe); const VkBindRecipeOp *op=&recipe->op[0];
    for (int check=0;check<=1;check++) {
        reset_textures(p,&ex); p->slot[0].texture.valid=0; texture_mutate_lookup=1;
        actual_prepared_images(&ex,op,p,check);
        assert(!texture_locked && !texture_mismatches);
        assert((uintptr_t)p->desc[1].image.imageView==texture_views[2]);
        assert(texture_lookups==2);
    }
    // The checking oracle must not compare a pre-lock hit against a table changed while it waited for the lock.
    reset_textures(p,&ex); texture_mutate_lock=1;
    actual_prepared_images(&ex,op,p,1);
    assert(!texture_mismatches && (uintptr_t)p->desc[0].image.imageView==texture_views[1]);
    // A deferred worker lookup is incomplete. Its placeholder must never become a reusable result.
    reset_textures(p,&ex); p->slot[0].texture.valid=0; texture_defer=1;
    actual_prepared_images(&ex,op,p,1);
    assert(!texture_mismatches && !p->slot[0].texture.valid && !p->slot[1].texture.valid);
    texture_defer=0; texture_lookups=0;
    actual_prepared_images(&ex,op,p,0);
    assert(texture_lookups==2 && (uintptr_t)p->desc[0].image.imageView==texture_views[1]);
    // The shared descriptor memo sits below the prepared slot. A deferred first use in a new submission must
    // invalidate that memo too, otherwise the retry would bypass the lookup through this second cache.
    reset_textures(p,&ex); VkDescriptorImageInfo out; VkImage image;
    vk_texture_lock(); actual_tex_info_lookup(&ex,&texture,1,0,0,&out,&image); vk_texture_unlock();
    assert(texture_lookups==1);
    test_texture_epoch++; texture_refresh_defer=1;
    vk_texture_lock(); actual_tex_info_lookup(&ex,&texture,1,0,0,&out,&image); vk_texture_unlock();
    assert(texture_defer);
    texture_defer=texture_refresh_defer=0;
    vk_texture_lock(); actual_tex_info_lookup(&ex,&texture,1,0,0,&out,&image); vk_texture_unlock();
    assert(texture_lookups==2 && (uintptr_t)out.imageView==texture_views[1]);
    // A completed storage refresh may replace the image rather than simply re-uploading it.
    test_texture_epoch++; texture_refresh_mutate=1;
    vk_texture_lock(); actual_tex_info_lookup(&ex,&texture,1,0,0,&out,&image); vk_texture_unlock();
    assert(texture_lookups==3 && (uintptr_t)out.imageView==texture_views[1]);
    // The memo's own checking mode also sees a deferred full lookup as incomplete, not an oracle mismatch.
    test_td=2; texture_full_defer=1;
    vk_texture_lock(); actual_tex_info_lookup(&ex,&texture,1,0,0,&out,&image); vk_texture_unlock();
    assert(texture_defer && texture_lookups==4);
    texture_defer=texture_full_defer=0;
    vk_texture_lock(); actual_tex_info_lookup(&ex,&texture,1,0,0,&out,&image); vk_texture_unlock();
    test_td=1;
    assert(texture_lookups==5 && (uintptr_t)out.imageView==texture_views[1]);
    context_free(cx); free(recipe);
    puts("prepared_texture: per-member invalidation, lock transitions, shared memo refresh and deferred retry passed");
    return 0;
}

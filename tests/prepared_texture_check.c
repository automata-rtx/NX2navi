// Exercise the production sampled-image loop and token validation with controlled table mutations.
#define main binding_fixture_main
#define PREP_TEXTURE_LOCK_TEST 1
#include "prepared_bindings_check.c"
#undef main
#include "texture_resource.h"

static uint64_t texture_tokens[3], texture_views[3];
static int texture_mutate_lookup, texture_mutate_lock, texture_refresh_defer, texture_refresh_mutate;
static int texture_full_defer;
#define texture_defer test_deferred
static unsigned texture_mismatches, texture_lookups, token_reads, selector_lookups;
static VkTexImage sample_image;
static VkTextureSampleStatus sample_status = VK_TEXTURE_SAMPLE_FALLBACK;
static unsigned sample_queries, sample_prepares, sample_uses, optional_prepares;
static int history_status;
typedef VkTexImage Tex;
static int tslots[3];
static const Tex *texture_projected_sample_report_locked(const int *slot,int native,VkImage expected,
    VkTextureSample *out,RcRegistryPosition *edge,const char **depth_gate) {
    (void)expected;(void)out;(void)edge;(void)depth_gate;
    assert(slot==&tslots[native]);
    return NULL; // The native projection fixture exercises the real provider.
}
static void sample_refusal_record(VkTextureSampleRefusal *refused,const int *slot,const Tex *source,const char *gate) {
    (void)refused;(void)slot;(void)source;(void)gate;assert(0);
}
static int texture_registration_source_attach(const int *slot, Tex *source) {
    assert(slot > tslots && slot < tslots + 3 && source == &sample_image);
    return 1;
}
static uint64_t texture_slot_registration(int id) { assert(id > 0); return 400 + (uint64_t)id; }
static const VkTexImage *texture_by_slot_locked(int id,view_t *v,int *have,VkTextureSampleStatus *status,
                                               VkTextureSampleRefusal *refused) {
    (void)refused;
    assert(*status==VK_TEXTURE_SAMPLE_FALLBACK);
    return vk_texture_by_id_locked(id,v,have);
}
uint64_t vk_exec_texture_pool(void) { return 8; }
int vk_texture_pool_slot_locked(uint64_t pool, int logical) { assert(pool == 8); selector_lookups++; return logical; }
static int texture_history_sample_locked(int id, VkTextureSample *sample, int *native, int recording) {
    assert(texture_locked && id > 0 && sample && native && recording);
    *native = vk_texture_pool_slot_locked(vk_exec_texture_pool(), id);
    if (history_status == VK_TEXTURE_REGISTRATION_EXACT) sample->image = &sample_image;
    return history_status;
}
VkTextureSampleStatus vk_texture_sample_query_locked(int id, VkTextureSample *out) {
    assert(texture_locked && id > 0); sample_queries++;
    out->image = &sample_image;
    if (sample_status == VK_TEXTURE_SAMPLE_READY) {
        out->native_slot = id;
        out->registration = texture_slot_registration(id);
    }
    return sample_status;
}
static VkTextureSampleStatus texture_sample_prepare_locked(int id, VkTextureSample *out, int budgeted,
                                                        VkTextureSampleRefusal *refused) {
    (void)refused;
    assert(texture_locked && id > 0); sample_prepares++;
    if (budgeted) {
        optional_prepares++;
        *out = (VkTextureSample){0};
        return VK_TEXTURE_SAMPLE_PREPARE;
    }
    out->image = &sample_image;
    out->native_slot = id;
    out->registration = texture_slot_registration(id);
    return VK_TEXTURE_SAMPLE_READY;
}
void vk_texture_sample_use_locked(const VkTextureSample *sample) {
    assert(texture_locked && sample->image == &sample_image); sample_uses++;
}

static uint64_t bt_now(void) { return 0; }
static uint64_t token_of(uint64_t h) { assert(h < 3); token_reads++; return texture_tokens[h]; }
void vk_texture_lock(void) {
    assert(!texture_locked); texture_locked = 1;
    if (texture_mutate_lock) {
        texture_mutate_lock = 0;
        test_textures++; texture_tokens[1]++; texture_views[1]++;
    }
}
void vk_texture_unlock(void) { assert(texture_locked); texture_locked = 0; }
static int tex_info_full(VkExec *, const GfxShaderRes *, uint64_t, int, int, VkDescriptorImageInfo *, VkImage *,int);
static void tex_info(VkExec *ex, int stage, uint32_t member, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                     VkDescriptorImageInfo *out, VkImage *image, VkTextureReadKey *key, int *recorded) {
    assert(stage == NVN_STAGE_VS && member < 2 && !recorded);
    tex_info_full(ex,r,h,storage,separate,out,image,1);
    *key = texture_defer || storage ? (VkTextureReadKey){0} :
        (VkTextureReadKey){.registration=400+h,.native_slot=(int)h};
}
const VkTexImage *vk_texture_by_id_locked(int id, view_t *v, int *have) {
    (void)v; (void)have;
    assert(texture_locked && id>0 && id<3);
    if (texture_refresh_defer) texture_defer=1;
    if (texture_refresh_mutate) {
        texture_refresh_mutate=0;
        test_textures++; texture_tokens[id]++; texture_views[id]++;
    }
    return &sample_image;
}
static int tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                         VkDescriptorImageInfo *out, VkImage *image, int demanded) {
    assert(demanded);
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
    texture_lookups = texture_mismatches = 0;
    for (unsigned j = 0; j < 2; j++) {
        unsigned h = j + 1;
        texture_tokens[h] = h + 100; texture_views[h] = h + 200;
        ex->tex[0][3+j] = h;
        p->slot[j].texture = (TexMemo){.authority = test_authority_epoch, .valid = 1, .h = h, .tok = texture_tokens[h],
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
    reset_textures(p,&ex);test_binding_allowed=0;
    actual_prepared_images(&ex,op,p,0);
    assert(texture_lookups==2 && !texture_locked);test_binding_allowed=1;
    // Two descriptor pools may expose an identical handle and even use the same image with different views.
    reset_textures(p,&ex);
    ex.tex_pool=9;
    texture_views[1]++;
    actual_prepared_images(&ex,op,p,0);
    assert(texture_lookups==2&&p->slot[0].texture.pool==9);
    assert((uintptr_t)p->desc[0].image.imageView==texture_views[1]);
    ex.tex_pool=0;
    // A deferred worker lookup is incomplete. Its placeholder must never become a reusable result.
    reset_textures(p,&ex); p->slot[0].texture.valid=0; texture_defer=1;
    actual_prepared_images(&ex,op,p,1);
    assert(!texture_mismatches && !p->slot[0].texture.valid && !p->slot[1].texture.valid);
    texture_defer=0; texture_lookups=0;
    actual_prepared_images(&ex,op,p,0);
    assert(texture_lookups==2 && (uintptr_t)p->desc[0].image.imageView==texture_views[1]);
    // Resolution owns the selected physical manifest before native packet publication. A later metadata or
    // completion-owner transition retains those captures, while permission and native installation are reset.
    assert(p->images&&p->images->image[0]==p->slot[0].texture.image);
    TextureImagePublication *selected=p->images;
    CbRes owners[2]={{0}};cx->res=&owners[0];
    assert(prepared_instance(&ex,recipe)==p&&p->images==selected&&!p->slot[0].texture.valid);
    actual_prepared_images(&ex,op,p,0);
    unsigned captured=test_publication_captures,committed=test_publication_commits;
    prepared_publish_images(&ex,p);
    assert(test_publication_captures==captured+1&&test_publication_commits==committed+1);
    assert(owners[0].images->image[0]==selected->image[0]);
    // Pending preparation must neither replace the old owned selection with its placeholder nor publish it.
    unsigned updated=test_publication_updates;
    VkImage retained=selected->image[0];
    p->slot[0].texture.valid=0;texture_defer=1;
    actual_prepared_images(&ex,op,p,0);
    assert(test_publication_updates==updated&&selected->image[0]==retained&&!p->slot[0].texture.valid);
    texture_defer=0;
    cx->res=&owners[1];test_capture++;ex.fence++;
    assert(prepared_instance(&ex,recipe)==p&&p->images==selected);
    assert(!p->slot[0].texture.valid&&!p->set[0].packet&&!p->set[0].allocated);
    actual_prepared_images(&ex,op,p,0);prepared_publish_images(&ex,p);
    assert(owners[1].images&&owners[1].images->image[0]==selected->image[0]);
    vk_texture_publication_release(owners[0].images);vk_texture_publication_release(owners[1].images);
    cx->res=NULL;
    // An ordered miss now has one resolver, with no TD layer able to keep an incomplete result beneath it.
    reset_textures(p,&ex);texture_full_defer=1;p->slot[0].texture.valid=p->slot[1].texture.valid=0;
    actual_prepared_images(&ex,op,p,0);
    assert(texture_defer && !p->slot[0].texture.valid && !p->slot[1].texture.valid);
    unsigned before_retry=texture_lookups;texture_defer=texture_full_defer=0;
    actual_prepared_images(&ex,op,p,0);
    assert(texture_lookups==before_retry+2 && p->slot[0].texture.valid && p->slot[1].texture.valid);
    assert(p->slot[0].texture.selected.registration==401 && p->slot[1].texture.selected.registration==402);
    unsigned selected_checks=test_selected_texture_checks;
    actual_prepared_images(&ex,op,p,0);
    assert(test_selected_texture_checks==selected_checks+2 && texture_lookups==before_retry+2);
    p->slot[0].texture.selected.canonical = p->slot[1].texture.selected.canonical = 1;
    unsigned prior_tokens = token_reads;
    test_authority_epoch++;
    test_textures++;
    actual_prepared_images(&ex,op,p,0);
    assert(test_selected_texture_checks==selected_checks+4 && token_reads==prior_tokens && texture_lookups==before_retry+2);
    test_binding_allowed = 0; // The exact source permission wins even with the same logical selector.
    actual_prepared_images(&ex,op,p,0);
    assert(texture_lookups==before_retry+4);
    test_binding_allowed = 1;

    // A qualified group renews the permissions carried by its installed manifest. A mixed group keeps the
    // old aggregate policy, and a changed member or closed ordered scope cannot inherit its old proof.
    p->slot[0].texture.selected.canonical = p->slot[1].texture.selected.canonical = 1;
    PreparedGroup *group = &p->set[0].group[VK_BR_GROUP_TEXTURE];
    PreparedStamp stamp = prepared_stamp(&ex,VK_BR_GROUP_TEXTURE);
    VkUsnapProof proof = {0};
    test_next_write=(RcRegistryPosition){test_position.scene,8};
    prepared_seal(&ex,p,0,VK_BR_GROUP_TEXTURE,&stamp,&proof);
    assert(group->sealed && (group->flags & PREP_QUALIFIED_TEXTURE));
    unsigned exact_checks=test_selected_texture_checks, preserved_lookups=texture_lookups;
    unsigned preserved_updates=test_publication_updates;
    prior_tokens=token_reads;
    test_textures++;test_authority_epoch++;test_registrations++;
    test_image_operation=actual_prepared_images;
    cx->prep_dirty[VK_BR_GROUP_TEXTURE]=0;
    uint64_t aggregate_skips=cx->prep_stats.aggregate_skipped;
    prepared_resolve(&ex,p,0);
    assert(cx->prep_stats.aggregate_skipped==aggregate_skips+op->count);
    assert(test_selected_texture_checks==exact_checks+2 && token_reads==prior_tokens &&
           texture_lookups==preserved_lookups && test_publication_updates==preserved_updates && p->images==selected);
    assert(prepared_group_current(&ex,p,0,VK_BR_GROUP_TEXTURE) && test_selected_texture_checks==exact_checks+2);
    test_position.ordinal=7;
    assert(!prepared_group_current(&ex,p,0,VK_BR_GROUP_TEXTURE)); // The actual writer's read edge was reached.
    test_next_write.ordinal=10;
    assert(prepared_group_current(&ex,p,0,VK_BR_GROUP_TEXTURE)); // New finite permission for the same exact input.
    assert(group->texture_position.ordinal==7 && group->texture_next_write.ordinal==9);
    test_registrations += UINT64_C(1) << 30; // A history window overflow cannot stand in for exact validation.
    test_binding_allowed=0;
    assert(!prepared_group_current(&ex,p,0,VK_BR_GROUP_TEXTURE));
    assert(group->stamp.registrations!=test_registrations && p->images==selected);
    test_binding_allowed=1;test_position_open=0;
    assert(!prepared_group_current(&ex,p,0,VK_BR_GROUP_TEXTURE));
    test_position_open=1;
    texture_mutate_lock=1;
    assert(!prepared_group_current(&ex,p,0,VK_BR_GROUP_TEXTURE)); // Mutation while the receipt acquires the texture lock.
    assert(prepared_group_current(&ex,p,0,VK_BR_GROUP_TEXTURE));
    test_mapping++;
    assert(!prepared_group_current(&ex,p,0,VK_BR_GROUP_TEXTURE));
    test_mapping--;
    test_samplers++;
    assert(!prepared_group_current(&ex,p,0,VK_BR_GROUP_TEXTURE));
    test_samplers--;
    p->slot[1].texture.selected.canonical=0;
    p->slot[1].texture.authority=test_authority_epoch;
    stamp=prepared_stamp(&ex,VK_BR_GROUP_TEXTURE);
    prepared_seal(&ex,p,0,VK_BR_GROUP_TEXTURE,&stamp,&proof);
    assert(group->sealed && !(group->flags & PREP_QUALIFIED_TEXTURE));
    test_textures++;
    assert(!prepared_group_current(&ex,p,0,VK_BR_GROUP_TEXTURE));
    test_position.ordinal=2;test_next_write=(RcRegistryPosition){0};
    // The integration consumes ready data without preparation, prepares exactly once when requested,
    // and records use only after a successful result. Special ordered images bypass this API.
    vk_texture_lock();
    sample_status=VK_TEXTURE_SAMPLE_READY;
    sample_queries=sample_prepares=sample_uses=0;
    VkTextureSample sample;
    assert(vk_texture_resolve_sample_locked(1,VK_TEXTURE_READ_REQUIRED,&sample)==VK_TEXTURE_SAMPLE_READY);
    assert(sample.image==&sample_image&&sample.native_slot==1&&!sample.historical&&sample.registration==401);
    assert(sample_queries==1 && sample_prepares==0 && sample_uses==1 && selector_lookups==1);
    assert(vk_texture_resolve_sample_locked(1,0,&sample)==VK_TEXTURE_SAMPLE_READY);
    assert(sample.image==&sample_image&&sample.native_slot==1&&sample_queries==2&&sample_prepares==0&&sample_uses==2);
    sample_status=VK_TEXTURE_SAMPLE_PREPARE;
    assert(vk_texture_resolve_sample_locked(1,VK_TEXTURE_READ_REQUIRED,&sample)==VK_TEXTURE_SAMPLE_READY);
    assert(sample.image==&sample_image&&sample_prepares==1&&sample_uses==3);
    assert(vk_texture_resolve_sample_locked(1,0,&sample)==VK_TEXTURE_SAMPLE_PREPARE);
    assert(sample_prepares==2&&optional_prepares==1&&sample_uses==3&&!sample.image);
    sample_status=VK_TEXTURE_SAMPLE_UNAVAILABLE;
    assert(vk_texture_resolve_sample_locked(1,VK_TEXTURE_READ_REQUIRED,&sample)==VK_TEXTURE_SAMPLE_UNAVAILABLE);
    assert(sample_uses==3);
    sample_status=VK_TEXTURE_SAMPLE_DEFERRED;
    assert(vk_texture_resolve_sample_locked(1,VK_TEXTURE_READ_REQUIRED,&sample)==VK_TEXTURE_SAMPLE_DEFERRED);
    assert(sample_uses==3);
    unsigned queried=sample_queries;
    assert(vk_texture_resolve_sample_locked(1,VK_TEXTURE_READ_FORCE_COMPAT,&sample)==VK_TEXTURE_SAMPLE_FALLBACK);
    history_status=VK_TEXTURE_REGISTRATION_EXACT;
    assert(vk_texture_resolve_sample_locked(1,VK_TEXTURE_READ_REQUIRED,&sample)==VK_TEXTURE_SAMPLE_READY);
    assert(sample.image==&sample_image&&sample.historical==VK_TEXTURE_REGISTRATION_EXACT&&sample_uses==3);
    history_status=VK_TEXTURE_REGISTRATION_GONE;
    assert(vk_texture_resolve_sample_locked(1,0,&sample)==VK_TEXTURE_SAMPLE_GONE&&!sample.image);
    assert(sample_queries==queried);
    vk_texture_unlock();
    unsigned released=test_publication_releases;
    context_free(cx);assert(test_publication_releases==released+1);free(recipe);
    puts("prepared_texture: per-member invalidation, lock transitions, shared memo refresh and deferred retry passed");
    return 0;
}

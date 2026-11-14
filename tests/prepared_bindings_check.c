// Production cache decisions and traversal are extracted unchanged. Resource resolution and the snapshot proof
// service are controlled fakes here; usnap_lease_check exercises the actual publication protocol separately.
#include "prepared_state_under_test.h"
#undef calloc
#include <assert.h>

static uint64_t test_mem = 10, test_snap = 20, test_capture = 30, test_drop, test_epoch = 100, test_serial = 1;
static uint32_t test_textures = 40, test_texture_epoch = 50, test_samplers = 60;
static int test_td = 1, test_token_ok = 1, test_fail_alloc, test_mutate_ordered;
static int test_snapshots, test_fail_proof_end, test_deferred;
static unsigned test_allocs, test_fail_at, test_begins, test_executed[VK_BR_MAX_BINDINGS], test_sequence[256], test_nsequence;
static uint64_t test_counted;
static uint64_t test_semantics = 1469598103934665603ull;
static void semantic_event(uint64_t value) { test_semantics = (test_semantics ^ value) * 1099511628211ull; }
VkExecFrame vk_exec_frame;
const VkTexImage *vk_dlaa_swap_img;

uint64_t vk_mem_gen(void) { return test_mem; }
uint64_t vk_usnap_gen(void) { return test_snap; }
uint64_t vk_usnap_capture(void) { return test_capture; }
uint32_t vk_texture_slots_gen(void) { return test_textures; }
uint32_t vk_texture_epoch(void) { return test_texture_epoch; }
uint32_t vk_sampler_gen(void) { return test_samplers; }
int vk_texture_deferred(void) { return test_deferred; }
int vk_usnap_lease_valid(const VkUsnapLease *lease) {
    return lease->capture && lease->capture == test_capture && (!test_drop || test_drop > test_epoch);
}
void vk_usnap_proof_begin(VkUsnapProof *p) {
    semantic_event(1);
    memset(p, 0, sizeof *p);
    p->capture = test_capture, p->epoch = test_epoch, p->serial = test_serial, p->building = !!test_capture;
    test_begins++;
}
int vk_usnap_proof_add(VkUsnapProof *p, const VkUsnapLease *lease) {
    semantic_event(2); semantic_event(lease->capture);
    if (!p->building || !vk_usnap_lease_valid(lease)) return p->building = 0;
    if (test_drop && (!p->horizon || test_drop < p->horizon)) p->horizon = test_drop;
    p->buckets |= 1;
    return 1;
}
int vk_usnap_proof_end(VkUsnapProof *p) {
    semantic_event(3);
    p->sealed = p->building && p->capture == test_capture && p->epoch == test_epoch && p->serial == test_serial;
    if (test_fail_proof_end) test_fail_proof_end--, p->sealed = 0;
    p->building = 0;
    return p->sealed;
}
int vk_usnap_proof_valid(VkUsnapProof *p) {
    semantic_event(4);
    return p->sealed && p->capture == test_capture && p->serial == test_serial &&
           (!p->horizon || test_epoch < p->horizon);
}
int vk_usnap_proof_inherit(VkUsnapProof *p, const VkUsnapProof *prior) {
    semantic_event(5);
    VkUsnapProof copy = *prior;
    if (!p->building || p->capture != prior->capture || p->epoch != test_epoch ||
        !vk_usnap_proof_valid(&copy) || copy.serial != p->serial) return p->building = 0;
    p->buckets |= prior->buckets;
    if (prior->horizon && (!p->horizon || prior->horizon < p->horizon)) p->horizon = prior->horizon;
    return 1;
}
void rt_fatal(const char *fmt, ...) { (void)fmt; abort(); }
void rt_log(const char *fmt, ...) { (void)fmt; }
static int td_mode(void) { return test_td; }
static int slot_still(uint32_t *gen, uint64_t token, uint64_t handle, uint32_t current) {
    if (!test_token_ok || token != handle + 100) return 0;
    *gen = current;
    return 1;
}
static void count_use(DrawCtx *cx, int cls, int use, uint64_t n) {
    semantic_event(6); semantic_event(n);
    (void)cx; assert(cls == VK_MCLS_HOST_CACHED && use == VK_MUSE_UBO); test_counted += n;
}
static void *prepared_test_calloc(size_t n, size_t size) {
    test_allocs++;
    if (test_fail_at == test_allocs) { test_fail_at = 0; return NULL; }
    if (test_fail_alloc) { test_fail_alloc--; return NULL; }
    return calloc(n, size);
}

static void execute_fake(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) {
    (void)check;
    unsigned i = (unsigned)(op - p->recipe->op);
    semantic_event(7); semantic_event(i);
    test_executed[i]++;
    assert(test_nsequence < sizeof test_sequence / sizeof *test_sequence);
    test_sequence[test_nsequence++] = i;
    if (!op->stable) {
        if (test_mutate_ordered) test_mem++;
        return;
    }
    for (uint32_t k = 0; k < op->count; k++) {
        uint32_t j = op->first + k;
        PreparedSlot *slot = &p->slot[j];
        const p_bind_buf *b = prepared_buffer(ex, &p->recipe->input[j]);
        VkDescSlot before = p->desc[j];
        int hit = op->kind == VK_BR_TEXTURE ? prepared_texture_valid(ex, &p->recipe->input[j], slot, &p->desc[j]) :
                  op->kind != VK_BR_TEXEL && prepared_buffer_valid(ex, &p->recipe->input[j], slot);
        if (hit && !check) {
            if (slot->count.has) count_use(ex->dc, slot->count.cls, slot->count.use, slot->count.n);
            continue;
        }
        prepared_refresh_record(p, op, j);
        if (op->kind == VK_BR_TEXEL) p->desc[j].view = (VkBufferView)(uintptr_t)99;
        else if (op->kind == VK_BR_TEXTURE) {
            uint64_t handle = prepared_handle(ex, &p->recipe->input[j]);
            slot->texture = (TexMemo){.valid = 1, .h = handle, .tok = handle + 100,
                .ep = test_texture_epoch, .sg = test_samplers, .gen = test_textures,
                .image = (VkImage)(uintptr_t)8,
                .out = {(VkSampler)(uintptr_t)2, (VkImageView)(uintptr_t)3, VK_IMAGE_LAYOUT_GENERAL}};
            p->desc[j].image = slot->texture.out;
            p->desc[j].w[2] = (uint32_t)VK_IMAGE_LAYOUT_GENERAL;
        }
        else {
            assert(b);
            slot->input[0] = b->addr, slot->input[1] = b->size;
            slot->valid = 1, slot->mem_gen = test_mem, slot->snap_gen = test_snap;
            memset(&slot->lease, 0, sizeof slot->lease);
            if (test_snapshots) slot->lease.capture = test_capture;
            slot->count = (PbCount){.cls = VK_MCLS_HOST_CACHED, .use = VK_MUSE_UBO, .n = b->size, .has = 1};
            p->desc[j].buffer = (VkDescriptorBufferInfo){(VkBuffer)(uintptr_t)1, b->addr, b->size};
            count_use(ex->dc, slot->count.cls, slot->count.use, slot->count.n);
        }
        prepared_changed(p, op, j, &before);
    }
}
static void prepared_support(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) { execute_fake(ex, op, p, check); }
static void prepared_uniforms(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) { execute_fake(ex, op, p, check); }
static void prepared_storage(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) { execute_fake(ex, op, p, check); }
static void prepared_images(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) { execute_fake(ex, op, p, check); }
static void prepared_texels(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) { execute_fake(ex, op, p, check); }

static DrawCtx *context_new(void) { DrawCtx *cx = calloc(1, sizeof *cx); assert(cx); return cx; }
static void directory_check(DrawCtx *cx) {
    uint16_t expected[PREP_BUCKETS] = {0};
    uint32_t bytes = 0;
    for (unsigned i = 0; i < PREP_N; i++) {
        PreparedBindings *p = &cx->prep[i];
        if (p->recipe) {
            assert(p->index == i);
            expected[prepared_bucket(p->recipe)] |= (uint16_t)(1u << i);
            for (unsigned j = i + 1; j < PREP_N; j++) assert(cx->prep[j].recipe != p->recipe);
        }
        bytes += p->capacity * (sizeof *p->slot + sizeof *p->desc);
    }
    assert(!memcmp(expected, cx->prep_lookup, sizeof expected));
    assert(bytes == cx->prep_bytes && bytes <= PREP_BYTES_LIMIT);
}
static void context_free(DrawCtx *cx) {
    directory_check(cx);
    for (unsigned i = 0; i < PREP_N; i++) prepared_evict(cx, &cx->prep[i]);
    directory_check(cx);
    assert(!cx->prep_bytes);
    if (dc == cx) dc = &dc_main;
    free(cx);
}
static void reset_observations(void) {
    memset(test_executed, 0, sizeof test_executed);
    test_counted = 0, test_nsequence = 0, test_begins = 0;
}
static VkPipeBinding uniform_binding(unsigned binding, GfxShaderRes *resource) {
    return (VkPipeBinding){.binding = binding, .kind = GFX_RES_UBO, .stage = GFX_STAGE_VS,
                          .res = resource, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
}

static void lifetimes_and_watchers(void) {
    DrawCtx *cx = context_new(), *other = context_new();
    VkExec ex = {.dc = cx, .cb = (VkCommandBuffer)(uintptr_t)1, .fence = 1};
    GfxShaderRes resource = {.slot = 3};
    VkPipeBinding binding = uniform_binding(0, &resource);
    VkPipeInfo info = {.n_bindings = 1, .bindings = &binding};
    VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0);
    assert(recipe);
    PreparedBindings *p = prepared_instance(&ex, recipe);
    uint32_t bit = 1u << (p->index * 2);
    assert(cx->prep_ubo[0][0] == bit && !other->prep_ubo[0][0]);
    assert(!prepared_watch(cx, (uint32_t)offsetof(VkExec, ubo) + 1));
    ex.ubo[0][0] = (p_bind_buf){.addr = 4096, .size = 256};
    prepared_resolve(&ex, p, 0);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].sealed && prepared_group_valid(&ex, &p->set[0].group[VK_BR_GROUP_BUFFER]));
    reset_observations();
    prepared_resolve(&ex, p, 0);
    assert(!test_executed[0] && !test_begins && test_counted == 256);
    dc = other; // adoption must use the explicit owner even while TLS still refers to a different context
    vk_exec_draw_state_adopted(&ex);
    assert(cx->prep_dirty[0] == UINT32_MAX && cx->prep_dirty[1] == UINT32_MAX && other->prep_dirty[0] == 0 && other->prep_dirty[1] == 0);
    prepared_resolve(&ex, p, 0);
    assert(test_executed[0] == !NX2NAVI_PREP_PARTIAL_REUSE);
    p->set[0].allocated = (VkDescriptorSet)(uintptr_t)7;
    dc = cx;
    vk_exec_draw_new_cb();
    ex.cb = (VkCommandBuffer)(uintptr_t)2;
    assert(prepared_instance(&ex, recipe) == p && p->slot[0].valid && p->set[0].allocated);
    test_capture++;
    assert(!prepared_instance(&ex, recipe)->slot[0].valid && !p->set[0].allocated && !p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    prepared_resolve(&ex, p, 0);
    ex.fence++;
    assert(!prepared_instance(&ex, recipe)->slot[0].valid);
    prepared_evict(cx, p);
    assert(!cx->prep_ubo[0][0] && !cx->prep_bytes);
    context_free(cx), context_free(other), free(recipe);

    uint64_t handles[128] = {0};
    uint32_t watches[128] = {0};
    watches[3] = 4, watches[4] = 8;
    p_bind_handles bind = {.stage = 0, .first = 3, .n = 2, .h = {11, 22}};
    assert(bind_handles(handles, 128, &bind, watches) == 12);
    assert(!bind_handles(handles, 128, &bind, watches));
    bind.h[1]++;
    assert(bind_handles(handles, 128, &bind, watches) == 8);
    bind.first = INT32_MAX;
    assert(!bind_handles(handles, 128, &bind, watches));
}

static void ordered_and_partial_sets(void) {
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 10};
    GfxShaderRes u0 = {.slot = 3}, u1 = {.slot = 4};
    VkPipeBinding bindings[] = {uniform_binding(0, &u0),
        {.binding = 1, .kind = VK_PIPE_SUPPORT, .stage = GFX_STAGE_VS, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
        uniform_binding(2, &u1)};
    VkPipeInfo info = {.n_bindings = 3, .bindings = bindings};
    VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0);
    PreparedBindings *p = prepared_instance(&ex, recipe);
    ex.ubo[0][0] = (p_bind_buf){.addr = 4096, .size = 128};
    ex.ubo[0][1] = (p_bind_buf){.addr = 8192, .size = 256};
    reset_observations();
    prepared_resolve(&ex, p, 0);
    assert(test_nsequence == 3 && test_sequence[0] == 0 && test_sequence[1] == 1 && test_sequence[2] == 2);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    reset_observations();
    prepared_resolve(&ex, p, 0);
    assert(test_nsequence == 1 && test_sequence[0] == 1 && test_counted == 384 && !test_begins);
    reset_observations();
    test_mutate_ordered = 1;
    prepared_resolve(&ex, p, 0);
    assert(!test_executed[0] && test_executed[1] == 1 && test_executed[2] == 1);
    assert(test_counted == 384 && !p->set[0].group[VK_BR_GROUP_BUFFER].sealed); // earlier cached live range was invalidated mid-set
    test_mutate_ordered = 0;
    reset_observations();
    prepared_resolve(&ex, p, 0);
    assert(test_executed[0] == 1 && test_executed[2] == 1 && p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    reset_observations();
    prepared_resolve(&ex, p, 1);
    assert(test_nsequence == 3 && test_counted == 384 && p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    context_free(cx), free(recipe);
}

static void proofs_and_texture_sealing(void) {
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 20};
    GfxShaderRes uniform = {.slot = 3};
    VkPipeBinding binding = uniform_binding(0, &uniform);
    VkPipeInfo info = {.n_bindings = 1, .bindings = &binding};
    VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0);
    PreparedBindings *p = prepared_instance(&ex, recipe);
    ex.ubo[0][0] = (p_bind_buf){.addr = 4096, .size = 256};
    prepared_resolve(&ex, p, 0);
    PreparedSlot *slot = &p->slot[0];
    slot->lease = (VkUsnapLease){.capture = test_capture, .addr = 4096, .size = 256, .align = 16};
    test_drop = test_epoch + 1;
    PreparedStamp stamp = prepared_stamp(&ex, VK_BR_GROUP_BUFFER);
    VkUsnapProof proof;
    vk_usnap_proof_begin(&proof);
    prepared_seal(&ex, p, 0, VK_BR_GROUP_BUFFER, &stamp, &proof);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].sealed && p->snapshot_count[0] == 1 && prepared_group_valid(&ex, &p->set[0].group[VK_BR_GROUP_BUFFER]));
    test_mem++, test_snap++; // an unrelated live-resource invalidation cannot discard a valid captured lease
    assert(prepared_group_valid(&ex, &p->set[0].group[VK_BR_GROUP_BUFFER]));
    test_epoch++;
    assert(!prepared_group_valid(&ex, &p->set[0].group[VK_BR_GROUP_BUFFER]));
    test_drop = 0;
    prepared_evict(cx, p);
    free(recipe);

    GfxShaderRes texture = {.slot = 3};
    binding = (VkPipeBinding){.kind = GFX_RES_TEX, .stage = GFX_STAGE_VS, .res = &texture,
                             .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER};
    recipe = vk_bind_recipe_create(&info, 0);
    p = prepared_instance(&ex, recipe), slot = &p->slot[0];
    ex.tex[0][3] = 17;
    slot->texture = (TexMemo){.valid = 1, .h = 17, .tok = 117, .ep = test_texture_epoch,
        .sg = test_samplers, .gen = test_textures, .image = (VkImage)(uintptr_t)8,
        .out = {(VkSampler)(uintptr_t)2, (VkImageView)(uintptr_t)3, VK_IMAGE_LAYOUT_GENERAL}};
    p->desc[0].image = slot->texture.out;
    p->desc[0].w[2] = (uint32_t)VK_IMAGE_LAYOUT_GENERAL;
    stamp = prepared_stamp(&ex, VK_BR_GROUP_TEXTURE); vk_usnap_proof_begin(&proof); prepared_seal(&ex, p, 0, VK_BR_GROUP_TEXTURE, &stamp, &proof);
    assert(p->set[0].group[VK_BR_GROUP_TEXTURE].sealed && prepared_group_valid(&ex, &p->set[0].group[VK_BR_GROUP_TEXTURE]));
    test_samplers++;
    assert(!prepared_group_valid(&ex, &p->set[0].group[VK_BR_GROUP_TEXTURE]));
    stamp = prepared_stamp(&ex, VK_BR_GROUP_TEXTURE); vk_usnap_proof_begin(&proof); prepared_seal(&ex, p, 0, VK_BR_GROUP_TEXTURE, &stamp, &proof);
    assert(!p->set[0].group[VK_BR_GROUP_TEXTURE].sealed); // unchanged construction stamps do not bless an old sampler memo
    test_samplers--; test_token_ok = 0;
    stamp = prepared_stamp(&ex, VK_BR_GROUP_TEXTURE); vk_usnap_proof_begin(&proof); prepared_seal(&ex, p, 0, VK_BR_GROUP_TEXTURE, &stamp, &proof);
    assert(!p->set[0].group[VK_BR_GROUP_TEXTURE].sealed);
    test_token_ok = 1;
    ex.nfbl = 1, ex.fbl_img[0] = slot->texture.image;
    assert(!prepared_texture_valid(&ex, &recipe->input[0], slot, &p->desc[0]));
    p->desc[0].w[2] = (uint32_t)VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
    assert(prepared_texture_valid(&ex, &recipe->input[0], slot, &p->desc[0]));
    stamp = prepared_stamp(&ex, VK_BR_GROUP_TEXTURE); vk_usnap_proof_begin(&proof); prepared_seal(&ex, p, 0, VK_BR_GROUP_TEXTURE, &stamp, &proof);
    assert(p->set[0].group[VK_BR_GROUP_TEXTURE].sealed);
    twin_dispatch = 1;
    assert(!prepared_group_valid(&ex, &p->set[0].group[VK_BR_GROUP_TEXTURE]));
    twin_dispatch = 0; cx->mo_ovr_on = 1;
    assert(!prepared_group_valid(&ex, &p->set[0].group[VK_BR_GROUP_TEXTURE]));
    cx->mo_ovr_on = 0;
    context_free(cx), free(recipe);
}

static void mixed_set_groups_and_texel_scratch(void) {
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 25};
    GfxShaderRes uniform = {.slot = 3}, texture = {.slot = 7};
    VkPipeBinding bindings[] = {uniform_binding(0, &uniform),
        {.binding = 1, .kind = GFX_RES_TEX, .stage = GFX_STAGE_VS, .res = &texture,
         .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
        {.binding = 2, .kind = GFX_RES_TEX, .stage = GFX_STAGE_VS, .res = &texture,
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER}};
    VkPipeInfo info = {.n_bindings = 3, .bindings = bindings};
    VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0);
    assert(recipe && recipe->need_w && recipe->sets[0].stable_groups == 3);
    PreparedBindings *p = prepared_instance(&ex, recipe);
    ex.ubo[0][0] = (p_bind_buf){.addr = 4096, .size = 128};
    ex.tex[0][7] = 17;
    prepared_resolve(&ex, p, 0);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].sealed && p->set[0].group[VK_BR_GROUP_TEXTURE].sealed);
    reset_observations();
    ex.ubo[0][0].addr += 256;
    cx->prep_dirty[VK_BR_GROUP_BUFFER] |= cx->prep_ubo[0][0];
    prepared_resolve(&ex, p, 0);
    assert(test_executed[0] == 1 && test_executed[1] == 0 && test_counted == 128);
    assert(p->desc[0].buffer.offset == ex.ubo[0][0].addr && p->changed_sets == 1);
    reset_observations();
    test_samplers++;
    prepared_resolve(&ex, p, 0);
    assert(test_executed[0] == 0 && test_executed[1] == 1 && test_counted == 128);
    reset_observations();
    cx->bind_bv[2] = (VkBufferView)(uintptr_t)12345;
    prepared_resolve(&ex, p, 0);
    assert(!test_nsequence && !test_begins && cx->bind_bv[2] == p->desc[2].view);
    assert(cx->bind_w[2].pTexelBufferView == &cx->bind_bv[2]);
    context_free(cx), free(recipe);
}

static void eviction_failure_packets_and_stats(void) {
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 30};
    enum { N = 80 };
    VkPipeInfo infos[N];
    VkBindRecipe *recipes[N];
    GfxShaderRes u = {.slot = 3};
    VkPipeBinding b = uniform_binding(0, &u);
    // Large arrays exercise the allocation bound even when the associative buckets differ.
    b.count = VK_BR_MAX_DESC;
    for (unsigned i = 0; i < N; i++) {
        infos[i] = (VkPipeInfo){.n_bindings = 1, .bindings = &b};
        recipes[i] = vk_bind_recipe_create(&infos[i], 0);
        assert(recipes[i] && prepared_instance(&ex, recipes[i]));
        assert(cx->prep_bytes <= PREP_BYTES_LIMIT);
        uint32_t expected = 0;
        for (unsigned j = 0; j < PREP_N; j++) if (cx->prep[j].recipe) expected |= 1u << (2 * j);
        assert(cx->prep_ubo[0][0] == expected); // eviction unregisters old masks before slot reuse
    }
    context_free(cx);
    for (unsigned i = 0; i < N; i++) free(recipes[i]);
    cx = context_new(); ex.dc = cx; b.count = 1;
    infos[0] = (VkPipeInfo){.n_bindings = 1, .bindings = &b};
    recipes[0] = vk_bind_recipe_create(&infos[0], 0);
    test_fail_alloc = 1;
    assert(!prepared_instance(&ex, recipes[0]) && !cx->prep_bytes && !cx->prep_ubo[0][0]);
    unsigned attempts = test_allocs;
    assert(!prepared_instance(&ex, recipes[0]) && test_allocs == attempts);
    ex.fence++;
    PreparedBindings *p = prepared_instance(&ex, recipes[0]);
    assert(p);
    DescriptorPacket *packet = malloc(sizeof *packet + sizeof(VkDescSlot));
    packet->bytes = sizeof(VkDescSlot), packet->data[0] = p->desc[0];
    assert(descriptor_packet_equal(packet, p->desc, sizeof(VkDescSlot)));
    p->desc[0].buffer.offset++;
    assert(!descriptor_packet_equal(packet, p->desc, sizeof(VkDescSlot)));
    cx->B.descriptors[0].push_packet[0] = packet;
    cx->B.descriptors[1].push_packet[0] = packet;
    cx->B.descriptors[1].pushed[0] = 1;
    compute_bindings_disturbed(cx);
    assert(!cx->B.descriptors[1].pushed[0] && cx->B.descriptors[0].push_packet[0] == packet);
    // Local accumulation never writes concurrently to the plain frame fields. Collection transfers each total once.
    dc = cx; vk_exec_draw_prepared_stats_collect();
    memset(&vk_exec_frame, 0, sizeof vk_exec_frame);
    cx->prep_stats.builds = 7, cx->prep_stats.reused = 123;
    cx->prep_stats.cache_hits = 9, cx->prep_stats.cache_misses = 2;
    cx->prep_stats.cache_evictions = 1, cx->prep_stats.cache_probes = 11;
    vk_exec_draw_prepared_stats_flush(&ex);
    assert(!cx->prep_stats.builds && !vk_exec_frame.prep_builds);
    vk_exec_draw_prepared_stats_collect();
    assert(vk_exec_frame.prep_builds == 7 && vk_exec_frame.prep_reused == 123);
    assert(vk_exec_frame.prep_cache_hits == 9 && vk_exec_frame.prep_cache_misses == 2);
    assert(vk_exec_frame.prep_cache_evictions == 1 && vk_exec_frame.prep_cache_probes == 11);
    vk_exec_draw_prepared_stats_collect();
    assert(vk_exec_frame.prep_builds == 7);
    assert(vk_exec_frame.prep_cache_hits == 9 && vk_exec_frame.prep_cache_probes == 11);
    free(packet); context_free(cx); free(recipes[0]);
}

static void defer_changed_descriptor(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) {
    (void)ex; (void)check;
    p->desc[op->first].w[1]++;
    test_deferred = 1;
}

static void incremental_group_renewal(void) {
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 70};
    enum { N = 8 };
    GfxShaderRes resources[N]; VkPipeBinding bindings[N];
    for (unsigned i=0;i<N;i++) {
        resources[i]=(GfxShaderRes){.slot=(int)i+3}; bindings[i]=uniform_binding(i,&resources[i]);
        ex.ubo[0][i]=(p_bind_buf){.addr=4096+i*256,.size=128};
    }
    VkPipeInfo info={.n_bindings=N,.bindings=bindings};
    VkBindRecipe *recipe=vk_bind_recipe_create(&info,0);
    assert(recipe);
    PreparedBindings *p=prepared_instance(&ex,recipe);
    prepared_resolve(&ex,p,0);
    uint64_t visits=cx->prep_stats.proof_descriptors;
    ex.ubo[0][3].addr+=4096; ex.ubo[0][3].size=256;
    cx->prep_dirty[VK_BR_GROUP_BUFFER] |= cx->prep_ubo[0][3];
    reset_observations(); prepared_resolve(&ex,p,0);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    assert(cx->prep_stats.proof_descriptors-visits==1);
    assert(test_counted==(N+1)*128);
    reset_observations(); prepared_resolve(&ex,p,0);
    assert(!test_nsequence && test_counted==(N+1)*128);
    // Global memory changes reject renewal even though the final addresses did not change.
    visits=cx->prep_stats.proof_descriptors; test_mem++;
    prepared_resolve(&ex,p,0);
    assert(cx->prep_stats.proof_descriptors-visits==N);
    // Adopting identical state dirties the inputs, but the old valid resources need no second descriptor walk.
    vk_exec_draw_state_adopted(&ex); visits=cx->prep_stats.proof_descriptors;
    prepared_resolve(&ex,p,0);
    assert(cx->prep_stats.proof_descriptors==visits && p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    prepared_op_check(&ex, &p->recipe->op[0], p, defer_changed_descriptor);
    assert(test_deferred && !p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    prepared_resolve(&ex,p,1);
    assert(!p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    test_deferred=0;
    prepared_resolve(&ex,p,1);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    context_free(cx); free(recipe);

    // A changed snapshot binding extends the old proof under the fresh publication bracket.
    cx=context_new(); ex.dc=cx; ex.fence++; test_snapshots=1; test_drop=0;
    recipe=vk_bind_recipe_create(&info,0); p=prepared_instance(&ex,recipe);
    prepared_resolve(&ex,p,0);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].flags & PREP_SNAPSHOT);
    visits=cx->prep_stats.proof_descriptors;
    ex.ubo[0][5].addr+=4096; cx->prep_dirty[VK_BR_GROUP_BUFFER]|=cx->prep_ubo[0][5];
    prepared_resolve(&ex,p,0);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].sealed && cx->prep_stats.proof_descriptors-visits==1);
    for (unsigned i=0;i<N;i++) assert(p->snapshot_count[i]==1);
    // Renewal edits two contributions before its closing bracket fails. The full fallback must rebuild counts
    // from scratch, including both snapshot-to-live transitions, and publish no successful renewal statistic.
    uint64_t renewals=cx->prep_stats.renewals, renewed=cx->prep_stats.renewed_descriptors;
    visits=cx->prep_stats.proof_descriptors; test_snapshots=0; test_fail_proof_end=1;
    ex.ubo[0][2].addr+=4096; ex.ubo[0][2].size+=128; ex.ubo[0][5].addr+=4096;
    cx->prep_dirty[VK_BR_GROUP_BUFFER]|=cx->prep_ubo[0][2] | cx->prep_ubo[0][5];
    prepared_resolve(&ex,p,0);
    assert(!test_fail_proof_end && p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    assert(cx->prep_stats.proof_descriptors-visits==N+2);
    assert(cx->prep_stats.renewals==renewals && cx->prep_stats.renewed_descriptors==renewed);
    for (unsigned i=0;i<N;i++) assert(p->snapshot_count[i]==(i!=2 && i!=5));
    reset_observations(); prepared_resolve(&ex,p,0);
    uint64_t expected_count=0; for (unsigned i=0;i<N;i++) expected_count+=ex.ubo[0][i].size;
    assert(!test_nsequence && test_counted==expected_count);
    // Moving one live contribution back into the snapshot proof updates the next all-skipped draw as well.
    test_snapshots=1; visits=cx->prep_stats.proof_descriptors;
    ex.ubo[0][2].addr+=4096; cx->prep_dirty[VK_BR_GROUP_BUFFER]|=cx->prep_ubo[0][2];
    prepared_resolve(&ex,p,0);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].sealed && cx->prep_stats.proof_descriptors-visits==1);
    assert(cx->prep_stats.renewals==renewals+1 && cx->prep_stats.renewed_descriptors==renewed+1);
    for (unsigned i=0;i<N;i++) assert(p->snapshot_count[i]==(i!=5));
    reset_observations(); prepared_resolve(&ex,p,0);
    assert(!test_nsequence && test_counted==expected_count);
    test_drop=test_epoch; test_serial++;
    prepared_resolve(&ex,p,0);
    assert(!p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    test_drop=0; test_snapshots=0; context_free(cx); free(recipe);
}

static void retained_storage_replacements(void) {
    enum { N = PREP_N + 1 };
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 100};
    GfxShaderRes u = {.slot = 3};
    VkPipeBinding binding = uniform_binding(0, &u);
    binding.count = 16;
    VkPipeInfo info = {.n_bindings = 1, .bindings = &binding};
    VkBindRecipe *recipes[N];
    for (unsigned i = 0; i < N; i++) {
        recipes[i] = vk_bind_recipe_create(&info, 0);
        assert(recipes[i] && recipes[i]->n_desc == 16);
    }
    unsigned initial = test_allocs;
    // A working set larger than the cache forces replacement after the cold fills. Poisoning detects stale proofs
    // and descriptors; smaller active ranges followed by larger ones exercise retained, previously unused tails.
    for (unsigned k = 0; k < 1000; k++) {
        VkBindRecipe *r = recipes[k % N];
        r->n_desc = k < N ? 16 : (k % 3 == 0 ? 4 : 16);
        PreparedBindings *p = prepared_instance(&ex, r);
        assert(p && p->capacity == 16 && p->recipe == r);
        PreparedSlot zero_slot = {0};
        VkDescSlot zero_desc = {0};
        for (unsigned j = 0; j < r->n_desc; j++) {
            assert(!memcmp(&p->slot[j], &zero_slot, sizeof zero_slot));
            assert(!memcmp(&p->desc[j], &zero_desc, sizeof zero_desc));
        }
        assert(!p->set[0].packet && !p->set[0].allocated && !p->set[0].group[0].sealed);
        assert(cx->prepared_last != p);
        uint32_t expected_bytes = 0, watch = 0;
        for (unsigned j = 0; j < PREP_N; j++) {
            expected_bytes += cx->prep[j].capacity * (sizeof *p->slot + sizeof *p->desc);
            if (cx->prep[j].recipe) watch |= 1u << (2 * j);
        }
        assert(cx->prep_bytes == expected_bytes && cx->prep_bytes <= PREP_BYTES_LIMIT);
        assert(cx->prep_ubo[0][0] == watch);
        memset(p->slot, 0xa5, p->capacity * sizeof *p->slot);
        memset(p->desc, 0xa5, p->capacity * sizeof *p->desc);
        p->set[0].packet = (const DescriptorPacket *)(uintptr_t)1;
        p->set[0].allocated = (VkDescriptorSet)(uintptr_t)1;
        p->set[0].group[0].sealed = 1;
        cx->prepared_last = p;
    }
    assert(test_allocs - initial == PREP_N * 2);
    context_free(cx);
    // Fail the second allocation during a replacement that requires growth. No partially allocated storage,
    // recipe watches or accounting may escape; the same submission falls back without retrying allocation.
    cx = context_new(); ex.dc = cx;
    for (unsigned i = 0; i < PREP_N; i++) {
        recipes[i]->n_desc = 4;
        assert(prepared_instance(&ex, recipes[i]));
    }
    recipes[PREP_N]->n_desc = 16;
    test_fail_at = test_allocs + 2;
    assert(!prepared_instance(&ex, recipes[PREP_N]) && !test_fail_at);
    unsigned attempts = test_allocs;
    assert(!prepared_instance(&ex, recipes[PREP_N]) && attempts == test_allocs);
    assert(cx->prep_bytes == (PREP_N - 1) * 4 * (sizeof(PreparedSlot) + sizeof(VkDescSlot)));
    ex.fence++;
    assert(prepared_instance(&ex, recipes[PREP_N]));
    context_free(cx);
    for (unsigned i = 0; i < N; i++) free(recipes[i]);
    puts("prepared storage: 1000 capacity replacements use 32 allocations; poisoned tails, budget and growth failure passed");
}

static void associative_residency(void) {
    enum { N = PREP_N + 1, ROUNDS = 500 };
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 120};
    GfxShaderRes resource = {.slot = 3};
    VkPipeBinding binding = uniform_binding(0, &resource);
    VkPipeInfo info = {.n_bindings = 1, .bindings = &binding};
    VkBindRecipe *recipes[N];
    PreparedBindings *resident[PREP_N];
    ex.ubo[0][0] = (p_bind_buf){.addr = 4096, .size = 128};
    for (unsigned i = 0; i < N; i++) {
        // Identical legacy residency keys deliberately exhaust one former four-way set.
        recipes[i] = vk_bind_recipe_create(&info, 0);
        assert(recipes[i]);
    }
    for (unsigned i = 0; i < PREP_N; i++) {
        resident[i] = prepared_instance(&ex, recipes[i]);
        assert(resident[i]);
        prepared_resolve(&ex, resident[i], 0);
    }
    unsigned allocs = test_allocs;
    reset_observations();
    for (unsigned k = 0; k < ROUNDS * PREP_N; k++) {
        unsigned i = k % PREP_N;
        assert(prepared_instance(&ex, recipes[i]) == resident[i]);
        prepared_resolve(&ex, resident[i], 0);
        directory_check(cx);
    }
    assert(test_allocs == allocs && !test_nsequence);
    assert(cx->prep_stats.cache_hits == ROUNDS * PREP_N && cx->prep_stats.cache_misses == PREP_N);
    assert(!cx->prep_stats.cache_evictions);
    // Retained proofs still observe changed inputs and every watch bit, including bit 30.
    assert(cx->prep_ubo[0][0] == 0x55555555u);
    ex.ubo[0][0].addr += 4096;
    cx->prep_dirty[VK_BR_GROUP_BUFFER] |= cx->prep_ubo[0][0];
    reset_observations();
    for (unsigned i = 0; i < PREP_N; i++) prepared_resolve(&ex, resident[i], 0);
    assert(test_nsequence == PREP_N);
    cx->prepared_last = resident[0];
    assert(prepared_instance(&ex, recipes[PREP_N]) == resident[0]);
    assert(!cx->prepared_last && !resident[0]->slot[0].valid);
    directory_check(cx);
    // Random accesses are checked against an independent move-to-front reference LRU, including explicit drops.
    unsigned lru[PREP_N], count = PREP_N;
    lru[0] = PREP_N;
    for (unsigned i = 1; i < PREP_N; i++) lru[i] = PREP_N - i;
    uint32_t random = 0x27ade123u;
    for (unsigned k = 0; k < 20000; k++) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        unsigned which = random % N, at = count;
        for (unsigned i = 0; i < count; i++) if (lru[i] == which) at = i;
        PreparedBindings *prior = NULL;
        for (unsigned i = 0; i < PREP_N; i++) if (cx->prep[i].recipe == recipes[which]) prior = &cx->prep[i];
        assert((prior != NULL) == (at < count));
        if ((random & 31u) == 0 && prior) {
            prepared_evict(cx, prior);
            memmove(&lru[at], &lru[at + 1], (count - at - 1) * sizeof *lru);
            count--;
        } else {
            PreparedBindings *p = prepared_instance(&ex, recipes[which]);
            assert(p && (!prior || p == prior));
            if (at == count) { if (count < PREP_N) count++; at = count - 1; }
            memmove(&lru[1], &lru[0], at * sizeof *lru);
            lru[0] = which;
        }
        directory_check(cx);
        for (unsigned i = 0; i < PREP_N; i++) if (cx->prep[i].recipe) {
            int found = 0;
            for (unsigned j = 0; j < count; j++) found |= cx->prep[i].recipe == recipes[lru[j]];
            assert(found);
        }
    }
    context_free(cx);
    for (unsigned i = 0; i < N; i++) free(recipes[i]);
    puts("prepared residency: 8000 conflict-free reuses, change invalidation and 20000 reference-LRU operations passed");
}

static void directory_collisions(void) {
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 121};
    GfxShaderRes resource = {.slot = 3};
    VkPipeBinding binding = uniform_binding(0, &resource);
    VkPipeInfo info = {.n_bindings = 1, .bindings = &binding};
    VkBindRecipe *recipes[PREP_BUCKETS + 1], *first[PREP_BUCKETS] = {0}, *a = NULL, *b = NULL;
    for (unsigned i = 0; i <= PREP_BUCKETS; i++) {
        recipes[i] = vk_bind_recipe_create(&info, 0);
        assert(recipes[i]);
        unsigned bucket = prepared_bucket(recipes[i]);
        if (first[bucket]) a = first[bucket], b = recipes[i];
        else first[bucket] = recipes[i];
    }
    assert(a && b && a != b);
    PreparedBindings *pa = prepared_instance(&ex, a), *pb = prepared_instance(&ex, b);
    assert(pa && pb && pa != pb);
    pa->desc[0].w[0] = 123, pb->desc[0].w[0] = 456;
    for (unsigned i = 0; i < 100; i++) {
        assert(prepared_instance(&ex, a) == pa && pa->desc[0].w[0] == 123);
        assert(prepared_instance(&ex, b) == pb && pb->desc[0].w[0] == 456);
    }
    prepared_evict(cx, pa);
    directory_check(cx);
    assert(prepared_instance(&ex, b) == pb && pb->desc[0].w[0] == 456);
    pa = prepared_instance(&ex, a);
    assert(pa && pa != pb && !pa->desc[0].w[0]);
    context_free(cx);
    for (unsigned i = 0; i <= PREP_BUCKETS; i++) free(recipes[i]);
    puts("prepared directory: forced collisions retain exact identity across lookup, eviction and readmission");
}

// Compare retained production traversal with a fresh resolution after each state transition. The fresh context
// cannot reuse descriptors or group proofs; ordered effects and resource accounting must nevertheless agree.
static void retained_against_fresh_resolution(void) {
    DrawCtx *cached = context_new(), *fresh = context_new();
    VkExec ex = {.dc = cached, .cb = (VkCommandBuffer)(uintptr_t)1, .fence = 1000};
    enum { OPS = 10 };
    GfxShaderRes resources[OPS] = {0};
    VkPipeBinding bindings[OPS] = {0};
    for (unsigned set = 0; set < 2; set++) {
        unsigned base = set * 5;
        int stage = set ? GFX_STAGE_FS : GFX_STAGE_VS;
        resources[base].slot = 3;
        resources[base + 2].slot = 7;
        resources[base + 3].slot = 5;
        resources[base + 4].slot = 8;
        bindings[base] = uniform_binding(0, &resources[base]);
        bindings[base].count = 2;
        bindings[base + 1] = (VkPipeBinding){.binding = 1, .kind = VK_PIPE_SUPPORT,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
        bindings[base + 2] = (VkPipeBinding){.binding = 2, .kind = GFX_RES_TEX,
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .res = &resources[base + 2]};
        bindings[base + 3] = uniform_binding(3, &resources[base + 3]);
        bindings[base + 4] = (VkPipeBinding){.binding = 4, .kind = GFX_RES_TEX,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, .res = &resources[base + 4]};
        for (unsigned i = base; i < base + 5; i++) bindings[i].stage = stage, bindings[i].set = set;
        for (unsigned i = 0; i < 3; i++)
            ex.ubo[set][i] = (p_bind_buf){.addr = 4096 + set * 8192 + i * 512, .size = 128};
        ex.tex[set][7] = 17 + set;
    }
    VkPipeInfo info = {.n_bindings = OPS, .n_sets = 2, .bindings = bindings};
    VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0);
    assert(recipe && recipe->need_w);
    uint32_t random = 0x6a09e667u;
    for (unsigned step = 0; step < 4096; step++) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        unsigned stage = (random >> 8) & 1, slot = (random >> 10) % 3;
        switch (random & 15) {
        case 0: case 1: case 2: case 3: break;
        case 4: case 5:
            ex.ubo[stage][slot].addr += 256;
            ex.ubo[stage][slot].size = 64 * (1 + ((random >> 16) & 7));
            cached->prep_dirty[VK_BR_GROUP_BUFFER] |= cached->prep_ubo[stage][slot];
            break;
        case 6:
            ex.tex[stage][7]++;
            cached->prep_dirty[VK_BR_GROUP_TEXTURE] |= cached->prep_tex[stage][7];
            break;
        case 7: test_mem++; break;
        case 8: test_snap++; break;
        case 9: test_samplers++; break;
        case 10: test_textures++; test_texture_epoch++; break;
        case 11: vk_exec_draw_state_adopted(&ex); break;
        case 12: ex.fence++; test_capture++; break;
        case 13:
            dc = cached; vk_exec_draw_new_cb();
            ex.cb = (VkCommandBuffer)((uintptr_t)ex.cb + 1);
            break;
        case 14: test_snapshots = !test_snapshots; test_mem++; break;
        case 15: test_epoch++; test_serial++; test_drop = test_epoch + 1; break;
        }
        test_mutate_ordered = (random >> 20) % 17 == 0;
        test_deferred = (random >> 16) % 29 == 0;
        cached->mo_ovr_on = (random >> 12) % 31 == 0;
        PreparedBindings *retained = prepared_instance(&ex, recipe);
        assert(retained);
        uint64_t before_memory = test_mem;
        reset_observations(); prepared_resolve(&ex, retained, 0);
        uint64_t after_memory = test_mem, retained_count = test_counted;
        unsigned retained_ordered[OPS];
        memcpy(retained_ordered, test_executed, sizeof retained_ordered);
        VkExec reference = ex;
        reference.dc = fresh; fresh->mo_ovr_on = cached->mo_ovr_on;
        for (unsigned i = 0; i < PREP_N; i++) if (fresh->prep[i].recipe) prepared_forget(fresh, &fresh->prep[i]);
        PreparedBindings *resolved = prepared_instance(&reference, recipe);
        assert(resolved);
        test_mem = before_memory;
        reset_observations(); prepared_resolve(&reference, resolved, 0);
        assert(test_mem == after_memory && test_counted == retained_count);
        assert(!memcmp(retained->desc, resolved->desc, recipe->n_desc * sizeof *retained->desc));
        for (unsigned i = 0; i < OPS; i++)
            if (!recipe->op[i].stable) assert(retained_ordered[i] == test_executed[i]);
        assert(!memcmp(cached->bind_bv, fresh->bind_bv, recipe->n_desc * sizeof *cached->bind_bv));
    }
    assert(cached->prep_stats.aggregate_skipped > 0);
    context_free(cached); context_free(fresh); free(recipe);
    test_mutate_ordered = test_deferred = test_snapshots = 0; test_drop = 0;
    puts("prepared differential: 4096 retained versus fresh resolutions preserve bytes, ordered effects and census");
}

static void dirty_group_unchanged_members(void) {
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 2000};
    GfxShaderRes resources[3] = {{.slot = 3}, {.slot = 4}, {.slot = 5}};
    VkPipeBinding bindings[] = {uniform_binding(0, &resources[0]),
        {.binding = 1, .kind = VK_PIPE_SUPPORT, .stage = GFX_STAGE_VS,
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
        uniform_binding(2, &resources[1]), uniform_binding(3, &resources[2])};
    VkPipeInfo info = {.n_bindings = 4, .bindings = bindings};
    VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0);
    assert(recipe);
    for (unsigned i = 0; i < 3; i++) ex.ubo[0][i] = (p_bind_buf){.addr = 4096 + i * 512, .size = 128};
    PreparedBindings *p = prepared_instance(&ex, recipe);
    prepared_resolve(&ex, p, 0);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    // One changed input makes the group dirty, but cannot force the other valid inputs through their resolvers.
    ex.ubo[0][2].addr += 4096;
    cx->prep_dirty[VK_BR_GROUP_BUFFER] |= cx->prep_ubo[0][2];
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(!test_executed[0] && test_executed[1] == 1 && !test_executed[2] && test_executed[3] == 1);
    assert(test_counted == 384 && p->desc[3].buffer.offset == ex.ubo[0][2].addr);
    // Dirty notification without an actual input change retains the same results, including resource accounting.
    vk_exec_draw_state_adopted(&ex);
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(test_nsequence == 1 && test_sequence[0] == 1 && test_counted == 384);
    // Reuse before an ordered invalidation must not grant permission to reuse after it in the same group.
    vk_exec_draw_state_adopted(&ex); test_mutate_ordered = 1;
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(!test_executed[0] && test_executed[1] == 1 && test_executed[2] == 1 && test_executed[3] == 1);
    assert(test_counted == 384 && !p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    test_mutate_ordered = 0;
    context_free(cx); free(recipe);

    // Empty renewal cannot extend a snapshot beyond its recorded future-write horizon, even with the same serial.
    cx = context_new(); ex.dc = cx; ex.fence++; test_snapshots = 1; test_drop = test_epoch + 2;
    info.n_bindings = 1; info.bindings = bindings;
    recipe = vk_bind_recipe_create(&info, 0); p = prepared_instance(&ex, recipe);
    prepared_resolve(&ex, p, 0);
    assert(p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    vk_exec_draw_state_adopted(&ex);
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(!test_nsequence && test_counted == 128 && p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    test_epoch = test_drop;
    vk_exec_draw_state_adopted(&ex);
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(test_executed[0] == 1 && !p->set[0].group[VK_BR_GROUP_BUFFER].sealed);
    test_snapshots = 0; test_drop = 0;
    context_free(cx); free(recipe);
    puts("prepared partial reuse: unchanged members skip resolution, ordered invalidation and snapshot horizons remain binding");
}

static void partial_sampled_arrays_and_source_free_inputs(void) {
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 3000};
    GfxShaderRes resources[] = {{.slot = 7}, {.slot = 9}, {.slot = 10}};
    VkPipeBinding bindings[] = {
        {.binding = 0, .kind = GFX_RES_TEX, .stage = GFX_STAGE_VS, .count = 2,
         .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .res = &resources[0]},
        {.binding = 1, .kind = GFX_RES_TEX, .stage = GFX_STAGE_VS,
         .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .res = &resources[1]},
        {.binding = 2, .kind = GFX_RES_TEX, .stage = GFX_STAGE_VS,
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, .res = &resources[2]}};
    VkPipeInfo info = {.n_bindings = 3, .bindings = bindings};
    VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0);
    assert(recipe);
    ex.tex[0][7] = 70; ex.tex[0][8] = 80; ex.tex[0][9] = 90;
    PreparedBindings *p = prepared_instance(&ex, recipe);
    prepared_resolve(&ex, p, 0);
    ex.tex[0][9]++;
    cx->prep_dirty[VK_BR_GROUP_TEXTURE] |= cx->prep_tex[0][9];
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(!test_executed[0] && test_executed[1] == 1 && !test_executed[2]);
    ex.tex[0][8]++;
    cx->prep_dirty[VK_BR_GROUP_TEXTURE] |= cx->prep_tex[0][8];
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(test_executed[0] == 1 && !test_executed[1] && !test_executed[2]);
    assert(p->slot[1].texture.h == ex.tex[0][8]);
    // Equal handles cannot override a changed sampler generation.
    test_samplers++; vk_exec_draw_state_adopted(&ex);
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(test_executed[0] == 1 && test_executed[1] == 1 && !test_executed[2]);
    context_free(cx); free(recipe);

    // Constants and zero descriptors do not have direct bound input addresses. Constants still require the
    // caller's program proof, and zero descriptors do not depend on the ordinary buffer-slot valid bit.
    cx = context_new(); ex.dc = cx; ex.fence++;
    prog_t program = {.serial = 991}; ex.prog = &program;
    GfxShaderRes constant = {.slot = 1};
    bindings[0] = uniform_binding(0, &constant);
    bindings[1] = bindings[2]; bindings[1].binding = 1;
    info.n_bindings = 2;
    recipe = vk_bind_recipe_create(&info, 0); p = prepared_instance(&ex, recipe);
    assert(recipe->input[0].source == VK_BR_NO_SOURCE && recipe->input[1].reuse == VK_BR_ZERO);
    p->slot[0].valid = 1;
    p->slot[0].prog = &program; p->slot[0].prog_serial = program.serial;
    PreparedGroup *group = &p->set[0].group[VK_BR_GROUP_BUFFER];
    group->sealed = 1; group->flags = PREP_PROGRAM; group->stamp = prepared_stamp(&ex, VK_BR_GROUP_BUFFER);
    assert(prepared_group_valid(&ex, group));
    assert(prepared_op_inputs_equal(&ex, &recipe->op[0], p));
    assert(!p->slot[1].valid && prepared_op_inputs_equal(&ex, &recipe->op[1], p));
    program.serial++;
    assert(!prepared_group_valid(&ex, group));
    recipe->input[1].reuse = VK_BR_ALWAYS;
    assert(!prepared_op_inputs_equal(&ex, &recipe->op[1], p));
    recipe->input[1].reuse = 255;
    assert(!prepared_op_inputs_equal(&ex, &recipe->op[1], p));
    recipe->input[1].reuse = VK_BR_ZERO; recipe->op[1].stable = 0;
    assert(!prepared_op_inputs_equal(&ex, &recipe->op[1], p));
    context_free(cx); free(recipe);
    puts("prepared partial inputs: sampled arrays, sampler invalidation, constants, zero and unsupported inputs passed");
}

static void promotion_effects(void) {
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 3000};
    GfxShaderRes resources[2] = {{.slot = 3}, {.slot = 4}};
    VkPipeBinding bindings[3] = {uniform_binding(0, &resources[0]),
        {.binding = 1, .kind = VK_PIPE_SUPPORT, .stage = GFX_STAGE_VS, .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
        uniform_binding(2, &resources[1])};
    VkPipeInfo info = {.n_bindings = 3, .bindings = bindings};
    VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0);
    assert(recipe);
    ex.ubo[0][0] = (p_bind_buf){.addr = 4096, .size = 64};
    ex.ubo[0][1] = (p_bind_buf){.addr = 8192, .size = 128};
    PreparedBindings *p = prepared_instance(&ex, recipe);
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(p->set[0].group[0].sealed);
    ex.ubo[0][1].addr += 256;
    cx->prep_dirty[0] = UINT32_MAX;
    p->promotion_pending = 1;
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(!p->promotion_pending && cx->prep_residency.promoted_valid_groups == 1);
    assert(cx->prep_residency.promoted_skipped == 1 && cx->prep_residency.promoted_resolved == 1);
    assert(test_nsequence == 2 && test_sequence[0] == 1 && test_sequence[1] == 2);
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(cx->prep_residency.promoted_valid_groups == 1 && cx->prep_residency.promoted_skipped == 1 &&
           cx->prep_residency.promoted_resolved == 1);
    p->promotion_pending = 1;
    cx->prep_dirty[0] = UINT32_MAX;
    reset_observations(); prepared_resolve(&ex, p, 1);
    assert(!p->promotion_pending && cx->prep_residency.promoted_valid_groups == 2);
    assert(cx->prep_residency.promoted_skipped == 1 && cx->prep_residency.promoted_resolved == 3);
    context_free(cx); free(recipe);
    puts("promotion counters: initial proof validity, actual stable decisions and one-shot consumption passed");
}

static void maximum_recipe_resolution(void) {
    enum { OPS = 128, COUNT = 8 };
    DrawCtx *cached = context_new(), *fresh = context_new();
    VkExec ex = {.dc = cached, .fence = 4000};
    GfxShaderRes resources[OPS] = {0};
    VkPipeBinding bindings[OPS] = {0};
    for (unsigned i = 0; i < OPS; i++) {
        unsigned set = i / 64;
        resources[i].slot = i % 4 == 1 ? (i % 16) * 8 : 3;
        bindings[i] = uniform_binding(i, &resources[i]);
        bindings[i].set = set;
        bindings[i].stage = set ? GFX_STAGE_FS : GFX_STAGE_VS;
        bindings[i].count = COUNT;
        if (i % 4 == 1) {
            bindings[i].kind = GFX_RES_TEX;
            bindings[i].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        } else if (i % 4 == 3) {
            resources[i].flags = 1;
            bindings[i].kind = GFX_RES_SSBO;
            bindings[i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
    }
    for (unsigned set = 0; set < 2; set++) {
        for (unsigned slot = 0; slot < 8; slot++) ex.ubo[set][slot] = (p_bind_buf){.addr = 4096 + set * 8192 + slot * 256, .size = 64};
        for (unsigned slot = 0; slot < 128; slot++) ex.tex[set][slot] = 100 + set * 128 + slot;
    }
    VkPipeInfo info = {.n_sets = 2, .n_bindings = OPS, .bindings = bindings};
    VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0);
    assert(recipe && recipe->n_desc == 1024);
    for (unsigned step = 0; step < 64; step++) {
        unsigned set = step % 2, slot = step % 8;
        if (step % 4 == 0) {
            ex.ubo[set][slot].addr += 256;
            cached->prep_dirty[0] |= cached->prep_ubo[set][slot];
        } else if (step % 4 == 1) {
            ex.tex[set][8 + slot]++;
            cached->prep_dirty[1] |= cached->prep_tex[set][8 + slot];
        }
        if (step % 11 == 0) { test_snapshots = !test_snapshots; test_mem++; }
        if (step % 17 == 0) test_fail_proof_end = 1;
        test_mutate_ordered = step % 13 == 0;
        unsigned fail_before = test_fail_proof_end;
        uint64_t memory_before = test_mem;
        PreparedBindings *retained = prepared_instance(&ex, recipe);
        reset_observations(); prepared_resolve(&ex, retained, step % 9 == 0);
        uint64_t memory_after = test_mem, counted = test_counted;
        unsigned ordered[OPS]; memcpy(ordered, test_executed, sizeof ordered);
        VkExec reference = ex; reference.dc = fresh;
        for (unsigned i = 0; i < PREP_N; i++) if (fresh->prep[i].recipe) prepared_forget(fresh, &fresh->prep[i]);
        PreparedBindings *resolved = prepared_instance(&reference, recipe);
        test_mem = memory_before; test_fail_proof_end = fail_before;
        reset_observations(); prepared_resolve(&reference, resolved, 0);
        assert(memory_after == test_mem && counted == test_counted);
        assert(!memcmp(retained->desc, resolved->desc, 1024 * sizeof *retained->desc));
        for (unsigned i = 0; i < OPS; i++) if (!recipe->op[i].stable) assert(ordered[i] == test_executed[i]);
        for (unsigned j = 0; j < 1024; j++) for (unsigned w = 0; w < 3; w++) semantic_event(retained->desc[j].w[w]);
    }
    context_free(cached); context_free(fresh); free(recipe);
    test_snapshots = test_mutate_ordered = test_fail_proof_end = 0;
    puts("maximum recipe: 128 operations and 1024 descriptors preserve fresh-resolution bytes, order and census");
}

int main(void) {
    lifetimes_and_watchers();
    ordered_and_partial_sets();
    proofs_and_texture_sealing();
    mixed_set_groups_and_texel_scratch();
    eviction_failure_packets_and_stats();
    incremental_group_renewal();
    retained_storage_replacements();
    associative_residency();
    directory_collisions();
    retained_against_fresh_resolution();
#if NX2NAVI_PREP_PARTIAL_REUSE
    dirty_group_unchanged_members();
    partial_sampled_arrays_and_source_free_inputs();
#endif
    promotion_effects();
    maximum_recipe_resolution();
    printf("semantic event digest: %016llx\n", (unsigned long long)test_semantics);
    puts("prepared_bindings: dirty sets, ordered invalidation, proof sealing, lifetime, bounded eviction and accounting passed");
    return 0;
}

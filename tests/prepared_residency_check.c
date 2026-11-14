// Exercise production residency functions and resolver traversal against controlled resource/capture services.
#define PREP_TEST_RESET_TRACE 1
#define main prepared_legacy_main
#include "prepared_bindings_check.c"
#undef main
#undef memset

static unsigned residency_object_resets;
static void *prepared_test_memset(void *p, int value, size_t bytes) {
    if (!value && bytes == sizeof(PreparedBindings)) residency_object_resets++;
    return memset(p, value, bytes);
}

static void residency_directory(DrawCtx *cx) {
    uint16_t hot[PREP_BUCKETS] = {0};
    uint64_t cold[PREP_BUCKETS] = {0};
    size_t arrays = 0, metadata = 0;
    PreparedBindings *seen[PREP_N + PREP_DORMANT_N];
    unsigned n = 0;
    for (unsigned i = 0; i < PREP_N + PREP_DORMANT_N; i++) {
        PreparedBindings *p = i < PREP_N ? prepared_hot(cx, i) : cx->prep_dormant[i - PREP_N];
        if (!p) continue;
        for (unsigned j = 0; j < n; j++) assert(seen[j] != p);
        seen[n++] = p;
        if (i < PREP_N) {
            assert(p->index == i && p->dormant_index == PREP_NOT_HOT);
            if (p->recipe) hot[prepared_bucket(p->recipe)] |= (uint16_t)(1u << i);
        } else {
            assert(p->index == PREP_NOT_HOT && p->dormant_index == i - PREP_N);
            if (p->recipe) cold[prepared_bucket(p->recipe)] |= 1ull << (i - PREP_N);
        }
        arrays += p->capacity * (sizeof *p->slot + sizeof *p->desc);
        if (p->heap_owned) metadata += sizeof *p;
        if (p->recipe) for (unsigned j = 0; j + 1 < n; j++) assert(seen[j]->recipe != p->recipe);
    }
    assert(!memcmp(hot, cx->prep_lookup, sizeof hot));
    assert(!memcmp(cold, cx->prep_dormant_lookup, sizeof cold));
    assert(arrays == cx->prep_bytes && metadata == cx->prep_metadata_bytes);
    assert(prepared_total_bytes(cx) <= PREP_BYTES_LIMIT);
    uint32_t watchers = 0;
    for (unsigned i = 0; i < PREP_N; i++) if (prepared_hot(cx, i)->recipe) watchers |= 1u << (2 * i);
    assert(cx->prep_ubo[0][0] == watchers);
}

static void residency_free(DrawCtx *cx) {
    residency_directory(cx);
    prepared_dispose(cx);
    assert(!cx->prep_bytes && !cx->prep_metadata_bytes && !cx->prepared_last);
    for (unsigned i = 0; i < PREP_N; i++) assert(!cx->prep_hot[i]);
    for (unsigned i = 0; i < PREP_DORMANT_N; i++) assert(!cx->prep_dormant[i]);
    prepared_dispose(cx); // disposal is idempotent before or after a lazy directory initialization
    if (dc == cx) dc = &dc_main;
    free(cx);
}

static void recipes_make(VkBindRecipe **recipes, unsigned n, VkPipeInfo *info) {
    for (unsigned i = 0; i < n; i++) { recipes[i] = vk_bind_recipe_create(info, 0); assert(recipes[i]); }
}

static void make_cold(VkExec *ex, VkBindRecipe **recipes, unsigned target) {
    for (unsigned i = 1; i <= PREP_N; i++) {
        PreparedBindings *p = prepared_instance(ex, recipes[(target + i) % (PREP_N + 1)]);
        assert(p);
    }
    assert(prepared_dormant_find(ex->dc, recipes[target]));
}

static void dormant_validity_and_ownership(void) {
    enum { N = PREP_N + 1 };
    DrawCtx *cx = context_new();
    cx->res = (CbRes *)(uintptr_t)0x1000;
    VkExec ex = {.dc = cx, .cb = (VkCommandBuffer)(uintptr_t)1, .fence = 301};
    GfxShaderRes resource = {.slot = 3};
    VkPipeBinding binding = uniform_binding(0, &resource);
    VkPipeInfo info = {.n_bindings = 1, .bindings = &binding};
    VkBindRecipe *recipes[N]; recipes_make(recipes, N, &info);
    ex.ubo[0][0] = (p_bind_buf){.addr = 4096, .size = 256};
    test_snapshots = 1;
    PreparedBindings *first = prepared_instance(&ex, recipes[0]);
    reset_observations(); prepared_resolve(&ex, first, 0);
    assert(first->set[0].group[0].sealed);
    first->set[0].packet = (const DescriptorPacket *)(uintptr_t)0xdead;
    first->set[0].allocated = (VkDescriptorSet)(uintptr_t)7;
    cx->prepared_last = first;
    make_cold(&ex, recipes, 0);
    assert(first->index == PREP_NOT_HOT && !cx->prepared_last);
    unsigned allocs = test_allocs;
    PreparedBindings *p = prepared_instance(&ex, recipes[0]);
    assert(p == first && test_allocs == allocs && p->promotion_pending);
    assert(p->owner == cx->res && p->set[0].packet && p->set[0].allocated);
    assert(cx->prep_dirty[0] & (1u << (p->index * 2)));
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(!test_nsequence && test_counted == 256 && p->set[0].packet);
    // A cold recipe has no watcher. A direct state change must still be observed on promotion.
    make_cold(&ex, recipes, 0);
    ex.ubo[0][0].addr = 8192;
    p = prepared_instance(&ex, recipes[0]);
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(test_executed[0] == 1 && p->desc[0].buffer.offset == 8192);
    assert(!p->set[0].packet && !p->set[0].allocated);
    // A publication change invalidates a retained certificate even with identical direct inputs.
    make_cold(&ex, recipes, 0); test_serial++;
    p = prepared_instance(&ex, recipes[0]);
    assert(!prepared_group_valid(&ex, &p->set[0].group[0]));
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(test_executed[0] == 1);
    // A new command buffer needs fresh bound state, but eligible data ownership stays unchanged.
    p->set[0].packet = (const DescriptorPacket *)(uintptr_t)0xdead;
    p->set[0].allocated = (VkDescriptorSet)(uintptr_t)7;
    dc = cx; vk_exec_draw_new_cb(); ex.cb = (VkCommandBuffer)(uintptr_t)2;
    assert(prepared_instance(&ex, recipes[0]) == p && p->set[0].packet && p->set[0].allocated);
    vk_exec_draw_state_adopted(&ex);
    ex.ubo[0][0].addr = 12288;
    reset_observations(); prepared_resolve(&ex, p, 0);
    assert(test_executed[0] == 1 && p->desc[0].buffer.offset == 12288);
    // Owner, fence and capture changes are independent gates; old packet pointers must never be dereferenced.
    for (unsigned change = 0; change < 3; change++) {
        p->set[0].packet = (const DescriptorPacket *)(uintptr_t)0xdead;
        p->set[0].allocated = (VkDescriptorSet)(uintptr_t)7;
        make_cold(&ex, recipes, 0);
        if (change == 0) cx->res = (CbRes *)(uintptr_t)0x2000;
        else if (change == 1) { ex.fence++; cx->res = (CbRes *)(uintptr_t)0x1000; } // reused owner address
        else test_capture++;
        uint64_t stale = cx->prep_residency.stale_epoch;
        p = prepared_instance(&ex, recipes[0]);
        assert(cx->prep_residency.stale_epoch == stale + 1);
        assert(p == first && !p->slot[0].valid && !p->set[0].group[0].sealed);
        assert(!p->set[0].packet && !p->set[0].allocated && p->owner == cx->res);
        reset_observations(); prepared_resolve(&ex, p, 0); assert(test_executed[0] == 1);
    }
    residency_free(cx);
    for (unsigned i = 0; i < N; i++) free(recipes[i]);
    test_snapshots = 0;
    puts("dormant residency: pointer promotion, missed watches, proof change, new buffer, state adoption and three ownership gates passed");
}

static void dormant_bound_failure_and_recycling(void) {
    enum { N = PREP_N + PREP_DORMANT_N + 7 };
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 401};
    GfxShaderRes resource = {.slot = 3};
    VkPipeBinding binding = uniform_binding(0, &resource);
    VkPipeInfo info = {.n_bindings = 1, .bindings = &binding};
    VkBindRecipe *recipes[N]; recipes_make(recipes, N, &info);
    for (unsigned i = 0; i < PREP_N; i++) assert(prepared_instance(&ex, recipes[i]));
    test_fail_alloc = 1;
    assert(prepared_instance(&ex, recipes[PREP_N])); // optional metadata failure retains the old hot fallback
    assert(cx->prep_residency.admission_failed == 1 && !cx->prep_metadata_bytes);
    unsigned attempts = test_allocs;
    for (unsigned i = PREP_N + 1; i < N; i++) assert(prepared_instance(&ex, recipes[i]));
    assert(test_allocs == attempts);
    ex.fence++;
    uint32_t random = 0x3861ab25;
    for (unsigned k = 0; k < 6000; k++) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        assert(prepared_instance(&ex, recipes[random % N]));
        residency_directory(cx);
    }
    assert(cx->prep_metadata_bytes <= PREP_DORMANT_N * sizeof(PreparedBindings));
    assert(cx->prep_residency.hits && cx->prep_residency.drops);
    residency_free(cx);
    // New dormant metadata succeeds, then the second backing allocation fails. Previously retained entries stay
    // owned and searchable; only this new recipe falls back, with retries suppressed for the failed fence.
    cx = context_new(); ex.dc = cx; ex.fence++;
    for (unsigned i = 0; i < PREP_N; i++) assert(prepared_instance(&ex, recipes[i]));
    recipes[PREP_N]->n_desc = 4;
    test_fail_at = test_allocs + 3;
    residency_object_resets = 0;
    assert(!prepared_instance(&ex, recipes[PREP_N]) && !test_fail_at);
    assert(!residency_object_resets); // new metadata and failed backing growth never acquired an identity
    for (unsigned i = 0; i < PREP_N; i++) if (!prepared_hot(cx, i)->recipe) {
        PreparedBindings *empty = prepared_hot(cx, i);
        assert(!empty->owner && !empty->fence && !empty->capture);
        for (unsigned g = 0; g < VK_BR_GROUPS; g++) assert(!(cx->prep_dirty[g] & (3u << (2 * i))));
        for (unsigned s = 0; s < 2; s++) {
            assert(!empty->set[s].packet && !empty->set[s].allocated);
            for (unsigned g = 0; g < VK_BR_GROUPS; g++) assert(!empty->set[s].group[g].sealed);
        }
    }
    residency_directory(cx);
    assert(prepared_dormant_find(cx, recipes[0]));
    attempts = test_allocs;
    assert(!prepared_instance(&ex, recipes[PREP_N]) && test_allocs == attempts);
    assert(!residency_object_resets);
    // A dormant hit never allocates, so it remains usable even after optional backing growth failed.
    assert(prepared_instance(&ex, recipes[0]) && test_allocs == attempts);
    ex.fence++;
    assert(prepared_instance(&ex, recipes[PREP_N]));
    assert(!residency_object_resets); // the vacant object left by failure needs arrays, not another identity reset
    residency_free(cx);
    // Large descriptors saturate the total accounting budget; retained metadata is charged too.
    cx = context_new(); ex.dc = cx;
    for (unsigned i = 0; i < N; i++) recipes[i]->n_desc = VK_BR_MAX_DESC;
    for (unsigned k = 0; k < 2 * N; k++) {
        assert(prepared_instance(&ex, recipes[k % N]));
        residency_directory(cx);
    }
    assert(prepared_total_bytes(cx) > cx->prep_bytes);
    residency_free(cx);
    for (unsigned i = 0; i < N; i++) free(recipes[i]);
    puts("dormant residency: optional allocation fallback, 6000 directory transitions, byte budget and idempotent retirement passed");
}

static void dormant_exact_bucket_collisions(void) {
    enum { N = PREP_BUCKETS + PREP_N + 2 };
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 501};
    GfxShaderRes resource = {.slot = 3};
    VkPipeBinding binding = uniform_binding(0, &resource);
    VkPipeInfo info = {.n_bindings = 1, .bindings = &binding};
    VkBindRecipe *recipes[N]; recipes_make(recipes, N, &info);
    unsigned a = N, b = N;
    for (unsigned i = 0; i < N && a == N; i++) for (unsigned j = i + 1; j < N; j++)
        if (prepared_bucket(recipes[i]) == prepared_bucket(recipes[j])) { a = i; b = j; break; }
    assert(a != N);
    PreparedBindings *pa = prepared_instance(&ex, recipes[a]), *pb = prepared_instance(&ex, recipes[b]);
    unsigned added = 0;
    for (unsigned i = 0; i < N && added < PREP_N; i++) if (i != a && i != b) {
        assert(prepared_instance(&ex, recipes[i])); added++;
    }
    assert(pa->index == PREP_NOT_HOT && pb->index == PREP_NOT_HOT);
    assert(prepared_dormant_find(cx, recipes[a]) == pa && prepared_dormant_find(cx, recipes[b]) == pb);
    assert(prepared_instance(&ex, recipes[b]) == pb);
    assert(prepared_dormant_find(cx, recipes[a]) == pa);
    assert(prepared_instance(&ex, recipes[a]) == pa);
    residency_free(cx);
    for (unsigned i = 0; i < N; i++) free(recipes[i]);
    puts("dormant residency: exact recipe identity survives cold-bucket collisions and promotion");
}

static void dormant_single_reset_ownership(void) {
    enum { N = PREP_N + PREP_DORMANT_N + 2 };
    DrawCtx *cx = context_new();
    VkExec ex = {.dc = cx, .fence = 701};
    GfxShaderRes resource = {.slot = 3};
    VkPipeBinding binding = uniform_binding(0, &resource);
    VkPipeInfo info = {.n_bindings = 1, .bindings = &binding};
    VkBindRecipe *recipes[N]; recipes_make(recipes, N, &info);
    residency_object_resets = 0;
    for (unsigned i = 0; i < PREP_N; i++) assert(prepared_instance(&ex, recipes[i]));
    assert(!residency_object_resets); // initial context storage is already zeroed
    unsigned before = test_allocs;
    assert(prepared_instance(&ex, recipes[PREP_N]));
    assert(test_allocs > before && !residency_object_resets); // calloc is the new object's initialization owner
    for (unsigned i = PREP_N + 1; i < PREP_N + PREP_DORMANT_N; i++)
        assert(prepared_instance(&ex, recipes[i]));
    assert(!residency_object_resets);
    residency_directory(cx);
    uint64_t drops = cx->prep_residency.drops;
    assert(prepared_instance(&ex, recipes[N - 2]));
    assert(residency_object_resets == 1 && cx->prep_residency.drops == drops + 1);
    residency_directory(cx); // the reset unlinked only the displaced cold identity; new watches are intact
    PreparedBindings *candidate = NULL;
    for (unsigned i = 0; i < PREP_DORMANT_N && !candidate; i++)
        if (cx->prep_dormant[i] && cx->prep_dormant[i]->recipe) candidate = cx->prep_dormant[i];
    assert(candidate);
    const VkBindRecipe *recipe = candidate->recipe;
    assert(prepared_instance(&ex, recipe) == candidate && residency_object_resets == 1);
    residency_directory(cx); // promotion never destroys the retained object's contents
    residency_free(cx);
    cx = context_new(); ex.dc = cx; ex.fence++;
    for (unsigned i = 0; i < PREP_N; i++) assert(prepared_instance(&ex, recipes[i]));
    residency_object_resets = 0; test_fail_alloc = 1;
    assert(prepared_instance(&ex, recipes[PREP_N]));
    assert(residency_object_resets == 1 && cx->prep_residency.admission_failed == 1);
    residency_directory(cx); // failed optional admission leaves the caller responsible for one hot reset
    residency_free(cx);
    for (unsigned i = 0; i < N; i++) free(recipes[i]);
    puts("dormant residency: exactly one reset per cold replacement/hot fallback, none for new or promoted objects");
}

int main(void) {
    assert(NX2NAVI_PREP_DORMANT);
    if (PREP_BYTES_LIMIT < sizeof(PreparedBindings) * PREP_N) {
        DrawCtx *cx = context_new();
        VkExec ex = {.dc = cx, .fence = 601};
        VkPipeInfo info = {0};
        VkBindRecipe *recipe = vk_bind_recipe_create(&info, 0); assert(recipe);
        unsigned before = test_allocs;
        residency_object_resets = 0;
        assert(!prepared_instance(&ex, recipe) && test_allocs == before && !cx->prep_bytes);
        assert(!prepared_instance(&ex, recipe) && test_allocs == before && !residency_object_resets);
        for (unsigned i = 0; i < PREP_N; i++) assert(!prepared_hot(cx, i)->recipe);
        prepared_dispose(cx); free(cx); free(recipe);
        puts("dormant residency: budget below fixed metadata fails closed without unsigned underflow");
        return 0;
    }
    dormant_validity_and_ownership();
    dormant_bound_failure_and_recycling();
    dormant_exact_bucket_collisions();
    dormant_single_reset_ownership();
    printf("dormant residency accounting: fixed %zu bytes per context, optional object %zu bytes, %u dormant entries, total cap %u bytes\n",
           prepared_fixed_bytes(&dc_main), sizeof(PreparedBindings), PREP_DORMANT_N, PREP_BYTES_LIMIT);
    return 0;
}

#define RWT_REFUSED() RWT_COUNT(RWT_COUNT_REFUSED, 1)
// Included by vk_exec_draw.c after its authoritative resource resolvers and PB helpers. Recipes supply fixed
// decisions; prepared instances hold their final packed descriptors. Ordered resolvers are shared with mode 0.

static const VkBindRecipe *binding_recipe(VkExec *ex, const VkPipeInfo *info) {
    DrawCtx *cx = ex->dc;
    RecipeKey key = {.info = info, .stage = cx->mfb_stage};
    if (cx->recipe_last_p && !memcmp(&key, &cx->recipe_last, sizeof key)) return cx->recipe_last_p;
    int locked = vk_exec_shared();
    if (locked) plat_mutex_lock(&rs_lock);
    if (!binding_recipes.klen) nvn_hm_init(&binding_recipes, sizeof key);
    VkBindRecipe *p = nvn_hm_get(&binding_recipes, &key);
    if (!p) {
        p = vk_bind_recipe_create(info, cx->mfb_stage);
        if (p) nvn_hm_put(&binding_recipes, &key, p);
    }
    if (locked) plat_mutex_unlock(&rs_lock);
    cx->recipe_last = key, cx->recipe_last_p = p;
    return p;
}

static uint32_t *prepared_watch(DrawCtx *cx, uint32_t source) {
#define WATCH(member, table) \
    if (source >= offsetof(VkExec, member) && source < offsetof(VkExec, member) + sizeof(((VkExec *)0)->member)) { \
        size_t off = source - offsetof(VkExec, member); \
        if (off % sizeof(((VkExec *)0)->member[0][0])) return NULL; \
        size_t index = off / sizeof(((VkExec *)0)->member[0][0]); \
        size_t columns = sizeof cx->table[0] / sizeof cx->table[0][0]; \
        return &cx->table[index / columns][index % columns]; \
    }
    WATCH(ubo, prep_ubo)
    WATCH(ssbo, prep_ssbo)
    WATCH(tex, prep_tex)
    WATCH(img, prep_img)
#undef WATCH
    return NULL;
}

static void prepared_watch_recipe(DrawCtx *cx, PreparedBindings *p, int add) {
    const VkBindRecipe *r = p->recipe;
    if (!r || p->index >= PREP_N) return;
    for (uint32_t i = 0; i < r->n_ops; i++) {
        const VkBindRecipeOp *op = &r->op[i];
        if (!op->stable) continue;
        uint32_t bit = 1u << (p->index * 2 + op->set);
        for (uint32_t k = 0; k < op->count; k++) {
            uint32_t *watch = prepared_watch(cx, r->input[op->first + k].source);
            if (watch) { if (add) *watch |= bit; else *watch &= ~bit; }
        }
    }
}

// Lookup buckets do not constrain residency. Colliding recipes can coexist in any of the bounded slots; exact
// pointer comparison resolves bucket collisions without walking the rest of the cache on ordinary hits.
static uint32_t prepared_bucket(const VkBindRecipe *recipe) {
    uintptr_t key = (uintptr_t)recipe;
    return (uint32_t)((key >> 4) ^ (key >> 13) ^ (key >> 23)) & (PREP_BUCKETS - 1);
}

static PreparedBindings *prepared_hot(DrawCtx *cx, uint32_t i) {
    return cx->prep_hot[i] ? cx->prep_hot[i] : &cx->prep[i];
}

static void prepared_directory_init(DrawCtx *cx) {
    if (cx->prep_hot[0]) return;
    for (uint32_t i = 0; i < PREP_N; i++) {
        cx->prep_hot[i] = &cx->prep[i];
        cx->prep[i].index = i;
        cx->prep[i].dormant_index = PREP_NOT_HOT;
    }
}

// The residency budget includes inline certificates, directory/watch storage and optional objects as well as
// descriptor backing. It applies independently to each context; the coarse log reports the context multiplier.
static size_t prepared_fixed_bytes(const DrawCtx *cx) {
    // Charge the whole contiguous region, including padding and adjacent diagnostic/pool metadata. Allocator
    // bookkeeping and the optional replay-work history after this region are separate process overhead.
    return offsetof(DrawCtx, prep_capture) + sizeof cx->prep_capture - offsetof(DrawCtx, prep_incarnation);
}

static size_t prepared_total_bytes(const DrawCtx *cx) {
    return prepared_fixed_bytes(cx) + cx->prep_metadata_bytes + cx->prep_bytes;
}

// Forgetting discards ownership associations, not their CbRes-owned packets. The arrays can serve a later recipe.
static void prepared_forget(DrawCtx *cx, PreparedBindings *p) {
    if (p->recipe) {
        if (p->index < PREP_N) {
            RWT_HISTORY_EVICT(cx, p->recipe);
            cx->prep_lookup[prepared_bucket(p->recipe)] &= (uint16_t)~(1u << p->index);
            cx->prep_stats.cache_evictions++;
        } else if (p->dormant_index < PREP_DORMANT_N) {
            cx->prep_dormant_lookup[prepared_bucket(p->recipe)] &= ~(1ull << p->dormant_index);
            cx->prep_residency.drops++;
        }
    }
    prepared_watch_recipe(cx, p, 0);
    PreparedSlot *slot = p->slot;
    VkDescSlot *desc = p->desc;
    uint32_t capacity = p->capacity;
    if (cx->prepared_last == p) cx->prepared_last = NULL;
    uint32_t index = p->index, dormant_index = p->dormant_index;
    uint8_t heap_owned = p->heap_owned;
    memset(p, 0, sizeof *p);
    p->index = index, p->dormant_index = dormant_index, p->heap_owned = heap_owned;
    p->slot = slot, p->desc = desc, p->capacity = capacity;
    if (index < PREP_N)
        for (unsigned g = 0; g < VK_BR_GROUPS; g++) cx->prep_dirty[g] &= ~(3u << (index * 2));
}

static void prepared_evict(DrawCtx *cx, PreparedBindings *p) {
    // Array growth also arrives here after identity has already been discarded, or on fresh zeroed storage.
    if (p->recipe) prepared_forget(cx, p);
    cx->prep_bytes -= p->capacity * (sizeof *p->slot + sizeof *p->desc);
    free(p->slot), free(p->desc);
    p->slot = NULL, p->desc = NULL, p->capacity = 0;
}

static int prepared_make_room(DrawCtx *cx, PreparedBindings *keep, size_t bytes) {
    while (prepared_total_bytes(cx) + bytes > PREP_BYTES_LIMIT) {
        PreparedBindings *oldest = NULL;
        // Discard dormant arrays first. Their metadata can still serve the next displaced entry.
        for (uint32_t i = 0; i < PREP_DORMANT_N; i++) {
            PreparedBindings *p = cx->prep_dormant[i];
            if (p && p != keep && p->capacity && (!oldest || p->touched < oldest->touched)) oldest = p;
        }
        if (!oldest) for (uint32_t i = 0; i < PREP_N; i++) {
            PreparedBindings *p = prepared_hot(cx, i);
            if (p != keep && p->capacity && (!oldest || p->touched < oldest->touched)) oldest = p;
        }
        if (!oldest) return 0;
        prepared_evict(cx, oldest);
    }
    return 1;
}

static int prepared_same_epoch(const VkExec *ex, const PreparedBindings *p, uint64_t capture) {
    return p->owner == ex->dc->res && p->fence == ex->fence && p->capture == capture;
}

// Remove hot watches before the slot changes identity. A dormant entry receives no notifications; promotion
// therefore always marks every group dirty so the ordinary resolver compares its current inputs exactly.
static void prepared_demote(DrawCtx *cx, PreparedBindings *p, uint32_t cold) {
    if (p->recipe) {
        RWT_HISTORY_EVICT(cx, p->recipe);
        cx->prep_lookup[prepared_bucket(p->recipe)] &= (uint16_t)~(1u << p->index);
        prepared_watch_recipe(cx, p, 0);
        cx->prep_stats.cache_evictions++;
        cx->prep_residency.demotions++;
        cx->prep_dormant_lookup[prepared_bucket(p->recipe)] |= 1ull << cold;
    }
    if (cx->prepared_last == p) cx->prepared_last = NULL;
    for (unsigned g = 0; g < VK_BR_GROUPS; g++) cx->prep_dirty[g] &= ~(3u << (p->index * 2));
    p->index = PREP_NOT_HOT, p->dormant_index = cold, p->promotion_pending = 0;
    cx->prep_dormant[cold] = p;
}

static PreparedBindings *prepared_dormant_find(DrawCtx *cx, const VkBindRecipe *recipe) {
    if (!NX2NAVI_PREP_DORMANT) return NULL;
    cx->prep_residency.lookups++;
    uint64_t mask = cx->prep_dormant_lookup[prepared_bucket(recipe)];
    for (; mask; mask &= mask - 1) {
        uint32_t i = (uint32_t)__builtin_ctzll(mask);
        PreparedBindings *p = cx->prep_dormant[i];
        if (p && p->recipe == recipe) { cx->prep_residency.hits++; return p; }
    }
    return NULL;
}

// Return an object for a genuinely new hot recipe, retaining the displaced entry when bounded storage permits.
// A failure only disables optional retention for this fence; the original hot replacement path remains usable.
static PreparedBindings *prepared_dormant_admit(VkExec *ex, PreparedBindings *victim) {
    DrawCtx *cx = ex->dc;
    if (!NX2NAVI_PREP_DORMANT || !victim->recipe ||
        (cx->prep_dormant_failed && cx->prep_dormant_failed_fence == ex->fence)) return victim;
    uint32_t cold = 0;
    for (uint32_t i = 0; i < PREP_DORMANT_N; i++) {
        if (!cx->prep_dormant[i]) { cold = i; break; }
        if (!cx->prep_dormant[i]->recipe) { cold = i; break; }
        if (cx->prep_dormant[i]->touched < cx->prep_dormant[cold]->touched) cold = i;
    }
    PreparedBindings *replacement = cx->prep_dormant[cold];
    if (!replacement) {
        if (prepared_total_bytes(cx) + sizeof *replacement <= PREP_BYTES_LIMIT)
            replacement = calloc(1, sizeof *replacement);
        if (!replacement) {
            cx->prep_residency.admission_failed++;
            cx->prep_dormant_failed = 1, cx->prep_dormant_failed_fence = ex->fence;
            return victim;
        }
        replacement->heap_owned = 1;
        replacement->index = PREP_NOT_HOT, replacement->dormant_index = cold;
        cx->prep_metadata_bytes += sizeof *replacement;
    } else if (replacement->recipe) prepared_forget(cx, replacement);
    uint32_t hot = victim->index;
    prepared_demote(cx, victim, cold);
    replacement->index = hot, replacement->dormant_index = PREP_NOT_HOT;
    cx->prep_hot[hot] = replacement;
    return replacement;
}

static PreparedBindings *prepared_promote(VkExec *ex, PreparedBindings *cold, PreparedBindings *victim) {
    DrawCtx *cx = ex->dc;
    uint32_t hot = victim->index, at = cold->dormant_index;
    cx->prep_dormant_lookup[prepared_bucket(cold->recipe)] &= ~(1ull << at);
    prepared_demote(cx, victim, at);
    cold->index = hot, cold->dormant_index = PREP_NOT_HOT, cold->promotion_pending = 1;
    cx->prep_hot[hot] = cold;
    cx->prep_lookup[prepared_bucket(cold->recipe)] |= (uint16_t)(1u << hot);
    prepared_watch_recipe(cx, cold, 1);
    for (unsigned g = 0; g < VK_BR_GROUPS; g++) cx->prep_dirty[g] |= 3u << (hot * 2);
    cx->prep_residency.promotions++;
    return cold;
}

// Called only after all recording has stopped, under the existing draw-shutdown contract. Worker contexts remain
// allocated because their worker objects still name them; their optional residency storage is released here too.
static void prepared_dispose(DrawCtx *cx) {
    prepared_directory_init(cx);
    for (uint32_t i = 0; i < PREP_N + PREP_DORMANT_N; i++) {
        PreparedBindings *p = i < PREP_N ? cx->prep_hot[i] : cx->prep_dormant[i - PREP_N];
        if (!p) continue;
        free(p->slot), free(p->desc);
        if (p->heap_owned) free(p);
    }
    memset(cx->prep, 0, sizeof cx->prep);
    memset(cx->prep_hot, 0, sizeof cx->prep_hot);
    memset(cx->prep_dormant, 0, sizeof cx->prep_dormant);
    memset(cx->prep_lookup, 0, sizeof cx->prep_lookup);
    memset(cx->prep_dormant_lookup, 0, sizeof cx->prep_dormant_lookup);
    memset(cx->prep_ubo, 0, sizeof cx->prep_ubo);
    memset(cx->prep_ssbo, 0, sizeof cx->prep_ssbo);
    memset(cx->prep_tex, 0, sizeof cx->prep_tex);
    memset(cx->prep_img, 0, sizeof cx->prep_img);
    memset(cx->prep_dirty, 0, sizeof cx->prep_dirty);
    cx->prep_bytes = cx->prep_metadata_bytes = cx->prep_alloc_failed = cx->prep_dormant_failed = 0;
    cx->prepared_last = NULL;
    cx->recipe_last_p = NULL;
    memset(&cx->recipe_last, 0, sizeof cx->recipe_last);
}

static PreparedBindings *prepared_instance(VkExec *ex, const VkBindRecipe *recipe) {
    DrawCtx *cx = ex->dc;
    prepared_directory_init(cx);
    uint32_t bucket = prepared_bucket(recipe);
    PreparedBindings *b = NULL;
    for (uint32_t members = cx->prep_lookup[bucket]; members; members &= members - 1) {
        uint32_t i = (uint32_t)__builtin_ctz(members);
        cx->prep_stats.cache_probes++;
        PreparedBindings *p = cx->prep_hot[i];
        if (p->recipe == recipe) { b = p; break; }
    }
    RWT_HISTORY_LOOKUP(cx, recipe, b != NULL);
    if (!b) {
        PC_COUNT(cx, PC_CACHE_MISS);
        cx->prep_stats.cache_misses++;
        PreparedBindings *victim = cx->prep_hot[0];
        // Victim selection is a miss cost, not a per-draw scan. Prefer vacant slots, then global least recent use.
        for (uint32_t i = 0; i < PREP_N; i++) {
            PreparedBindings *p = cx->prep_hot[i];
            if (!p->recipe) { victim = p; break; }
            if (p->touched < victim->touched) victim = p;
        }
        b = prepared_dormant_find(cx, recipe);
        if (b) {
            if (prepared_same_epoch(ex, b, vk_usnap_capture())) cx->prep_residency.same_epoch++;
            else cx->prep_residency.stale_epoch++;
            b = prepared_promote(ex, b, victim);
            goto resident;
        }
        if (cx->prep_alloc_failed && cx->prep_alloc_failed_fence == ex->fence) return (RWT_REFUSED(), NULL);
        b = prepared_dormant_admit(ex, victim);
        RWT_COUNT(RWT_COUNT_REBUILD_DESCRIPTORS, recipe->n_desc);
        // A retained cold replacement was unlinked and reset by prepared_dormant_admit; a newly allocated or
        // vacant object is already empty. Only the unchanged hot fallback still owns a recipe to discard here.
        if (b->recipe) prepared_forget(cx, b);
        size_t bytes = recipe->n_desc * (sizeof *b->slot + sizeof *b->desc);
        size_t fixed = prepared_fixed_bytes(cx);
        if (fixed > PREP_BYTES_LIMIT || bytes > PREP_BYTES_LIMIT - fixed) return (RWT_REFUSED(), NULL);
        if (b->capacity >= recipe->n_desc) {
            RWT_COUNT(RWT_COUNT_REUSED_STORAGE, 1);
            // Only the new recipe's active range is accessed. Clear it even when the previous recipe was smaller:
            // unused retained elements may still contain an older recipe's descriptors and resource proofs.
            if (recipe->n_desc) {
                memset(b->slot, 0, recipe->n_desc * sizeof *b->slot);
                memset(b->desc, 0, recipe->n_desc * sizeof *b->desc);
            }
        } else {
            RWT_COUNT(RWT_COUNT_ALLOCATE_STORAGE, 1);
            prepared_evict(cx, b);
            if (!prepared_make_room(cx, b, bytes)) return (RWT_REFUSED(), NULL);
            b->capacity = recipe->n_desc;
            if (b->capacity) {
                b->slot = calloc(b->capacity, sizeof *b->slot);
                b->desc = calloc(b->capacity, sizeof *b->desc);
                if (!b->slot || !b->desc) {
                    free(b->slot), free(b->desc);
                    b->slot = NULL, b->desc = NULL, b->capacity = 0;
                    cx->prep_alloc_failed = 1, cx->prep_alloc_failed_fence = ex->fence;
                    return (RWT_REFUSED(), NULL);   // optional admission failed; the ordered renderer remains available
                }
            }
            cx->prep_bytes += (uint32_t)bytes;
        }
        b->recipe = recipe;
        cx->prep_lookup[bucket] |= (uint16_t)(1u << b->index);
        prepared_watch_recipe(cx, b, 1);
        for (unsigned g = 0; g < VK_BR_GROUPS; g++) cx->prep_dirty[g] |= 3u << (b->index * 2);
    } else { cx->prep_stats.cache_hits++; PC_COUNT(cx, PC_CACHE_HIT); }
resident:;
    b->touched = ++cx->prep_clock;
    uint64_t capture = vk_usnap_capture();
    if (!prepared_same_epoch(ex, b, capture)) {
        if (b->fence != ex->fence) { PC_COUNT(cx, PC_FENCE_RESET); RWT_COUNT(RWT_COUNT_FENCE_RESET, 1); }
        if (b->capture != capture) { PC_COUNT(cx, PC_CAPTURE_RESET); RWT_COUNT(RWT_COUNT_CAPTURE_RESET, 1); }
        PC_ADD(cx, PC_RESET_DESCRIPTORS, b->recipe->n_desc);
        RWT_COUNT(RWT_COUNT_RESET_DESCRIPTORS, b->recipe->n_desc);
        for (uint32_t i = 0; i < b->recipe->n_desc; i++) b->slot[i].valid = b->slot[i].texture.valid = 0;
        for (uint32_t s = 0; s < 2; s++) {
            b->set[s].packet = NULL, b->set[s].allocated = VK_NULL_HANDLE;
            for (unsigned g = 0; g < VK_BR_GROUPS; g++) b->set[s].group[g].sealed = 0;
        }
        b->owner = cx->res, b->fence = ex->fence, b->capture = capture;
        for (unsigned g = 0; g < VK_BR_GROUPS; g++) cx->prep_dirty[g] |= 3u << (b->index * 2);
    }
    // These record the current consumer, not the resource lifetime. A new command buffer must emit bindings again,
    // but the same submission's resolved resources remain eligible for their ordinary lease/token checks.
    b->cb = ex->cb, b->incarnation = cx->prep_incarnation;
    return b;
}

static const p_bind_buf *prepared_buffer(const VkExec *ex, const VkBindRecipeInput *in) {
    return in->source == VK_BR_NO_SOURCE ? NULL : (const p_bind_buf *)((const uint8_t *)ex + in->source);
}

static uint64_t prepared_handle(const VkExec *ex, const VkBindRecipeInput *in) {
    return in->source == VK_BR_NO_SOURCE ? 0 : *(const uint64_t *)((const uint8_t *)ex + in->source);
}

static void prepared_count(DrawCtx *cx, const PreparedSlot *s) {
    if (s->count.has) count_use(cx, s->count.cls, s->count.use, s->count.n);
    vk_exec_frame.bind_reused++;
    cx->prep_stats.reused++;
    if (s->lease.capture) cx->prep_stats.snapshot_reused++;
}

// The input address and size are direct reads at compiled offsets, not a reconstructed four-word PB key. A lease
// proves the exact snapshot range, so a write dropping some other region does not discard this descriptor.
static int prepared_buffer_valid(VkExec *ex, const VkBindRecipeInput *in, const PreparedSlot *s) {
    if (!s->valid || in->reuse == VK_BR_ALWAYS || ex->dc->mo_ovr_on) return 0;
    const p_bind_buf *b = prepared_buffer(ex, in);
    if (b && (s->input[0] != b->addr || s->input[1] != b->size)) return 0;
    if (s->lease.capture) return vk_usnap_lease_valid(&s->lease);
    if (s->mem_gen != vk_mem_gen() || s->snap_gen != vk_usnap_gen()) return 0;
    if (in->reuse == VK_BR_CONSTANT && (s->prog != ex->prog || s->prog_serial != (ex->prog ? ex->prog->serial : 0)))
        return 0;
    return 1;
}

static void prepared_buffer_begin(VkExec *ex, const VkBindRecipeInput *in, PreparedSlot *s) {
    const p_bind_buf *b = prepared_buffer(ex, in);
    s->input[0] = b ? b->addr : 0, s->input[1] = b ? b->size : 0;
    s->mem_gen = vk_mem_gen(), s->snap_gen = vk_usnap_gen();
    s->prog = ex->prog, s->prog_serial = ex->prog ? ex->prog->serial : 0;
    memset(&s->lease, 0, sizeof s->lease);
    s->count = (PbCount){0};
    ex->dc->pb_cap = &s->count, ex->dc->pb_spoiled = 0;
}

static void prepared_buffer_end(VkExec *ex, const VkBindRecipeInput *in, PreparedSlot *s) {
    ex->dc->pb_cap = NULL, ex->dc->prep_capture = NULL;
    s->valid = in->reuse != VK_BR_ALWAYS && !ex->dc->pb_spoiled && !ex->dc->mo_ovr_on;
    vk_exec_frame.bind_resolved++;
    ex->dc->prep_stats.resolved++;
}

static void prepared_check(VkExec *ex, const VkBindRecipeOp *op, const PreparedBindings *p, uint32_t j,
                           int hit, const PreparedSlot *prior, const VkDescSlot *before, const VkDescSlot *after) {
    if (hit) ex->dc->prep_stats.checked++;
    if (hit && memcmp(before, after, sizeof *before)) {
        const PreparedSlot *now = &p->slot[j];
        const VkBindRecipeInput *in = &p->recipe->input[j];
        rt_log("prepared mismatch: descriptor %u kind %u stage %d slot %d reuse %u set %u operation %u; "
               "old words %#llx %#llx %#llx; new words %#llx %#llx %#llx", j, op->kind, op->stage,
               in->slot, in->reuse, op->set, (unsigned)(op - p->recipe->op),
               (unsigned long long)before->w[0], (unsigned long long)before->w[1], (unsigned long long)before->w[2],
               (unsigned long long)after->w[0], (unsigned long long)after->w[1], (unsigned long long)after->w[2]);
        rt_log("prepared mismatch: fence %llu packet fence %llu capture %llu current %llu; "
               "input old %#llx/%llu new %#llx/%llu; memory %llu/%llu/current %llu; snapshots %llu/%llu/current %llu; "
               "valid %u/%u; program %llu/%llu/current %llu", (unsigned long long)ex->fence,
               (unsigned long long)p->fence, (unsigned long long)p->capture, (unsigned long long)vk_usnap_capture(),
               (unsigned long long)prior->input[0], (unsigned long long)prior->input[1],
               (unsigned long long)now->input[0], (unsigned long long)now->input[1],
               (unsigned long long)prior->mem_gen, (unsigned long long)now->mem_gen, (unsigned long long)vk_mem_gen(),
               (unsigned long long)prior->snap_gen, (unsigned long long)now->snap_gen, (unsigned long long)vk_usnap_gen(),
               prior->valid, now->valid, (unsigned long long)prior->prog_serial,
               (unsigned long long)now->prog_serial, (unsigned long long)(ex->prog ? ex->prog->serial : 0));
        rt_log("prepared mismatch: lease capture %llu/%llu valid now %d/%d; texture handle %#llx/%#llx "
               "token %#llx/%#llx/current %#llx; generation %u/%u/current %u; epoch %u/%u/current %u; "
               "samplers %u/%u/current %u; image %p/%p; deferred %d",
               (unsigned long long)prior->lease.capture, (unsigned long long)now->lease.capture,
               vk_usnap_lease_valid(&prior->lease), vk_usnap_lease_valid(&now->lease),
               (unsigned long long)prior->texture.h, (unsigned long long)now->texture.h,
               (unsigned long long)prior->texture.tok, (unsigned long long)now->texture.tok,
               (unsigned long long)token_of(prior->texture.h), prior->texture.gen, now->texture.gen, vk_texture_slots_gen(),
               prior->texture.ep, now->texture.ep, vk_texture_epoch(), prior->texture.sg, now->texture.sg, vk_sampler_gen(),
               (void *)prior->texture.image, (void *)now->texture.image, vk_texture_deferred());
        rt_log("prepared mismatch: lease ranges old %#llx/%llu align %llu region %u; new %#llx/%llu align %llu region %u",
               (unsigned long long)prior->lease.addr, (unsigned long long)prior->lease.size,
               (unsigned long long)prior->lease.align, prior->lease.region,
               (unsigned long long)now->lease.addr, (unsigned long long)now->lease.size,
               (unsigned long long)now->lease.align, now->lease.region);
        rt_fatal("vk_exec: prepared descriptor %u differs from ordered resolution", j);
    }
}

typedef void (*PreparedOp)(VkExec *, const VkBindRecipeOp *, PreparedBindings *, int);

// The large reference copy exists only in checking mode. Keeping it out of the hot caller also avoids imposing
// its stack commitment/probing on ordinary resolution on platforms with small stack guard pages.
static __attribute__((noinline)) void prepared_op_check(VkExec *ex, const VkBindRecipeOp *op,
                                                       PreparedBindings *p, PreparedOp execute) {
    VkDescSlot expected[VK_BR_MAX_DESC];
    memcpy(expected, p->desc + op->first, op->count * sizeof *expected);
    execute(ex, op, p, 1);
    if (vk_texture_deferred()) {
        // An incomplete lookup will be retried in order. No result from this attempt can extend a reusable proof.
        for (uint32_t s = 0; s < p->recipe->n_sets; s++)
            for (unsigned g = 0; g < VK_BR_GROUPS; g++) p->set[s].group[g].sealed = 0;
        return;
    }
    if (memcmp(expected, p->desc + op->first, op->count * sizeof *expected))
        rt_fatal("vk_exec: aggregate prepared proof differs at operation %u", (unsigned)(op - p->recipe->op));
}

// Mark only a set whose actual descriptor bytes changed. Resource validation and ordered side effects still run
// even when the resulting bytes are equal; equality permits packet reuse, never skipping those side effects.
static void prepared_changed(PreparedBindings *p, const VkBindRecipeOp *op, uint32_t j, const VkDescSlot *before) {
    if (memcmp(before, &p->desc[j], sizeof *before)) p->changed_sets |= 1u << op->set;
}

static void prepared_refresh_record(PreparedBindings *p, const VkBindRecipeOp *op, uint32_t j) {
    unsigned group = vk_bind_recipe_group(op);
    if (!op->stable || !(p->renewable & (1u << (op->set * VK_BR_GROUPS + group)))) return;
    if (NX2NAVI_PREP_PARTIAL_REUSE) p->refreshed_groups |= 1u << (op->set * VK_BR_GROUPS + group);
    uint64_t bit = 1ull << (j % 64);
    if (p->refreshed[j / 64] & bit) return;
    p->refreshed[j / 64] |= bit;
    p->slot[j].previous_count = p->slot[j].count;
    p->slot[j].previous_snapshot = p->slot[j].lease.capture != 0;
}

static void prepared_support(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) {
    (void)check;
    VkDescSlot before = p->desc[op->first];
    DrawCtx *cx = ex->dc;
    float in[3];
    support_in(ex, in);
    if (!cx->B.sup_valid || memcmp(in, cx->B.sup_in, sizeof in)) {
        uint8_t tmp[SUPPORT_BYTES];
        support_fill(tmp, in[0], in[1], in[2]);
        memcpy(cx->B.sup_in, in, sizeof in);
        if (!cx->B.sup_valid || memcmp(tmp, cx->B.support, SUPPORT_BYTES)) {
            VkBuffer sb;
            VkDeviceSize so;
            memcpy(scratch(ex, SUPPORT_BYTES, &sb, &so), tmp, SUPPORT_BYTES);
            memcpy(cx->B.support, tmp, SUPPORT_BYTES);
            cx->B.sup_buf = sb, cx->B.sup_off = so, cx->B.sup_valid = 1;
        }
    }
    p->desc[op->first].buffer = (VkDescriptorBufferInfo){cx->B.sup_buf, cx->B.sup_off, SUPPORT_BYTES};
    prepared_changed(p, op, op->first, &before);
}

static void prepared_uniforms(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) {
    uint64_t start = bt_now();
    for (uint32_t k = 0; k < op->count; k++) {
        uint32_t j = op->first + k;
        const VkBindRecipeInput *in = &p->recipe->input[j];
        PreparedSlot *s = &p->slot[j];
        VkDescSlot *out = &p->desc[j];
        int hit = prepared_buffer_valid(ex, in, s);
        if (hit && !check) { prepared_count(ex->dc, s); continue; }
        VkDescSlot before = *out;
        PreparedSlot prior;
        if (check && hit) prior = *s;
        prepared_refresh_record(p, op, j);
        prepared_buffer_begin(ex, in, s);
        if (in->reuse == VK_BR_UNIFORM && !ex->dc->mo_ovr_on) ex->dc->prep_capture = &s->lease;
        out->buffer = ubo_info(ex, op->stage, in->slot);
        prepared_buffer_end(ex, in, s);
        prepared_check(ex, op, p, j, check && hit && s->valid, &prior, &before, out);
        prepared_changed(p, op, j, &before);
    }
    vk_exec_frame.bind_buf_ns += bt_now() - start;
}

static void prepared_storage(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) {
    uint64_t start = bt_now();
    for (uint32_t k = 0; k < op->count; k++) {
        uint32_t j = op->first + k;
        const VkBindRecipeInput *in = &p->recipe->input[j];
        PreparedSlot *s = &p->slot[j];
        VkDescSlot *out = &p->desc[j];
        int hit = prepared_buffer_valid(ex, in, s);
        if (hit && !check) { prepared_count(ex->dc, s); continue; }
        VkDescSlot before = *out;
        PreparedSlot prior;
        if (check && hit) prior = *s;
        prepared_refresh_record(p, op, j);
        prepared_buffer_begin(ex, in, s);
        out->buffer = ssbo_info(ex, op->stage, op->binding->res, op->binding->res->flags & 1);
        prepared_buffer_end(ex, in, s);
        prepared_check(ex, op, p, j, check && hit && s->valid, &prior, &before, out);
        prepared_changed(p, op, j, &before);
        if (in->reuse == VK_BR_ALWAYS) ex->dc->prep_stats.ordered++;
    }
    vk_exec_frame.bind_buf_ns += bt_now() - start;
}

static void prepared_images(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) {
    const GfxShaderRes *r = op->binding->res;
    int storage = op->kind == VK_BR_IMAGE, locked = 0;
    // These are the same dependency boundaries as TexMemo. Exact-level promotion, DLSS substitution and depth twins
    // retain their ordered resolver; storage images must mark their writes even when their descriptor is identical.
    int reusable = !storage && !r->exact && td_mode() == 1 && !vk_dlaa_swap_img && !twin_dispatch && !ex->dc->mo_ovr_on;
    uint64_t start = bt_now();
    for (uint32_t k = 0; k < op->count; k++) {
        uint32_t j = op->first + k;
        uint64_t h = prepared_handle(ex, &p->recipe->input[j]);
        PreparedSlot *s = &p->slot[j];
        VkDescSlot *out = &p->desc[j];
        TexMemo *m = &s->texture;
        // A preceding member can create, take back or retire an image during tex_info. Its table changes must be
        // visible to the next member even when this call already holds the recursive texture lock.
        uint32_t gen = reusable ? vk_texture_slots_gen() : 0, epoch = reusable ? vk_texture_epoch() : 0;
        uint32_t samplers = reusable ? vk_sampler_gen() : 0;
        int hit = reusable && m->valid && m->h == h && m->ep == epoch && m->sg == samplers &&
                  slot_still(&m->gen, m->tok, h, gen);
        if (check && hit && !locked) {
            // The oracle resolves under the lock. A table change while waiting makes the earlier hit ineligible
            // for comparison; it must be tested again against the same table state the resolver will observe.
            vk_texture_lock(), locked = 1;
            gen = vk_texture_slots_gen(), epoch = vk_texture_epoch(), samplers = vk_sampler_gen();
            hit = m->valid && m->h == h && m->ep == epoch && m->sg == samplers &&
                  slot_still(&m->gen, m->tok, h, gen);
        }
        VkDescSlot before = *out, expected = *out;
        PreparedSlot prior;
        if (check && hit) prior = *s;
        if (hit) {
            expected.image = m->out;
            expected.image.imageLayout = m->image && ex->nfbl && in_fbl(ex, m->image)
                ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT : VK_IMAGE_LAYOUT_GENERAL;
            expected.w[2] = (uint32_t)expected.image.imageLayout;
            if (!check) {
                // The packed descriptor already lives where the replay consumes it. Only its layout can change.
                out->w[2] = expected.w[2];
                prepared_changed(p, op, j, &before);
                vk_exec_frame.bind_reused++, ex->dc->prep_stats.reused++;
                continue;
            }
        }
        prepared_refresh_record(p, op, j);
        if (!locked) {
            vk_texture_lock(), locked = 1;
            gen = reusable ? vk_texture_slots_gen() : 0, epoch = reusable ? vk_texture_epoch() : 0;
            samplers = reusable ? vk_sampler_gen() : 0;
        }
        VkImage image = VK_NULL_HANDLE;
        uint64_t token = reusable ? token_of(h) : 0;
        tex_info(ex, r, h, storage, r->separate, &out->image, &image);
        out->w[2] = (uint32_t)out->image.imageLayout;
        int incomplete = vk_texture_deferred();
        *m = (TexMemo){.h = h, .tok = token, .gen = gen, .ep = epoch, .sg = samplers,
                       .valid = (uint8_t)(reusable && token && !incomplete), .image = image, .out = out->image};
        // The worker retries a deferred draw in stream order. Its temporary result is neither an oracle nor a
        // reusable binding; retaining it could let the retry skip the resolution it originally deferred.
        prepared_check(ex, op, p, j, check && hit && !incomplete, &prior, &expected, out);
        prepared_changed(p, op, j, &before);
        vk_exec_frame.bind_resolved++, ex->dc->prep_stats.resolved++;
        if (storage) ex->dc->prep_stats.ordered++;
    }
    if (locked) vk_texture_unlock();
    vk_exec_frame.bind_tex_ns += bt_now() - start;
}

static void prepared_texels(VkExec *ex, const VkBindRecipeOp *op, PreparedBindings *p, int check) {
    (void)check;
    for (uint32_t k = 0; k < op->count; k++) {
        uint32_t j = op->first + k;
        VkDescSlot before = p->desc[j];
        p->desc[j] = (VkDescSlot){0};
        p->desc[j].view = Z.zview;
        prepared_changed(p, op, j, &before);
        if (p->recipe->need_w) ex->dc->bind_bv[j] = Z.zview;
    }
}

enum { PREP_LIVE = 1, PREP_TEXTURE = 2, PREP_PROGRAM = 4, PREP_SNAPSHOT = 8 };

static PreparedStamp prepared_stamp(const VkExec *ex, unsigned group) {
    PreparedStamp s = {0};
    s.mem = vk_mem_gen();
    if (group == VK_BR_GROUP_BUFFER) {
        s.snapshot = vk_usnap_gen();
        s.program = ex->prog, s.program_serial = ex->prog ? ex->prog->serial : 0;
    } else {
        s.texture_pool = ex->tex_pool, s.sampler_pool = ex->samp_pool;
        s.textures = vk_texture_slots_gen(), s.texture_epoch = vk_texture_epoch(), s.samplers = vk_sampler_gen();
        s.nfbl = ex->nfbl;
        if (s.nfbl > 0 && s.nfbl <= 9) memcpy(s.feedback, ex->fbl_img, (size_t)s.nfbl * sizeof *s.feedback);
    }
    return s;
}

static int prepared_stamp_equal(const PreparedStamp *a, const PreparedStamp *b, uint32_t flags) {
    if ((flags & PREP_LIVE) && (a->mem != b->mem || a->snapshot != b->snapshot)) return 0;
    if ((flags & PREP_PROGRAM) && (a->program != b->program || a->program_serial != b->program_serial)) return 0;
    if (flags & PREP_TEXTURE) {
        if (a->mem != b->mem || a->textures != b->textures || a->texture_epoch != b->texture_epoch ||
            a->samplers != b->samplers || a->texture_pool != b->texture_pool || a->sampler_pool != b->sampler_pool ||
            a->nfbl != b->nfbl || a->nfbl < 0 || a->nfbl > 9) return 0;
        if (a->nfbl && memcmp(a->feedback, b->feedback, (size_t)a->nfbl * sizeof *a->feedback)) return 0;
    }
    return 1;
}

static int prepared_texture_reusable(const VkExec *ex) {
    return td_mode() == 1 && !vk_dlaa_swap_img && !twin_dispatch && !ex->dc->mo_ovr_on;
}

// Sealing may follow an ordered op that changed state after an earlier contribution was skipped. Revalidate each
// retained memo against current inputs/tokens, not merely its valid bit. This walk occurs only when building a proof.
static int prepared_texture_valid(VkExec *ex, const VkBindRecipeInput *in, PreparedSlot *slot,
                                  const VkDescSlot *descriptor) {
    TexMemo *m = &slot->texture;
    uint64_t handle = prepared_handle(ex, in);
    if (!prepared_texture_reusable(ex) || !m->valid || m->h != handle || m->ep != vk_texture_epoch() ||
        m->sg != vk_sampler_gen() || !slot_still(&m->gen, m->tok, handle, vk_texture_slots_gen())) return 0;
    VkImageLayout layout = m->image && ex->nfbl && in_fbl(ex, m->image)
        ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT : VK_IMAGE_LAYOUT_GENERAL;
    return descriptor->image.sampler == m->out.sampler && descriptor->image.imageView == m->out.imageView &&
           descriptor->image.imageLayout == layout && descriptor->w[2] == (uint32_t)layout;
}

static int prepared_group_valid(VkExec *ex, PreparedGroup *s) {
    uint64_t prep_guard_start = PC_NOW(ex->dc);
    if (!s->sealed) PC_GUARD_RETURN(ex, PC_GUARD_UNSEALED, 0);
    if (ex->dc->mo_ovr_on) PC_GUARD_RETURN(ex, PC_GUARD_OVERRIDE, 0);
    if ((s->flags & PREP_TEXTURE) && !prepared_texture_reusable(ex)) PC_GUARD_RETURN(ex, PC_GUARD_TEXTURE_MODE, 0);
    const PreparedStamp *saved = &s->stamp;
    if ((s->flags & (PREP_LIVE | PREP_TEXTURE)) && saved->mem != vk_mem_gen()) PC_GUARD_RETURN(ex, PC_GUARD_MEMORY, 0);
    if ((s->flags & PREP_LIVE) && saved->snapshot != vk_usnap_gen()) PC_GUARD_RETURN(ex, PC_GUARD_SNAPSHOT, 0);
    if ((s->flags & PREP_PROGRAM) &&
        (saved->program != ex->prog || saved->program_serial != (ex->prog ? ex->prog->serial : 0))) PC_GUARD_RETURN(ex, PC_GUARD_PROGRAM, 0);
    if (s->flags & PREP_TEXTURE) {
        if (saved->textures != vk_texture_slots_gen()) PC_GUARD_RETURN(ex, PC_GUARD_TEXTURES, 0);
        if (saved->texture_epoch != vk_texture_epoch()) PC_GUARD_RETURN(ex, PC_GUARD_EPOCH, 0);
        if (saved->samplers != vk_sampler_gen()) PC_GUARD_RETURN(ex, PC_GUARD_SAMPLERS, 0);
        if (saved->texture_pool != ex->tex_pool || saved->sampler_pool != ex->samp_pool) PC_GUARD_RETURN(ex, PC_GUARD_POOLS, 0);
        if (saved->nfbl != ex->nfbl || ex->nfbl < 0 || ex->nfbl > 9) PC_GUARD_RETURN(ex, PC_GUARD_FEEDBACK, 0);
        if (ex->nfbl && memcmp(saved->feedback, ex->fbl_img, (size_t)ex->nfbl * sizeof *ex->fbl_img)) PC_GUARD_RETURN(ex, PC_GUARD_FEEDBACK, 0);
    }
    if ((s->flags & PREP_SNAPSHOT) && !vk_usnap_proof_valid(&s->proof)) PC_GUARD_RETURN(ex, PC_GUARD_PROOF, 0);
    PC_GUARD_RETURN(ex, PC_GUARD_OK, 1);
}

// The group's live proof already validates retained resources. A binding notification can concern another member
// of that group; compare only this op's direct inputs before reusing it. The caller must recheck the proof after
// every ordered resolver. This helper never substitutes for a lifetime, snapshot, texture or program proof.
static int prepared_op_inputs_equal(VkExec *ex, const VkBindRecipeOp *op, const PreparedBindings *p) {
    if (!op->stable) return 0;
    for (uint32_t k = 0; k < op->count; k++) {
        uint32_t j = op->first + k;
        const VkBindRecipeInput *in = &p->recipe->input[j];
        const PreparedSlot *slot = &p->slot[j];
        switch (in->reuse) {
        case VK_BR_UNIFORM:
        case VK_BR_STORAGE: {
            const p_bind_buf *bound = prepared_buffer(ex, in);
            if (!bound || !slot->valid || slot->input[0] != bound->addr || slot->input[1] != bound->size) return 0;
            break;
        }
        case VK_BR_CONSTANT:
            if (!slot->valid) return 0;
            break;
        case VK_BR_ZERO:
            break;
        case VK_BR_SAMPLED:
            if (!slot->texture.valid || slot->texture.h != prepared_handle(ex, in)) return 0;
            break;
        default:
            return 0;
        }
    }
    return 1;
}

static int prepared_census_adjust(PreparedGroup *group, const PbCount *count, int add) {
    if (!count->has || !count->n) return 1;
    uint32_t i = 0;
    for (; i < group->ncounts; i++)
        if (group->counts[i].cls == count->cls && group->counts[i].use == count->use) break;
    if (i == group->ncounts) {
        if (!add || i == sizeof group->counts / sizeof group->counts[0]) return 0;
        group->counts[group->ncounts++] = (RsCount){count->cls, count->use, 0};
    }
    RsCount *item = &group->counts[i];
    if (add) {
        if (UINT64_MAX - item->bytes < count->n) return 0;
        item->bytes += count->n;
    } else {
        if (item->bytes < count->n) return 0;
        item->bytes -= count->n;
        if (!item->bytes) *item = group->counts[--group->ncounts];
    }
    return 1;
}

// Validate one refreshed descriptor in the same order under either immutable membership walk.
static inline int prepared_renew_member(VkExec *ex, PreparedBindings *p, uint32_t i, uint32_t j,
                                        PreparedGroup *next, VkUsnapProof *proof) {
    const VkBindRecipeInput *in = &p->recipe->input[j];
    PreparedSlot *slot = &p->slot[j];
    ex->dc->prep_stats.proof_descriptors++;
    if (in->reuse == VK_BR_SAMPLED) {
        next->flags |= PREP_TEXTURE;
        if (!prepared_texture_valid(ex, in, slot, &p->desc[j])) return 0;
    } else if (in->reuse != VK_BR_ZERO) {
        if (!prepared_buffer_valid(ex, in, slot)) return 0;
        if (slot->previous_snapshot) {
            if (!p->snapshot_count[i]) return 0;
            p->snapshot_count[i]--;
        }
        if (slot->lease.capture) {
            p->snapshot_count[i]++;
            next->flags |= PREP_SNAPSHOT;
            if (!vk_usnap_proof_add(proof, &slot->lease)) return 0;
        } else next->flags |= PREP_LIVE;
        if (in->reuse == VK_BR_CONSTANT) next->flags |= PREP_PROGRAM;
        if (!prepared_census_adjust(next, &slot->previous_count, 0) ||
            !prepared_census_adjust(next, &slot->count, 1)) return 0;
    }
    return 1;
}

// The old aggregate proves untouched contributions only if it remains valid across the ordinary resolvers.
// Changed contributions are revalidated here. Retained snapshot buckets can reject extra reuse, never admit it.
// Keep the working aggregate off the common resolver's stack and copy it only on a genuine renewal attempt.
static __attribute__((noinline)) int prepared_try_renew(VkExec *ex, PreparedBindings *p, uint32_t set, unsigned group,
                                                       const PreparedStamp *before, VkUsnapProof *proof) {
    PreparedGroup *prior = &p->set[set].group[group];
    if (!prepared_group_valid(ex, prior)) return 0;
    if (NX2NAVI_PREP_PARTIAL_REUSE && !(p->refreshed_groups & (1u << (set * VK_BR_GROUPS + group)))) {
        PC_COUNT(ex->dc, PC_EMPTY_RENEW_TRY);
        PreparedStamp after = prepared_stamp(ex, group);
        if (!prepared_stamp_equal(before, &after, prior->flags) || !prepared_group_valid(ex, prior)) return 0;
        // End the original snapshot publication bracket even though no new leases were added. Keep the prior
        // certificate, including its buckets and future-drop horizon; an empty new proof must not replace it.
        if ((prior->flags & PREP_SNAPSHOT) && !vk_usnap_proof_end(proof)) return 0;
        ex->dc->prep_stats.renewals++;
        PC_COUNT(ex->dc, PC_EMPTY_RENEW_OK);
        return 1;
    }
    PreparedGroup next = *prior;
    uint64_t renewed = 0;
    if ((prior->flags & PREP_SNAPSHOT) && !vk_usnap_proof_inherit(proof, &prior->proof)) return 0;
#if NX2NAVI_PREP_COMPILED_RESOLVE
    const VkBindRecipeMembers *members = &p->recipe->members[set][group];
    VkBindRecipeRefreshWalk walk = {.members = members, .refreshed = p->refreshed, .words = members->words};
    for (uint32_t j; (j = vk_bind_recipe_next_refreshed(&walk)) < VK_BR_MAX_DESC;) {
        uint32_t i = p->recipe->input[j].owner_op;
        if (!prepared_renew_member(ex, p, i, j, &next, proof)) return 0;
        renewed++;
    }
#else
    for (uint32_t i = p->recipe->sets[set].lo; i < p->recipe->sets[set].hi; i++) {
        const VkBindRecipeOp *op = &p->recipe->op[i];
        if (!op->stable || vk_bind_recipe_group(op) != group) continue;
        uint32_t end = op->first + op->count;
        for (uint32_t word = op->first / 64; word <= (end - 1) / 64; word++) {
            uint64_t mask = p->refreshed[word];
            if (word == op->first / 64) mask &= UINT64_MAX << (op->first % 64);
            if (word == (end - 1) / 64 && (end % 64)) mask &= (1ull << (end % 64)) - 1;
            for (; mask; mask &= mask - 1) {
                uint32_t j = word * 64 + (uint32_t)__builtin_ctzll(mask);
                if (!prepared_renew_member(ex, p, i, j, &next, proof)) return 0;
                renewed++;
            }
        }
    }
#endif
    PreparedStamp after = prepared_stamp(ex, group);
    if (!prepared_stamp_equal(before, &after, next.flags) || !prepared_group_valid(ex, prior)) return 0;
    if (next.flags & PREP_SNAPSHOT) {
        if (!vk_usnap_proof_end(proof)) return 0;
        next.proof = *proof;
    }
    next.stamp = after, next.sealed = 1;
    *prior = next;
    ex->dc->prep_stats.renewals++;
    ex->dc->prep_stats.renewed_descriptors += renewed;
    return 1;
}

// Construction is deliberately bounded by stamps taken before resolving and after collecting every dependency.
// A concurrent writer or an ordered op may prevent sealing; descriptors remain usable via their original checks.
static void prepared_seal(VkExec *ex, PreparedBindings *p, uint32_t set, unsigned group, const PreparedStamp *before,
                          VkUsnapProof *proof) {
    if (p->renewable & (1u << (set * VK_BR_GROUPS + group))) {
        PC_COUNT(ex->dc, PC_RENEW_TRY);
        uint64_t prep_renew_start = PC_NOW(ex->dc);
        int renewed = prepared_try_renew(ex, p, set, group, before, proof);
        PC_TIME(ex->dc, PCT_RENEW, prep_renew_start);
        if (renewed) { PC_COUNT(ex->dc, PC_RENEW_OK); return; }
        PC_COUNT(ex->dc, PC_RENEW_FAIL);
        // An attempted extension may have rejected its bracket. The full path visits every slot under a new one.
        if (group == VK_BR_GROUP_BUFFER) vk_usnap_proof_begin(proof);
    }
    PreparedGroup *s = &p->set[set].group[group];
    s->sealed = 0, s->flags = 0, s->ncounts = 0;
    if (ex->dc->mo_ovr_on) return;
    PC_COUNT(ex->dc, PC_FULL_SEAL);
    uint64_t prep_seal_start = PC_NOW(ex->dc);
    int ok = 1;
#if NX2NAVI_PREP_COMPILED_RESOLVE
    const VkBindRecipeMembers *members = &p->recipe->members[set][group];
    uint64_t remaining[2] = {members->ops[0], members->ops[1]};
    uint32_t i = vk_bind_recipe_next_op(remaining);
#else
    uint32_t i = p->recipe->sets[set].lo;
#endif
    for (; i < p->recipe->sets[set].hi;
#if NX2NAVI_PREP_COMPILED_RESOLVE
         i = vk_bind_recipe_next_op(remaining)
#else
         i++
#endif
    ) {
        const VkBindRecipeOp *op = &p->recipe->op[i];
#if !NX2NAVI_PREP_COMPILED_RESOLVE
        if (!op->stable || vk_bind_recipe_group(op) != group) continue;
#endif
        ex->dc->prep_stats.proof_descriptors += op->count;
        PC_ADD(ex->dc, PC_SEAL_DESCRIPTORS, op->count);
        p->snapshot_count[i] = 0;
        for (uint32_t k = 0; k < op->count; k++) {
            uint32_t j = op->first + k;
            const VkBindRecipeInput *in = &p->recipe->input[j];
            PreparedSlot *slot = &p->slot[j];
            if (in->reuse == VK_BR_SAMPLED) {
                s->flags |= PREP_TEXTURE;
                if (!prepared_texture_valid(ex, in, slot, &p->desc[j])) ok = 0;
            } else if (in->reuse != VK_BR_ZERO) {
                if (!prepared_buffer_valid(ex, in, slot)) ok = 0;
                if (slot->lease.capture) {
                    p->snapshot_count[i]++;
                    s->flags |= PREP_SNAPSHOT;
                    if (!vk_usnap_proof_add(proof, &slot->lease)) ok = 0;
                } else s->flags |= PREP_LIVE;
                if (in->reuse == VK_BR_CONSTANT) s->flags |= PREP_PROGRAM;
            }
            if (slot->count.has && in->reuse != VK_BR_SAMPLED && in->reuse != VK_BR_ZERO) {
                uint32_t c = 0;
                for (; c < s->ncounts; c++)
                    if (s->counts[c].cls == slot->count.cls && s->counts[c].use == slot->count.use) break;
                if (c == s->ncounts) {
                    if (c == sizeof s->counts / sizeof s->counts[0]) { ok = 0; continue; }
                    s->counts[s->ncounts++] = (RsCount){slot->count.cls, slot->count.use, 0};
                }
                s->counts[c].bytes += slot->count.n;
            }
        }
    }
    int snapshot_ok = group == VK_BR_GROUP_BUFFER ? vk_usnap_proof_end(proof) : 1;
    PreparedStamp after = prepared_stamp(ex, group);
    if (!ok || !prepared_stamp_equal(before, &after, s->flags) ||
        ((s->flags & PREP_SNAPSHOT) && !snapshot_ok)) {
        PC_TIME(ex->dc, PCT_FULL_SEAL, prep_seal_start);
        PC_COUNT(ex->dc, PC_SEAL_FAIL);
        return;
    }
    if (s->flags & PREP_SNAPSHOT) memcpy(&s->proof, proof, sizeof s->proof);
    s->stamp = after, s->sealed = 1;
    PC_TIME(ex->dc, PCT_FULL_SEAL, prep_seal_start);
    PC_COUNT(ex->dc, PC_SEAL_OK);
}

static void prepared_resolve(VkExec *ex, PreparedBindings *p, int check) {
    static const PreparedOp execute[VK_BR_KINDS] = {
        prepared_support, prepared_uniforms, prepared_storage, prepared_images, prepared_images, prepared_texels
    };
    DrawCtx *cx = ex->dc;
    uint64_t prep_resolve_start = PC_NOW(cx), prep_initial_start = prep_resolve_start;
    int prep_any_stable_resolved = 0;
    const int promoted = p->promotion_pending;
    cx->prep_stats.builds++;
    p->changed_sets = 0, p->renewable = 0;
    if (NX2NAVI_PREP_PARTIAL_REUSE) p->refreshed_groups = 0;
#if NX2NAVI_PREP_COMPILED_RESOLVE
    unsigned fast = 0, all_skipped = 15, reusable_proof = 0, guard_pending = 0, constructing = 0;
#define PG_GET(state, s, g, bit) ((state) & (bit))
#define PG_SET(state, s, g, bit) ((state) |= (bit))
#define PG_CLEAR(state, s, g, bit) ((state) &= ~(bit))
#else
    int fast[2][VK_BR_GROUPS] = {{0}}, all_skipped[2][VK_BR_GROUPS] = {{1, 1}, {1, 1}};
    int reusable_proof[2][VK_BR_GROUPS] = {{0}};
    int guard_pending[2][VK_BR_GROUPS] = {{0}}, constructing[2][VK_BR_GROUPS] = {{0}};
#define PG_GET(state, s, g, bit) ((state)[s][g])
#define PG_SET(state, s, g, bit) ((state)[s][g] = 1)
#define PG_CLEAR(state, s, g, bit) ((state)[s][g] = 0)
#endif
    uint64_t skipped[2] = {0};
    PreparedStamp before[2][VK_BR_GROUPS];
    VkUsnapProof proof[2];   // only buffer groups can hold snapshot leases
    for (uint32_t s = 0; s < p->recipe->n_sets; s++) for (unsigned g = 0; g < VK_BR_GROUPS; g++) {
        if (!(p->recipe->sets[s].stable_groups & (1u << g))) continue;
        unsigned group_bit = 1u << (s * VK_BR_GROUPS + g);
        (void)group_bit;
        uint32_t bit = 1u << (p->index * 2 + s);
        int valid = prepared_group_valid(ex, &p->set[s].group[g]);
        if (promoted && valid) cx->prep_residency.promoted_valid_groups++;
        if (NX2NAVI_PREP_PARTIAL_REUSE && valid) PG_SET(reusable_proof, s, g, group_bit);
        if (!(cx->prep_dirty[g] & bit) && valid) PG_SET(fast, s, g, group_bit);
        if (valid) PC_COUNT(cx, (cx->prep_dirty[g] & bit) ? PC_GROUP_DIRTY : PC_GROUP_FAST);
        else PC_COUNT(cx, (cx->prep_dirty[g] & bit) ? PC_GROUP_BOTH : PC_GROUP_STALE);
        cx->prep_dirty[g] &= ~bit;
        if (!PG_GET(fast, s, g, group_bit) || check) {
            before[s][g] = prepared_stamp(ex, g);
            if (g == VK_BR_GROUP_BUFFER) vk_usnap_proof_begin(&proof[s]);
            PG_SET(constructing, s, g, group_bit);
            if (valid && !check) p->renewable |= 1u << (s * VK_BR_GROUPS + g);
        }
    }
    if (p->renewable) memset(p->refreshed, 0, sizeof p->refreshed);
    PC_TIME(cx, PCT_INITIAL_GUARDS, prep_initial_start);
    uint64_t prep_ops_start = PC_NOW(cx);
    for (uint32_t i = 0; i < p->recipe->n_ops; i++) {
        const VkBindRecipeOp *op = &p->recipe->op[i];
        uint32_t s = op->set;
        unsigned g = vk_bind_recipe_group(op);
#if NX2NAVI_PREP_COMPILED_RESOLVE
        unsigned group_bit = op->group_bit;
#else
        unsigned group_bit = 1u << (s * VK_BR_GROUPS + g);
        (void)group_bit;
#endif
        PC_COUNT(cx, PC_OPS);
        int partial = NX2NAVI_PREP_PARTIAL_REUSE && op->stable && !PG_GET(fast, s, g, group_bit) && PG_GET(reusable_proof, s, g, group_bit) &&
                      prepared_op_inputs_equal(ex, op, p);
        if ((PG_GET(fast, s, g, group_bit) || partial) && PG_GET(guard_pending, s, g, group_bit)) PC_COUNT(cx, PC_GROUP_RECHECK);
        // An earlier ordered resolver can change a generation or drop a snapshot lease. Never move its effects
        // past this op: a failed guard takes the ordinary per-slot path for the rest of this group on this draw.
        // Dirty groups need this recheck only when this op could reuse a retained binding. Ordered or changed
        // inputs still execute their ordinary resolver, which makes the guard pending again before later reuse.
        if ((PG_GET(fast, s, g, group_bit) || partial) && PG_GET(guard_pending, s, g, group_bit) && !prepared_group_valid(ex, &p->set[s].group[g])) {
            PG_CLEAR(fast, s, g, group_bit);
            PG_CLEAR(reusable_proof, s, g, group_bit);
            partial = 0;
            if (!PG_GET(constructing, s, g, group_bit)) {
                before[s][g] = prepared_stamp(ex, g);
                if (g == VK_BR_GROUP_BUFFER) vk_usnap_proof_begin(&proof[s]);
                PG_SET(constructing, s, g, group_bit);
            }
        }
        PG_CLEAR(guard_pending, s, g, group_bit);
        if (op->stable && (PG_GET(fast, s, g, group_bit) || partial) && !check) {
            if (partial) { PC_COUNT(cx, PC_PARTIAL_SKIP); PC_ADD(cx, PC_PARTIAL_DESCRIPTORS, op->count); }
            PC_COUNT(cx, PC_STABLE_SKIP);
            skipped[i / 64] |= 1ull << (i % 64);
            if (promoted) cx->prep_residency.promoted_skipped += op->count;
            cx->prep_stats.aggregate_skipped += op->count;
            cx->prep_stats.reused += op->count;
            cx->prep_stats.snapshot_reused += p->snapshot_count[i];
            vk_exec_frame.bind_reused += op->count;
        } else {
            if (op->stable) {
                if (promoted) cx->prep_residency.promoted_resolved += op->count;
                PC_COUNT(cx, PC_STABLE_RESOLVE); prep_any_stable_resolved = 1;
            }
            else PC_COUNT(cx, PC_ORDERED);
            if (op->stable) PG_CLEAR(all_skipped, s, g, group_bit);
            if (op->stable && (PG_GET(fast, s, g, group_bit) || partial) && check) {
                if (partial) PC_COUNT(cx, PC_PARTIAL_CHECKED);
                prepared_op_check(ex, op, p, execute[op->kind]);
            }
            else execute[op->kind](ex, op, p, check);
#if NX2NAVI_PREP_COMPILED_RESOLVE
            guard_pending = 15;
#else
            for (unsigned ss = 0; ss < 2; ss++) for (unsigned gg = 0; gg < VK_BR_GROUPS; gg++)
                guard_pending[ss][gg] = 1;
#endif
        }
        cx->prep_stats.descriptors += op->count;
        if (p->recipe->need_w) {
            PC_COUNT(cx, PC_WRITE_STRUCTS);
            VkWriteDescriptorSet *w = &ex->dc->bind_w[i];
            *w = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                  .dstBinding = op->binding->binding, .descriptorCount = op->count, .descriptorType = op->binding->type};
            VkDescSlot *d = &p->desc[op->first];
            if (op->kind == VK_BR_TEXEL) {
                for (uint32_t k = 0; k < op->count; k++) cx->bind_bv[op->first + k] = p->desc[op->first + k].view;
                w->pTexelBufferView = &cx->bind_bv[op->first];
            }
            else if (op->kind == VK_BR_TEXTURE || op->kind == VK_BR_IMAGE) w->pImageInfo = &d->image;
            else w->pBufferInfo = &d->buffer;
        }
    }
    PC_TIME(cx, PCT_OP_LOOP, prep_ops_start);
    uint64_t prep_finish_start = PC_NOW(cx);
    for (uint32_t s = 0; s < p->recipe->n_sets; s++) for (unsigned g = 0; g < VK_BR_GROUPS; g++) {
        if (!(p->recipe->sets[s].stable_groups & (1u << g))) continue;
        unsigned group_bit = 1u << (s * VK_BR_GROUPS + g);
        (void)group_bit;
        // Count skipped contributions before resealing replaces their census. A rare mid-set invalidation needs
        // only the already-skipped spans; the usual whole-group hit uses the small aggregate class/use list.
        PreparedGroup *group = &p->set[s].group[g];
        if (PG_GET(all_skipped, s, g, group_bit) && !check) {
            for (uint32_t c = 0; c < group->ncounts; c++) {
                const RsCount *n = &group->counts[c];
                count_use(cx, n->cls, n->use, n->bytes);
            }
        } else {
#if NX2NAVI_PREP_COMPILED_RESOLVE
            const VkBindRecipeMembers *members = &p->recipe->members[s][g];
            uint64_t remaining[2] = {members->ops[0] & skipped[0], members->ops[1] & skipped[1]};
            uint32_t i = vk_bind_recipe_next_op(remaining);
#else
            uint32_t i = p->recipe->sets[s].lo;
#endif
            for (; i < p->recipe->sets[s].hi;
#if NX2NAVI_PREP_COMPILED_RESOLVE
                 i = vk_bind_recipe_next_op(remaining)
#else
                 i++
#endif
            ) {
#if !NX2NAVI_PREP_COMPILED_RESOLVE
                if (!(skipped[i / 64] >> (i % 64) & 1)) continue;
#endif
                const VkBindRecipeOp *op = &p->recipe->op[i];
#if !NX2NAVI_PREP_COMPILED_RESOLVE
                if (vk_bind_recipe_group(op) != g) continue;
#endif
                for (uint32_t k = 0; k < op->count; k++) {
                    const PbCount *n = &p->slot[op->first + k].count;
                    if (n->has) count_use(cx, n->cls, n->use, n->n);
                }
            }
        }
        // A proof surviving every ordered boundary can be retained without revisiting any descriptor.
        if (vk_texture_deferred()) group->sealed = 0;
        else if (PG_GET(constructing, s, g, group_bit)) prepared_seal(ex, p, s, g, &before[s][g], &proof[s]);
        else if (PG_GET(guard_pending, s, g, group_bit) && !prepared_group_valid(ex, group)) group->sealed = 0;
    }
    for (uint32_t s = 0; s < p->recipe->n_sets; s++)
        if (p->changed_sets & (1u << s)) p->set[s].packet = NULL, p->set[s].allocated = VK_NULL_HANDLE;
    p->renewable = 0;
    p->promotion_pending = 0;
    PC_TIME(cx, PCT_FINISH, prep_finish_start);
    int prep_has_stable = p->recipe->sets[0].stable_groups ||
                         (p->recipe->n_sets > 1 && p->recipe->sets[1].stable_groups);
    PC_COUNT(cx, !prep_has_stable ? PC_NO_STABLE : prep_any_stable_resolved ? PC_MIXED_RESOLVE : PC_ALL_STABLE_SKIP);
    PC_TIME_PAIR(cx, PCT_RESOLVE, !prep_has_stable ? PCT_RESOLVE_NO_STABLE :
                 prep_any_stable_resolved ? PCT_RESOLVE_MIXED : PCT_RESOLVE_ALL_STABLE_SKIP, prep_resolve_start);
    (void)prep_any_stable_resolved;
#undef PG_GET
#undef PG_SET
#undef PG_CLEAR
}

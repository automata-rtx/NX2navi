// The texture lock owns namespace membership. Descriptor payload and history remain in the registration service.
#pragma once

typedef struct {
    nvn_hmap extra_slots; // logical id to physical TexSlot index + 1
    int count;
} TexturePoolNames;

static nvn_hmap texture_pool_names = {.klen = sizeof(uint64_t)};
static uint64_t texture_primary_pool = UINT64_MAX;
static int texture_next_extra_slot = VK_POOL_TEX;
static int texture_free_extra_slot = -1;

int vk_texture_pool_primary(uint64_t owner) {
    return __atomic_load_n(&texture_primary_pool, __ATOMIC_ACQUIRE) == owner;
}

static TexturePoolNames *texture_pool_open(uint64_t owner) {
    TexturePoolNames *names = nvn_hm_get(&texture_pool_names, &owner);
    if (names) return names;
    names = calloc(1, sizeof *names);
    if (!names) rt_fatal("vk: allocating texture pool namespace");
    nvn_hm_init(&names->extra_slots, sizeof(int));
    nvn_hm_put(&texture_pool_names, &owner, names);
    if (__atomic_load_n(&texture_primary_pool, __ATOMIC_RELAXED) == UINT64_MAX)
        __atomic_store_n(&texture_primary_pool, owner, __ATOMIC_RELEASE);
    return names;
}

int vk_texture_pool_slot_locked(uint64_t owner, int logical) {
    if (logical < 0 || logical >= 1 << 20) return -1;
    int id = logical;
    if (logical >= VK_POOL_TEX || !vk_texture_pool_primary(owner)) {
        TexturePoolNames *names = nvn_hm_get(&texture_pool_names, &owner);
        if (!names) return -1;
        id = (int)(intptr_t)nvn_hm_get(&names->extra_slots, &logical) - 1;
    }
    // A primary selector has an arithmetic address even before its TexSlot is allocated or published.
    // Readers look up published slots only; claim reserves the address independently below.
    if (id < 0 || id >= ntslots) return -1;
    const TexSlot *slot = &tslots[id];
    return slot->used && slot->pool_owner == owner && slot->logical_id == logical ? id : -1;
}

int vk_texture_pool_count_locked(uint64_t owner) {
    TexturePoolNames *names = nvn_hm_get(&texture_pool_names, &owner);
    return names ? names->count : 0;
}

int vk_texture_pool_logical_locked(uint64_t owner, int native_slot) {
    if (native_slot < 0 || native_slot >= ntslots) return -1;
    const TexSlot *slot = &tslots[native_slot];
    return slot->used && slot->pool_owner == owner ? slot->logical_id : -1;
}

static int texture_pool_claim(uint64_t owner, int logical) {
    TexturePoolNames *names = texture_pool_open(owner);
    if (logical >= names->count) names->count = logical + 1;
    if (logical < VK_POOL_TEX && vk_texture_pool_primary(owner)) return logical;
    int slot = vk_texture_pool_slot_locked(owner, logical);
    if (slot >= 0) return slot;
    if (texture_free_extra_slot >= 0) {
        slot = texture_free_extra_slot;
        texture_free_extra_slot = tslots[slot].free_slot;
    } else {
        if (texture_next_extra_slot >= 1 << 20) rt_fatal("vk: native texture selector capacity exhausted");
        slot = texture_next_extra_slot++;
    }
    nvn_hm_put(&names->extra_slots, &logical, (void *)(intptr_t)(slot + 1));
    return slot;
}

static void texture_pool_names_free(void *arg) {
    TexturePoolNames *names = arg;
    free(names->extra_slots.keys);
    free(names->extra_slots.vals);
    free(names);
}

static void texture_pool_names_shutdown(void) {
    nvn_hm_clear(&texture_pool_names, texture_pool_names_free);
    free(texture_pool_names.keys);
    free(texture_pool_names.vals);
    nvn_hm_init(&texture_pool_names, sizeof(uint64_t));
    __atomic_store_n(&texture_primary_pool, UINT64_MAX, __ATOMIC_RELEASE);
    texture_free_extra_slot = -1;
    texture_next_extra_slot = VK_POOL_TEX;
}

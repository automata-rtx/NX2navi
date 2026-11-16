#include "motion_owner.h"
#include "host.h"
#include "layout.h"
#include "nvn_api.h"
#include "platform/platform.h"
#include <string.h>
#include <limits.h>
#include <stdlib.h>

// The supported game build exposes a player equipment component with generation-checked actor links.
// Selection follows those links and the equipment's model-binding parent; proximity is never an identity.
#define OWNER_MODELS 128
#define OWNER_ACTORS 22
typedef struct {
    uint64_t actor, link;
    uint32_t generation, kind;
} OwnerActor;
static struct {
    MotionOwner models[OWNER_MODELS];
    unsigned count;
    uint64_t frame, producer_frame, serial;
    int valid;
    MotionOwnerSnapshot cached;
    MotionOwnerSnapshot *published;
    unsigned capacity_misses;
    unsigned model_bind_failures;
} O;
static PlatMutex owner_lock = PLAT_MUTEX_INIT;
static PlatOnce owner_once = PLAT_ONCE_INIT;
static int owner_on, player_motion;

static void owner_init(void *unused) {
    (void)unused;
    player_motion = rt_knob("PLAYER_MOTION", "1");
    owner_on = RT_V111 && rt_knob("SHIELD_MOTION", "1");
}
int motion_owner_enabled(void) {
    plat_once(&owner_once, owner_init, NULL);
    return owner_on;
}
static int read_at(uint64_t a, void *out, size_t n) {
    if (!a || a > UINT64_MAX - n) return 0;
    mem_kernel_access(a, n);
    if (!mem_read_hold(a, n)) return 0;
    memcpy(out, (const void *)(uintptr_t)a, n);
    mem_read_release();
    return 1;
}
static uint64_t ptr(uint64_t a) {
    uint64_t p = 0;
    return read_at(a, &p, sizeof p) ? p : 0;
}
static uint64_t field(uint64_t object, uint32_t offset) { return object ? ptr(object + offset) : 0; }
static uint64_t component(uint64_t actor, uint32_t offset) { return field(field(actor, 0x270), offset); }
static uint64_t player_actor(void) {
    uint64_t p = field(ptr(MOD_MAIN_BASE + 0x4e371d0), 0x480);
    char name[7];
    return p && read_at(field(p, 0x20), name, sizeof name) && !memcmp(name, "Player", 7) ? p : 0;
}
// Ocarina's player query follows this component to determine whether a recenter is blending.
int motion_owner_player_camera_interpolating(void) {
    uint64_t camera = component(player_actor(), 0x368);
    uint8_t blending;
    return camera && read_at(camera + 0x43c, &blending, sizeof blending) && blending == 1;
}
static int actor_link(uint64_t link, OwnerActor *out) {
    struct {
        uint64_t table;
        uint32_t generation, padding;
    } a, b;
    uint32_t generation, state;
    if (!link || !read_at(link + 8, &a, sizeof a) || !a.table || a.generation == UINT32_MAX ||
        !read_at(a.table + 0x48, &generation, 4) || generation != a.generation)
        return 0;
    uint64_t actor = field(a.table, 0x40);
    if (!actor || !read_at(actor + 0x2c, &state, 4) || (uint32_t)(state - 7) < 5) return 0;
    // Reject a slot replaced during the read. This is not a replacement for the game's actor locking:
    // unknown layouts and any observed disagreement fail closed, and runtime lifetime checks remain required.
    if (!read_at(link + 8, &b, sizeof b) || a.table != b.table || a.generation != b.generation ||
        !read_at(a.table + 0x48, &generation, 4) || generation != a.generation || field(a.table, 0x40) != actor)
        return 0;
    *out = (OwnerActor){actor, a.table, generation, 0};
    return 1;
}
static int parent_is_player(uint64_t actor, uint64_t player) {
    uint64_t seen[4] = {actor};
    for (unsigned depth = 0; depth < 4; ++depth) {
        uint64_t bind = component(actor, 0x18);
        OwnerActor parent;
        if (!bind || !actor_link(bind + 0x70, &parent)) return 0;
        if (parent.actor == player) return 1;
        for (unsigned i = 0; i <= depth; ++i)
            if (seen[i] == parent.actor) return 0;
        if (depth + 1 < 4) seen[depth + 1] = parent.actor;
        actor = parent.actor;
    }
    return 0;
}
// Equipped actors have different names from inventory items, shop displays and world props.
static unsigned equipment_kind(uint64_t actor) {
    static const struct {
        const char *name;
        enum MotionEquipmentKind kind;
    } items[] = {
        {"Weapon_Sword_Kokiri", MOTION_SWORD}, {"Weapon_Sword_Master", MOTION_SWORD},
        {"Weapon_Sword_Giant", MOTION_SWORD}, {"Weapon_Sword_BrokenGiant", MOTION_SWORD},
        {"Weapon_Sword_Biggoron", MOTION_SWORD},
        {"Weapon_Shield_Deku", MOTION_SHIELD}, {"Weapon_Shield_Hylian", MOTION_SHIELD},
        {"Weapon_Shield_Mirror", MOTION_SHIELD},
        {"Weapon_Sheath_Kokiri", MOTION_SHEATH}, {"Weapon_Sheath_070", MOTION_SHEATH},
        {"Weapon_Lsheath_059", MOTION_SHEATH},
    };
    char name[32];
    if (!read_at(field(actor, 0x20), name, sizeof name) || !memchr(name, 0, sizeof name)) return 0;
    for (size_t i = 0; i < sizeof items / sizeof *items; ++i)
        if (!strcmp(name, items[i].name)) return items[i].kind;
    return 0;
}
// The instance owns both the model resource and its named binding. Check the active asset,
// since the same Player actor can also render Adult Link or a cutscene replacement.
static uint64_t child_resource(uint64_t model) {
    uint64_t resource = field(model, 0xd8);
    char tag[4], name[sizeof "Player_Child"];
    if (!read_at(resource, tag, sizeof tag) || memcmp(tag, "FMDL", sizeof tag) ||
        !read_at(field(field(model, 0x78), 0x30), name, sizeof name) || memcmp(name, "Player_Child", sizeof name))
        return 0;
    return resource;
}
static int collect_models(uint64_t player, OwnerActor actor) {
    uint64_t model_component = component(actor.actor, 8);
    if (!model_component) return 1;
    // Only decode the implementation whose getter directly returns its model collection.
    if (field(ptr(model_component), 0xf8) != MOD_MAIN_BASE + 0x9d7160) return 1;
    uint64_t collection = field(model_component, 0x18), list = field(collection, 0x28);
    int32_t count;
    if (!collection || !read_at(collection + 0x20, &count, 4) || count < 0 || count > 16 || (count && !list)) return 0;
    for (int i = 0; i < count; ++i) {
        uint64_t model = ptr(ptr(list + (uint64_t)i * 8));
        if (!model) continue;
        // Match the model implementation used by the traced skeleton and shape binding path.
        if (field(ptr(model), 0x40) != MOD_MAIN_BASE + 0x735430) continue;
        uint64_t resource = actor.kind == MOTION_PLAYER ? child_resource(model) : 0;
        if (actor.kind == MOTION_PLAYER && !resource) continue;
        if (O.count == OWNER_MODELS) return 0;
        O.models[O.count++] = (MotionOwner){.player = player,
                                            .actor = actor.actor,
                                            .link = actor.link,
                                            .model = model, .resource = resource,
                                            .generation = actor.generation,
                                            .kind = actor.kind};
    }
    return field(model_component, 0x18) == collection && field(collection, 0x28) == list;
}
static void refresh(uint64_t frame, int force) {
    if (!force && O.valid && O.frame == frame) return;
    MotionOwner previous[OWNER_MODELS];
    unsigned nprevious = O.count;
    memcpy(previous, O.models, nprevious * sizeof *previous);
    O.frame = frame;
    O.valid = 1;
    O.count = 0;
    uint64_t player = player_actor();
    if (!player) return;
    OwnerActor actors[OWNER_ACTORS] = {{player, 0, 0, MOTION_PLAYER}};
    unsigned count = 1;
    uint64_t equipment = component(player, 0x200);
    if (equipment) {
        // Eight static and thirteen dynamic slots, each a 24-byte actor link.
        for (unsigned slot = 0; slot < 21; ++slot) {
            uint64_t link = equipment + (slot < 8 ? 0x20 + slot * 24 : 0xe0 + (slot - 8) * 24);
            OwnerActor a;
            if (!actor_link(link, &a)) continue;
            a.kind = equipment_kind(a.actor);
            if (!a.kind) continue;
            if (!parent_is_player(a.actor, player)) {
                ++O.model_bind_failures;
                continue;
            }
            unsigned i;
            for (i = 0; i < count && actors[i].actor != a.actor; ++i) {
            }
            if (i != count) continue;
            actors[count++] = a;
        }
    }
    for (unsigned i = 1; i < count; ++i)
        if (!collect_models(player, actors[i])) {
            O.count = 0;
            break;
        }
    // Reserve equipment identities before enumerating the additional player shapes.
    unsigned equipment_models = O.count;
    if (player_motion && !collect_models(player, actors[0])) O.count = equipment_models;
    if (player_actor() != player || component(player, 0x200) != equipment) O.count = 0;
    for (unsigned i = 0; i < O.count; ++i) {
        MotionOwner *m = &O.models[i];
        for (unsigned j = 0; j < nprevious; ++j) {
            const MotionOwner *p = &previous[j];
            if (m->player == p->player && m->actor == p->actor && m->link == p->link &&
                m->generation == p->generation && m->model == p->model && m->resource == p->resource) {
                m->lifetime = p->lifetime;
                break;
            }
        }
        if (!m->lifetime) {
            m->lifetime = ++O.serial;
            if (rt_knob("SHIELD_TRACE", "0")) {
                char name[32] = {0};
                if (read_at(field(m->actor, 0x20), name, sizeof name))
                    rt_log("equipped motion: %.*s model lifetime %llu", (int)sizeof name, name,
                           (unsigned long long)m->lifetime);
            }
        }
    }
}
static int binding_address_order(const void *a, const void *b) {
    uint64_t x = ((const MotionOwnerBinding *)a)->address, y = ((const MotionOwnerBinding *)b)->address;
    return x < y ? -1 : x > y;
}

static void capture_model_buffers(MotionOwnerSnapshot *out) {
    unsigned equipment_bindings = 0;
    for (unsigned i = 0; i < O.count; ++i) {
        MotionOwner owner = O.models[i];
        uint32_t count = 0;
        if (!read_at(owner.model + 0x20, &count, 4) || count > 128) continue;
        uint64_t shapes = field(owner.model, 0x100);
        for (unsigned shape = 0; shape < count; ++shape) {
            uint64_t blocks = field(shapes + (uint64_t)shape * 0x70, 0x20);
            for (unsigned slot = 0; slot < 2; ++slot) {
                uint64_t buffer = field(blocks + (uint64_t)slot * 0x48, 8);
                buf_t b;
                pool_t pool;
                if (!read_at(buffer, &b, sizeof b) || b.magic != MAGIC('B', 'U', 'F', 0) ||
                    !read_at((uint64_t)(uintptr_t)b.pool, &pool, sizeof pool) || pool.magic != NVN_POOL_MAGIC ||
                    b.off < 0 || b.size < 256 || b.size > 65536 || !pool.mem || (uint64_t)b.off > pool.size ||
                    b.size > pool.size - (uint64_t)b.off)
                    continue;
                uint64_t address = (uint64_t)(uintptr_t)pool.mem + (uint64_t)b.off;
                if (out->count == MOTION_OWNER_BINDINGS) {
                    ++O.capacity_misses;
                    out->count = owner.kind == MOTION_PLAYER ? equipment_bindings : 0;
                    return;
                }
                owner.part = (uint64_t)shape + 1;
                out->bindings[out->count++] = (MotionOwnerBinding){
                    .address = address, .size = b.size, .observed = O.producer_frame, .buffer = buffer, .owner = owner};
            }
        }
        if (owner.kind != MOTION_PLAYER) equipment_bindings = out->count;
    }
}

void motion_owner_release(MotionOwnerSnapshot *snapshot) {
    if (snapshot && __atomic_sub_fetch(&snapshot->refs, 1, __ATOMIC_ACQ_REL) == 0) free(snapshot);
}

void motion_owner_shutdown(void) {
    plat_mutex_lock(&owner_lock);
    motion_owner_release(O.published);
    O.published = NULL;
    O.valid = 0;
    O.count = O.cached.count = 0;
    plat_mutex_unlock(&owner_lock);
}

// Submissions retain the same immutable identity list until the producer discovers
// another generation. A queued replay can finish using its original list after a swap.
MotionOwnerSnapshot *motion_owner_acquire(void) {
    if (!motion_owner_enabled()) return NULL;
    plat_mutex_lock(&owner_lock);
    if (!O.valid) {
        motion_owner_release(O.published);
        O.published = NULL;
        refresh(O.producer_frame, 1);
        O.cached.count = 0;
        capture_model_buffers(&O.cached);
        if (O.producer_frame % 30 == 0 && rt_knob("SHIELD_TRACE", "0"))
            rt_log("shield owner: frame %llu models %u bindings %u", (unsigned long long)O.producer_frame, O.count,
                   O.cached.count);
        qsort(O.cached.bindings, O.cached.count, sizeof *O.cached.bindings, binding_address_order);
        unsigned kept = 0;
        for (unsigned i = 0; i < O.cached.count;) {
            const MotionOwnerBinding *binding = &O.cached.bindings[i];
            uint64_t end = binding->address + binding->size;
            unsigned next = i + 1;
            while (next < O.cached.count && O.cached.bindings[next].address < end) {
                const MotionOwnerBinding *overlap = &O.cached.bindings[next++];
                if (end < overlap->address + overlap->size) end = overlap->address + overlap->size;
            }
            if (next == i + 1) O.cached.bindings[kept++] = *binding;
            i = next;
        }
        O.cached.count = kept;
    }
    if (!O.published && O.cached.count) {
        O.published = malloc(sizeof *O.published);
        if (O.published) {
            O.published->count = O.cached.count;
            O.published->refs = 1;
            memcpy(O.published->bindings, O.cached.bindings, O.cached.count * sizeof *O.cached.bindings);
        }
    }
    MotionOwnerSnapshot *snapshot = O.published;
    if (snapshot) __atomic_add_fetch(&snapshot->refs, 1, __ATOMIC_RELAXED);
    plat_mutex_unlock(&owner_lock);
    return snapshot;
}

void motion_owner_present(void) {
    if (!motion_owner_enabled()) return;
    plat_mutex_lock(&owner_lock);
    ++O.producer_frame;
    O.valid = 0;
    plat_mutex_unlock(&owner_lock);
}

static void forget(uint64_t buffer, uint64_t address, uint64_t size) {
    if (!motion_owner_enabled()) return;
    plat_mutex_lock(&owner_lock);
    for (unsigned i = 0; i < O.cached.count; ++i) {
        MotionOwnerBinding *b = &O.cached.bindings[i];
        if (b->buffer != buffer && !(size && b->address < address + size && address < b->address + b->size)) continue;
        O.cached.count = O.count = 0;
        O.valid = 0;
        break;
    }
    plat_mutex_unlock(&owner_lock);
}
void motion_owner_forget(uint64_t buffer) { forget(buffer, 0, 0); }
void motion_owner_forget_range(uint64_t address, uint64_t size) {
    if (address <= UINT64_MAX - size) forget(0, address, size);
}

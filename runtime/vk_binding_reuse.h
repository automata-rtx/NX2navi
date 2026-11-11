#ifndef VK_BINDING_REUSE_H
#define VK_BINDING_REUSE_H

#include <stdint.h>
#include <string.h>

enum { PB_BUFFER = 1, PB_TEXTURE = 2, PB_ALL = 3, PB_G = 17, PB_SETS = 16, PB_WAYS = 4, PB_N = PB_SETS * PB_WAYS };
typedef struct { uintptr_t key; uint64_t used; } PbTag;

// Keep submission and memory freshness boundaries for both classes. Texture slot
// changes are checked by the individual slot token in the descriptor's key.
static inline unsigned pb_invalidated(const uint64_t old[PB_G], const uint64_t now[PB_G]) {
    if (old[0] != now[0] || old[1] != now[1] || old[2] != now[2] || old[5] != now[5]) return PB_ALL;
    if ((uint32_t)old[3] != (uint32_t)now[3] || old[4] != now[4] ||
        memcmp(old + 6, now + 6, (PB_G - 6) * sizeof *now)) return PB_TEXTURE;
    return 0;
}

// Four ways prevent alternating layouts that share a hash bucket from evicting
// each other on every draw. Replacement changes reuse only, never validity.
static inline unsigned pb_select(PbTag tags[PB_N], uintptr_t key, uint64_t tick) {
    unsigned first = (unsigned)((key >> 6 ^ key >> 13) % PB_SETS) * PB_WAYS, victim = first;
    for (unsigned i = first; i < first + PB_WAYS; i++) {
        if (tags[i].key == key) { tags[i].used = tick; return i; }
        if (tags[i].used < tags[victim].used) victim = i;
    }
    tags[victim] = (PbTag){key, tick};
    return victim;
}
#endif

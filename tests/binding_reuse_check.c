#include <assert.h>
#include <stdio.h>
#include "vk_binding_reuse.h"

int main(void) {
    uint64_t old[PB_G] = {0}, now[PB_G] = {0};
    assert(pb_invalidated(old, now) == 0);
    now[3] = 42ull << 32;   // an unrelated texture slot changed
    assert(pb_invalidated(old, now) == 0);
    now[3]++;
    assert(pb_invalidated(old, now) == PB_TEXTURE); // new texture freshness epoch
    for (unsigned j = 0; j < PB_G; j++) {
        memset(now, 0, sizeof now);
        now[j] = 1;
        assert(pb_invalidated(old, now) == ((j <= 2 || j == 5) ? PB_ALL : PB_TEXTURE));
    }
    PbTag tags[PB_N] = {0};
    uintptr_t keys[5];
    unsigned n = 0;
    for (uintptr_t k = 64; n < 5; k += 64)
        if ((k >> 6 ^ k >> 13) % PB_SETS == 0) keys[n++] = k;
    unsigned slots[4];
    for (unsigned i = 0; i < 4; i++) slots[i] = pb_select(tags, keys[i], i + 1);
    for (unsigned i = 0; i < 4; i++)
        for (unsigned j = i + 1; j < 4; j++) assert(slots[i] != slots[j]);
    assert(pb_select(tags, keys[0], 5) == slots[0]);
    assert(pb_select(tags, keys[4], 6) == slots[1]); // only the least recently used way is replaced
    assert(pb_select(tags, keys[0], 7) == slots[0]);
    assert(pb_select(tags, keys[2], 8) == slots[2]);
    puts("Binding dependency and associative replacement checks passed");
    return 0;
}

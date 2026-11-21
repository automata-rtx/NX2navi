#pragma once
#include <stddef.h>
#include <stdint.h>

enum { RC_INVENTORY_BUFFER = 1, RC_INVENTORY_IMAGE = 2, RC_INVENTORY_GPA = 3 };
typedef struct {
    uint32_t kind, set, binding, count;
} RcWriteBinding;
typedef int (*RcWriteBindingCovered)(void *context, const RcWriteBinding *binding);
// This is structural output coverage, not a write-extent proof. The callback must cover every descriptor-array
// element of each possible destination. Unknown pointer provenance or effects refuse the entire inventory.
int rc_write_inventory_covered(const uint32_t *words, size_t count,
                               RcWriteBindingCovered covered, void *context);
typedef int (*RcWriteSiteCovered)(void *context, size_t instruction, const RcWriteBinding *binding);
int rc_write_inventory_sites_covered(const uint32_t *words, size_t count,
                                     RcWriteSiteCovered covered, void *context);

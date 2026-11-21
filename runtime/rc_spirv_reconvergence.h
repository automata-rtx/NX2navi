#pragma once

#include <stdint.h>
#include <string.h>

enum { RC_SPV_MAXIMALLY_RECONVERGES_KHR = 6023 };

static inline int rc_spirv_maximal_reconvergence_extension(const uint32_t *words, uint32_t count) {
    static const char extension[32] = "SPV_KHR_maximal_reconvergence";
    return count == 1 + sizeof extension / sizeof *words &&
        !memcmp(words + 1, extension, sizeof extension);
}

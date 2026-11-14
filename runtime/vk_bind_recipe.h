// Fixed resource decisions for a pipeline. Instances and their resolved resources belong to a recording context.
#pragma once
#include "vk_exec.h"

#ifndef NX2NAVI_PREP_PARTIAL_REUSE
#define NX2NAVI_PREP_PARTIAL_REUSE 1
#endif
#ifndef NX2NAVI_PREP_COMPILED_RESOLVE
#define NX2NAVI_PREP_COMPILED_RESOLVE 1
#endif

enum {
    VK_BR_SUPPORT, VK_BR_UBO, VK_BR_SSBO, VK_BR_TEXTURE, VK_BR_IMAGE, VK_BR_TEXEL, VK_BR_KINDS
};
enum { VK_BR_ALWAYS, VK_BR_UNIFORM, VK_BR_CONSTANT, VK_BR_STORAGE, VK_BR_SAMPLED, VK_BR_ZERO };
enum { VK_BR_GROUP_BUFFER, VK_BR_GROUP_TEXTURE, VK_BR_GROUPS };
#define VK_BR_NO_SOURCE UINT32_MAX
#define VK_BR_MAX_BINDINGS 128u
#define VK_BR_MAX_DESC 1024u

typedef struct {
    uint32_t source;                     // byte offset in VkExec, or VK_BR_NO_SOURCE
    int32_t slot;
    uint8_t reuse;                       // validity contract, not a promise that a result is reusable
#if NX2NAVI_PREP_COMPILED_RESOLVE
    uint16_t owner_op;                   // packed into existing input padding; immutable descriptor ownership
#endif
} VkBindRecipeInput;
typedef struct {
    const VkPipeBinding *binding;
    uint32_t first, count;
    uint8_t kind, stage, set, stable;     // stable ops may be skipped only under their set's resource proof
#if NX2NAVI_PREP_COMPILED_RESOLVE
    uint8_t group_bit;                   // one of the four set/group bits, including ordered operations
#endif
} VkBindRecipeOp;

static inline unsigned vk_bind_recipe_group(const VkBindRecipeOp *op) {
    return op->kind == VK_BR_TEXTURE ? VK_BR_GROUP_TEXTURE : VK_BR_GROUP_BUFFER;
}
typedef struct {
    uint32_t lo, hi, first, count, stable_groups;
} VkBindRecipeSet;
#if NX2NAVI_PREP_COMPILED_RESOLVE
typedef struct {
    uint64_t ops[(VK_BR_MAX_BINDINGS + 63) / 64];
    uint64_t descriptors[(VK_BR_MAX_DESC + 63) / 64];
    uint16_t words;                      // nonempty descriptor-mask words, visited in ascending order
} VkBindRecipeMembers;
_Static_assert(sizeof(VkBindRecipeInput) == 12, "descriptor ownership must fit existing input padding");
_Static_assert(sizeof(VkBindRecipeOp) == 24, "group bit must fit existing operation padding");
_Static_assert(VK_BR_MAX_DESC <= 1024 && VK_BR_MAX_BINDINGS <= 128, "member masks cover every recipe");
#endif

typedef struct VkBindRecipe {
    const VkPipeInfo *info;
    uint32_t n_sets, n_ops, n_desc;
    int need_w, stage_override;
    VkBindRecipeSet sets[2];
    VkBindRecipeOp op[VK_BR_MAX_BINDINGS];
#if NX2NAVI_PREP_COMPILED_RESOLVE
    VkBindRecipeMembers members[2][VK_BR_GROUPS];
#endif
    VkBindRecipeInput input[];
} VkBindRecipe;

// stage_override is 0 normally, or the task/mesh NVN stage whose fallback executes a compute module.
// All resource kinds have a recipe; an ordered resource has VK_BR_ALWAYS instead of rejecting its whole pipeline.
VkBindRecipe *vk_bind_recipe_create(const VkPipeInfo *info, int stage_override);

#if NX2NAVI_PREP_COMPILED_RESOLVE
// These walks describe membership only. Callers still establish the live resource proof at every ordered boundary.
static inline uint32_t vk_bind_recipe_next_op(uint64_t pending[2]) {
    unsigned word = pending[0] ? 0 : 1;
    if (!pending[word]) return VK_BR_MAX_BINDINGS;
    uint32_t index = word * 64 + (uint32_t)__builtin_ctzll(pending[word]);
    pending[word] &= pending[word] - 1;
    return index;
}

typedef struct {
    const VkBindRecipeMembers *members;
    const uint64_t *refreshed;
    uint64_t pending;
    uint32_t words, word;
} VkBindRecipeRefreshWalk;

static inline uint32_t vk_bind_recipe_next_refreshed(VkBindRecipeRefreshWalk *walk) {
    while (!walk->pending) {
        if (!walk->words) return VK_BR_MAX_DESC;
        walk->word = (uint32_t)__builtin_ctz(walk->words);
        walk->words &= walk->words - 1;
        walk->pending = walk->members->descriptors[walk->word] & walk->refreshed[walk->word];
    }
    uint32_t index = walk->word * 64 + (uint32_t)__builtin_ctzll(walk->pending);
    walk->pending &= walk->pending - 1;
    return index;
}
#endif

// Stable recording-scope storage for one deduplicated material instance use. The texture adapter owns acquisition,
// submission and completion under its lock. Do not relocate an active use: its capture guards its owning address.
#ifndef VK_TEXTURE_LIFETIME_H
#define VK_TEXTURE_LIFETIME_H
#include "resource_core_lifetime.h"
#include <stddef.h>

struct TextureMaterialInstance;
typedef struct {
    struct TextureMaterialInstance *instance;
    RcLifetimeCapture capture;
} TextureMaterialUse;

enum { TEXTURE_MATERIAL_UPLOAD_DOMAIN = 1, TEXTURE_MATERIAL_GRAPHICS_DOMAIN = 2 };

// VkImage is supplied by the Vulkan adapter (or the isolated fixture) before including this header.
typedef struct TextureUseBatch TextureUseBatch;
typedef struct TextureImagePublication TextureImagePublication;
TextureImagePublication *vk_texture_publication_create(const VkImage *images, size_t count,
                                                       const TextureImagePublication *previous);
// A descriptor-set owner stores only entries naming ordinary material instances. Update before the descriptor
// write, while the backend permits that set to change. Earlier scene captures remain independent. Sparse index
// growth never moves active captures. Allocation failure is fatal, as for other required publication ownership.
TextureImagePublication *vk_texture_publication_create_indexed(uint32_t entry_limit);
void vk_texture_publication_update(TextureImagePublication *publication, uint32_t entry, VkImage image);
// Start a new selection in a reusable manifest builder. Old exact holds remain pending until the caller either
// transfers the completed selection and commits, or releases the builder after abandoning preparation.
void vk_texture_publication_clear(TextureImagePublication *publication);
// Commit after the queued native descriptor writes finish. Removed instances remain held until then, including
// intermediate A->B->A writes. Release also safely cancels all uncommitted holds when the descriptor owner is gone.
void vk_texture_publication_commit_updates(TextureImagePublication *publication);
void vk_texture_publication_use(TextureImagePublication *publication);
// Capture an already selected manifest into a descriptor-packet owner and the actual recording batch. This
// transfers exact instance holds, never resolves numeric image handles again. The destination is an unindexed
// completion-owned union; an indexed source contributes only its currently selected entries.
void vk_texture_publication_capture(TextureImagePublication *destination, const TextureImagePublication *source);
void vk_texture_publication_add_and_use(TextureImagePublication *publication, const VkImage *images, size_t count);
void vk_texture_publication_release(TextureImagePublication *publication);
int vk_texture_publication_has_retired(TextureImagePublication *publication);
void vk_texture_use_images(const VkImage *images, size_t count);
TextureUseBatch *vk_texture_uses_seal(uint64_t fence, uint64_t upload_submitted);
void vk_texture_uses_handed(TextureUseBatch *batch);
void vk_texture_uses_abandon(TextureUseBatch *batch);
void vk_texture_uses_cancel(uint64_t fence);
void vk_texture_uses_complete(uint64_t fence, uint64_t upload_completed);
// A successful device/queue idle wait also covers handed split parts with no final scene signal. It releases
// physical captures only; it does not advance guest-visible scene completion or execute guest side effects.
void vk_texture_uses_idle(uint64_t last_handed, uint64_t upload_completed);
void vk_texture_uses_cancel_unsubmitted(uint64_t last_handed);
void vk_texture_uses_shutdown(void);
#endif

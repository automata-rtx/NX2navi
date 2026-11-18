// Explicit preparation boundary for CPU-provided sampled textures. All calls require the texture lock.
#pragma once
#include "vk.h"

typedef enum {
    VK_TEXTURE_SAMPLE_READY,
    VK_TEXTURE_SAMPLE_PREPARE,
    VK_TEXTURE_SAMPLE_FALLBACK,
    VK_TEXTURE_SAMPLE_DEFERRED,
    VK_TEXTURE_SAMPLE_UNAVAILABLE
} VkTextureSampleStatus;

typedef struct {
    const VkTexImage *image;
    view_t view;
    int has_view;
} VkTextureSample;

// Query never allocates, uploads, revives an entry or changes its lifetime. A ready result remains valid only
// while the caller holds the texture lock; recording use is explicit and must precede releasing that lock.
VkTextureSampleStatus vk_texture_sample_query_locked(int id, VkTextureSample *out);
// Preparation may allocate/upload/revive. Cold eligible slots enter this path; unrelated resource classes use
// FALLBACK. Deferred work preserves the replay's existing ordered retry signal and is never published as ready.
VkTextureSampleStatus vk_texture_sample_prepare_locked(int id, VkTextureSample *out);
void vk_texture_sample_use_locked(const VkTextureSample *sample);

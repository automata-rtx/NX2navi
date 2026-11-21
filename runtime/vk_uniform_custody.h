#pragma once
#include "vk.h"

// As for the ordinary native-fragment query. The output array owns one reference for each non-NULL entry,
// including when UNSUPPORTED returns the exact winner partition for a disallowed native transfer.
RcRegistryResult vk_usnap_buffer_fragments_with_payloads(const VkMemSpanCapture *, const VkMemSpan *,
    uint64_t address, uint64_t size, RcRegistryPosition,
    const VkMemBufferInlineProof *inline_proofs, uint32_t inline_count,
    RcRegistryBufferFragment *, RcRegistryBufferPayload **payloads, uint32_t capacity, uint32_t *fragment_count);
RcRegistryResult vk_mem_span_buffer_fragments_with_payloads(const VkMemSpanCapture *, const VkMemSpan *,
    uint64_t address, uint64_t size, RcRegistryGpuCapture, RcRegistryPosition,
    const VkMemBufferInlineProof *inline_proofs, uint32_t inline_count,
    RcRegistryBufferFragment *, RcRegistryBufferPayload **payloads, uint32_t capacity, uint32_t *fragment_count);

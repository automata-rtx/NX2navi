#ifndef VK_UNIFORM_PAYLOAD_H
#define VK_UNIFORM_PAYLOAD_H

#include "vk.h"
#include "vk_recording_part.h"

typedef struct VkUniformPayload VkUniformPayload;

// Copy the immutable inline-update source and its complete normalized destination extents.
// The caller supplies the physical location of the first byte and the authored output.
// Success takes the producer part reference and clears the caller slot.
VkResult vk_uniform_payload_create(uint64_t address, uint64_t bytes, const void *data,
    const RcInterval *extents, uint32_t count, RcRegistryBufferLocation native_base,
    RcRegistryPosition output, VkRecordingPartDisposition **part, VkUniformPayload **out);

// Acquire while another owner keeps the object alive, for example under the custody lock.
// A borrowed pointer must stay protected until retain returns.
VkUniformPayload *vk_uniform_payload_retain(VkUniformPayload *payload);
// Last release only queues retirement. Reap outside registry and recording-part locks.
void vk_uniform_payload_release(VkUniformPayload *payload);
void vk_uniform_payload_reap(void);
int vk_uniform_payload_handed(const VkUniformPayload *payload);

// The producer part must have reached native handoff. The caller supplies a current, authoritative
// winner for the consuming read and retains the payload until the returned bytes have been copied.
// Another guest alias may use those same bytes.
const uint8_t *vk_uniform_payload_match(const VkUniformPayload *payload, RcInterval demand,
    RcRegistryBufferLocation native, RcRegistryPosition output);

RcRegistryBufferPayload *vk_uniform_payload_registry(VkUniformPayload *payload);
const uint8_t *vk_uniform_payload_registry_match(const RcRegistryBufferPayload *payload, RcInterval demand,
    RcRegistryBufferLocation native, RcRegistryPosition output);

#endif

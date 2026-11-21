#include "vk_uniform_payload.h"
#include "host.h"
#include <string.h>

struct VkUniformPayload {
    RcRegistryBufferPayload registry;
    RcBudgetTicket budget;
    VkRecordingPartDisposition *part;
    VkUniformPayload *retired_next;
    RcRegistryBufferLocation native_base;
    RcRegistryPosition output;
    uint64_t address, bytes;
    uint32_t refs, count;
    RcInterval extents[];
};

static VkUniformPayload *uniform_payload_retired;

static int uniform_payload_normalized(RcInterval interval) {
    return interval.mapping && interval.range.backing != RC_GUEST_COMPATIBILITY_BACKING &&
        rc_range_valid(interval.range) && interval.range.size <= UINT64_MAX - interval.address;
}

static void uniform_payload_registry_retain(RcRegistryBufferPayload *ref) {
    if (!vk_uniform_payload_retain((VkUniformPayload *)ref))
        rt_fatal("vk: uniform payload reference exhausted");
}

static void uniform_payload_registry_release(RcRegistryBufferPayload *ref) {
    vk_uniform_payload_release((VkUniformPayload *)ref);
}

VkResult vk_uniform_payload_create(uint64_t address, uint64_t bytes, const void *data,
    const RcInterval *extents, uint32_t count, RcRegistryBufferLocation native_base,
    RcRegistryPosition output, VkRecordingPartDisposition **part, VkUniformPayload **out) {
    if (!out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = NULL;
    if (!address || !bytes || bytes > UINT64_MAX - address || !data || !extents || !count || count > bytes ||
        !part || !*part ||
        !native_base.allocation || bytes > UINT64_MAX - native_base.offset ||
        !rc_registry_position_valid(output) || !(output.ordinal & 1))
        return VK_ERROR_INITIALIZATION_FAILED;
    size_t extent_bytes, metadata;
    if (__builtin_mul_overflow((size_t)count, sizeof(RcInterval), &extent_bytes) ||
        __builtin_add_overflow(sizeof(VkUniformPayload), extent_bytes, &metadata))
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (bytes > SIZE_MAX - metadata) return VK_ERROR_OUT_OF_HOST_MEMORY;

    const uint64_t end = address + bytes;
    uint64_t covered = address;
    for (uint32_t i = 0; i < count; ++i) {
        const RcInterval extent = extents[i];
        if (!uniform_payload_normalized(extent) || extent.address != covered ||
            extent.range.size > end - covered) return VK_ERROR_INITIALIZATION_FAILED;
        covered += extent.range.size;
    }
    if (covered != end) return VK_ERROR_INITIALIZATION_FAILED;

    RcBudgetTicket budget = {0};
    void *storage;
    const VkResult result = vk_replay_metadata_allocate(metadata + (size_t)bytes, 0, RC_BUDGET_SNAPSHOT,
        &budget, &storage);
    if (result != VK_SUCCESS) return result;
    VkUniformPayload *payload = storage;
    *payload = (VkUniformPayload){.registry = {uniform_payload_registry_retain, uniform_payload_registry_release},
        .budget = budget, .part = *part, .native_base = native_base, .output = output,
        .address = address, .bytes = bytes, .refs = 1, .count = count};
    memcpy(payload->extents, extents, extent_bytes);
    memcpy(payload->extents + count, data, (size_t)bytes);
    *part = NULL;
    *out = payload;
    return VK_SUCCESS;
}

VkUniformPayload *vk_uniform_payload_retain(VkUniformPayload *payload) {
    if (!payload) return NULL;
    uint32_t refs = __atomic_load_n(&payload->refs, __ATOMIC_RELAXED);
    while (refs && refs != UINT32_MAX) {
        if (__atomic_compare_exchange_n(&payload->refs, &refs, refs + 1, 1,
                __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return payload;
    }
    return NULL;
}

void vk_uniform_payload_release(VkUniformPayload *payload) {
    if (!payload || __atomic_fetch_sub(&payload->refs, 1, __ATOMIC_ACQ_REL) != 1) return;
    VkUniformPayload *head = __atomic_load_n(&uniform_payload_retired, __ATOMIC_RELAXED);
    do {
        payload->retired_next = head;
    } while (!__atomic_compare_exchange_n(&uniform_payload_retired, &head, payload, 1,
                                          __ATOMIC_RELEASE, __ATOMIC_RELAXED));
}

void vk_uniform_payload_reap(void) {
    VkUniformPayload *payload = __atomic_exchange_n(&uniform_payload_retired, NULL, __ATOMIC_ACQUIRE);
    while (payload) {
        VkUniformPayload *next = payload->retired_next;
        RcBudgetTicket budget = payload->budget;
        vk_recording_part_disposition_release(payload->part);
        vk_replay_metadata_free(payload, &budget);
        payload = next;
    }
}

int vk_uniform_payload_handed(const VkUniformPayload *payload) {
    return payload && vk_recording_part_disposition_handed(payload->part);
}

const uint8_t *vk_uniform_payload_match(const VkUniformPayload *payload, RcInterval demand,
    RcRegistryBufferLocation native, RcRegistryPosition output) {
    if (!vk_uniform_payload_handed(payload) || !uniform_payload_normalized(demand) ||
        rc_registry_position_compare(output, payload->output) ||
        native.allocation != payload->native_base.allocation ||
        native.incarnation != payload->native_base.incarnation ||
        native.offset < payload->native_base.offset) return NULL;
    const uint64_t delta = native.offset - payload->native_base.offset;
    const RcSourceRange range = demand.range;
    if (delta >= payload->bytes || range.size > payload->bytes - delta) return NULL;

    const uint64_t end = range.offset + range.size;
    uint64_t covered = range.offset;
    for (uint32_t i = 0; i < payload->count; ++i) {
        const RcInterval extent = payload->extents[i];
        if (!rc_range_overlap(extent.range, range)) continue;
        const uint64_t start = extent.range.offset > range.offset ? extent.range.offset : range.offset;
        const uint64_t limit = extent.range.offset + extent.range.size;
        const uint64_t from = extent.address - payload->address + (start - extent.range.offset);
        // Every overlapping canonical byte must identify just one affine location in the copied write.
        if (start != covered || from != delta + (start - range.offset)) return NULL;
        covered = limit < end ? limit : end;
    }
    const uint8_t *data = (const uint8_t *)(payload->extents + payload->count);
    return covered == end ? data + (size_t)delta : NULL;
}

RcRegistryBufferPayload *vk_uniform_payload_registry(VkUniformPayload *payload) {
    return payload ? &payload->registry : NULL;
}

const uint8_t *vk_uniform_payload_registry_match(const RcRegistryBufferPayload *ref, RcInterval demand,
    RcRegistryBufferLocation native, RcRegistryPosition output) {
    if (!ref || ref->retain != uniform_payload_registry_retain || ref->release != uniform_payload_registry_release)
        return NULL;
    return vk_uniform_payload_match((const VkUniformPayload *)ref, demand, native, output);
}

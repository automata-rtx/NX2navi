#pragma once
#include "resource_core_registry.h"
#include "vk.h"

typedef struct {
    RcInterval *intervals;
    RcRegistryWrite *tickets;
    RcRegistryWrite guard;
    uint32_t count;
} ResourceHostWrite;

// A vector is owned by the caller and must cover the whole requested address span. Zero means unavailable.
uint32_t resource_host_interval_slice(const RcInterval *, uint32_t count, uint64_t address, uint64_t size, RcInterval **);
uint32_t resource_host_current_intervals(uint64_t address, uint64_t size, RcInterval **);
uint32_t vk_mem_host_intervals(const VkMemSpan *, uint64_t address, uint64_t size, RcInterval **);
uint32_t vk_mem_device_intervals(const VkMemSpan *, uint64_t address, uint64_t size, RcInterval **);
uint32_t vk_mem_device_writer_intervals(const VkMemSpan *, uint64_t address, uint64_t size, RcInterval **);
static inline uint32_t vk_mem_writer_intervals(const VkMemSpan *span, uint64_t address, uint64_t size, RcInterval **out) {
    return span && span->alloc ? vk_mem_device_writer_intervals(span, address, size, out) :
                                vk_mem_host_intervals(span, address, size, out);
}
struct MemBackingCapture;
typedef enum VkMemCanonicalSourceQualificationStage {
    VK_MEM_CANONICAL_SOURCE_INVALID, VK_MEM_CANONICAL_SOURCE_GPU_ONLY,
    VK_MEM_CANONICAL_SOURCE_LEASE_BEFORE, VK_MEM_CANONICAL_SOURCE_PROVIDER_MISSING,
    VK_MEM_CANONICAL_SOURCE_PROVIDER_UNPUBLISHED, VK_MEM_CANONICAL_SOURCE_PROVIDER_KIND,
    VK_MEM_CANONICAL_SOURCE_PROVIDER_BUFFER, VK_MEM_CANONICAL_SOURCE_PROVIDER_EXTENT,
    VK_MEM_CANONICAL_SOURCE_PROVIDER_HOST, VK_MEM_CANONICAL_SOURCE_VECTOR,
    VK_MEM_CANONICAL_SOURCE_LEASE_AFTER, VK_MEM_CANONICAL_SOURCE_EXACT
} VkMemCanonicalSourceQualificationStage;
typedef struct {
    VkMemCanonicalSourceQualificationStage stage;
    int provider_kind;
    uint32_t source_count, provider_count;
    uint64_t provider_publication, provider_address, provider_bytes;
    uint64_t provider_import_lo, provider_import_hi, provider_mapping_cutoff;
    uint64_t mismatch_address;
    RcInterval source_interval, provider_interval;
} VkMemCanonicalSourceQualification;
// Qualifies an ordinary tracked import and its retained mapping. Source permission still comes from the registry.
RcRegistryResult vk_mem_canonical_source_qualify(const struct MemBackingCapture *, uint64_t address, uint64_t size);
// Optional report copies the original pool selection and first vector conflict. A zero interval means absent coverage.
RcRegistryResult vk_mem_canonical_source_qualify_report(const struct MemBackingCapture *, uint64_t address, uint64_t size,
    VkMemCanonicalSourceQualification *);
// Qualify the exact range of an expected retained native import. The source copy still needs its own read hold
// and writer exclusion. ABSENT source coverage cannot establish authority for this import.
RcRegistryResult vk_mem_canonical_source_qualify_expected(const struct MemBackingCapture *, const VkMemSpanCapture *);
int vk_mem_recorded_write_mirrored(const VkMemSpan *);
void *vk_mem_recorded_provider_acquire(const VkMemSpan *);
void vk_mem_recorded_provider_release(void *);
void vk_mem_buffer_write_completed(void *provider, VkBuffer buffer, uint64_t address, uint64_t size, int mirror);
void vk_mem_cpu_result_written(uint64_t address, uint64_t size);

// Adopt the vector and reserve an unpositioned fallback when no ordered declaration covers the actual destination.
// An empty vector uses a canonical-only unknown guard when this operation may reach guest CPU storage.
ResourceHostWrite resource_host_write_prepare(RcInterval *, uint32_t count, RcRegistryWriterKind, int declared);
void resource_host_write_dispose(ResourceHostWrite *, uint64_t completed_fence, int may_have_executed, int cpu_visible);

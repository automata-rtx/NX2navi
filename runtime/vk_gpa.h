// The device address table (vk_gpa.c): GPU address pages to buffer device addresses for shaders' global memory accesses.
#pragma once
#include <stdint.h>

#include "vk.h"
#include "resource_core_backing.h"
#include "resource_core_registry.h"

#define VK_GPA_CB_SLOT 0x40   // res.json cb_slot of the table (translator ResourceManager.DeviceAddressTableCbSlot)

int vk_gpa_wanted(void);                   // the device has buffer device addresses and NX2NAVI_GPA is not 0
int vk_gpa_wanted_ctx(const VkCtx *c);     // the same during device creation, before vk_ctx() returns it
typedef enum { VK_GPA_READY, VK_GPA_DISABLED, VK_GPA_NO_CAPACITY, VK_GPA_ERROR } VkGpaPrepareResult;
// Serialized device startup, before publication and before shader host capabilities are selected.
VkGpaPrepareResult vk_gpa_prepare(VkResult *failure);
int vk_gpa_on(void);                       // the prepared table exists; does not allocate
uint64_t vk_gpa_device_address(VkBuffer b);
VkDescriptorBufferInfo vk_gpa_info(void);  // the whole table, for a storage descriptor
// Maps the pages [lo, hi) holds whole to buf, whose byte 0 is GPU address buf_lo. Returns the record (0: not mapped).
uint32_t vk_gpa_map(uint64_t lo, uint64_t hi, VkBuffer buf, uint64_t buf_lo, int cpu_visible_after_fence,
                    const RcInterval *host, uint32_t host_count);
typedef struct {
    uint64_t lo, hi;
    VkBuffer buffer;
    RcSourceRange backing;
    uint64_t delta;
} VkGpaVirtualBinding;
typedef struct {
    const VkGpaVirtualBinding *bindings;
    uint32_t count;
} VkGpaVirtualUpdate;
// Called under the pool topology lock. Successful prepare keeps native admission exclusive until commit/cancel.
// A disabled table admits an inert update. Keep bindings alive and unchanged until the update is released.
RcBackingResult vk_gpa_virtual_prepare(VkGpaVirtualBinding *, uint32_t count, VkGpaVirtualUpdate *);
void vk_gpa_virtual_commit(VkGpaVirtualUpdate *);
void vk_gpa_virtual_cancel(VkGpaVirtualUpdate *);
void vk_gpa_virtual_unmap(uint64_t lo, uint64_t hi);
// Unmaps the pages of [lo, hi) that name rec; cover (may be NULL) gives the record of another live buffer that holds a
// page whole, or 0. rec is reused once the GPU finished the submissions issued until now.
void vk_gpa_unmap(uint64_t lo, uint64_t hi, uint32_t rec, uint32_t (*cover)(uint64_t page_addr));
// Replace one whole import projection, optionally retaining the original import as its write twin.
// Return the new record on success or zero with the old projection intact. Caller owns native queue ordering.
uint32_t vk_gpa_remap_exact(uint64_t lo, uint64_t hi, uint32_t old_rec, VkBuffer buf, uint64_t buf_lo,
                            int keep_twin, int cpu_visible_after_fence, int *uses_twin);
// A physical-store command captures all reachable mappings. Additions to the mutable table join every live scope
// before their page entries become visible. End only at command cancellation or actual submission completion.
typedef struct VkGpaWriteScope VkGpaWriteScope;
typedef struct { uint64_t addr, size; } VkGpaEffectRange;
typedef struct {
    uint64_t program;
    uint32_t opcode, stage, set, binding;
    uint8_t shader_key[32]; // selected shader if its reflection supplied the actual table descriptor
} VkGpaWriteOrigin;
typedef struct VkGpaPrefixRequest {
    RcRegistryWrite guard;
    RcRegistryPosition producer, read;
    RcInterval source;
} VkGpaPrefixRequest;
VkGpaWriteScope *vk_gpa_write_begin(void);
VkGpaWriteScope *vk_gpa_write_begin_at(const VkGpaEffectRange *ranges, uint32_t count, int complete, RcRegistryPosition position);
// Complete proven guest-address effects, not descriptor extents. Mapping additions retain only their intersection
// in guest address space. The backing owner covers any canonical CPU address. Invalid/unproved ranges use the whole domain.
VkGpaWriteScope *vk_gpa_write_begin_ranges(const VkGpaEffectRange *ranges, uint32_t count);
void vk_gpa_write_cancel(VkGpaWriteScope *scope);
void vk_gpa_write_abandon(VkGpaWriteScope *scope, int may_have_executed);
void vk_gpa_write_complete(VkGpaWriteScope *scope, uint64_t fence);
void vk_gpa_write_note_recorded(VkGpaWriteScope *, const VkGpaWriteOrigin *);
// Optional donor proof from an already recorded, fully matched native primary/twin store plan.
void vk_gpa_write_record_definite(VkGpaWriteScope *, const VkGpaEffectRange *, uint32_t count);
void vk_gpa_report_write_blocker(RcRegistryWrite, RcSourceRange);
int vk_gpa_write_prefix_ready(const VkGpaPrefixRequest *);
int vk_gpa_write_scene_may_need_prefix(uint64_t scene);
// The caller's actual private queue completion includes the producer and an explicit host visibility barrier.
int vk_gpa_write_prefix_complete(const VkGpaPrefixRequest *);
void vk_gpa_report(void);
void vk_gpa_effects_report(uint64_t frame);
void vk_gpa_shutdown(void);

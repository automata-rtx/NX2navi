// The mesh fallback's buffers and its records (vk_mesh_fb.c); the modules are mesh_fb.c's, the draws
// vk_exec_draw.c's ("mesh fallback").
#pragma once
#include "resource_core_vma.h"

typedef enum {
    VK_MESHFB_READY, VK_MESHFB_PENDING, VK_MESHFB_STOPPED, VK_MESHFB_UNSUPPORTED, VK_MESHFB_TERMINAL,
} VkMeshFbResult;
typedef struct {
    VkResult error;
    const char *reason;
    RcAdmissionStatus admission;
    RcAdmissionVector peak;
    RcBudgetReadiness before;
    RcVmaDependency child;
} VkMeshFbFailure;
// Without allow_start this only looks up readiness. The sole real producer
// attempts pending prerequisites from outside the recorder and provider locks.
VkMeshFbResult vk_meshfb_prepare(int allow_start, VkMeshFbFailure *failure);
// The buffer a module's extra storage binding reads or writes (mesh_fb.h MESHFB_CB_*), whole or as much of it as one
// storage binding may span.
VkDescriptorBufferInfo vk_meshfb_buffer(int cb_slot);
uint64_t vk_meshfb_mesh_words(void);   // the mesh buffer's size as a binding sees it, in words

enum { VK_MESHFB_ARGUMENT_TASK = 1, VK_MESHFB_ARGUMENT_PROBE_PARTIAL = 2 };
typedef struct {
    uint64_t scene, ordinal;
    uint32_t flags, task_words, task_fan, slot_words, vertices;
} VkMeshFbArguments;
// Exact native outer-dispatch limit for a Ready provider. Scene/ordinal are unnecessary before recording.
uint32_t vk_meshfb_requested_limit(const VkMeshFbArguments *args);

typedef enum {
    VK_MESHFB_CAPACITY_LIMIT_NONE, VK_MESHFB_CAPACITY_LIMIT_TASK_COUNT,
    VK_MESHFB_CAPACITY_LIMIT_MESH_COUNT, VK_MESHFB_CAPACITY_LIMIT_TASK_PACKING,
    VK_MESHFB_CAPACITY_LIMIT_EXPANSION,
} VkMeshFbCapacityLimit;
typedef enum {
    VK_MESHFB_CAPACITY_READY, VK_MESHFB_CAPACITY_OVERFLOW, VK_MESHFB_CAPACITY_STOPPED, VK_MESHFB_CAPACITY_UNPROVEN,
} VkMeshFbCapacityResult;
typedef struct {
    uint64_t scene, ordinal, requested_mesh;
    uint32_t requested_outer, available, detail;
    VkMeshFbCapacityLimit limit;
    VkResult error;
} VkMeshFbCapacity;

// Initialize the required, first-failure report once per actual transformed scene.
// Returns one recorded reset command, zero for an already open scene, or -1 for an owner mismatch.
int vk_meshfb_capacity_begin(VkCommandBuffer cb, uint64_t scene);
// A 64-byte queue-ordered record. The original outer count is copied or written exactly once.
// Normalize before the first indirect dispatch. A task draw assembles again after its actual task dispatch.
// A full replay barrier separates the record, normalization, each actual dispatch and assembly.
VkDeviceSize vk_meshfb_record(VkCommandBuffer cb, VkBuffer src, VkDeviceSize src_off, uint32_t requested,
                              const VkMeshFbArguments *args, VkBuffer *rec, int *cmds);
void vk_meshfb_normalize(VkCommandBuffer cb, VkDeviceSize at, const VkMeshFbArguments *args);
void vk_meshfb_scan(VkCommandBuffer cb, VkDeviceSize at, const VkMeshFbArguments *args);
// Copy and make host-visible the required outcome. Returns three commands, zero for no transformed
// work, or -1 on an owner mismatch. The owner ends rendering before recording this copy.
int vk_meshfb_capacity_copy(VkCommandBuffer cb, uint64_t scene);
// Read only after the native replay prefix containing this copy has completed successfully.
// The current scene keeps the mapped row until the exact GPU cookie validates or shutdown settles it.
VkMeshFbCapacityResult vk_meshfb_capacity_read(uint64_t scene, VkMeshFbCapacity *out);
// Copy the draw's report for an actual completed recording result. Returns two
// commands for a copy and host barrier, or zero while that report row is still owned.
int vk_meshfb_note_for_scene(VkCommandBuffer cb, VkDeviceSize at, uint64_t scene);
// Count through the last completed reports, after the recorder is drained.
uint64_t vk_meshfb_short(void);
// Diagnostic probe, after replay drain or from the active recorder.
unsigned vk_meshfb_probe_pending_reports(void);
// After replay and completion drain, before allocator teardown. Failure keeps
// all possibly executing resources with the current owner.
VkResult vk_meshfb_shutdown(void);

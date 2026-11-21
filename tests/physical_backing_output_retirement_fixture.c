#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "resource_core_backing.h"
#define rc_registry_runtime standalone_registry_runtime
#include "resource_core_registry.c"
#undef rc_registry_runtime
#include "nvn_hmap.h"

#define OK(call) assert((call) == 0)

typedef uint64_t VkImage, VkImageView, VkBuffer, VkDeviceMemory, VkDeviceSize;
typedef void *VmaAllocation; typedef uint64_t VmaAllocator;
#define VK_NULL_HANDLE 0
typedef struct { uint64_t device; VmaAllocator vma; uint64_t timeline; } VkCtx;
typedef struct {
    RcBackingId backing;
    uint32_t native_refs;
    VkBuffer buf, mbuf;
    VkDeviceMemory mem;
    VmaAllocation alloc, malloc_, salloc;
    void *host_backing, *physical_raw_uses;
} PoolRec;
static VkCtx context = {1, 2, 3};
static RcBackingStore *pool_backings;
static RcRegistry *registry;
RcRegistry *rc_registry_runtime(void) { return registry; }

typedef int PlatMutex, PlatRMutex, PlatOnce;
static PlatMutex pool_lock;
static PlatRMutex cmd_lock;
static PlatOnce cmd_once;
static void plat_mutex_lock_shared(PlatMutex *lock) { assert(!(*lock)++); }
static void plat_mutex_unlock_shared(PlatMutex *lock) { assert((*lock)-- == 1); }
static void plat_rmutex_lock(PlatRMutex *lock) { ++*lock; }
static void plat_rmutex_unlock(PlatRMutex *lock) { assert((*lock)-- > 0); }
static void cmd_init(void *unused) { (void)unused; }
static void plat_once(PlatOnce *once, void (*cb)(void *), void *arg) { if (!*once) cb(arg), *once = 1; }
static void rt_fatal(const char *message) { fprintf(stderr, "%s\n", message); abort(); }

static unsigned original_pool_buffer_destroyed, original_pool_memory_freed, image_freed, image_gone;
static unsigned allocation_retired, metadata_live, publication_reclaim, budget_progress;
static RcBackingId original_identity;
int vk_mem_device_backing_reclaimed(RcBackingId *);
static void vkDestroyBuffer(uint64_t device, VkBuffer buffer, void *cb) {
    assert(device == 1 && buffer == 21 && !cb); ++original_pool_buffer_destroyed;
}
static void resource_core_vma_free_memory(VmaAllocator vma, VmaAllocation allocation) {
    assert(vma == 2 && allocation == (VmaAllocation)(uintptr_t)22 && original_pool_buffer_destroyed == 1);
    assert(!vk_mem_device_backing_reclaimed(&original_identity));
    ++original_pool_memory_freed;
}
static void vkDestroyImage(uint64_t device, VkImage image, void *cb) { (void)device; (void)image; (void)cb; abort(); }
static void vkFreeMemory(uint64_t device, VkDeviceMemory mem, void *cb) { (void)device; (void)mem; (void)cb; abort(); }
static void resource_core_vma_destroy_buffer(VmaAllocator vma, VkBuffer buffer, VmaAllocation allocation) {
    (void)vma; (void)buffer; (void)allocation; abort();
}
static void vkDestroyImageView(uint64_t device, VkImageView view, void *cb) { (void)device; (void)view; (void)cb; abort(); }
static void resource_core_vma_destroy_image(VmaAllocator vma, VkImage image, VmaAllocation allocation) {
    assert(vma == 2 && image == 100 && allocation == (VmaAllocation)(uintptr_t)51 && original_pool_memory_freed == 1);
    ++image_freed;
}
static void shadow_metadata_destroy(PoolRec *r) { assert(!r->native_refs); }
static void pool_authority_end(PoolRec *r) { assert(!r->native_refs); }
static void mem_backing_release(void **host_backing) { assert(!*host_backing); }

#include "dead_types.h"

typedef struct { VkImage image; uint64_t addr; } VkTexImage;
typedef struct { int unused; } nvn_texdesc;
typedef struct { int unused; } TexVkFormat;
typedef struct { int live; } RcBudgetTicket;
typedef struct NativeTextureOutput NativeTextureOutput;
typedef struct Tex {
    VkTexImage pub;
    struct { nvn_texdesc d; struct { uint64_t size; } L; } t;
    TexVkFormat f;
    uint64_t bytes;
    VmaAllocation alloc;
    RcInterval *image_private_mapping;
    uint32_t image_private_mapping_count;
    RcRegistryWrite *image_mapped_holds;
    uint32_t image_mapped_hold_count;
    int bound_to, material, pin_owned, alias_fragment_owner;
    NativeTextureOutput *native_output;
} Tex;
static nvn_hmap image_holds = {.klen = sizeof(VkImage)};
static uint32_t image_readers;
static uint64_t next_seq = 7, done_seq, tl_noted, timeline_done;
static Dead *dead;
static int ndead, cap_dead, npwait, open_active;
static struct { uint64_t seq; } open_b;
enum { VK_SUCCESS, RC_BUDGET_TEXTURE, RC_ADMISSION_OK = 0, REC_IMAGE, REC_VIEW, REC_BUFFER };
static int vk_replay_metadata_allocate(size_t bytes, int clear, int producer, RcBudgetTicket *ticket, void **out) {
    assert(clear && producer == RC_BUDGET_TEXTURE);
    *out = calloc(1, bytes);
    assert(*out);
    ++metadata_live;
    ticket->live = 1;
    return VK_SUCCESS;
}
static void vk_replay_metadata_free(void *allocation, RcBudgetTicket *ticket) {
    assert(ticket->live && metadata_live);
    --metadata_live;
    free(allocation);
}
static void vk_exec_image_gone(VkImage image) { assert(image == 100); ++image_gone; }
static void vk_exec_texture_publications_request_reclaim(void) { ++publication_reclaim; }
static void resource_core_budget_progress(void) { ++budget_progress; }
static int resource_core_vma_retire(VmaAllocation allocation) { assert(allocation == (VmaAllocation)(uintptr_t)51); ++allocation_retired; return 0; }
static void grow(void **p, int *capacity, size_t element) { *capacity += 8; *p = realloc(*p, *capacity * element); assert(*p); }
static int rt_knob(const char *name, const char *value) { assert(!strcmp(name,"DEFER_STAMP") && !strcmp(value,"1")); return 1; }
static uint64_t vk_exec_recording_fence(void) { return 12; }
static uint64_t vk_sync_issued(void) { return 11; }
static uint64_t vk_sync_fence_done(void) { return timeline_done; }
static void rec_check_handle(int kind, uint64_t handle, int destroyed, uint64_t tl, uint64_t completed) {
    assert(cmd_lock && kind == REC_IMAGE && handle == 100 && tl == 12);
    if (destroyed) assert(completed >= tl);
}
static void image_placement_remove(VkMemoryImageHold *hold) { (void)hold; abort(); }
static void retire(VkCtx *c, int wait) { assert(c == &context && !wait && cmd_lock); }
static void pool_wait_collect(uint64_t tl) { (void)tl; abort(); }
static int vkGetSemaphoreCounterValue(uint64_t device, uint64_t timeline, uint64_t *value) {
    assert(device == 1 && timeline == 3); *value = timeline_done; return VK_SUCCESS;
}

#include "extract.h"

static RcRegistryPosition at(uint64_t ordinal) { return (RcRegistryPosition){1, ordinal}; }

int main(void) {
    const uint64_t virtual_address = 0x180000;
    const VkImage physical_image = 100;
    registry = rc_registry_create((RcRegistryConfig){32, 32, 4});
    pool_backings = rc_backing_store_create((RcBackingConfig){2, 2, 4, 2, 4});
    PoolRec *old = calloc(1, sizeof *old);
    assert(registry && pool_backings && old);
    *old = (PoolRec){.native_refs = 1, .buf = 21, .alloc = (VmaAllocation)(uintptr_t)22};
    OK(rc_backing_create(pool_backings, 128, (uintptr_t)old, &old->backing));
    original_identity = (RcBackingId){.id = old->backing.id};
    RcBackingMapId window;
    OK(rc_backing_map_create(pool_backings, virtual_address, 128, &window));
    OK(rc_backing_map_replace(pool_backings, window, virtual_address, 64, old->backing, 0));
    OK(rc_backing_map_replace(pool_backings, window, virtual_address + 64, 64, old->backing, 64));
    RcBackingCapture *capture;
    OK(rc_backing_capture(pool_backings, window, virtual_address, 128, RC_BACKING_REQUIRE_MAPPED, &capture));
    pool_native_acquire(old);
    Tex logical = {.pub = {physical_image, virtual_address}, .bytes = 8192, .alloc = (VmaAllocation)(uintptr_t)51,
                   .image_private_mapping_count = 2, .image_mapped_hold_count = 2};
    logical.image_private_mapping = malloc(2 * sizeof *logical.image_private_mapping);
    logical.image_mapped_holds = calloc(2, sizeof *logical.image_mapped_holds);
    assert(logical.image_private_mapping && logical.image_mapped_holds);
    uint32_t count;
    const RcBackingSegment *segments = rc_backing_capture_segments(capture, &count);
    assert(count == 2);
    for (uint32_t i = 0; i < count; ++i)
        logical.image_private_mapping[i] = (RcInterval){segments[i].address, segments[i].mapping, segments[i].range};
    const RcInterval lookup = logical.image_private_mapping[0];
    OK(rc_registry_image_hold_begin_extents(registry, physical_image, logical.image_private_mapping, count,
                                            logical.image_mapped_holds));
    const RcRegistryImageBinding binding = {physical_image, logical.image_mapped_holds[0]};
    const RcRegistryImageBinding second_binding = {physical_image, logical.image_mapped_holds[1]};
    RcRegistryNativeImageWrite extents[] = {{{lookup, binding.hold}, &lookup.range, 1},
        {{logical.image_private_mapping[1], second_binding.hold}, NULL, 0}};
    RcRegistryWrite producer;
    OK(rc_registry_write_begin_at(registry, lookup.range, RC_REGISTRY_IMAGE, at(3), &producer));
    OK(rc_registry_image_producer_add(registry, producer, at(3)));
    OK(rc_registry_declare_through(registry, at(5)));
    OK(rc_registry_native_image_possible(registry, physical_image, extents, 2));
    OK(rc_registry_native_image_write_fragments(registry, physical_image, extents, 1, at(3)));
    OK(rc_registry_image_producer_resolve(registry, producer, at(3), &binding, 0));
    OK(rc_registry_handoff_through(registry, at(3)));
    rc_registry_image_producer_native_completed(registry, producer, at(3), binding);
    OK(rc_registry_write_cancel(registry, producer));
    assert(rc_registry_native_image_covers(registry, physical_image, &lookup, 1, at(4)));
    VkMemoryImageHold *reader = vk_mem_image_hold(physical_image);
    assert(native_texture_output_reserve(&logical));
    logical.native_output->possible = 1;
    native_texture_output_publish(&logical);
    assert(native_texture_output_transfer(&logical));
    native_texture_output_collect(32);
    assert(metadata_live == 1 && !allocation_retired && !image_freed);

    // Losing the alias and finalizing the pool do not settle an existing native owner.
    OK(rc_backing_map_replace(pool_backings, window, virtual_address, 128, (RcBackingId){0}, 0));
    native_texture_output_collect(32);
    assert(metadata_live == 1 && !allocation_retired);
    OK(rc_backing_finalize(pool_backings, old->backing));
    destroy_rec(&context, old);
    assert(old->native_refs == 1 && !original_pool_memory_freed);
    native_texture_output_collect(32);
    assert(metadata_live == 1 && !allocation_retired && rc_registry_image_binding_current(registry, binding));
    assert(!vk_mem_device_backing_reclaimed(&original_identity));
    assert(original_identity.slot == old->backing.slot);
    rc_backing_capture_release(&capture);
    const RcBackingId previous = old->backing;
    destroy_rec(&context, old);
    assert(original_pool_memory_freed == 1);

    // The identical VA and recycled slab slot now select an unrelated backing incarnation.
    RcBackingId replacement;
    OK(rc_backing_create(pool_backings, 128, 9, &replacement));
    assert(replacement.slot == previous.slot && replacement.id != previous.id);
    OK(rc_backing_map_replace(pool_backings, window, virtual_address, 128, replacement, 0));
    RcInterval fresh;
    OK(rc_backing_map_intervals(pool_backings, window, virtual_address, 64, &fresh, 1, &count));
    assert(count == 1 && fresh.range.backing == replacement.id);
    RcRegistryImageBinding replacement_binding = {.image = 101};
    OK(rc_registry_image_hold_begin_extents(registry, replacement_binding.image, &fresh, 1, &replacement_binding.hold));
    assert(vk_mem_device_backing_reclaimed(&original_identity));
    assert(rc_registry_native_image_discard_reclaimed(registry, physical_image, replacement.id, &binding.hold, 1) == RC_REGISTRY_STALE);
    assert(rc_registry_image_binding_current(registry, binding));
    // Even with a valid dead ID, a mixed vector cannot consume either backing's authority.
    RcRegistryWrite mixed[2] = {binding.hold};
    OK(rc_registry_image_hold_begin_extents(registry, physical_image, &fresh, 1, &mixed[1]));
    assert(rc_registry_native_image_discard_reclaimed(registry, physical_image, previous.id, mixed, 2) == RC_REGISTRY_STALE);
    assert(rc_registry_image_binding_current(registry, binding));
    assert(rc_registry_image_binding_current(registry, (RcRegistryImageBinding){physical_image, mixed[1]}));
    OK(rc_registry_hold_release(registry, mixed[1], 1));
    native_texture_output_collect(32);
    assert(!metadata_live && allocation_retired == 1 && ndead == 1 && image_gone == 1);
    assert(!rc_registry_image_binding_current(registry, binding));
    assert(!rc_registry_image_binding_current(registry, second_binding));
    assert(rc_registry_image_binding_current(registry, replacement_binding));
    assert(rc_registry_stats(registry).writer_foreign == 0);
    assert(vk_mem_image_hold_retired(reader) && publication_reclaim == 1 && !native_outputs_destroyed);
    assert(reclaim_pending_bytes == 8192);
    // Retention follows the reader plus actual upload and scene progress.
    VkMemoryImageHold *inherited = vk_mem_image_hold_retain_exact(reader, physical_image);
    assert(inherited == reader);
    done_seq = 7;
    timeline_done = 12;
    plat_rmutex_lock(&cmd_lock);
    collect_locked_out(&context, NULL, NULL, NULL);
    plat_rmutex_unlock(&cmd_lock);
    assert(ndead == 1 && !image_freed);
    vk_mem_image_release(reader);
    vk_mem_image_release(inherited);
    assert(budget_progress == 1 && !image_readers);
    done_seq = 6;
    plat_rmutex_lock(&cmd_lock);
    collect_locked_out(&context, NULL, NULL, NULL);
    assert(ndead == 1 && !image_freed);
    done_seq = 7;
    timeline_done = 11;
    collect_locked_out(&context, NULL, NULL, NULL);
    assert(ndead == 1 && !image_freed);
    timeline_done = 12;
    collect_locked_out(&context, NULL, NULL, NULL);
    plat_rmutex_unlock(&cmd_lock);
    assert(!ndead && image_freed == 1 && native_outputs_destroyed == 1 && !reclaim_pending_bytes);
    OK(rc_registry_hold_release(registry, replacement_binding.hold, 1));
    assert(!rc_registry_stats(registry).writers);
    rc_registry_destroy(registry);
    OK(rc_backing_finalize(pool_backings, replacement));
    uintptr_t provider;
    OK(rc_backing_reclaim(pool_backings, replacement, &provider));
    assert(provider == 9);
    OK(rc_backing_map_close(pool_backings, window));
    OK(rc_backing_store_destroy(pool_backings));
    free(image_holds.keys); free(image_holds.vals);
    free(alias_fragment_owners.keys); free(alias_fragment_owners.vals);
    free(dead);
    puts("Original physical backing reclaimed; replacement isolated; detached VkImage allocation freed only after its reader, upload and scene retired");
}

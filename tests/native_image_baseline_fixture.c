#define main shared_vma_fixture_main
#include "resource_core_vma_check.c"
#undef main
#include "resource_core_backing.h"
typedef struct VkMemImageAliasBaseline VkMemImageAliasBaseline;

enum { POOL_IMPORT = 1, POOL_PHYS, POOL_VIRT, PHYSICAL_ORIGIN_POISON_LIMIT = 32 };
typedef struct { uint64_t lo, hi; } FlushRange;
typedef struct PhysicalOriginBinding { struct PhysicalOriginBinding *next; uint64_t lo, hi; } PhysicalOriginBinding;
typedef struct PoolRec {
    int kind, unpublished;
    volatile int32_t used;
    uint32_t refs, physical_origin_poison_count;
    int physical_origin_unknown;
    uint64_t lo, size, publication, physical_origin_upload, physical_origin_bytes;
    VkDeviceSize physical_memory_offset;
    VkBuffer buf;
    VmaAllocation alloc;
    RcBackingId backing;
    RcBackingMapId map;
    FlushRange physical_origin_poison[PHYSICAL_ORIGIN_POISON_LIMIT];
    FlushRange *physical_raw_uses;
    uint32_t physical_raw_use_count;
    int physical_raw_use_unknown;
    const MemBackingCapture *host_backing;
    VkMemImageAliasBaseline *physical_origin_borrows;
    PhysicalOriginBinding *physical_origin_pending;
} PoolRec;
typedef struct { PoolRec *buf; } pidx_ent;

static PlatMutex pool_lock;
static int pool_index, fills, barriers, uploads, release_calls;
static const char *trace;
static pidx_ent selected_window;
static RcBackingStore *pool_backings;
static RcRegistry *registry;
static RcRegistryPosition read = {7, 2};
static uint64_t actual_recorded, submitted, completed, fill_bytes;
static int allow_zero;
static PoolRec physical, window, replacement;
static RcInterval mapping[4];
static uint32_t map_count;
static const uint64_t guest = 0x1000;

volatile int32_t vk_lost;
static RcRegistry *fixture_registry(void) { return registry; }
#define rc_registry_runtime fixture_registry
RcRegistryPosition vk_exec_resource_position(void) { return read; }
void plat_mutex_lock_shared(PlatMutex *lock) { plat_mutex_lock(lock); }
void plat_mutex_unlock_shared(PlatMutex *lock) { plat_mutex_unlock(lock); }
static const pidx_ent *pidx_find(const int *index, uint64_t address, uint64_t size) {
    assert(index == &pool_index);
    PoolRec *r = selected_window.buf;
    return r && address >= r->lo && address - r->lo <= r->size && size <= r->size - (address - r->lo) ?
        &selected_window : NULL;
}
static void pool_native_acquire(PoolRec *p) { assert(p->refs); p->refs++; }
static void destroy_rec(VkCtx *c, PoolRec *p) { assert(c == &context && p->refs > 1); p->refs--; release_calls++; }
const char *cfg_get(const char *name) { assert(!strcmp(name, "NX2NAVI_IMAGE_BASELINE_TRACE")); return trace; }
void rt_log(const char *fmt, ...) { assert(strstr(fmt, "image baseline read") || strstr(fmt, "image placement raw guard")); }
int vk_fix_zero(void) { return allow_zero; }
VkCommandBuffer vk_upload_begin(void) { uploads++; return (VkCommandBuffer)(uintptr_t)5; }
void vk_upload_end(void) { assert(uploads == fills && fills == barriers); }
uint64_t vk_upload_recording_sequence(VkCommandBuffer cb) {
    assert(cb == (VkCommandBuffer)(uintptr_t)5 && fills && barriers);
    return actual_recorded;
}
void vk_upload_progress(uint64_t *sub, uint64_t *done) {
    if (sub) *sub = submitted;
    if (done) *done = completed;
}
VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(VkCommandBuffer cb, VkBuffer buf, VkDeviceSize offset, VkDeviceSize size, uint32_t data) {
    assert(cb == (VkCommandBuffer)(uintptr_t)5 && buf == physical.buf && !offset && !data);
    fills++;
    fill_bytes = size;
}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *dependency) {
    assert(cb == (VkCommandBuffer)(uintptr_t)5 && dependency->memoryBarrierCount == 1);
    assert(dependency->pMemoryBarriers[0].srcAccessMask == VK_ACCESS_2_TRANSFER_WRITE_BIT);
    barriers++;
}
struct VkMemSpanCapture {
    uint64_t address, size;
    int native_input;
    PoolRec *window, *provider;
    VkMemSpan selected;
    const MemBackingCapture *host;
    const RcInterval *device_intervals;
    uint32_t device_count;
};
const RcInterval *mem_backing_intervals(const MemBackingCapture *owner, uint32_t *count) {
    (void)owner; (void)count; abort();
}
static int span_capture_match(const VkMemSpanCapture *span, const VkMemSpan *selected) {
    assert(map_lock_depth && span->window == selected_window.buf && selected == &span->selected);
    return rc_backing_map_intervals_current(pool_backings, span->window->map,
        span->device_intervals, span->device_count, span->address, span->size);
}
#include "origin_extract.h"
struct VkMemoryImageHold {
    PoolRec *pool;
    VkDeviceMemory pool_memory;
    VkDeviceSize pool_begin, pool_end;
    VkImage image;
    unsigned readers;
    int binding_pending;
};
static PlatOnce cmd_once;
static PlatRMutex cmd_lock;
static int command_depth, driver_calls, placements, remap_in_driver, raw_writer_in_driver;
static VkMemoryImageHold *image_holds;
static VkResult driver_result;
static uint64_t driver_allocation_offset;
static int check_rounded_binding;
static RcRegistryResult capture(VkMemImageAliasBaseline **out);
static void cmd_init(void *arg) { (void)arg; }
void plat_once(PlatOnce *once, void (*fn)(void *), void *arg) { (void)once; fn(arg); }
void plat_rmutex_lock(PlatRMutex *lock) { assert(lock == &cmd_lock && !command_depth++ && !map_lock_depth); }
void plat_rmutex_unlock(PlatRMutex *lock) { assert(lock == &cmd_lock && command_depth-- == 1); }
void vmaGetAllocationInfo(VmaAllocator allocator, VmaAllocation allocation, VmaAllocationInfo *out) {
    assert(allocator == context.vma && allocation == physical.alloc);
    *out = (VmaAllocationInfo){.deviceMemory = (VkDeviceMemory)(uintptr_t)22, .offset = driver_allocation_offset, .size = physical.size};
}
static int image_placement_available(VkDeviceMemory memory, uint64_t lo, uint64_t hi,
                                      VkMemoryImageHold **before, VkMemoryImageHold **after) {
    assert(command_depth && memory == (VkDeviceMemory)(uintptr_t)22 && lo == 80 && hi == 112);
    *before = *after = NULL;
    return !placements;
}
static VkMemoryImageHold *image_hold_find_or_add(VkImage image) {
    assert(command_depth && !image_holds);
    image_holds = calloc(1, sizeof *image_holds);
    assert(image_holds);
    image_holds->image = image;
    return image_holds;
}
static void image_placement_insert(VkMemoryImageHold *hold, VkMemoryImageHold *before, VkMemoryImageHold *after) {
    assert(command_depth && !placements && hold == image_holds && !before && !after);
    placements++;
}
static void image_placement_remove(VkMemoryImageHold *hold) {
    assert(command_depth && placements == 1 && hold == image_holds);
    placements--;
}
static VkMemoryImageHold *nvn_hm_del(VkMemoryImageHold **head, const VkImage *image) {
    assert(command_depth && *head && (*head)->image == *image);
    VkMemoryImageHold *out = *head;
    *head = NULL;
    return out;
}
VkResult vmaBindImageMemory2(VmaAllocator allocator, VmaAllocation allocation,
    VkDeviceSize offset, VkImage image, const void *next) {
    assert(allocator == context.vma && allocation == physical.alloc && offset == 16 && image && !next);
    assert(!command_depth && !map_lock_depth && image_holds->binding_pending && physical.physical_origin_pending);
    if (check_rounded_binding) {
        assert(physical.physical_origin_pending->lo == 0 && physical.physical_origin_pending->hi == 64);
        VkMemSpan adjacent = {.buffer = physical.buf, .alloc = physical.alloc, .offset = 48, .alloc_offset = 48, .avail = 83};
        RcInterval *intervals = NULL;
        assert(!vk_mem_device_writer_intervals(&adjacent, guest + 48, 16, &intervals) && !intervals);
    }
    VkMemImageAliasBaseline *owner = NULL;
    assert(capture(&owner) == RC_REGISTRY_OK && !owner);
    if (raw_writer_in_driver) {
        VkMemSpan writer = {.buffer = physical.buf, .alloc = physical.alloc, .offset = 16, .alloc_offset = 16, .avail = 115};
        RcInterval *intervals = NULL;
        assert(!vk_mem_device_writer_intervals(&writer, guest + 16, 16, &intervals) && !intervals);
        VkMemSpanCapture held = {.address = guest + 16, .size = 16, .native_input = 1, .window = selected_window.buf,
            .provider = &physical, .selected = writer, .device_intervals = mapping, .device_count = map_count};
        assert(!vk_mem_span_input_intervals(&held, guest + 16, 16, &intervals) && !intervals);
        assert(!physical.physical_raw_use_count);
    }
    driver_calls++;
    if (remap_in_driver)
        assert(rc_backing_map_replace(pool_backings, window.map, guest + 16, 32, physical.backing, 16) == RC_BACKING_OK);
    return driver_result;
}
#include "origin_bind_extract.h"

static void open_origin(int virtual_pool, uint64_t bytes) {
    start(256);
    context.props.limits.bufferImageGranularity = 1;
    driver_allocation_offset = 64;
    release_calls = fills = barriers = uploads = 0;
    allow_zero = 1;
    trace = "1";
    actual_recorded = 3;
    submitted = completed = fill_bytes = 0;
    read = (RcRegistryPosition){7, 2};
    physical = (PoolRec){.kind = POOL_PHYS, .refs = 1, .lo = virtual_pool ? 0x8000 : guest, .size = bytes,
        .publication = 20, .buf = (VkBuffer)(uintptr_t)9, .alloc = (VmaAllocation)(uintptr_t)10,
        .physical_memory_offset = driver_allocation_offset};
    window = (PoolRec){.kind = POOL_VIRT, .refs = 1, .lo = guest, .size = physical.size, .publication = 31};
    selected_window.buf = virtual_pool ? &window : &physical;
    pool_backings = rc_backing_store_create((RcBackingConfig){8, 8, 16, 16, 16});
    registry = rc_registry_create((RcRegistryConfig){8, 16, 3});
    assert(pool_backings && registry);
    assert(rc_registry_declare_through(registry, read) == RC_REGISTRY_OK);
    assert(rc_backing_create(pool_backings, physical.size, (uintptr_t)&physical, &physical.backing) == RC_BACKING_OK);
    assert(rc_backing_map_create(pool_backings, physical.lo, physical.size, &physical.map) == RC_BACKING_OK);
    assert(rc_backing_map_replace(pool_backings, physical.map, physical.lo, physical.size, physical.backing, 0) == RC_BACKING_OK);
    if (virtual_pool) {
        assert(rc_backing_map_create(pool_backings, guest, physical.size, &window.map) == RC_BACKING_OK);
        assert(rc_backing_map_replace(pool_backings, window.map, guest, physical.size, physical.backing, 0) == RC_BACKING_OK);
    }
    assert(rc_backing_map_intervals(pool_backings, selected_window.buf->map, guest, physical.size,
        mapping, 4, &map_count) == RC_BACKING_OK && map_count == 1);
}
static void close_origin(void) {
    assert(physical.refs == 1 && window.refs == 1 && !physical.physical_origin_borrows && !physical.physical_origin_pending);
    free(physical.physical_raw_uses);
    RcBudgetSnapshot usage;
    resource_core_budget_snapshot(&usage);
    assert(!usage.producer[RC_BUDGET_SNAPSHOT].retained.amount[RC_BUDGET_RAM]);
    assert(!usage.producer[RC_BUDGET_SNAPSHOT].reserved.amount[RC_BUDGET_RAM]);
    if (window.map.id) assert(rc_backing_map_close(pool_backings, window.map) == RC_BACKING_OK);
    assert(rc_backing_map_close(pool_backings, physical.map) == RC_BACKING_OK);
    uintptr_t token;
    assert(rc_backing_finalize(pool_backings, physical.backing) == RC_BACKING_OK);
    assert(rc_backing_reclaim(pool_backings, physical.backing, &token) == RC_BACKING_OK);
    if (replacement.backing.id) {
        assert(rc_backing_finalize(pool_backings, replacement.backing) == RC_BACKING_OK);
        assert(rc_backing_reclaim(pool_backings, replacement.backing, &token) == RC_BACKING_OK);
        replacement = (PoolRec){0};
    }
    assert(rc_backing_store_destroy(pool_backings) == RC_BACKING_OK);
    rc_registry_destroy(registry);
}
static RcRegistryResult capture(VkMemImageAliasBaseline **out) {
    return vk_mem_image_alias_baseline_capture(guest, physical.size, mapping, map_count, (RcRegistryPosition){7, 2}, out);
}
static void complete_origin(void) {
    physical_origin_initialize(&physical);
    assert(fills == 1 && barriers == 1 && fill_bytes == 128 && physical.physical_origin_upload == 3);
    submitted = completed = actual_recorded;
}
static void fill_and_fragment(void) {
    open_origin(0, 131);
    VkMemImageAliasBaseline *owner = NULL;
    allow_zero = 0;
    physical_origin_initialize(&physical);
    assert(!fills && capture(&owner) == RC_REGISTRY_OK && !owner);
    allow_zero = 1;
    actual_recorded = 0;
    physical_origin_initialize(&physical);
    assert(fills == 1 && !physical.physical_origin_upload && capture(&owner) == RC_REGISTRY_OK && !owner);
    actual_recorded = 3;
    physical_origin_initialize(&physical);
    assert(fills == 2 && fill_bytes == 128 && physical.physical_origin_bytes == 128);
    assert(capture(&owner) == RC_REGISTRY_OK && !owner);
    submitted = 3;
    assert(capture(&owner) == RC_REGISTRY_OK && !owner);
    completed = 3;
    assert(capture(&owner) == RC_REGISTRY_OK && owner && physical.refs == 2);
    uint32_t n;
    const RcRegistryBufferSpan *out = vk_mem_image_alias_baseline_spans(owner, &n);
    assert(out && n == 1 && out[0].interval.range.size == 128 && out[0].interval.mapping == mapping[0].mapping);
    assert(out[0].native.allocation == physical.publication && !out[0].native.incarnation && !out[0].native.offset);
    VkBuffer buffer;
    VkDeviceSize offset;
    RcRegistryBufferLocation source = out[0].native;
    source.offset = 124;
    assert(vk_mem_image_alias_baseline_source(owner, 0, source, 4, &buffer, &offset));
    assert(buffer == physical.buf && offset == 124);
    assert(!vk_mem_image_alias_baseline_source(owner, 0, source, 5, &buffer, &offset) && !buffer && !offset);
    source.incarnation++;
    assert(!vk_mem_image_alias_baseline_source(owner, 0, source, 4, &buffer, &offset));
    read.ordinal += 2;
    assert(!vk_mem_image_alias_baseline_current(owner));
    assert(!vk_mem_image_alias_baseline_source(owner, 0, out[0].native, 4, &buffer, &offset));
    vk_mem_image_alias_baseline_release(&owner);
    vk_mem_image_alias_baseline_release(&owner);
    assert(release_calls == 1);
    close_origin();
}
static void poison_and_borrow(void) {
    open_origin(0, 131);
    complete_origin();
    VkMemImageAliasBaseline *owner = NULL;
    PhysicalOriginBinding binding = {NULL, 16, 48};
    // Simulate a bind reservation that predates the placement policy, then its rollback.
    physical.physical_origin_pending = &binding;
    assert(capture(&owner) == RC_REGISTRY_OK && !owner);
    physical_origin_end_image(&physical, &binding, 0);
    assert(!physical.physical_origin_poison_count && !physical.physical_origin_pending);
    assert(capture(&owner) == RC_REGISTRY_OK && owner);
    assert(!physical_origin_image_allowed(&physical, 16, 48));
    assert(physical_origin_image_allowed(&physical, 128, 131));
    assert(!physical.physical_origin_pending && vk_mem_image_alias_baseline_current(owner));
    vk_mem_image_alias_baseline_release(&owner);

    // A real previous bind recorded in this physical incarnation poisons only its original range.
    physical.physical_origin_pending = &binding;
    physical_origin_end_image(&physical, &binding, 1);
    assert(capture(&owner) == RC_REGISTRY_OK && owner);
    uint32_t n;
    const RcRegistryBufferSpan *out = vk_mem_image_alias_baseline_spans(owner, &n);
    assert(n == 2 && out[0].native.offset == 0 && out[0].interval.range.size == 16);
    assert(out[1].native.offset == 48 && out[1].interval.range.size == 80);
    assert(physical_origin_image_allowed(&physical, 16, 48));
    assert(!physical_origin_image_allowed(&physical, 47, 49));
    vk_mem_image_alias_baseline_release(&owner);
    assert(release_calls == 2);
    close_origin();
}

static void remap_rejects_recycled_identity(void) {
    open_origin(1, 131);
    complete_origin();
    VkMemImageAliasBaseline *owner = NULL;
    assert(capture(&owner) == RC_REGISTRY_OK && owner && window.refs == 2 && physical.refs == 2);
    replacement = (PoolRec){.kind = POOL_PHYS, .refs = 1, .size = physical.size, .publication = 21,
        .buf = physical.buf, .alloc = physical.alloc};
    assert(rc_backing_create(pool_backings, replacement.size, (uintptr_t)&replacement, &replacement.backing) == RC_BACKING_OK);
    assert(rc_backing_map_replace(pool_backings, window.map, guest, replacement.size, replacement.backing, 0) == RC_BACKING_OK);
    assert(!vk_mem_image_alias_baseline_current(owner));
    vk_mem_image_alias_baseline_release(&owner);
    assert(release_calls == 2 && physical.refs == 1 && window.refs == 1 && replacement.refs == 1);
    assert(capture(&owner) == RC_REGISTRY_STALE && !owner);
    close_origin();
}
static const VkMemoryRequirements placement_requirements = {.size = 32, .alignment = 16, .memoryTypeBits = 1};

static VkMemSpan selected_placement(void) {
    return (VkMemSpan){.buffer = physical.buf, .alloc = physical.alloc, .offset = 16, .alloc_offset = 16, .avail = 115};
}

static void physical_placement_preserves_recorded_and_completed_fill(void) {
    open_origin(0, 131);
    physical_origin_initialize(&physical);
    const VkMemSpan selected = selected_placement();
    const VkImage image = (VkImage)(uintptr_t)29;
    VkResult native = VK_ERROR_UNKNOWN;
    driver_calls = placements = 0;
    assert(!submitted && !completed);
    assert(vk_mem_image_alias_preplacement(guest + 16, &placement_requirements, &selected, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE);
    assert(vk_mem_bind_image_alias(image, guest + 16, &placement_requirements, &selected, mapping, map_count, &native) ==
        VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE && native == VK_SUCCESS);
    assert(!driver_calls && !placements && !image_holds && physical.refs == 1 && !release_calls);

    submitted = completed = actual_recorded;
    VkMemImageAliasBaseline *owner = NULL;
    assert(capture(&owner) == RC_REGISTRY_OK && owner);
    assert(vk_mem_bind_image_alias(image, guest + 16, &placement_requirements, &selected, mapping, map_count, &native) ==
        VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE && native == VK_SUCCESS);
    assert(!driver_calls && !placements && !image_holds && physical.refs == 2);
    assert(vk_mem_image_alias_baseline_current(owner) && !physical.physical_origin_poison_count);
    vk_mem_image_alias_baseline_release(&owner);

    const VkMemSpan tail = {.buffer = physical.buf, .alloc = physical.alloc, .offset = 128, .alloc_offset = 128, .avail = 3};
    const VkMemoryRequirements tail_requirements = {.size = 3, .alignment = 16, .memoryTypeBits = 1};
    assert(vk_mem_image_alias_preplacement(guest + 128, &tail_requirements, &tail, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_DIRECT);
    assert(release_calls == 1);
    close_origin();
}

static void physical_placement_protects_adjacent_linear_bytes(void) {
    open_origin(0, 131);
    complete_origin();
    const VkMemSpan selected = selected_placement();
    const VkImage image = (VkImage)(uintptr_t)29;
    VkResult native;
    driver_calls = placements = 0;
    // An earlier bound image left [0,16) initialized and invalidated [16,128).
    PhysicalOriginBinding previous = {NULL, 16, 128};
    physical.physical_origin_pending = &previous;
    physical_origin_end_image(&physical, &previous, 1);
    assert(vk_mem_image_alias_preplacement(guest + 16, &placement_requirements, &selected, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_DIRECT);
    context.props.limits.bufferImageGranularity = 128;
    // Absolute image [80,112) aliases [0,128) after padding. Clip to [0,64) in the buffer.
    assert(vk_mem_image_alias_preplacement(guest + 16, &placement_requirements, &selected, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE);
    assert(vk_mem_bind_image_alias(image, guest + 16, &placement_requirements, &selected, mapping, map_count, &native) ==
        VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE && native == VK_SUCCESS);
    assert(!driver_calls && !placements && physical.physical_origin_poison_count == 1);

    FlushRange clipped = {0};
    assert(physical_image_alias_range(&physical, 128, 3, 128, &clipped) && clipped.lo == 64 && clipped.hi == 131);
    assert(!physical_image_alias_range(&physical, 128, 4, 128, &clipped));
    assert(!physical_image_alias_range(&physical, 16, 32, 3, &clipped));
    PoolRec overflow = {.size = 131, .physical_memory_offset = UINT64_MAX - 130};
    assert(!physical_image_alias_range(&overflow, 16, 32, 128, &clipped));
    close_origin();

    open_origin(0, 131);
    context.props.limits.bufferImageGranularity = 128;
    VkMemSpan far = {.buffer = physical.buf, .alloc = physical.alloc, .offset = 64, .alloc_offset = 64, .avail = 67};
    RcInterval *intervals = NULL;
    assert(vk_mem_device_writer_intervals(&far, guest + 64, 16, &intervals) == 1);
    free(intervals);
    assert(vk_mem_image_alias_preplacement(guest + 16, &placement_requirements, &selected, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_DIRECT);
    driver_allocation_offset += 16;
    assert(vk_mem_bind_image_alias(image, guest + 16, &placement_requirements, &selected, mapping, map_count, &native) ==
        VK_MEM_IMAGE_ALIAS_STALE && !driver_calls && !placements && physical.refs == 1);
    driver_allocation_offset -= 16;
    check_rounded_binding = 1;
    driver_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    assert(vk_mem_bind_image_alias(image, guest + 16, &placement_requirements, &selected, mapping, map_count, &native) ==
        VK_MEM_IMAGE_ALIAS_FAILED && native == driver_result);
    check_rounded_binding = 0;
    assert(driver_calls == 1 && !physical.physical_origin_pending && !physical.physical_origin_poison_count);
    VkMemSpan adjacent = {.buffer = physical.buf, .alloc = physical.alloc, .offset = 48, .alloc_offset = 48, .avail = 83};
    assert(vk_mem_device_writer_intervals(&adjacent, guest + 48, 16, &intervals) == 1);
    free(intervals);
    assert(vk_mem_image_alias_preplacement(guest + 16, &placement_requirements, &selected, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE);
    close_origin();
}

static void exercise_real_bind_rollback(void) {
    open_origin(0, 131); // A genuinely uninitialized physical allocation.
    const VkMemSpan selected = selected_placement();
    const VkImage image = (VkImage)(uintptr_t)29;
    VkResult native;
    driver_calls = placements = 0;
    assert(vk_mem_image_alias_preplacement(guest + 16, &placement_requirements, &selected, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_DIRECT);
    raw_writer_in_driver = 1;
    driver_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    assert(vk_mem_bind_image_alias(image, guest + 16, &placement_requirements, &selected, mapping, map_count, &native) ==
        VK_MEM_IMAGE_ALIAS_FAILED && native == driver_result);
    assert(driver_calls == 1 && !placements && !image_holds && physical.refs == 1 && release_calls == 1);
    assert(!physical.physical_origin_poison_count && !physical.physical_origin_pending);

    driver_result = VK_SUCCESS;
    assert(vk_mem_bind_image_alias(image, guest + 16, &placement_requirements, &selected, mapping, map_count, &native) ==
        VK_MEM_IMAGE_ALIAS_DIRECT && native == VK_SUCCESS);
    assert(driver_calls == 2 && placements == 1 && physical.refs == 2 && !image_holds->binding_pending);
    assert(!physical.physical_origin_poison_count && !physical.physical_origin_pending);
    const VkImage second = (VkImage)(uintptr_t)30;
    assert(vk_mem_bind_image_alias(second, guest + 16, &placement_requirements, &selected, mapping, map_count, &native) ==
        VK_MEM_IMAGE_ALIAS_OCCUPIED && native == VK_SUCCESS);
    assert(driver_calls == 2 && placements == 1 && physical.refs == 2 && image_holds->image == image);
    assert(!image_holds->binding_pending && !physical.physical_origin_poison_count && !physical.physical_origin_pending);
    plat_rmutex_lock(&cmd_lock);
    image_placement_remove(image_holds);
    free(nvn_hm_del(&image_holds, &image));
    plat_rmutex_unlock(&cmd_lock);
    destroy_rec(&context, &physical);
    assert(release_calls == 3 && !placements && !image_holds);
    raw_writer_in_driver = 0;
    close_origin();
}

static void driver_failure_requires_current_mapping(void) {
    open_origin(1, 131);
    const VkMemSpan selected = selected_placement();
    const VkImage image = (VkImage)(uintptr_t)29;
    VkResult native;
    driver_calls = placements = 0;
    remap_in_driver = 1;
    driver_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    assert(vk_mem_bind_image_alias(image, guest + 16, &placement_requirements, &selected, mapping, map_count, &native) ==
        VK_MEM_IMAGE_ALIAS_STALE && native == driver_result);
    assert(driver_calls == 1 && !placements && !image_holds && physical.refs == 1 && release_calls == 1);
    assert(!physical.physical_origin_poison_count && !physical.physical_origin_pending);
    assert(vk_mem_image_alias_preplacement(guest + 16, &placement_requirements, &selected, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_STALE);
    remap_in_driver = 0;
    close_origin();
}

static void successful_stale_bind_keeps_original_owner(void) {
    open_origin(1, 131);
    const VkMemSpan selected = selected_placement();
    const VkImage image = (VkImage)(uintptr_t)29;
    VkResult native;
    driver_calls = placements = 0;
    remap_in_driver = 1;
    driver_result = VK_SUCCESS;
    assert(vk_mem_bind_image_alias(image, guest + 16, &placement_requirements, &selected, mapping, map_count, &native) ==
        VK_MEM_IMAGE_ALIAS_STALE && native == VK_SUCCESS);
    assert(driver_calls == 1 && placements == 1 && image_holds->pool == &physical && physical.refs == 2);
    assert(!image_holds->binding_pending && !physical.physical_origin_pending);
    // The unchanged driver allocation does not make the original mapping current again.
    assert(vk_mem_image_alias_preplacement(guest + 16, &placement_requirements, &selected, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_STALE);
    plat_rmutex_lock(&cmd_lock);
    image_placement_remove(image_holds);
    free(nvn_hm_del(&image_holds, &image));
    plat_rmutex_unlock(&cmd_lock);
    destroy_rec(&context, &physical);
    assert(physical.refs == 1 && release_calls == 1 && !placements && !image_holds);
    remap_in_driver = 0;
    close_origin();
}

static void qualified_raw_writer_preserves_only_its_extent(void) {
    open_origin(1, 1024);
    const VkMemSpan selected = selected_placement();
    RcInterval *intervals = NULL;
    assert(vk_mem_device_intervals(&selected, guest + 16, 16, &intervals) == 1);
    free(intervals);
    assert(vk_mem_image_alias_preplacement(guest + 16, &placement_requirements, &selected, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_DIRECT);
    assert(vk_mem_device_writer_intervals(&selected, guest + 16, 16, &intervals) == 1);
    assert(intervals[0].mapping == mapping[0].mapping && intervals[0].range.offset == 16);
    free(intervals);
    assert(physical.physical_raw_use_count == 1 && physical.physical_raw_uses[0].lo == 16 &&
           physical.physical_raw_uses[0].hi == 32);
    assert(vk_mem_image_alias_preplacement(guest + 16, &placement_requirements, &selected, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE);
    VkMemSpan disjoint = selected;
    disjoint.offset = disjoint.alloc_offset = 64;
    assert(vk_mem_image_alias_preplacement(guest + 64, &placement_requirements, &disjoint, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_DIRECT);
    VkMemSpanCapture held = {.address = guest + 64, .size = 32, .native_input = 1, .window = &window,
        .provider = &physical, .selected = disjoint, .device_intervals = mapping, .device_count = map_count};
    assert(vk_mem_span_input_intervals(&held, guest + 76, 4, &intervals) == 1);
    assert(intervals[0].mapping == mapping[0].mapping && intervals[0].range.offset == 76);
    free(intervals);
    assert(physical.physical_raw_use_count == 2 && physical.physical_raw_uses[1].lo == 76 &&
           physical.physical_raw_uses[1].hi == 80);
    assert(vk_mem_image_alias_preplacement(guest + 64, &placement_requirements, &disjoint, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE);
    VkMemImageAliasBaseline *owner = NULL;
    assert(capture(&owner) == RC_REGISTRY_OK && !owner); // A selected or canceled write does not establish data.
    close_origin();
}

static void exhausted_raw_extent_history_keeps_optional_binding_private(void) {
    open_origin(0, 1024);
    const VkMemSpan selected = selected_placement();
    for (uint32_t i = 0; i <= PHYSICAL_ORIGIN_POISON_LIMIT; ++i) {
        VkMemSpan target = selected;
        target.offset = target.alloc_offset = 4 + 8 * i;
        RcInterval *intervals = NULL;
        assert(vk_mem_device_writer_intervals(&target, guest + target.alloc_offset, 4, &intervals) == 1);
        free(intervals);
    }
    assert(physical.physical_raw_use_unknown);
    VkMemSpan disjoint = selected;
    disjoint.offset = disjoint.alloc_offset = 512;
    assert(vk_mem_image_alias_preplacement(guest + 512, &placement_requirements, &disjoint, mapping, map_count) ==
        VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE);
    close_origin();
}

static void one_affine_15_mib_extent(void) {
    open_origin(1, 0xf00000);
    physical_origin_initialize(&physical);
    assert(fills == 1 && fill_bytes == 0xf00000 && map_count == 1);
    submitted = completed = 3;
    VkMemImageAliasBaseline *owner = NULL;
    assert(capture(&owner) == RC_REGISTRY_OK && owner);
    uint32_t count;
    const RcRegistryBufferSpan *spans = vk_mem_image_alias_baseline_spans(owner, &count);
    assert(count == 1 && spans[0].interval.range.size == 0xf00000 && spans[0].interval.mapping == mapping[0].mapping);
    vk_mem_image_alias_baseline_release(&owner);
    assert(release_calls == 2);
    close_origin();
}
static void multi_native_run_leaves_image_only(void) {
    open_origin(1, 131);
    complete_origin();
    replacement = (PoolRec){.kind = POOL_PHYS, .refs = 1, .size = physical.size, .publication = 21,
        .buf = (VkBuffer)(uintptr_t)12, .alloc = (VmaAllocation)(uintptr_t)13};
    assert(rc_backing_create(pool_backings, replacement.size, (uintptr_t)&replacement, &replacement.backing) == RC_BACKING_OK);
    assert(rc_backing_map_replace(pool_backings, window.map, guest + 64, 67, replacement.backing, 0) == RC_BACKING_OK);
    assert(rc_backing_map_intervals(pool_backings, window.map, guest, physical.size, mapping, 4, &map_count) == RC_BACKING_OK);
    assert(map_count == 2);
    VkMemImageAliasBaseline *owner = NULL;
    assert(capture(&owner) == RC_REGISTRY_OK && !owner && !release_calls);
    close_origin();
}
static void nonaffine_current_same_provider_leaves_image_only(void) {
    open_origin(1, 131);
    complete_origin();
    assert(rc_backing_map_replace(pool_backings, window.map, guest + 64, 67, physical.backing, 32) == RC_BACKING_OK);
    assert(rc_backing_map_intervals(pool_backings, window.map, guest, physical.size, mapping, 4, &map_count) == RC_BACKING_OK);
    assert(map_count == 2 && mapping[0].range.backing == mapping[1].range.backing && mapping[1].range.offset == 32);
    assert(rc_backing_map_intervals_current(pool_backings, window.map, mapping, map_count, guest, physical.size));
    VkMemImageAliasBaseline *owner = NULL;
    assert(capture(&owner) == RC_REGISTRY_OK && !owner && !release_calls);
    close_origin();
}

static void mixed_registry_physical_identity(void) {
    open_origin(0, 131);
    complete_origin();
    VkMemImageAliasBaseline *owner = NULL;
    assert(capture(&owner) == RC_REGISTRY_OK && owner);
    uint32_t count, fragments;
    const RcRegistryBufferSpan *spans = vk_mem_image_alias_baseline_spans(owner, &count);
    assert(count == 1 && spans[0].native.allocation == physical.publication && !spans[0].native.incarnation);
    RcRegistryNativeImageSnapshotFragment output[3];
    RcRegistryNativeImageFailure failure;
    assert(rc_registry_native_image_snapshot_mixed(registry, &spans[0].interval, 1, spans, count,
        read, output, 3, &fragments, &failure) == RC_REGISTRY_OK);
    assert(fragments == 1 && output[0].baseline == 0 && output[0].fragment.interval.range.size == 128 &&
        !rc_registry_position_valid(output[0].fragment.output));
    vk_mem_image_alias_baseline_release(&owner);

    const RcRegistryPosition write = {7, 3};
    const RcSourceRange written = {mapping[0].range.backing, 32, 16};
    RcRegistryWrite ticket;
    assert(rc_registry_write_begin_at(registry, written, RC_REGISTRY_BUFFER, write, &ticket) == RC_REGISTRY_OK);
    read.ordinal = 4;
    assert(rc_registry_declare_through(registry, read) == RC_REGISTRY_OK);
    assert(vk_mem_image_alias_baseline_capture(guest, physical.size, mapping, map_count, read, &owner) ==
        RC_REGISTRY_OK && owner);
    spans = vk_mem_image_alias_baseline_spans(owner, &count);
    assert(rc_registry_native_image_snapshot_mixed(registry, &spans[0].interval, 1, spans, count,
        read, output, 3, &fragments, &failure) == RC_REGISTRY_BLOCKED && !fragments && failure.blocker.id == ticket.id);

    RcRegistryBufferSpan placed = spans[0];
    placed.interval.address += 32;
    placed.interval.range.offset += 32;
    placed.interval.range.size = 16;
    placed.native.offset += 32;
    assert(rc_range_equal(placed.interval.range, written));
    assert(rc_registry_buffer_write_record(registry, ticket, placed, write) == RC_REGISTRY_OK);
    assert(rc_registry_native_image_snapshot_mixed(registry, &spans[0].interval, 1, spans, count,
        read, output, 3, &fragments, &failure) == RC_REGISTRY_OK);
    assert(fragments == 3 && output[0].fragment.interval.range.size == 32 &&
        output[1].baseline == 0 && output[1].fragment.interval.range.size == 16 &&
        !rc_registry_position_compare(output[1].fragment.output, write) &&
        output[2].fragment.interval.range.size == 80 && !rc_registry_position_valid(output[2].fragment.output));
    VkBuffer buffer;
    VkDeviceSize offset;
    assert(vk_mem_image_alias_baseline_source(owner, output[1].baseline, output[1].native_buffer,
        output[1].fragment.interval.range.size, &buffer, &offset) && buffer == physical.buf && offset == 32);

    RcRegistryBufferSpan mismatched = spans[0];
    mismatched.native.allocation++;
    assert(rc_registry_native_image_snapshot_mixed(registry, &mismatched.interval, 1, &mismatched, 1,
        read, output, 3, &fragments, &failure) == RC_REGISTRY_BLOCKED && !fragments);
    assert(failure.blocker.id == ticket.id);
    assert(rc_registry_write_cancel(registry, ticket) == RC_REGISTRY_OK);
    vk_mem_image_alias_baseline_release(&owner);
    close_origin();
}

static void cpu_handed_raw_receipt(void) {
    open_origin(1, 131);
    complete_origin();
    const RcRegistryPosition cut = {7, 3};
    assert(rc_registry_declare_through(registry, cut) == RC_REGISTRY_OK);
    assert(rc_registry_handoff_through(registry, cut) == RC_REGISTRY_OK);
    assert(!rc_registry_position_compare(rc_registry_handed_cutoff(registry), cut));

    VkMemImageAliasBaseline *owner = NULL;
    uint64_t supplier = 0;
    completed = actual_recorded - 1;
    submitted = actual_recorded - 1;
    assert(vk_mem_image_alias_baseline_capture_handed(guest, physical.size, mapping, map_count, cut, &owner,
        &supplier) == RC_REGISTRY_OK && !owner && !supplier);
    submitted = actual_recorded;
    assert(vk_mem_image_alias_baseline_capture_handed(guest, physical.size, mapping, map_count, cut, &owner,
        &supplier) == RC_REGISTRY_OK && owner && supplier == actual_recorded && physical.refs == 2 && window.refs == 2);
    uint32_t count;
    assert(!vk_mem_image_alias_baseline_spans(owner, &count) && !count);
    assert(!vk_mem_image_alias_baseline_submitted_spans(owner, supplier, &count) && !count);
    VkBuffer buffer;
    VkDeviceSize offset;
    const RcRegistryBufferLocation native = {physical.publication, 0, 24};
    assert(!vk_mem_image_alias_baseline_source(owner, 0, native, 8, &buffer, &offset));

    completed = actual_recorded;
    const RcRegistryBufferSpan *spans = vk_mem_image_alias_baseline_spans(owner, &count);
    assert(spans && count == 1 && vk_mem_image_alias_baseline_current(owner));
    assert(vk_mem_image_alias_baseline_source(owner, 0, native, 8, &buffer, &offset) &&
        buffer == physical.buf && offset == 24);
    const RcRegistryBufferLocation wrong_allocation = {physical.publication + 1, 0, 24};
    assert(!vk_mem_image_alias_baseline_source(owner, 0, wrong_allocation, 8, &buffer, &offset));
    ++physical.physical_origin_upload;
    assert(!vk_mem_image_alias_baseline_current(owner));
    --physical.physical_origin_upload;
    assert(vk_mem_image_alias_baseline_current(owner));
    const RcRegistryPosition later = {8, 3};
    assert(rc_registry_declare_through(registry, later) == RC_REGISTRY_OK);
    assert(rc_registry_handoff_through(registry, later) == RC_REGISTRY_OK);
    assert(!vk_mem_image_alias_baseline_current(owner));
    assert(!vk_mem_image_alias_baseline_source(owner, 0, native, 8, &buffer, &offset));
    vk_mem_image_alias_baseline_release(&owner);
    close_origin();
}

static void ordered_submitted_raw_receipt(void) {
    open_origin(1, 131);
    complete_origin();
    submitted = completed = actual_recorded - 1;
    VkMemImageAliasBaseline *owner = NULL;
    uint64_t supplier = 0;
    assert(vk_mem_image_alias_baseline_capture_submitted(guest, physical.size, mapping, map_count, read, &owner,
        &supplier) == RC_REGISTRY_OK && !owner && !supplier);

    submitted = actual_recorded;
    assert(vk_mem_image_alias_baseline_capture_submitted(guest, physical.size, mapping, map_count, read, &owner,
        &supplier) == RC_REGISTRY_OK && owner && supplier == actual_recorded && physical.refs == 2 && window.refs == 2);
    uint32_t count;
    assert(!vk_mem_image_alias_baseline_submitted_spans(owner, supplier + 1, &count) && !count);
    const RcRegistryBufferSpan *before = vk_mem_image_alias_baseline_submitted_spans(owner, supplier, &count);
    assert(before && count == 1 && before[0].native.allocation == physical.publication);
    VkBuffer buffer;
    VkDeviceSize offset;
    assert(!vk_mem_image_alias_baseline_current(owner) && !vk_mem_image_alias_baseline_spans(owner, &count) && !count);
    assert(!vk_mem_image_alias_baseline_source(owner, 0, before[0].native, 8, &buffer, &offset));
    read.ordinal += 2;
    assert(!vk_mem_image_alias_baseline_submitted_spans(owner, supplier, &count) && !count);
    read.ordinal -= 2;
    ++physical.physical_origin_upload;
    assert(!vk_mem_image_alias_baseline_submitted_spans(owner, supplier, &count) && !count);
    --physical.physical_origin_upload;

    completed = supplier;
    const RcRegistryBufferSpan *after = vk_mem_image_alias_baseline_spans(owner, &count);
    assert(after == before && count == 1 && vk_mem_image_alias_baseline_current(owner));
    assert(vk_mem_image_alias_baseline_source(owner, 0, after[0].native, 8, &buffer, &offset) &&
        buffer == physical.buf && !offset);
    vk_mem_image_alias_baseline_release(&owner);
    assert(vk_mem_image_alias_baseline_capture_submitted(guest, physical.size, mapping, map_count, read, &owner,
        &supplier) == RC_REGISTRY_OK && owner && !supplier);
    assert(vk_mem_image_alias_baseline_current(owner));
    vk_mem_image_alias_baseline_release(&owner);
    close_origin();
}

int main(void) {
    trace = NULL;
    assert(!rt_knob("IMAGE_BASELINE_TRACE", "0"));
    trace = "0";
    assert(!rt_knob("IMAGE_BASELINE_TRACE", "0"));
    trace = "1";
    assert(rt_knob("IMAGE_BASELINE_TRACE", "0") == 1);
    fill_and_fragment();
    poison_and_borrow();
    remap_rejects_recycled_identity();
    physical_placement_preserves_recorded_and_completed_fill();
    physical_placement_protects_adjacent_linear_bytes();
    exercise_real_bind_rollback();
    driver_failure_requires_current_mapping();
    successful_stale_bind_keeps_original_owner();
    qualified_raw_writer_preserves_only_its_extent();
    exhausted_raw_extent_history_keeps_optional_binding_private();
    one_affine_15_mib_extent();
    multi_native_run_leaves_image_only();
    nonaffine_current_same_provider_leaves_image_only();
    mixed_registry_physical_identity();
    cpu_handed_raw_receipt();
    ordered_submitted_raw_receipt();
    puts("physical image baseline: RAW origin, granularity, native bind, remap, mixed-registry and exact CPU/GPU fill custody passed");
    return 0;
}

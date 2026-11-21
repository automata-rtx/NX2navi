// Vulkan memory of the graphics layer: NVN memory pools as Vulkan
// buffers, the virtual-pool page table, GPU address resolution, page write generations for re-uploads, and the
// upload batch (command buffer, staging buffers, deferred destruction) that vk_texture.c records into.
//
// Contains code Copyright (c) 2026 jhaizhou, Copyright (c) 2026 Peiikai (MIT; see THIRD_PARTY_NOTICES.md).
// Virtual pools have 64 KiB pages. A texture is stale when a covered page carries a generation newer than its upload;
// every write-watch query resets the OS bits, so each query stamps the pages it saw with a new generation shared by all
// users. Virtual pools are a GPU-side page table
// over device-local memory and dirty pages come from GetWriteWatch (mem_written_pages), so neither host VM aliasing
// nor page protection (a fault handler) is needed.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "nvn_pool_index.h"
#include "texture_source.h"
#include "resource_core_registry.h"
#include "resource_core_vma.h"
#include "vk.h"
#include "vk_uniform_custody.h"
#include "replay_phase_trace.h"
#include "resource_core_backing.h"
#include "vk_dlaa.h"
#include "vk_exec.h"
#include "vk_gpa.h"
#include "resource_host_write.h"
#include "wb_lazy_choice.h"
#include "wb_merge_spv.h"
#include "native_rmw_word_spv.h"
#include "native_rmw_bitmap_spv.h"
#include "platform/platform.h"

// Timestamps for the tracking cost, in ns; qpc_ms converts.
static uint64_t qpc(void) { return plat_now_ns(); }

static double qpc_ms(uint64_t ticks) { return (double)ticks / 1e6; }

// ---------------------------------------------------------------- pools
enum { POOL_IMPORT = 1, POOL_PHYS, POOL_VIRT };
enum { MS_NONE, MS_ON, MS_REFUSED, MS_CREATING, MS_PENDING, MS_DORMANT };

#define PHYSICAL_ORIGIN_POISON_LIMIT 32
#define FLUSH_RANGES 32                // flush ranges a mirrored pool keeps between two syncs (flush_add)
typedef struct { uint64_t lo, hi; } FlushRange;
typedef struct PhysicalOriginBinding {
    struct PhysicalOriginBinding *next;
    uint64_t lo, hi;
} PhysicalOriginBinding;

typedef struct PoolRec {
    const pool_t *key;
    int kind;
    uint32_t flags;
    ResourceHostWrite physical_host_authority;
    RcRegistryWrite physical_exclusion; // CPU-visible physical storage remains outside ordinary source certification
    uint64_t lo, size;                 // GPU address range of the pool
    uint64_t import_lo, import_hi;     // POOL_IMPORT: the imported host range (aligned outwards)
    uint64_t publication;              // exact pool-index incarnation, immutable once published
    VkBuffer buf;
    VkDeviceMemory mem;                // POOL_IMPORT
    VmaAllocation alloc;               // POOL_PHYS
    uint32_t native_refs;             // Native owner and recorded provider references.
    uint32_t mirror_input_refs;       // Exact selections and recorded work name the installed allocation.
    int unpublished;                  // pool_lock, provider removed from logical selection.
    uint64_t host_mapping_cutoff;      // Compatibility proof where the host topology cannot supply an exact vector.
    MemBackingCapture *host_backing;   // Native import/shadow guest extent, retained with this pool.
    RcBackingId backing;               // POOL_PHYS: device storage, separate from any CPU-visible guest span
    RcBackingMapId map;                // POOL_PHYS/POOL_VIRT: current device-address window
    VkDeviceSize physical_memory_offset;  // dedicated physical buffer's VkDeviceMemory base
    uint64_t physical_origin_upload, physical_origin_bytes;
    uint32_t physical_origin_poison_count;
    int physical_origin_unknown;
    FlushRange physical_origin_poison[PHYSICAL_ORIGIN_POISON_LIMIT];
    // Placement-only exclusions for already selected raw uses. These never certify buffer data.
    FlushRange *physical_raw_uses;
    uint32_t physical_raw_use_count;
    int physical_raw_use_unknown;
    VkMemImageAliasBaseline *physical_origin_borrows;   // pool_lock, exact read-time raw leases
    PhysicalOriginBinding *physical_origin_pending;   // pool_lock, synchronous native image binds
    volatile int32_t used;                // resolved at least once: the GPU may still read it at finalize
    // NX2NAVI_TRACE=perf dirty census (vk_mem_dirty_submit), replay thread under the submit lock only:
    uint32_t bound_sub;                // last submit that bound it as a buffer (dirty_sub), 0 never
    uint32_t synced_gen;               // page generation its lazy-mirror count covers; DIRTY_UNSYNCED until scanned
    uint8_t bufbound;                  // ever bound as a vertex, index, uniform, storage or indirect buffer
    uint8_t untracked;                 // the last dirty scan found no write watch over it
    // Parallel replay creates mirrors under use_lock. Publish a final state only after all backing fields are ready.
    _Atomic uint8_t mstate;            // MS_CREATING makes another worker wait instead of retaining an imported span
    uint32_t gpa_rec;                  // POOL_IMPORT: its record in the device address table (vk_gpa.c), 0 none
    uint8_t gpa_host_coherent;         // actual memory properties of the imported guest pointer
    VkBuffer mbuf;                     // MS_ON: device-local copy of [import_lo, import_hi), same offsets as buf
    VmaAllocation malloc_;
    uint64_t mirror_incarnation;       // pool_lock, increments with each published mirror allocation
    _Atomic uint64_t mirror_origin_checked_scene; // original recording scene with complete write-watch coverage
    _Atomic uint64_t mirror_origin_full_queued; // scene whose prologue owns the entire first copy
    _Atomic uint64_t mirror_origin_full_copy;   // scene of a recorded full copy into this incarnation
    _Atomic uint64_t mirror_origin_last_copy;   // scene of its newest recorded refresh
    _Atomic uint64_t mirror_origin_queued;     // scene that owns uploads awaiting the prologue
    uint32_t acc_sub;                  // replay: the last submission that synced the mirror (mirror_sync)
    uint32_t pm_frame;                 // replay: the frame (pframe) of pm_cur
    uint8_t pm_cur, pm_prev;           // replay: the places in their frame (vk_sub_in_frame, 7 the rest) of the submissions
                                       // that synced it, in frame pm_frame and in the one before (NX2NAVI_PRESNAP_PLACE)
    uint32_t wb_sub;                   // replay: the last submission that wrote mirror bytes back to guest memory
    uint64_t wb_fence;                 // replay: that submission's fence (mirror_capture waits for no write-back)
    uint8_t wb_kept;                   // replay: a write-back left bytes in the mirror alone (wb_skip): never evicted
    RcVmaDependency mirror_dependency; // exact placement and pressure evidence from the last refused attempt
    _Atomic uint32_t mirror_checked_sub; // one optional decision at first demand in a submission
    _Atomic uint64_t mirror_import_only_scene; // an ordered demand retained the import for this mirror scene
    _Atomic uint64_t import_writer_scene;
    _Atomic uint32_t import_writer_refs;
    _Atomic uint64_t import_writer_fence;
    unsigned mirror_retire_completed, mirror_import_transitioned;
    struct PoolRec *mirror_pressure_prev, *mirror_pressure_next; // pool_lock, one notification per refused attempt
    uint8_t refused_why;               // the MR_* reason of its refusal, for the counts when it is asked again
    uint8_t twin;                      // MS_ON: the table's writes reach guest memory too (twin_wanted); 0: the mirror only
    uint32_t vd_wrote_sub, vd_use_sub; // verify mode: the last submission that bound it written / used its mirror
    uint64_t vd_wrote_n;               // verify mode: written bindings of it so far
    uint32_t mgen;                     // gen_lock: every page write stamped up to this generation is in the mirror
                                       // (or in a copy queued before its next use)
    FlushRange fl[FLUSH_RANGES + 1];   // flush_lock: nvnMemoryPoolFlushMappedRange pages not copied yet (flush_add)
    int fl_n;                          // flush_lock: ranges in fl; read without the lock to skip an empty list
    uint64_t fl_calls, fl_sum;         // flush_lock: the calls and bytes behind fl, for mirror_sync's statistics
    // Shadow storage (POOL_IMPORT in shadow mode, "shadow storage" below): buf is a host-visible copy of the guest pages.
    VmaAllocation salloc;
    uint8_t *smap;                     // its mapping; offsets as in buf (from import_lo)
    uint32_t sgen;                     // gen_lock: every page write stamped up to this generation is in the shadow
    uint64_t sborn;                    // the value of the first submission that can have used the shadow (shadow_pool)
    volatile uint8_t stouched;         // on the touched list: the GPU may use it in the submission being recorded
    uint32_t stab_i;                   // shadow_lock: 1 + its index in stab (the table's shadows), 0 when not there
    // shadow_lock: base copies of the pages a noted GPU write covers (merge_page), open addressing by page address;
    // csb a power of two, page 0 an empty slot
    struct SBase { uint64_t page; uint8_t *base; } *sb;
    uint32_t nsb, csb;
    // shadow_lock ("Pages a shader stores into through the device address table" below): a mark per page of
    // [import_lo, import_hi), the hash of the shadow page as its last whole copy-in left it, 0 not known; the pages
    // found stored into (guest addresses), and whether the pool is on the list of pools that have such pages
    uint32_t *smark;
    uint64_t *tpg;
    uint32_t ntpg, ctpg;
    uint8_t tlisted;
    uint64_t t_over;                   // whole copy-ins that found such a store in the page they overwrote, so far
    uint64_t t_over_log;               // t_over at the last log line
    uint64_t s_in;                     // NX2NAVI_SHADOW_CHECK: bytes copied in whole since the last log line
    // NX2NAVI_TRACE=perf per-pool mirror traffic since the last mirror_pool_log, replay thread (the log reads and clears
    // them without a lock: a count lost to the race only blurs a diagnostic)
    uint64_t st_pages, st_flush, st_wb, st_merge;   // bytes: written pages and flush ranges copied up, GPU writes back
    uint32_t st_syncs;                              // syncs
} PoolRec;

static void mirror_forget(PoolRec *r);   // pool_lock exclusive: off the mirrored list
static void mirror_pressure_remove(PoolRec *r); // pool_lock exclusive
static int shadow_on;                    // shadow mode: imported pools get shadow storage instead (vk_mem_class_log)
static int shadow_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size);
static void shadow_touch(PoolRec *r);
static void shadow_forget(PoolRec *r);
static void pool_retiring(PoolRec *r);  // pool_fini: kept until pool_retire (or vk_mem_shutdown)
static void shadow_table_add(PoolRec *r);
static void pool_retire(void *arg);    // completion thread: destroys a finalized pool's record
static void pool_fini(pool_t *p);
static int pool_fini_idle(void);
static void pool_fini_wait(void);
static void pool_wait_add(PoolRec *r);
static void grow(void **arr, int *cap, size_t elem);

static PlatMutex pool_lock = PLAT_MUTEX_INIT;
static RcBackingStore *pool_backings;
static uint32_t copy_pool_host_intervals(const PoolRec *, uint64_t addr, uint64_t size, RcInterval **out);

static void pool_host_visibility(const RcRegistryVisibility *proof, const PoolRec *pool, uint64_t addr, uint64_t size) {
    RcInterval *intervals;
    uint32_t count = copy_pool_host_intervals(pool, addr, size, &intervals);
    for (uint32_t i = 0; i < count; i++)
        rc_registry_visibility_apply(rc_registry_runtime(), proof, intervals[i].range);
    free(intervals);
}
// Bumped under pool_lock (exclusive) by every change to the index or a page table: vk_mem_resolve's per-thread hits
// stay valid while it is unchanged.
static uint64_t pool_gen = 1;
// Bumped when a buffer use may get another buffer for the same span (a mirror created, refused or forgotten) and at
// every submission's start (mirror syncs are per submission). vk_mem_gen: both together.
static uint64_t mirror_gen;
static pidx pool_index;                // POOL_IMPORT, POOL_PHYS and POOL_VIRT records by GPU address
static PoolRec **phys_list;            // physical pools by key, for MapVirtual requests
static int nphys, cap_phys;
static int32_t n_imports, n_import_failures;
static PFN_vkGetMemoryHostPointerPropertiesEXT p_host_props;

static VkBufferUsageFlags pool_usage(const VkCtx *c) {
    VkBufferUsageFlags u = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                           VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (c->has_transform_feedback)
        u |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT | VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT;
    if (c->has_conditional_rendering) u |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    if (vk_gpa_wanted()) u |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;   // pages of the device address table
    return u;
}

static int memory_type(const VkCtx *c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++)
        if ((bits >> i & 1) && (c->mem_props.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    for (uint32_t i = 0; i < c->mem_props.memoryTypeCount; i++)
        if (bits >> i & 1) return (int)i;
    return -1;
}

// CPU-visible storage: the guest pages themselves become the buffer's memory (VK_EXT_external_memory_host).
static int import_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size) {
    uint64_t al = shadow_on || !c->min_imported_host_alignment ? 0x1000 : c->min_imported_host_alignment;
    r->import_lo = addr & ~(al - 1);
    r->import_hi = (addr + size + al - 1) & ~(al - 1);
    uint64_t n = r->import_hi - r->import_lo;
    r->host_mapping_cutoff = mem_mapping_serial();
    mem_backing_capture(r->import_lo, n, &r->host_backing);
    if (shadow_on) return shadow_pool(c, r, addr, size);
    if (!p_host_props)
        p_host_props = (PFN_vkGetMemoryHostPointerPropertiesEXT)vk_dpa(c->device, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult res = p_host_props ? p_host_props(c->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                               (void *)(uintptr_t)r->import_lo, &hp)
                                : VK_ERROR_EXTENSION_NOT_PRESENT;
    if (res != VK_SUCCESS || !hp.memoryTypeBits) {
        rt_log("vk: pool %#llx+%#llx: host pointer not importable (%d)", (unsigned long long)addr, (unsigned long long)size, (int)res);
        return 0;
    }
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = n,
                             .usage = pool_usage(c), .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if ((res = vkCreateBuffer(c->device, &bi, NULL, &r->buf)) != VK_SUCCESS) {
        rt_log("vk: pool %#llx: vkCreateBuffer for the import failed (%d)", (unsigned long long)addr, (int)res);
        return 0;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, r->buf, &req);
    int type = memory_type(c, req.memoryTypeBits & hp.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type < 0 || req.size > n) {
        rt_log("vk: pool %#llx: no memory type for the import (buffer %#x, host %#x, size %#llx of %#llx)",
               (unsigned long long)addr, req.memoryTypeBits, hp.memoryTypeBits, (unsigned long long)req.size,
               (unsigned long long)n);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        return 0;
    }
    VkImportMemoryHostPointerInfoEXT im = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                           .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                           .pHostPointer = (void *)(uintptr_t)r->import_lo};
    // A buffer with device address usage needs memory allocated for device addresses (vk_gpa.c).
    VkMemoryAllocateFlagsInfo fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .pNext = &im,
                                    .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = vk_gpa_wanted() ? (void *)&fi : (void *)&im,
                               .allocationSize = n, .memoryTypeIndex = (uint32_t)type};
    if ((res = vkAllocateMemory(c->device, &ai, NULL, &r->mem)) != VK_SUCCESS ||
        (res = vkBindBufferMemory(c->device, r->buf, r->mem, 0)) != VK_SUCCESS) {
        rt_log("vk: pool %#llx+%#llx: host memory import failed (%d)", (unsigned long long)addr, (unsigned long long)size, (int)res);
        if (r->mem) vkFreeMemory(c->device, r->mem, NULL);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        r->mem = VK_NULL_HANDLE;
        return 0;
    }
    const VkMemoryPropertyFlags coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    r->gpa_host_coherent = (c->mem_props.memoryTypes[type].propertyFlags & coherent) == coherent;
    plat_inc32(&n_imports);
    return 1;
}

// Physical pool: device-local memory of the pool size that both buffers and optimal-tiling images can be bound to.
static int alloc_physical(VkCtx *c, PoolRec *r) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = r->size, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkResult res = vkCreateBuffer(c->device, &bi, NULL, &r->buf);
    if (res != VK_SUCCESS) { rt_log("vk: physical pool: vkCreateBuffer failed (%d)", (int)res); return 0; }
    VkMemoryRequirements rb;
    vkGetBufferMemoryRequirements(c->device, r->buf, &rb);
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {256, 256, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    VkDeviceImageMemoryRequirements di = {.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = &ii};
    VkMemoryRequirements2 ri = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    vkGetDeviceImageMemoryRequirements(c->device, &di, &ri);
    uint32_t bits = rb.memoryTypeBits & ri.memoryRequirements.memoryTypeBits;
    if (!bits) bits = rb.memoryTypeBits;
    VkDeviceSize align = NVN_VPOOL_PAGE;
    if (rb.alignment > align) align = rb.alignment;
    if (ri.memoryRequirements.alignment > align) align = ri.memoryRequirements.alignment;
    VkMemoryRequirements req = {.size = (rb.size + align - 1) & ~(align - 1), .alignment = align, .memoryTypeBits = bits};
    VmaAllocationCreateInfo aci = {.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                   .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    RcVmaRequest admission = {.producer = RC_BUDGET_BUFFER, .required = 1};
    VmaAllocationInfo allocation;
    if ((res = resource_core_vma_allocate_memory(&admission, c->vma, &req, &aci, &r->alloc, &allocation, NULL)) != VK_SUCCESS ||
        (res = vmaBindBufferMemory(c->vma, r->alloc, r->buf)) != VK_SUCCESS) {
        rt_log("vk: physical pool of %#llx bytes: device-local allocation failed (%d)", (unsigned long long)r->size, (int)res);
        if (r->alloc) resource_core_vma_free_memory(c->vma, r->alloc);
        vkDestroyBuffer(c->device, r->buf, NULL);
        r->buf = VK_NULL_HANDLE;
        r->alloc = NULL;
        return 0;
    }
    r->physical_memory_offset = allocation.offset;
    return 1;
}

// Called after backing topology succeeds and before publication. This actual upload is the only initial
// contents proof. Small tails outside the VkCmdFillBuffer extent remain unknown.
static void physical_origin_initialize(PoolRec *r) {
    const uint64_t bytes = r->size & ~UINT64_C(3);
    if (!bytes || !vk_fix_zero()) return;
    VkCommandBuffer cb = vk_upload_begin();
    vkCmdFillBuffer(cb, r->buf, 0, bytes, 0);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    r->physical_origin_upload = vk_upload_recording_sequence(cb);
    if (r->physical_origin_upload) r->physical_origin_bytes = bytes;
    vk_upload_end();
}

int vk_fix_zero(void) {
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("NX2NAVI_FIX_ZERO");
    }
    return on;
}

static void index_put(PoolRec *r) {
    static uint64_t next_publication;  // pool_lock exclusive
    if (next_publication == UINT64_MAX) rt_fatal("vk: pool publication identity exhausted");
    r->publication = ++next_publication;
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    pidx_ent e = {.lo = r->lo, .hi = r->lo + r->size, .base = r->lo, .buf = r, .gpu = r->lo, .key = r->key};
    if (!pidx_insert(&pool_index, &e)) rt_fatal("vk: out of memory for the pool index");
}

static PoolRec *phys_find(const pool_t *key) {
    for (int i = 0; i < nphys; i++) if (phys_list[i]->key == key) return phys_list[i];
    return NULL;
}

// pool_lock exclusive. An unpublished pool can use this to unwind a partially opened topology.
static void pool_topology_close(PoolRec *r) {
    if (r->backing.id && rc_backing_finalize(pool_backings, r->backing) != RC_BACKING_OK)
        rt_fatal("vk: could not finalize physical pool backing");
    if (r->map.id && rc_backing_map_close(pool_backings, r->map) != RC_BACKING_OK)
        rt_fatal("vk: could not close pool address map");
    r->map = (RcBackingMapId){0};
}

static int pool_topology_open(PoolRec *r) {
    if (r->kind == POOL_IMPORT) return 1;
    plat_mutex_lock(&pool_lock);
    // The guest GPU window admits 1024 reservations. Allow for CPU-addressed physical pools and pending retirement.
    if (!pool_backings) pool_backings = rc_backing_store_create((RcBackingConfig){4096, 4096, 65536, 4096, 65536});
    RcBackingResult result = pool_backings ? rc_backing_map_create(pool_backings, r->lo, r->size, &r->map)
                                          : RC_BACKING_CAPACITY;
    if (result == RC_BACKING_OK && r->kind == POOL_PHYS) {
        result = rc_backing_create(pool_backings, r->size, (uintptr_t)r, &r->backing);
        if (result == RC_BACKING_OK)
            result = rc_backing_map_replace(pool_backings, r->map, r->lo, r->size, r->backing, 0);
    }
    if (result != RC_BACKING_OK) pool_topology_close(r);
    plat_mutex_unlock(&pool_lock);
    if (result != RC_BACKING_OK)
        rt_log("vk: pool address map at %#llx failed (%d)", (unsigned long long)r->lo, result);
    return result == RC_BACKING_OK;
}

// pool_lock exclusive. GPA is a native projection of the current topology. Only the aliases of the departing
// provider are cleared, so an earlier remap to a different backing survives physical pool finalization.
static void pool_topology_unpublish(PoolRec *r) {
    if (r->kind == POOL_VIRT) vk_gpa_virtual_unmap(r->lo, r->lo + r->size);
    if (r->kind == POOL_PHYS) {
        for (int i = 0; i < pool_index.n; i++) {
            PoolRec *v = pool_index.e[i].buf;
            if (v->kind != POOL_VIRT) continue;
            uint64_t at = v->lo, end = at + v->size;
            while (at < end) {
                RcBackingRun runs[32];
                uint32_t count;
                if (rc_backing_map_runs(pool_backings, v->map, at, runs, 32, &count) != RC_BACKING_OK || !count)
                    rt_fatal("vk: could not find physical pool aliases for removal");
                for (uint32_t j = 0; j < count; j++) {
                    RcBackingRun run = runs[j];
                    at = run.address + run.range.size;
                    if (run.range.backing == r->backing.id)
                        vk_gpa_virtual_unmap(run.address, at);
                }
            }
        }
    }
    pool_topology_close(r);
}

// Virtual images, transfers and physical-address stores may reach this separate GPU backing. Its CPU address
// cannot certify an ordinary source, including an overlapping registration, until real guest visibility is proven.
static void pool_authority_begin(PoolRec *r) {
    if (r->kind != POOL_PHYS || !r->size || nvn_gpu_only(r->lo)) return;
    RcInterval *intervals;
    uint32_t count = resource_host_current_intervals(r->lo, r->size, &intervals);
    r->physical_host_authority = resource_host_write_prepare(intervals, count, RC_REGISTRY_UNKNOWN, 0);
    rc_registry_hold_begin(rc_registry_runtime(), (RcSourceRange){1, r->lo, r->size},
                           RC_REGISTRY_UNKNOWN, &r->physical_exclusion);
}
static void pool_authority_end(PoolRec *r) {
    resource_host_write_dispose(&r->physical_host_authority, 0, 1, 0);
    if (r->physical_exclusion.id) {
        // Destruction proves the GPU stopped using the backing, not that its last bytes reached the CPU address.
        rc_registry_hold_release(rc_registry_runtime(), r->physical_exclusion, 0);
        r->physical_exclusion = (RcRegistryWrite){0};
    }
}

static void shadow_metadata_destroy(PoolRec *r);

static void pool_native_acquire(PoolRec *r) {
    uint32_t previous = __atomic_fetch_add(&r->native_refs, 1, __ATOMIC_RELAXED);
    if (!previous || previous == UINT32_MAX) rt_fatal("vk: invalid captured pool reference");
}

int vk_mem_device_backing_reclaimed(RcBackingId *identity) {
    if (!identity || identity->id <= RC_GUEST_COMPATIBILITY_BACKING) return 0;
    plat_mutex_lock_shared(&pool_lock);
    const int reclaimed = pool_backings && !rc_backing_explicit_retained(pool_backings, identity);
    plat_mutex_unlock_shared(&pool_lock);
    return reclaimed;
}

static void destroy_rec(VkCtx *c, PoolRec *r) {
    if (__atomic_sub_fetch(&r->native_refs, 1, __ATOMIC_ACQ_REL)) return;
    if (r->mbuf) resource_core_vma_destroy_buffer(c->vma, r->mbuf, r->malloc_);
    if (r->salloc) resource_core_vma_destroy_buffer(c->vma, r->buf, r->salloc), r->buf = VK_NULL_HANDLE;
    if (r->buf) vkDestroyBuffer(c->device, r->buf, NULL);
    if (r->mem) vkFreeMemory(c->device, r->mem, NULL);
    if (r->alloc) resource_core_vma_free_memory(c->vma, r->alloc);
    if (r->backing.id) {
        uintptr_t provider;
        // Publish reclamation after the last native owner destroyed the original physical allocation.
        if (rc_backing_reclaim(pool_backings, r->backing, &provider) != RC_BACKING_OK || provider != (uintptr_t)r)
            rt_fatal("vk: could not reclaim the retired physical pool backing");
    }
    shadow_metadata_destroy(r);
    pool_authority_end(r);
    mem_backing_release(&r->host_backing);
    free(r->physical_raw_uses);
    free(r);
}

static int pool_init(pool_t *p, const pool_t *setup) {
    VkCtx *c = vk_ctx();
    if (!c) { rt_log("vk: MemoryPoolInitialize before vk_init"); return 0; }
    // The GPU never sees a GPU_NO_ACCESS pool, except a physical one (0x189; 1.4.3 also 0x18c, CPU-visible) through the
    // virtual pools that map it. Either gets device-local memory: 1.4.3's CPU never writes its 0x18c pool's storage
    // (unchanged over 3600 frames of the intro), so the guest storage and the GPU copy need no sync.
    if ((setup->flags & NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS) && !(setup->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL)) {
        pool_fini(p);
        return 1;
    }
    // A pool of zero bytes (NX2navi makes some now and then) holds no address the GPU can use: it gets no record. A record
    // would import 0 bytes (invalid usage; the driver refuses the memory) and put an empty range into the index at the
    // pool's address, where a lookup of the pool that starts there found the empty one first. vk_pool_fini finds no
    // record for it and does nothing.
    if (!setup->size) {
        pool_fini(p);
        return 1;
    }
    PoolRec *r = calloc(1, sizeof *r);
    if (!r) rt_fatal("vk: out of memory for a pool record");
    r->native_refs = 1;
    r->key = p;
    r->flags = setup->flags;
    r->lo = (uintptr_t)setup->mem;
    r->size = setup->size;
    int ok;
    if (r->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) {
        r->kind = POOL_VIRT;
        ok = 1;
    } else if (r->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL) {
        r->kind = POOL_PHYS;
        ok = alloc_physical(c, r);
    } else {
        r->kind = POOL_IMPORT;
        mem_thaw(r->lo, r->size);   // the import may need write access to the pages
        ok = import_pool(c, r, r->lo, r->size);
        if (!ok && !shadow_on) {
            // Not fatal for the game: the pool works for the CPU, and vk_mem_resolve reports its addresses as unbacked.
            long n = plat_inc32(&n_import_failures);
            if (n <= 8) rt_log("vk: pool %p (flags %#x, %#llx bytes) has no GPU buffer (%ld so far)", (void *)(uintptr_t)r->lo,
                               r->flags, (unsigned long long)r->size, n);
            r->buf = VK_NULL_HANDLE;
            ok = 1;
        }
    }
    if (!ok) {
        mem_backing_release(&r->host_backing);
        free(r);
        return 0;
    }
    if (r->host_backing && (!r->buf || !mem_backing_current(r->host_backing)))
        mem_backing_release(&r->host_backing);
    if (r->kind == POOL_IMPORT &&
        !mem_mapping_range_unchanged(r->import_lo, r->import_hi - r->import_lo, r->host_mapping_cutoff))
        r->host_mapping_cutoff = 0;
    if (!pool_topology_open(r)) { destroy_rec(c, r); return 0; }
    if (r->kind == POOL_PHYS) physical_origin_initialize(r);
    pool_authority_begin(r); // Exclusion is visible before any address index or MapVirtual can publish this backing.
    pool_fini(p); // A successful replacement retires the previous incarnation before publishing the new one.
    plat_mutex_lock(&pool_lock);
    index_put(r);
    if (r->kind == POOL_PHYS) {
        if (nphys == cap_phys) {
            cap_phys = cap_phys ? cap_phys * 2 : 16;
            phys_list = realloc(phys_list, (size_t)cap_phys * sizeof *phys_list);
            if (!phys_list) rt_fatal("vk: out of memory for the physical pool list");
        }
        phys_list[nphys++] = r;
    }
    plat_mutex_unlock(&pool_lock);
    // The imported pages, or the shadow that stands for them, go into the device address table: shaders that read a
    // pointer from memory reach them there. A shadow in the table is kept up to date for the table's readers whether
    // or not the replay resolves it (vk_mem_shadow_sync).
    if (r->kind == POOL_IMPORT && (r->mem || r->smap) && r->buf && r->import_hi > r->import_lo) {
        uint32_t host_count;
        const RcInterval *host = mem_backing_intervals(r->host_backing, &host_count);
        r->gpa_rec = vk_gpa_map(r->import_lo, r->import_hi, r->buf, r->import_lo, r->gpa_host_coherent, host, host_count);
        if (r->smap && r->gpa_rec) shadow_table_add(r);
    }
    return 1;
}

// vk_gpa_unmap: the record of another live imported pool that holds the 4 KiB page at a whole (overlapping imports
// of the same guest memory), 0 for none.
static uint32_t gpa_cover(uint64_t a) {
    uint32_t rec = 0;
    plat_mutex_lock_shared(&pool_lock);
    for (int i = pidx_upper(&pool_index, a) - 1; i >= 0 && !rec; i--) {
        const PoolRec *o = pool_index.e[i].buf;
        if (o->kind == POOL_IMPORT && o->gpa_rec && o->import_lo <= a && o->import_hi >= a + 4096) rec = o->gpa_rec;
        if (i && pool_index.maxhi[i - 1] <= a) break;
    }
    plat_mutex_unlock_shared(&pool_lock);
    return rec;
}

static void pool_fini(pool_t *p) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    pidx_ent e;
    plat_mutex_lock(&pool_lock);
    int had = pidx_remove(&pool_index, p, &e);
    PoolRec *r = had ? e.buf : NULL;
    if (r) {
        r->unpublished = 1;
        mirror_pressure_remove(r);
        __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
        if (r->mstate == MS_ON) {
            // Every lazy segment borrows this exact pool's native buffers and immutable mapping.
            mem_lazy_release(r->import_lo, r->import_hi - r->import_lo);
            mirror_forget(r);
        }
        if (r->kind == POOL_PHYS) {
            for (int i = 0; i < nphys; i++)
                if (phys_list[i] == r) { phys_list[i] = phys_list[--nphys]; break; }
        }
        pool_topology_unpublish(r);
    }
    plat_mutex_unlock(&pool_lock);
    if (!r) return;
    if (r->smap) shadow_forget(r);
    // Off the device address table first; the buffer outlives the submissions that may still read the old pages.
    if (r->gpa_rec) vk_gpa_unmap(r->import_lo, r->import_hi, r->gpa_rec, gpa_cover);
    if (r->kind == POOL_IMPORT) mem_thaw(r->lo, r->size);   // the storage goes back to the guest writable
    // Images cut from this range go first; a used buffer or memory waits until the GPU is done with it, since the
    // guest gets the storage back when this returns.
    vk_texture_range_gone(r->lo, r->size, r->alloc);
    int narrow = 0;
    if (r->used && r->kind != POOL_VIRT) {
        if (pool_fini_idle()) {
            vk_upload_flush(1);
            vk_queue_lock();
            vkQueueWaitIdle(c->gfx_queue);
            vk_queue_unlock();
        } else {
            narrow = 1;
            pool_fini_wait();
        }
    }
    vk_mem_collect();
    if (narrow) {   // the Vulkan objects outlive the upload batches and the submission being recorded (pool_wait_add)
        pool_retiring(r);
        pool_wait_add(r);
        return;
    }
    // A record that waits for the next submission is on the retiring list until its callback runs: when no
    // submission follows (a pool freed at the end), vk_mem_shutdown destroys it. Left to the callback alone, a
    // shadow copy's buffer was still allocated when the allocator went, which stops the process there.
    // A mirrored pool's record and buffers outlive the submission being recorded, whose sync copies and write-backs may
    // name them (the game frees a pool only once the GPU is done with it, but its CPU writes may still be queued).
    if (r->mbuf) pool_retiring(r), vk_completion_add(vk_sync_issued() + 1, pool_retire, r);
    // the queue was not drained
    else if (r->gpa_rec && !r->used) pool_retiring(r), vk_completion_add(vk_sync_issued() + 1, pool_retire, r);
    // an unused physical pool may still be read through the virtual pages that mapped it (device addresses, which do not
    // mark it used): its buffer outlives the submission being recorded too
    else if (r->kind == POOL_PHYS && !r->used) pool_retiring(r), vk_completion_add(vk_sync_issued() + 1, pool_retire, r);
    else destroy_rec(c, r);
}

// NX2NAVI_TRACE=perf: a pool's initialize and finalize, with their imports, allocations, uploads and GPU waits, marked
// as work a frame can wait for (perf.c perf_mark_add).
int vk_pool_init_setup(pool_t *p, const pool_t *setup) {
    uint64_t t0 = perf_mark_begin();
    int r = pool_init(p, setup);
    perf_mark_end(PERF_M_POOL, t0, setup->size);
    return r;
}

int vk_pool_init(pool_t *p) { return vk_pool_init_setup(p, p); }

void vk_pool_fini(pool_t *p) {
    uint64_t t0 = perf_mark_begin(), size = p->size;
    pool_fini(p);
    perf_mark_end(PERF_M_POOL, t0, size);
}

// pool_lock exclusive. A prepared GPA update holds its lock through the pure canonical topology operation.
static RcBackingResult pool_topology_replace(PoolRec *pool, const nvn_mapping_t *requests, uint32_t count) {
    if (!count) return RC_BACKING_OK;
    RcBackingReplace *changes = calloc(count, sizeof *changes);
    VkGpaVirtualBinding *bindings = calloc(count, sizeof *bindings);
    RcBackingResult result = changes && bindings ? RC_BACKING_OK : RC_BACKING_CAPACITY;
    for (uint32_t i = 0; result == RC_BACKING_OK && i < count; ++i) {
        const nvn_mapping_t *req = &requests[i];
        uint64_t offset = req->virtual_offset, size = req->size, source = req->physical_offset;
        PoolRec *physical = req->physical_pool ? phys_find((const pool_t *)(const void *)req->physical_pool) : NULL;
        if (req->physical_pool && !physical) result = RC_BACKING_STALE;
        else if (!size || offset > pool->size || size > pool->size - offset ||
                 (offset | size | source) % NVN_VPOOL_PAGE ||
                 (physical && (source > physical->size || size > physical->size - source))) result = RC_BACKING_INVALID;
        else {
            uint64_t lo = pool->lo + offset;
            RcBackingId backing = physical ? physical->backing : (RcBackingId){0};
            changes[i] = (RcBackingReplace){lo, size, backing, physical ? source : 0};
            bindings[i] = (VkGpaVirtualBinding){.lo=lo, .hi=lo + size, .buffer=physical ? physical->buf : VK_NULL_HANDLE,
                .backing=physical ? (RcSourceRange){backing.id, source, size} : (RcSourceRange){0}};
        }
    }
    if (result == RC_BACKING_OK) {
        VkGpaVirtualUpdate update;
        result = vk_gpa_virtual_prepare(bindings, count, &update);
        if (result == RC_BACKING_OK) {
            result = rc_backing_map_replace_many(pool_backings, pool->map, changes, count);
            if (result == RC_BACKING_OK) {
                vk_gpa_virtual_commit(&update);
                __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
            } else vk_gpa_virtual_cancel(&update);
        }
    }
    free(bindings);
    free(changes);
    return result;
}

int vk_pool_map_virtual(pool_t *v, int n, const void *requests) {
    if (n < 0 || (n && !requests)) return 0;
    plat_mutex_lock(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)v->mem, 0);
    PoolRec *r = e && e->key == v ? e->buf : NULL;
    RcBackingResult result = r && r->kind == POOL_VIRT ? pool_topology_replace(r, requests, (uint32_t)n) : RC_BACKING_STALE;
    plat_mutex_unlock(&pool_lock);
    if (result != RC_BACKING_OK)
        rt_log("vk: MapVirtual on pool %p could not publish its mapping requests (%d)", (void *)v->mem, result);
    return result == RC_BACKING_OK;
}

// Per-thread copies of resolved import and physical pools that overlap no other index entry (so pidx_find picks them for
// any range inside), valid while pool_gen is unchanged: the replay resolves a few dozen pools thousands of times per
// frame (16 vertex streams and every buffer binding of each draw). Direct-mapped by the 4 KiB page of the address, so
// a hit is one compare however many pools a frame touches. NX2NAVI_RESOLVE_CHECK=1 compares every hit with the locked
// lookup and stops on a difference.
static int rec_class(const PoolRec *r) {
    if (r->kind != POOL_IMPORT) return VK_MCLS_DEVICE;
    return r->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS ? VK_MCLS_HOST_NOACC
         : r->flags & NVN_MEMORY_POOL_FLAGS_CPU_CACHED ? VK_MCLS_HOST_CACHED : VK_MCLS_HOST_UNC;
}

typedef struct { uint64_t gen, lo, hi, base, end; VkBuffer buf; VmaAllocation alloc; int cls; void *pool; } ResolveHit;
#define RHIT_SLOTS 512
static _Thread_local ResolveHit rhit[RHIT_SLOTS];

static ResolveHit *rhit_slot(uint64_t addr) { return &rhit[(addr >> 12 ^ addr >> 21) & (RHIT_SLOTS - 1)]; }

static int resolve_check_on(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("NX2NAVI_RESOLVE_CHECK");
        on = e && *e && strcmp(e, "0");
    }
    return on;
}

static int resolve_locked(uint64_t addr, uint64_t size, VkMemSpan *out, int touch);

// pool_lock shared. Native recording uses the selected buffer's existing submission lifetime; the store's
// borrowed provider token remains valid while resolving and marking that exact physical pool as used.
static int pool_device_span(PoolRec *r, uint64_t addr, uint64_t size, VkMemSpan *out, PoolRec **provider) {
    RcBackingRun run;
    uint32_t count;
    if (r->kind == POOL_PHYS && !size && addr == r->lo + r->size) {
        // The address index accepts the end of a physical range for zero-width queries.
        run = (RcBackingRun){addr, {r->backing.id, r->size, 0}, (uintptr_t)r};
    } else if (rc_backing_map_runs(pool_backings, r->map, addr, &run, 1, &count) != RC_BACKING_OK || !count) {
        return 0;
    }
    if (!run.provider || run.range.size < size) return 0;
    PoolRec *ph = (PoolRec *)run.provider;
    if (provider) *provider = ph;
    out->buffer = ph->buf;
    out->offset = out->alloc_offset = run.range.offset;
    out->avail = run.range.size;
    out->alloc = ph->alloc;
    out->cls = VK_MCLS_DEVICE;
    if (!ph->used) plat_xchg32(&ph->used, 1);
    return 1;
}

// touch: the GPU may read the pool's buffer in the submission being recorded, so its shadow goes on the touched list
// (shadow storage). A lookup (vk_mem_lookup) leaves it off: the caller reads guest memory itself or only asks what
// backs the address. Shaders that reach a pool through the device address table resolve nothing: the part they run in
// brings every pool of the table up to date (vk_mem_shadow_sync).
static int mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *out, int touch) {
    memset(out, 0, sizeof *out);
    uint64_t g = __atomic_load_n(&pool_gen, __ATOMIC_ACQUIRE), end = addr + size;
    const ResolveHit *h = rhit_slot(addr);
    if (end >= addr && h->gen == g && addr >= h->lo && end <= h->hi) {
        out->buffer = h->buf;
        out->offset = addr - h->base;
        out->avail = h->end - addr;
        out->alloc = h->alloc;
        out->cls = h->cls;
        out->pool = h->pool;
        if (h->alloc) out->alloc_offset = out->offset;
        if (shadow_on && touch && h->pool && !((PoolRec *)h->pool)->stouched) shadow_touch(h->pool);
        if (resolve_check_on()) {
            VkMemSpan s;
            if (!resolve_locked(addr, size, &s, touch) || memcmp(&s, out, sizeof s))
                rt_fatal("vk: resolve cache of 0x%llx differs from the pool index", (unsigned long long)addr);
        }
        return 1;
    }
    return resolve_locked(addr, size, out, touch);
}

int vk_mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *out) { return mem_resolve(addr, size, out, 1); }
int vk_mem_lookup(uint64_t addr, uint64_t size, VkMemSpan *out) { return mem_resolve(addr, size, out, 0); }

// The pool index lookup under the lock; a non-overlapped import or physical pool goes into the address's cache slot.
static int resolve_locked(uint64_t addr, uint64_t size, VkMemSpan *out, int touch) {
    memset(out, 0, sizeof *out);
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)addr, size);
    PoolRec *r = e ? e->buf : NULL;
    int ok = 0;
    if (r && (r->kind == POOL_IMPORT ? r->buf != VK_NULL_HANDLE : r->kind == POOL_PHYS)) {
        int i = (int)(e - pool_index.e);
        if ((i == 0 || pool_index.maxhi[i - 1] <= e->lo) && (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= e->hi)) {
            ResolveHit *h = rhit_slot(addr);
            h->gen = __atomic_load_n(&pool_gen, __ATOMIC_RELAXED);   // stable: changed only under the exclusive lock
            h->lo = e->lo, h->hi = e->hi, h->buf = r->buf, h->cls = rec_class(r);
            if (r->kind == POOL_IMPORT) h->base = r->import_lo, h->end = r->import_hi, h->alloc = NULL, h->pool = r;
            else h->base = r->lo, h->end = r->lo + r->size, h->alloc = r->alloc, h->pool = NULL;
        }
    }
    if (r && r->kind == POOL_IMPORT && r->buf) {
        out->cls = rec_class(r);
        out->pool = r;
        out->buffer = r->buf;
        out->offset = addr - r->import_lo;
        out->avail = r->import_hi - addr;
        if (r->smap && touch && !r->stouched) shadow_touch(r);
        ok = 1;
    } else if (r && (r->kind == POOL_PHYS || r->kind == POOL_VIRT)) {
        ok = pool_device_span(r, addr, size, out, NULL);
    }
    if (ok && !r->used) plat_xchg32(&r->used, 1);
    plat_mutex_unlock_shared(&pool_lock);
    if (!ok) memset(out, 0, sizeof *out);
    return ok;
}

struct VkMemSpanCapture {
    PoolRec *window, *provider;
    RcBackingCapture *mapping;
    MemBackingCapture *host;
    VkMemSpan selected;
    uint64_t address, size, mirror_incarnation, import_mirror_scene;
    uint64_t transition_mirror, writer_issued_scene;
    int native_input, input_recorded;
    int cold_writer, writer_published, retire_mirror;
    uint32_t references, device_count;
    RcInterval *device_intervals;
    RcInterval device_single;
};

static uint64_t mirror_import_scene(const PoolRec *pool);
static uint64_t mirror_scene(void);
static int mirror_pool_entry(const PoolRec *);
static int mirror_pool_alone(int);
static int mirror_transition_ready_locked(const PoolRec *, RcRegistryPosition);
static void mirror_retire_check(PoolRec *, int completed);
static void mirror_retire_completed(void *);
static void mirror_retire_canceled(void *, int may_have_executed);
static int host_interval_vectors_match(const MemBackingCapture *source, const MemBackingCapture *provider,
                                       uint64_t address, uint64_t size, VkMemCanonicalSourceQualification *report);

// The native import maps the original pointer affinely. An outward alignment hole can prevent retaining the
// whole import vector, but an exact fully mapped subrange is usable if its identity stayed stable since import.
static RcRegistryResult import_mapping_current(const PoolRec *pool, const MemBackingCapture *source,
                                              uint64_t address, uint64_t size) {
    const MemBackingCapture *expected = source ? source : pool->host_backing;
    if (!expected) return RC_REGISTRY_UNSUPPORTED;
    if (!mem_backing_current_range(expected, address, size)) return RC_REGISTRY_STALE;
    if (pool->host_backing)
        return !source || host_interval_vectors_match(source, pool->host_backing, address, size, NULL) ?
               RC_REGISTRY_OK : RC_REGISTRY_STALE;
    if (!pool->host_mapping_cutoff) return RC_REGISTRY_UNSUPPORTED;
    return mem_mapping_range_unchanged(address, size, pool->host_mapping_cutoff) ?
           RC_REGISTRY_OK : RC_REGISTRY_STALE;
}

// pool_lock held. Compare only the actual argument span and current winner at this address.
static int span_capture_match(const VkMemSpanCapture *capture, const VkMemSpan *selected) {
    const VkMemSpan *old = &capture->selected;
    if (selected && (old->buffer != selected->buffer || old->offset != selected->offset || old->alloc != selected->alloc ||
                     old->alloc_offset != selected->alloc_offset || old->pool != selected->pool ||
                     selected->avail < capture->size)) return 0;
    const pidx_ent *entry = pidx_find(&pool_index, capture->address, capture->size);
    PoolRec *window = entry ? entry->buf : NULL;
    if (!window || window != capture->window) return 0;
    if (window->kind == POOL_IMPORT) {
        if (capture->native_input && !capture->mirror_incarnation && vk_mirror_on) {
            if (capture->import_mirror_scene) {
                if (mirror_import_scene(window) != capture->import_mirror_scene) return 0;
            } else if (window->mstate != MS_REFUSED) return 0;
        }
        VkBuffer buffer = window->buf;
        if (capture->mirror_incarnation) {
            if (window->mstate != MS_ON || window->mirror_incarnation != capture->mirror_incarnation) return 0;
            buffer = window->mbuf;
        }
        return window == old->pool && old->buffer == buffer && capture->address >= window->import_lo &&
               old->offset == capture->address - window->import_lo &&
               import_mapping_current(window, capture->host, capture->address, capture->size) == RC_REGISTRY_OK;
    }
    if (capture->native_input)
        return rc_backing_map_intervals_current(pool_backings, window->map, capture->device_intervals,
                                                 capture->device_count, capture->address, capture->size);
    return rc_backing_capture_current(pool_backings, capture->mapping);
}

// The native ref pins the allocation itself. The mapping receipt is immutable metadata and does not
// occupy a backing-store capture slot for every draw in a preplanned group.
static RcBackingResult span_input_device_map(VkMemSpanCapture *capture) {
    PoolRec *window = capture->window;
    uint32_t count;
    RcBackingResult result = rc_backing_map_intervals(pool_backings, window->map, capture->address, capture->size,
                                                       NULL, 0, &count);
    if (result != RC_BACKING_OK) return result;
    if (!count) return RC_BACKING_INVALID;
    RcInterval *intervals = count == 1 ? &capture->device_single : calloc(count, sizeof *intervals);
    if (!intervals) return RC_BACKING_CAPACITY;
    result = rc_backing_map_intervals(pool_backings, window->map, capture->address, capture->size,
                                     intervals, count, &capture->device_count);
    if (result == RC_BACKING_OK) capture->device_intervals = intervals;
    else if (intervals != &capture->device_single) free(intervals);
    return result;
}

static VkMemSpanCapture *span_capture_create(uint64_t address, uint64_t size, const VkMemSpan *selected,
                                              int native_input, RcRegistryResult *result) {
    *result = RC_REGISTRY_INVALID;
    if (!size || !selected || !selected->buffer || selected->avail < size) return NULL;
    VkMemSpanCapture *capture = calloc(1, sizeof *capture);
    if (!capture) { *result = RC_REGISTRY_CAPACITY; return NULL; }
    capture->address = address;
    capture->size = size;
    capture->references = 1;
    capture->selected = *selected;
    capture->native_input = native_input;
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *entry = pidx_find(&pool_index, address, size);
    PoolRec *window = entry ? entry->buf : NULL;
    int captured = 0;
    capture->window = window;
    *result = RC_REGISTRY_STALE;
    if (native_input && window && window->kind == POOL_IMPORT && vk_mirror_on)
        capture->import_mirror_scene = mirror_import_scene(window);
    if (native_input && window && window->kind == POOL_IMPORT && vk_mirror_on &&
        window->mstate != MS_ON && window->mstate != MS_REFUSED && !capture->import_mirror_scene) {
        *result = RC_REGISTRY_UNSUPPORTED;
    } else if (window && window->kind == POOL_IMPORT) {
        capture->provider = window;
        if (native_input && vk_mirror_on && window->mstate == MS_ON && selected->pool == window &&
            selected->buffer == window->buf) capture->selected.buffer = window->mbuf;
        if (window->mstate == MS_ON && capture->selected.buffer == window->mbuf)
            capture->mirror_incarnation = window->mirror_incarnation;
        if (!window->host_backing && window->host_mapping_cutoff) {
            RcBackingResult mapped = mem_backing_capture(address, size, &capture->host);
            if (native_input && (mapped == RC_BACKING_CAPACITY || mapped == RC_BACKING_EXHAUSTED))
                *result = RC_REGISTRY_CAPACITY;
        } else if (native_input && !window->host_backing) {
            *result = RC_REGISTRY_UNSUPPORTED;
        }
        captured = span_capture_match(capture, NULL);
    } else if (window && (window->kind == POOL_PHYS || window->kind == POOL_VIRT)) {
        VkMemSpan span = {0};
        if (pool_device_span(window, address, size, &span, &capture->provider) &&
            span.buffer == selected->buffer && span.offset == selected->offset &&
            span.alloc == selected->alloc && span.alloc_offset == selected->alloc_offset) {
            RcBackingResult mapped = native_input ? span_input_device_map(capture) :
                rc_backing_capture(pool_backings, window->map, address, size, RC_BACKING_REQUIRE_MAPPED,
                                   &capture->mapping);
            captured = mapped == RC_BACKING_OK;
            if (native_input && (mapped == RC_BACKING_CAPACITY || mapped == RC_BACKING_EXHAUSTED))
                *result = RC_REGISTRY_CAPACITY;
        }
    }
    if (captured) {
        if (native_input && capture->mirror_incarnation &&
            __atomic_fetch_add(&window->mirror_input_refs, 1, __ATOMIC_RELAXED) == UINT32_MAX)
            rt_fatal("vk: mirror input reference exhausted");
        pool_native_acquire(window);
        if (capture->provider != window) pool_native_acquire(capture->provider);
    }
    plat_mutex_unlock_shared(&pool_lock);
    if (captured) { *result = RC_REGISTRY_OK; return capture; }
    mem_backing_release(&capture->host);
    free(capture);
    return NULL;
}

VkMemSpanCapture *vk_mem_span_capture(uint64_t address, uint64_t size, const VkMemSpan *selected) {
    RcRegistryResult result;
    return span_capture_create(address, size, selected, 0, &result);
}

// pool_lock held. Indexed containment is the logical pool range, independent of import padding.
static PoolRec *snapshot_import_window(const VkMemSpan *selected, const MemBackingCapturePlan *host) {
    const pidx_ent *entry = pidx_find(&pool_index, host->address, host->size);
    PoolRec *window = entry ? entry->buf : NULL;
    if (!window || window->kind != POOL_IMPORT || window->unpublished || !window->buf ||
        selected->pool != window || selected->buffer != window->buf || selected->alloc ||
        selected->avail < host->size || host->address < window->import_lo ||
        selected->offset != host->address - window->import_lo) return NULL;
    return window;
}

RcRegistryResult vk_mem_span_capture_plan(const VkMemSpan *selected, const MemBackingCapturePlan *host,
                                          VkMemSpanCapturePlan *out) {
    if (!selected || !host || !host->bytes || !out) return RC_REGISTRY_INVALID;
    *out = (VkMemSpanCapturePlan){0};
    plat_mutex_lock_shared(&pool_lock);
    PoolRec *window = snapshot_import_window(selected, host);
    RcRegistryResult result = RC_REGISTRY_STALE;
    if (window) {
        int nested = !window->host_backing;
        if (nested) {
            if (!window->host_mapping_cutoff) result = RC_REGISTRY_UNSUPPORTED;
            else if (mem_mapping_range_unchanged(host->address, host->size, window->host_mapping_cutoff))
                result = RC_REGISTRY_OK;
        } else result = import_mapping_current(window, NULL, host->address, host->size);
        if (result == RC_REGISTRY_OK) {
            if (nested && host->bytes > SIZE_MAX - sizeof(VkMemSpanCapture)) result = RC_REGISTRY_CAPACITY;
            else *out = (VkMemSpanCapturePlan){.selected = *selected, .nested_host = nested,
                .publication = window->publication,
                .bytes = sizeof(VkMemSpanCapture) + (nested ? host->bytes : 0)};
        }
    }
    plat_mutex_unlock_shared(&pool_lock);
    return result;
}

RcRegistryResult vk_mem_span_capture_bounded(const VkMemSpanCapturePlan *plan, const MemBackingCapturePlan *host,
                                             VkMemSpanCapture **out) {
    if (!plan || !host || !host->bytes || !out || *out || plan->bytes < sizeof(VkMemSpanCapture) ||
        plan->bytes - sizeof(VkMemSpanCapture) != (plan->nested_host ? host->bytes : 0))
        return RC_REGISTRY_INVALID;
    plat_mutex_lock_shared(&pool_lock);
    PoolRec *window = snapshot_import_window(&plan->selected, host);
    RcRegistryResult result = RC_REGISTRY_STALE;
    VkMemSpanCapture value = {.selected = plan->selected, .address = host->address, .size = host->size,
                              .window = window, .provider = window, .references = 1};
    if (window && plan->publication == window->publication &&
        plan->nested_host == !window->host_backing) {
        RcBackingResult mapped = plan->nested_host
            ? mem_backing_capture_bounded(host, &value.host) : RC_BACKING_OK;
        if (mapped == RC_BACKING_CAPACITY) result = RC_REGISTRY_CAPACITY;
        else if (mapped == RC_BACKING_OK && span_capture_match(&value, &plan->selected)) {
            VkMemSpanCapture *capture = malloc(sizeof *capture);
            if (capture) {
                *capture = value;
                pool_native_acquire(window);
                *out = capture;
                result = RC_REGISTRY_OK;
            } else result = RC_REGISTRY_CAPACITY;
        }
    }
    plat_mutex_unlock_shared(&pool_lock);
    if (result != RC_REGISTRY_OK) mem_backing_release(&value.host);
    return result;
}

static int import_writer_candidate(const PoolRec *pool) {
    if (!pool || !vk_mirror_on || pool->smap) return VK_MEM_IMPORT_WRITER_NONE;
    if (pool->mstate == MS_ON) return VK_MEM_IMPORT_WRITER_ACTIVE;
    if (pool->mstate == MS_DORMANT ||
        ((!pool->mirror_incarnation || pool->mirror_import_transitioned) &&
         (pool->mstate == MS_NONE || pool->mstate == MS_PENDING || pool->mstate == MS_CREATING)))
        return VK_MEM_IMPORT_WRITER_COLD;
    return VK_MEM_IMPORT_WRITER_NONE;
}

int vk_mem_span_import_writer_candidate(uint64_t address, const VkMemSpan *span) {
    if (!span || !address || !vk_mirror_on) return 0;
    const MemBackingCapturePlan input = {.address = address, .size = 1};
    plat_mutex_lock_shared(&pool_lock);
    PoolRec *pool = snapshot_import_window(span, &input);
    int candidate = import_writer_candidate(pool);
    if (candidate && !mirror_pool_alone(mirror_pool_entry(pool))) candidate = VK_MEM_IMPORT_WRITER_NONE;
    plat_mutex_unlock_shared(&pool_lock);
    return candidate;
}

// pool_lock held. The raw import is still the mapping authority when a mirror supplies the native buffer.
static RcRegistryResult native_import_plan_locked(const VkMemSpan *imported, const MemBackingCapturePlan *host,
    int cold_writer, VkMemImportInputPlan *plan, PoolRec **window) {
    *window = snapshot_import_window(imported, host);
    PoolRec *pool = *window;
    if (!pool) return RC_REGISTRY_STALE;
    RcRegistryResult result = pool->host_backing ? import_mapping_current(pool, NULL, host->address, host->size) :
        pool->host_mapping_cutoff && mem_mapping_range_unchanged(host->address, host->size, pool->host_mapping_cutoff)
        ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
    if (result != RC_REGISTRY_OK) return result;
    VkMemImportInputPlan next = {.imported = *imported, .selected = *imported, .publication = pool->publication,
        .bytes = sizeof(VkMemSpanCapture) + host->bytes, .cold_writer = cold_writer};
    next.selected.avail = host->size;
    if (cold_writer) {
        int mode = import_writer_candidate(pool);
        if (!mode || pool->mstate == MS_CREATING) return RC_REGISTRY_STALE;
        if (!pool->gpa_host_coherent || !mirror_pool_alone(mirror_pool_entry(pool))) return RC_REGISTRY_UNSUPPORTED;
        if (mode == VK_MEM_IMPORT_WRITER_ACTIVE) {
            if (!mirror_transition_ready_locked(pool, vk_exec_resource_position())) return RC_REGISTRY_BLOCKED;
            next.transition_mirror = pool->mirror_incarnation;
        }
        next.import_scene = mirror_scene();
    } else if (vk_mirror_on && pool->mstate == MS_ON) {
        if (!pool->mbuf || !pool->mirror_incarnation) return RC_REGISTRY_UNSUPPORTED;
        next.selected.buffer = pool->mbuf;
        next.mirror_incarnation = pool->mirror_incarnation;
    } else if (vk_mirror_on && pool->mstate != MS_REFUSED) {
        next.import_scene = mirror_import_scene(pool);
        if (!next.import_scene) return RC_REGISTRY_UNSUPPORTED;
    }
    *plan = next;
    return RC_REGISTRY_OK;
}

static RcRegistryResult import_input_plan(const VkMemSpan *imported, const MemBackingCapturePlan *host,
                                          int cold_writer, VkMemImportInputPlan *out) {
    if (!out) return RC_REGISTRY_INVALID;
    *out = (VkMemImportInputPlan){0};
    if (!imported || !host || !host->bytes || !host->count || !host->size ||
        host->size > UINT64_MAX - host->address || host->bytes > SIZE_MAX - sizeof(VkMemSpanCapture))
        return RC_REGISTRY_INVALID;
    PoolRec *window;
    plat_mutex_lock_shared(&pool_lock);
    RcRegistryResult result = native_import_plan_locked(imported, host, cold_writer, out, &window);
    plat_mutex_unlock_shared(&pool_lock);
    return result;
}

RcRegistryResult vk_mem_span_import_input_plan(const VkMemSpan *imported, const MemBackingCapturePlan *host,
                                               VkMemImportInputPlan *out) {
    return import_input_plan(imported, host, 0, out);
}

RcRegistryResult vk_mem_span_import_writer_plan(const VkMemSpan *imported, const MemBackingCapturePlan *host,
                                                VkMemImportInputPlan *out) {
    return import_input_plan(imported, host, 1, out);
}

RcRegistryResult vk_mem_span_import_input_capture(const VkMemImportInputPlan *plan, const MemBackingCapturePlan *host,
                                                   VkMemSpanCapture **out) {
    if (!plan || !host || !out || *out || !host->bytes || !host->count || !host->size ||
        host->size > UINT64_MAX - host->address || host->bytes > SIZE_MAX - sizeof(VkMemSpanCapture) ||
        plan->bytes != sizeof(VkMemSpanCapture) + host->bytes)
        return RC_REGISTRY_INVALID;
    MemBackingCapture *mapping = NULL;
    RcBackingResult mapped = mem_backing_capture_bounded(host, &mapping);
    if (mapped != RC_BACKING_OK)
        return mapped == RC_BACKING_CAPACITY || mapped == RC_BACKING_EXHAUSTED ? RC_REGISTRY_CAPACITY : RC_REGISTRY_STALE;
    VkMemSpanCapture *capture = calloc(1, sizeof *capture);
    if (!capture) {
        mem_backing_release(&mapping);
        return RC_REGISTRY_CAPACITY;
    }
    VkMemImportInputPlan current;
    PoolRec *window;
    plat_mutex_lock_shared(&pool_lock);
    RcRegistryResult result = native_import_plan_locked(&plan->imported, host, plan->cold_writer, &current, &window);
    if (result == RC_REGISTRY_OK && (current.publication != plan->publication ||
        current.mirror_incarnation != plan->mirror_incarnation || current.import_scene != plan->import_scene ||
        current.transition_mirror != plan->transition_mirror ||
        current.selected.buffer != plan->selected.buffer || current.selected.pool != plan->selected.pool ||
        current.selected.offset != plan->selected.offset || current.selected.avail != plan->selected.avail ||
        current.selected.alloc != plan->selected.alloc || current.selected.alloc_offset != plan->selected.alloc_offset ||
        current.selected.cls != plan->selected.cls ||
        import_mapping_current(window, mapping, host->address, host->size) != RC_REGISTRY_OK)) result = RC_REGISTRY_STALE;
    if (result == RC_REGISTRY_OK) {
        *capture = (VkMemSpanCapture){.window = window, .provider = window, .host = mapping,
            .selected = current.selected, .address = host->address, .size = host->size, .references = 1, .native_input = 1,
            .mirror_incarnation = current.mirror_incarnation, .import_mirror_scene = current.import_scene,
            .cold_writer = current.cold_writer, .transition_mirror = current.transition_mirror};
        if (capture->mirror_incarnation &&
            __atomic_fetch_add(&window->mirror_input_refs, 1, __ATOMIC_RELAXED) == UINT32_MAX)
            rt_fatal("vk: mirror import write reference exhausted");
        pool_native_acquire(window);
    }
    plat_mutex_unlock_shared(&pool_lock);
    if (result == RC_REGISTRY_OK) *out = capture;
    else {
        mem_backing_release(&mapping);
        free(capture);
    }
    return result;
}

uint32_t vk_mem_span_import_input_ranges(const VkMemSpanCapture *capture, const RcInterval **ranges) {
    if (!ranges) return 0;
    uint32_t count;
    *ranges = mem_backing_intervals(capture && capture->native_input ? capture->host : NULL, &count);
    return count;
}

RcRegistryResult vk_mem_span_import_writers_publish(VkMemSpanCapture *const *batch, uint32_t count,
                                                    RcRegistryPosition read) {
    if (!batch || !count || vk_exec_shared() ||
        rc_registry_position_compare(read, vk_exec_resource_position()) ||
        !rc_registry_provisional_read_current(rc_registry_runtime(), read)) return RC_REGISTRY_INVALID;
    RcRegistryResult result = RC_REGISTRY_OK;
    const uint64_t scene = mirror_scene();
    PoolRec *projected = NULL;
    int transitioning = 0, lazy_locked = 0;
    plat_mutex_lock(&pool_lock);
    for (uint32_t i = 0; i < count; ++i) {
        VkMemSpanCapture *capture = batch[i];
        if (!capture || !capture->cold_writer || capture->writer_published || capture->import_mirror_scene != scene) {
            result = RC_REGISTRY_INVALID;
            break;
        }
        const MemBackingCapturePlan span = {.address = capture->address, .size = capture->size};
        PoolRec *pool = snapshot_import_window(&capture->selected, &span);
        int candidate = import_writer_candidate(pool);
        if (pool != capture->window || !candidate || pool->mstate == MS_CREATING ||
            (capture->transition_mirror ? candidate != VK_MEM_IMPORT_WRITER_ACTIVE ||
                capture->transition_mirror != pool->mirror_incarnation : candidate != VK_MEM_IMPORT_WRITER_COLD) ||
            !pool->gpa_host_coherent || !mirror_pool_alone(mirror_pool_entry(pool)) ||
            (pool->import_writer_refs && pool->import_writer_scene != scene) ||
            import_mapping_current(pool, capture->host, capture->address, capture->size) != RC_REGISTRY_OK) {
            result = RC_REGISTRY_STALE;
            break;
        }
        if (capture->transition_mirror) {
            if (!mirror_transition_ready_locked(pool, read) || (pool->gpa_rec && projected && projected != pool)) {
                result = RC_REGISTRY_BLOCKED;
                break;
            }
            transitioning = 1;
            if (pool->gpa_rec) projected = pool;
        }
        if (pool->import_writer_refs > UINT32_MAX - count) {
            result = RC_REGISTRY_CAPACITY;
            break;
        }
        for (uint32_t j = 0; j < i; ++j)
            if (batch[j] == capture) result = RC_REGISTRY_INVALID;
        if (result != RC_REGISTRY_OK) break;
    }
    if (result == RC_REGISTRY_OK && transitioning) {
        lazy_locked = mem_lazy_provider_transition_try();
        if (!lazy_locked) result = RC_REGISTRY_BLOCKED;
        for (uint32_t i = 0; result == RC_REGISTRY_OK && i < count; ++i) {
            const VkMemSpanCapture *capture = batch[i];
            PoolRec *pool = capture->window;
            if (capture->transition_mirror && mem_lazy_provider_borrowed(pool, (uint64_t)(uintptr_t)pool->mbuf))
                result = RC_REGISTRY_BLOCKED;
        }
    }
    if (result == RC_REGISTRY_OK && projected) {
        uint32_t replacement = vk_gpa_remap_exact(projected->import_lo, projected->import_hi, projected->gpa_rec,
            projected->buf, projected->import_lo, 0, projected->gpa_host_coherent, NULL);
        if (replacement) projected->gpa_rec = replacement;
        else result = RC_REGISTRY_BLOCKED;
    }
    if (result == RC_REGISTRY_OK) {
        for (uint32_t i = 0; i < count; ++i) {
            VkMemSpanCapture *capture = batch[i];
            PoolRec *pool = capture->window;
            if (capture->transition_mirror && pool->mstate == MS_ON) {
                // Existing descriptors still name the original VkBuffer. Only future selections switch.
                mirror_forget(pool);
                pool->mirror_retire_completed = 0;
                pool->mirror_import_transitioned = 1;
                pool->mstate = MS_DORMANT;
                pool_native_acquire(pool);
                capture->retire_mirror = 1;
            }
            pool->import_writer_scene = scene;
            pool->import_writer_refs++;
            capture->writer_published = 1;
        }
    }
    if (lazy_locked) mem_lazy_provider_transition_end();
    plat_mutex_unlock(&pool_lock);
    if (result == RC_REGISTRY_OK)
        for (uint32_t i = 0; i < count; ++i)
            if (batch[i]->retire_mirror)
                vk_completion_add_owned(read.scene, mirror_retire_completed, mirror_retire_canceled, batch[i]->window);
    return result;
}

static void import_writer_release(VkMemSpanCapture *capture, uint64_t issued) {
    if (!capture || !capture->writer_published) return;
    PoolRec *pool = capture->window;
    plat_mutex_lock(&pool_lock);
    if (!pool->import_writer_refs || pool->import_writer_scene != capture->import_mirror_scene)
        rt_fatal("vk: import writer lost its selected scene");
    if (issued) {
        if (pool->import_writer_fence < issued) pool->import_writer_fence = issued;
        pool->mirror_import_only_scene = capture->import_mirror_scene;
        // The held descriptor survives release of its pending publication. Later effect commit
        // checks the exact issued scene through this retained capture.
        capture->writer_issued_scene = issued;
    }
    pool->import_writer_refs--;
    capture->writer_published = 0;
    plat_mutex_unlock(&pool_lock);
}

void vk_mem_span_import_writer_issued(VkMemSpanCapture *capture, uint64_t fence) {
    if (!fence) rt_fatal("vk: import writer issued without a scene fence");
    import_writer_release(capture, fence);
}

RcRegistryResult vk_mem_span_capture_bound_input(uint64_t address, uint64_t size, const VkMemSpan *selected,
                                                  VkMemSpanCapture **out) {
    if (!out || *out) return RC_REGISTRY_INVALID;
    RcRegistryResult result;
    VkMemSpanCapture *capture = span_capture_create(address, size, selected, 1, &result);
    if (!capture) return result;
    if (capture->selected.buffer != selected->buffer) {
        vk_mem_span_capture_release(&capture);
        return RC_REGISTRY_STALE;
    }
    *out = capture;
    return RC_REGISTRY_OK;
}

static int mirror_prepare_input(const VkMemSpanCapture *, const VkMemSpan *, VkMemMirrorInputPolicy);

RcRegistryResult vk_mem_span_prepare_native_input(uint64_t address, uint64_t size, int clamp, VkMemMirrorInputPolicy policy,
                                                  VkMemSpan *span, VkMemSpanCapture **owned) {
    VkMemSpan selected;
    if (*owned || !address || (!clamp && !size)) return RC_REGISTRY_INVALID;
    if (!vk_mem_lookup(address, clamp ? 1 : size, &selected)) return RC_REGISTRY_UNSUPPORTED;
    uint64_t extent = clamp && (!size || size > selected.avail) ? selected.avail : size;
    RcRegistryResult result;
    VkMemSpanCapture *capture = span_capture_create(address, extent, &selected, 1, &result);
    if (!capture && result == RC_REGISTRY_UNSUPPORTED && policy != VK_MEM_MIRROR_EXISTING &&
        vk_mirror_on && selected.pool) {
        RcRegistryResult pinned;
        VkMemSpanCapture *import = span_capture_create(address, extent, &selected, 0, &pinned);
        if (import && mirror_prepare_input(import, &selected, policy))
            capture = span_capture_create(address, extent, &selected, 1, &result);
        else if (pinned != RC_REGISTRY_OK) result = pinned;
        vk_mem_span_capture_release(&import);
    }
    if (!capture) return result;
    *owned = capture;
    *span = capture->selected;
    span->avail = extent;
    return RC_REGISTRY_OK;
}

RcRegistryResult vk_mem_span_native_input(uint64_t address, uint64_t size, int clamp, VkMemSpan *span,
                                          VkMemSpanCapture **owned) {
    return vk_mem_span_prepare_native_input(address, size, clamp, VK_MEM_MIRROR_EXISTING, span, owned);
}

static int span_record_mirror_input(const VkMemSpanCapture *capture, const VkMemSpan *selected);

int vk_mem_span_record_input(VkMemSpanCapture *capture, const VkMemSpan *selected) {
    if (!capture || !capture->native_input || (capture->cold_writer && !capture->writer_published)) return 0;
    if (capture->mirror_incarnation)
        return capture->input_recorded = span_record_mirror_input(capture, selected);
    plat_mutex_lock_shared(&pool_lock);
    int current = span_capture_match(capture, selected);
    if (current && capture->window->smap && !capture->window->stouched) shadow_touch(capture->window);
    plat_mutex_unlock_shared(&pool_lock);
    return capture->input_recorded = current;
}

// The ordinary argument hold already identifies its provider. Record its late shadow use without
// choosing a mirror or letting an installed mirror replace a previously selected import.
static int held_argument_matches(const VkMemSpanCapture *capture, uint64_t address, uint64_t size,
                                 const VkMemSpan *selected) {
    return capture && selected && capture->address == address && capture->size == size;
}

static int held_plain_argument_current(const VkMemSpanCapture *capture, const VkMemSpan *selected) {
    const PoolRec *window = capture->window;
    return span_capture_match(capture, selected) &&
           (window->kind != POOL_IMPORT || !vk_mirror_on ||
            (window->mstate != MS_ON && window->mstate != MS_CREATING));
}

int vk_mem_span_held_argument_current(const VkMemSpanCapture *capture,uint64_t address,uint64_t size,
                                      const VkMemSpan *selected) {
    if (!held_argument_matches(capture, address, size, selected)) return 0;
    plat_mutex_lock_shared(&pool_lock);
    int current = capture->native_input ? span_capture_match(capture, selected)
                                       : held_plain_argument_current(capture, selected);
    plat_mutex_unlock_shared(&pool_lock);
    return current;
}

RcRegistryResult vk_mem_span_prepare_held_argument(const VkMemSpanCapture *capture, const VkMemSpan *selected,
                                                    int ordered) {
    if (!capture || !selected) return RC_REGISTRY_INVALID;
    plat_mutex_lock_shared(&pool_lock);
    const PoolRec *window = capture->window;
    const int current = capture->native_input ? span_capture_match(capture, selected)
                                              : held_plain_argument_current(capture, selected);
    const int needs_import_scope = current && !capture->native_input && vk_mirror_on &&
        window->kind == POOL_IMPORT && window->mstate != MS_REFUSED && !mirror_import_scene(window);
    plat_mutex_unlock_shared(&pool_lock);
    if (!current) return RC_REGISTRY_STALE;
    if (!needs_import_scope) return RC_REGISTRY_OK;
    if (!ordered || vk_exec_shared()) return RC_REGISTRY_BUSY;
    // Keep the exact retained import for this recording scene. This uses the existing import-only
    // policy, including any pending earlier writers, without looking up or moving the argument.
    if (!mirror_prepare_input(capture, selected, VK_MEM_MIRROR_IMPORT) ||
        !vk_mem_span_held_argument_current(capture, capture->address, capture->size, selected))
        return RC_REGISTRY_STALE;
    return RC_REGISTRY_OK;
}

int vk_mem_span_record_held_argument(VkMemSpanCapture *capture,uint64_t address,uint64_t size,
                                     const VkMemSpan *selected) {
    if (!held_argument_matches(capture, address, size, selected)) return 0;
    if (capture->native_input) return vk_mem_span_record_input(capture, selected);
    plat_mutex_lock_shared(&pool_lock);
    int current = held_plain_argument_current(capture, selected);
    if (current && capture->window->smap && !capture->window->stouched) shadow_touch(capture->window);
    plat_mutex_unlock_shared(&pool_lock);
    return current;
}

int vk_mem_span_capture_current(const VkMemSpanCapture *capture, const VkMemSpan *selected) {
    if (!capture) return 0;
    plat_mutex_lock_shared(&pool_lock);
    int current = span_capture_match(capture, selected);
    plat_mutex_unlock_shared(&pool_lock);
    return current;
}

static void mirror_input_release(PoolRec *pool, uint64_t incarnation) {
    if (!incarnation) return;
    if (incarnation != pool->mirror_incarnation) rt_fatal("vk: releasing a different native mirror incarnation");
    uint32_t prior = __atomic_fetch_sub(&pool->mirror_input_refs, 1, __ATOMIC_ACQ_REL);
    if (!prior) rt_fatal("vk: releasing an unowned native mirror");
    if (prior == 1) mirror_retire_check(pool, 0);
}

static void span_capture_dispose(VkMemSpanCapture *capture, int keep_provider) {
    import_writer_release(capture, 0);
    rc_backing_capture_release(&capture->mapping);
    mem_backing_release(&capture->host);
    if (capture->device_intervals != &capture->device_single) free(capture->device_intervals);
    VkCtx *c = vk_ctx();
    if (!keep_provider && capture->native_input) mirror_input_release(capture->provider, capture->mirror_incarnation);
    if (!keep_provider && capture->provider != capture->window) destroy_rec(c, capture->provider);
    if (!keep_provider || capture->provider != capture->window) destroy_rec(c, capture->window);
    free(capture);
}

VkMemSpanCapture *vk_mem_span_capture_retain(VkMemSpanCapture *capture) {
    if (capture && __atomic_add_fetch(&capture->references, 1, __ATOMIC_RELAXED) == 0)
        rt_fatal("vk: native span capture reference exhausted");
    return capture;
}

void vk_mem_span_capture_release(VkMemSpanCapture **owned) {
    if (!owned || !*owned) return;
    VkMemSpanCapture *capture = *owned;
    *owned = NULL;
    if (__atomic_sub_fetch(&capture->references, 1, __ATOMIC_ACQ_REL) == 0)
        span_capture_dispose(capture, 0);
}

VkMemInputProvider vk_mem_span_take_input_provider(VkMemSpanCapture **owned) {
    if (!owned || !*owned || !(*owned)->input_recorded) return (VkMemInputProvider){0};
    VkMemSpanCapture *capture = *owned;
    VkMemInputProvider provider = {capture->provider, capture->mirror_incarnation};
    if (__atomic_load_n(&capture->references, __ATOMIC_ACQUIRE) == 1) {
        *owned = NULL;
        span_capture_dispose(capture, 1);
    } else {
        // The shared mapping and mirror remain held by the capture. The scene needs an independent physical pin.
        pool_native_acquire(provider.pool);
        if (provider.mirror_incarnation &&
            __atomic_add_fetch(&capture->provider->mirror_input_refs, 1, __ATOMIC_RELAXED) == 0)
            rt_fatal("vk: mirror input reference exhausted");
        vk_mem_span_capture_release(owned);
    }
    return provider;
}

void vk_mem_input_provider_release(VkMemInputProvider provider) {
    if (!provider.pool) return;
    mirror_input_release(provider.pool, provider.mirror_incarnation);
    destroy_rec(vk_ctx(), provider.pool);
}

// pool_lock held. A reused Vulkan handle cannot substitute for a current normalized mapping in placement admission.
static PoolRec *pool_image_alias_provider_locked(uint64_t addr, uint64_t size, const VkMemSpan *selected,
                                                 const PoolRec **window) {
    VkMemSpan current = {0};
    PoolRec *provider = NULL;
    const pidx_ent *entry = pidx_find(&pool_index, addr, size);
    PoolRec *r = entry ? entry->buf : NULL;
    if (!r || r->unpublished || (r->kind != POOL_PHYS && r->kind != POOL_VIRT) ||
        !pool_device_span(r, addr, size, &current, &provider) || !provider || provider->unpublished ||
        current.alloc != selected->alloc || current.alloc_offset != selected->alloc_offset ||
        current.buffer != selected->buffer) return NULL;
    if (window) *window = r;
    return provider;
}

static PoolRec *pool_image_alias_acquire(uint64_t addr, uint64_t size, const VkMemSpan *selected) {
    plat_mutex_lock_shared(&pool_lock);
    PoolRec *provider = pool_image_alias_provider_locked(addr, size, selected, NULL);
    if (provider) pool_native_acquire(provider);
    plat_mutex_unlock_shared(&pool_lock);
    return provider;
}

uint64_t vk_mem_gen(void) {
    return __atomic_load_n(&pool_gen, __ATOMIC_ACQUIRE) + __atomic_load_n(&mirror_gen, __ATOMIC_ACQUIRE);
}

int vk_mem_class_of(uint64_t addr) {
    VkMemSpan m;
    return vk_mem_lookup(addr, 1, &m) ? m.cls : VK_MCLS_NONE;
}

// ---------------------------------------------------------------- census
static volatile int64_t traffic[VK_MCLS_N][VK_MUSE_N];

// Every caller is the replay (vk_exec*, under vk_sync_submit's lock), about ten adds per draw: one writer at a time, so
// a plain add, not a locked one. An add racing the census's reset may be lost.
void vk_mem_count(int cls, int use, uint64_t bytes) {
    if ((unsigned)cls < VK_MCLS_N && (unsigned)use < VK_MUSE_N)
        __atomic_store_n(&traffic[cls][use], __atomic_load_n(&traffic[cls][use], __ATOMIC_RELAXED) + (int64_t)bytes,
                         __ATOMIC_RELAXED);
}

static void flag_names(uint32_t f, char *buf, size_t n) {
    static const char *const names[] = {"CPU_NO_ACCESS", "CPU_UNCACHED", "CPU_CACHED", "GPU_NO_ACCESS", "GPU_UNCACHED",
                                        "GPU_CACHED", "SHADER_CODE", "COMPRESSIBLE", "PHYSICAL", "VIRTUAL"};
    size_t k = 0;
    buf[0] = 0;
    for (int i = 0; i < 10; i++)
        if (f >> i & 1) k += (size_t)snprintf(buf + k, k < n ? n - k : 0, "%s%s", k ? "|" : "", names[i]);
    if (!k) snprintf(buf, n, "none");
}

// One line per (kind, flags) group of the live pools, then the bytes the replay bound or copied per frame by pool
// class since the previous census (the first one: since frame 1200). Logged at frame 1800 (in the world from the
// run-game saves), at frame 600, and with NX2NAVI_TRACE=perf every 1800 frames.
static void dirty_frame(uint64_t frame);

static void mirror_log(uint64_t frame);
static void mirror_pool_log(uint64_t frame);
static void shadow_log(uint64_t frame);
static int shadow_check_on(void);
static int dead_pending(void);

static void vram_log(uint64_t frame);
static void outside_watch(uint64_t frame);

static volatile uint64_t mirror_frame;   // the frame vk_mem_census last saw, for mirror_sync's log lines

// NX2NAVI_TRACE=perf: the 600-frame memory lines now, at a segment's end and once a scene has settled (perf.c).
void vk_mem_snapshot(uint64_t frame) {
    if (rt_trace & RT_TRACE_PERF) vram_log(frame);
}

void vk_mem_census(uint64_t frame) {
    mirror_frame = frame;
    static uint64_t window_start;
    if ((rt_trace & RT_TRACE_PERF) && frame % 600 == 0) vram_log(frame), mirror_pool_log(frame);
    if ((rt_trace & RT_TRACE_PERF) && frame % 60 == 0) outside_watch(frame);
    if ((rt_trace & RT_TRACE_PERF) && !vk_mirror_on) dirty_frame(frame);
    if ((rt_trace & RT_TRACE_PERF) ? frame % 60 == 0 : frame == 600 || frame == 1800)
        mirror_log(frame), shadow_log(frame);   // mirror_log also logs the write-watch line
    else if (shadow_check_on() && frame % 60 == 0)
        shadow_log(frame);   // NX2NAVI_SHADOW_CHECK: the shadow lines at that rate without the perf trace
    if (frame == 1200) {   // loading is over: the first window starts here
        for (int c = 0; c < VK_MCLS_N; c++)
            for (int u = 0; u < VK_MUSE_N; u++) plat_xchg64(&traffic[c][u], 0);
        window_start = frame;
        return;
    }
    // Frame 600 as well: runs on slow machines end before 1800.
    if (frame != 600 && (frame % 1800 || !(frame == 1800 || (rt_trace & RT_TRACE_PERF)))) return;
    typedef struct { int kind; uint32_t flags; uint64_t n, bytes; } Group;
    Group g[64];
    int ng = 0;
    uint64_t total[4] = {0}, count[4] = {0};
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) {
        const PoolRec *r = pool_index.e[i].buf;
        int j = 0;
        while (j < ng && (g[j].kind != r->kind || g[j].flags != r->flags)) j++;
        if (j == ng && ng < 64) g[ng++] = (Group){r->kind, r->flags, 0, 0};
        if (j < ng) g[j].n++, g[j].bytes += r->size;
        total[r->kind] += r->size, count[r->kind]++;
    }
    plat_mutex_unlock_shared(&pool_lock);
    rt_log("mem: census at frame %llu: import %llu pools %.1f MiB (%d without a GPU buffer), physical %llu pools %.1f MiB, "
           "virtual %llu pools %.1f MiB of address space",
           (unsigned long long)frame, (unsigned long long)count[POOL_IMPORT], total[POOL_IMPORT] / 1048576.0,
           n_import_failures, (unsigned long long)count[POOL_PHYS], total[POOL_PHYS] / 1048576.0,
           (unsigned long long)count[POOL_VIRT], total[POOL_VIRT] / 1048576.0);
    static const char *const kinds[] = {"", "import", "physical", "virtual"};
    for (int j = 0; j < ng; j++) {
        char f[160];
        flag_names(g[j].flags, f, sizeof f);
        rt_log("mem:   %s %#x %s: %llu pools, %.1f MiB", kinds[g[j].kind], g[j].flags, f, (unsigned long long)g[j].n,
               g[j].bytes / 1048576.0);
    }
    uint64_t frames = window_start && frame > window_start ? frame - window_start : frame;
    static const char *const cls[] = {"", "host CPU_NO_ACCESS", "host CPU_UNCACHED", "host CPU_CACHED", "device", "scratch"};
    rt_log("mem: traffic per frame over %llu frames, MiB (bound ranges, not bytes read): class: vertex index uniform "
           "storage indirect copy-src gpu-write cpu-read", (unsigned long long)frames);
    for (int c = 1; c < VK_MCLS_N; c++) {
        double v[VK_MUSE_N];
        for (int u = 0; u < VK_MUSE_N; u++) v[u] = plat_xchg64(&traffic[c][u], 0) / 1048576.0 / (double)(frames ? frames : 1);
        rt_log("mem:   %s: %.2f %.2f %.2f %.2f %.3f %.2f %.2f %.3f", cls[c], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    }
    window_start = frame;
}

// NX2NAVI_VRAM_BUDGET_MB (test; unset or 0: the device's): the device-local budget every decision here and in
// vk_texture.c compares with is at most this many MiB, so a large card behaves as a small one does under pressure
// (mirrors refused or evicted, textures dropped). The heap lines still log what the driver reports.
static uint64_t budget_cap(void) {
    static int64_t mb = -1;
    if (mb < 0) {
        const char *s = cfg_get("NX2NAVI_VRAM_BUDGET_MB");
        int64_t v = s && *s ? atoll(s) : 0;
        if (v > 0) rt_log("mem: the VRAM budget is taken as at most %lld MiB (NX2NAVI_VRAM_BUDGET_MB, a test)", (long long)v);
        mb = v > 0 ? v : 0;
    }
    return (uint64_t)mb << 20;
}

// NX2NAVI_VRAM_BALLAST_MB (test; unset or 0: none): that much device-local memory is taken when the allocator is made
// and held until it goes, in buffers of BALLAST_PIECE of their own memory. NX2NAVI_VRAM_BUDGET_MB above changes only the
// number the decisions compare with; this fills the card for real, so the driver's own behaviour at its limit (slow
// allocations, memory moved to system RAM, refusals) shows on a large card as it does on an 8 GB one or beside another
// program's use.
#define BALLAST_PIECE (256ull << 20)
#define BALLAST_MAX 512
static VkBuffer ballast_buf[BALLAST_MAX];
static VmaAllocation ballast_alloc[BALLAST_MAX];
static int nballast;

void vk_mem_ballast(VmaAllocator vma) {   // vk_device.c, right after the allocator is made (before vk_ctx answers)
    const char *s = cfg_get("NX2NAVI_VRAM_BALLAST_MB");
    uint64_t want = s && *s && atoll(s) > 0 ? (uint64_t)atoll(s) << 20 : 0, got = 0;
    if (!want) return;
    while (got < want && nballast < BALLAST_MAX) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                 .size = want - got < BALLAST_PIECE ? want - got : BALLAST_PIECE,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                      .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                      .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
        if (vmaCreateBuffer(vma, &bi, &ai, &ballast_buf[nballast], &ballast_alloc[nballast], NULL) != VK_SUCCESS) break;
        got += bi.size, nballast++;
    }
    rt_log("mem: %llu of %llu MiB of video memory taken and held for the session (NX2NAVI_VRAM_BALLAST_MB, a test)",
           (unsigned long long)(got >> 20), (unsigned long long)(want >> 20));
}

void vk_mem_ballast_release(VmaAllocator vma) {   // before the allocator goes
    for (int i = 0; i < nballast; i++) resource_core_vma_destroy_buffer(vma, ballast_buf[i], ballast_alloc[i]);
    nballast = 0;
}

static int vram_query(uint64_t *used, uint64_t *budget, int *exact) {
    VkCtx *c = vk_ctx();
    *used = *budget = 0, *exact = 0;
    if (!c) return 0;
    const VkPhysicalDeviceMemoryProperties *mp = &c->mem_props;
    if (c->has_memory_budget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &b};
        vkGetPhysicalDeviceMemoryProperties2(c->phys, &p2);
        for (uint32_t h = 0; h < p2.memoryProperties.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
            if (p2.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                *used += b.heapUsage[h], *budget += b.heapBudget[h];
        *exact = 1;
        return *budget != 0;
    }
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    for (uint32_t h = 0; h < mp->memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (mp->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            *used += vb[h].statistics.blockBytes, *budget += mp->memoryHeaps[h].size;
    return *budget != 0;
}

int vk_mem_vram(uint64_t *used, uint64_t *budget, int *exact) {
    int ok = vram_query(used, budget, exact);
    uint64_t cap = budget_cap();
    if (ok && cap && *budget > cap) *budget = cap;
    return ok;
}

// vk_mem_vram's use less the free space inside VMA's device-local blocks: the driver counts a block whole for as long
// as one allocation lives in it, and the next allocation of its type takes a hole first. The evictions measure with
// this: a texture or mirror freed in the middle of a block left the driver's figure where it was, so an eviction that
// waited for that figure to fall went on until nearly every idle texture and mirror was gone (16.2 GiB in use, 9.2 GiB
// of it free inside blocks after the evictions).
int vk_mem_vram_live(uint64_t *used, uint64_t *budget) {
    int exact;
    if (!vk_mem_vram(used, budget, &exact)) return 0;
    VkCtx *c = vk_ctx();
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    uint64_t slack = 0;
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if ((c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            vb[h].statistics.blockBytes > vb[h].statistics.allocationBytes)
            slack += vb[h].statistics.blockBytes - vb[h].statistics.allocationBytes;
    *used = *used > slack ? *used - slack : 0;
    return 1;
}

// ---------------------------------------------------------------- page write generations
// Two-level table over the guest address space (below 2^39): 256 MiB chunks of 65,536 page generations, allocated
// when a page of the chunk is first stamped. 0 = never stamped.
#define GEN_CHUNK_SHIFT 28
#define GEN_PAGE_SHIFT 12
#define GEN_CHUNKS (RT_ADDRESS_LIMIT >> GEN_CHUNK_SHIFT)
#define GEN_PER_CHUNK (1u << (GEN_CHUNK_SHIFT - GEN_PAGE_SHIFT))
static uint32_t *gen_chunk[GEN_CHUNKS];
static uint32_t gen_now;                 // last generation handed out
static uint64_t source_generation_lineage = 1;

// gen_lock held. Preserve the existing page-generation values while giving long-lived source certificates a
// separate identity for each wrap. Exhaustion disables certificates rather than admitting an ancient generation.
static uint32_t generation_next(void) {
    ++gen_now;
    if (!gen_now && source_generation_lineage) source_generation_lineage++;
    return gen_now;
}
static PlatMutex gen_lock = PLAT_MUTEX_INIT;
static uint64_t gen_pages[1024];
// The newest generation stamped on any page of each 2 MiB region (the write-watch snapshot regions below): the mirror
// sync skips regions with nothing newer than its pool's mirror.
#define REG_SHIFT 21
static uint32_t reg_max[RT_ADDRESS_LIMIT >> REG_SHIFT];
// The same two tables without the stamps of a mirror's own write-back (vk_gpu_written_back): what the CPU and the host
// paths wrote. The mirror sync reads these, so the bytes a mirror wrote back into guest memory are not copied up into it
// again at its next sync: the storage buffers the game's compute shaders write went down and came back up every frame,
// 20 MiB a frame over PCIe in Kokiri Forest (2026-11-09).
static uint32_t *cgen_chunk[GEN_CHUNKS];
static uint32_t creg_max[RT_ADDRESS_LIMIT >> REG_SHIFT];

// Every write-watch query of the generation table (gen_lock held exclusively): calls, bytes and time since the last
// mirror summary, all consumers together; the queries that found written pages (they also flush TLBs) apart.
static uint64_t ww_calls, ww_bytes, ww_ticks, ww_dirty_calls, ww_dirty_ticks, ww_pages;

// NX2NAVI_TRACE=perf: a write-watch query, freeze or its follow-up query that took over 4 ms gets a line (the first
// 200). A frame some ten frames after an area load took 51.6 ms of replay with 1.2 ms of it in the replay's ops; the
// replaying thread's samples sat in GetWriteWatch under vk_mem_mirror_begin and in the freeze's VirtualProtect, and no
// thread of the process could be sampled for 28 ms in between.
static void slow_mem_call(const char *what, uint64_t addr, uint64_t size, uint64_t pages, uint64_t ns) {
    static uint32_t logged;
    if (ns < 4000000ull || !(rt_trace & RT_TRACE_PERF) || __atomic_add_fetch(&logged, 1, __ATOMIC_RELAXED) > 200) return;
    rt_log("mem: slow %s: %.1f ms over %#llx + %.1f MiB, %lld pages written (frame %llu)", what, ns / 1e6,
           (unsigned long long)addr, size / 1048576.0, pages == UINT64_MAX ? -1ll : (long long)pages,
           (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
}

static uint64_t ww_query(uint64_t addr, uint64_t size) {
    uint64_t t0 = qpc();
    uint64_t n = mem_written_pages(addr, size, gen_pages, sizeof gen_pages / sizeof *gen_pages);
    uint64_t dt = qpc() - t0;
    ww_ticks += dt, ww_calls++, ww_bytes += size;
    if (n && n != UINT64_MAX) ww_dirty_calls++, ww_dirty_ticks += dt, ww_pages += n;
    slow_mem_call("write-watch query", addr, size, n, dt);
    return n;
}

// Frozen regions: a 2 MiB region whose pages nothing wrote for freeze_idle submissions is made read-only
// (memory.c mem_freeze), and write-watch queries skip it until a write or a file read thaws it. Most of the pool memory
// the mirror and the texture checks cover is static (meshes, textures), and a query costs about 1 us per MiB (4-5 on an
// 8th-gen i7), so this keeps the queries to the regions the CPU writes. The region need not lie wholly in pools: NVN
// pools are small and not 2 MiB aligned, so most regions also hold other guest memory, and a rule that refused those
// left 340 of 436 queried MiB a frame unfrozen in town although nothing had written them for 300 submissions. The
// other memory is safe to freeze for the same reason pools are: the only kernel writes into guest memory are file
// reads, which thaw their buffer first (fs_host.c, romfs.c), and every user-mode write faults and thaws.
#define FREEZE_PER_SUB 32
static int freeze_idle;                       // 0: off
static uint32_t ww_ep = 1;                    // gen_lock: the submission being recorded (vk_mem_mirror_begin)
static uint32_t reg_last[RT_ADDRESS_LIMIT >> REG_SHIFT];    // gen_lock: ww_ep of the region's last stamp
static uint32_t reg_tried[RT_ADDRESS_LIMIT >> REG_SHIFT];   // gen_lock: ww_ep of its last freeze candidacy
typedef struct { uint32_t region, idle; } FreezeCandidate;
static FreezeCandidate *fcand;                       // gen_lock: regions to try at the next freeze pass
static int nfcand, cfcand;
static uint64_t n_frz_refused;                // gen_lock: candidates mem_freeze refused

static uint32_t *slot_in(uint32_t **tab, uint64_t page_addr, int create) {
    uint64_t ci = page_addr >> GEN_CHUNK_SHIFT;
    if (ci >= GEN_CHUNKS) return NULL;
    if (!tab[ci]) {
        if (!create) return NULL;
        tab[ci] = calloc(GEN_PER_CHUNK, sizeof(uint32_t));
        if (!tab[ci]) rt_fatal("vk: out of memory for page generations");
    }
    return &tab[ci][(page_addr >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)];
}

static uint32_t *gen_slot(uint64_t page_addr, int create) { return slot_in(gen_chunk, page_addr, create); }

static uint64_t dependency_serial = 1;
static uint64_t dependency_region_serial[RT_ADDRESS_LIMIT >> REG_SHIFT];
static ResourceEvents dependency_events;
static int dependency_watch, dependency_reset;

void vk_mem_dependency_watch(void) { __atomic_store_n(&dependency_watch, 1, __ATOMIC_RELEASE); }
uint64_t vk_mem_dependency_serial(void) {
    return __atomic_load_n(&dependency_reset, __ATOMIC_ACQUIRE) ? 0 :
           __atomic_load_n(&dependency_serial, __ATOMIC_ACQUIRE);
}
static void dependency_changed(uint32_t region) {   // gen_lock held exclusively by every publisher
    uint64_t next = __atomic_load_n(&dependency_serial, __ATOMIC_RELAXED) + 1;
    if (!next)
        __atomic_store_n(&dependency_reset, 1, __ATOMIC_RELEASE);
    // Publish the affected region before the global watermark. A captured watermark under gen_lock includes every
    // earlier regional publication. Zero in the regional table means no observation since the watcher was enabled.
    __atomic_store_n(&dependency_region_serial[region], next, __ATOMIC_RELEASE);
    __atomic_store_n(&dependency_serial, next, __ATOMIC_RELEASE);
    resource_events_publish(&dependency_events, region, region);
}

int vk_mem_dependency_events_locked(uint64_t *cursor, ResourceEventVisit visit, void *user) {
    return resource_events_read(&dependency_events, cursor, visit, user) && vk_mem_dependency_serial() != 0;
}

int vk_mem_dependency_range_unchanged(uint64_t addr, uint64_t size, uint64_t observed) {
    if (!observed || !size || addr >= RT_ADDRESS_LIMIT || size > RT_ADDRESS_LIMIT - addr ||
        !vk_mem_dependency_serial()) return 0;
    uint64_t first = addr >> REG_SHIFT, last = (addr + size - 1) >> REG_SHIFT;
    for (uint64_t r = first; r <= last; r++)
        if (__atomic_load_n(&dependency_region_serial[r], __ATOMIC_ACQUIRE) > observed) return 0;
    // The caller already established tracked snapshots in this submission. This check only preserves that proof
    // across observations elsewhere; it never establishes freshness or tracking by itself.
    return vk_mem_dependency_serial() != 0;
}

// gen_lock held exclusively; g is the newest generation handed out. back: a mirror's write-back, which leaves the
// CPU-side tables alone.
static void stamp_page_as(uint64_t a, uint32_t g, int back) {
    uint32_t *s = gen_slot(a, 1);
    if (!s) return;
    *s = g;
    if (reg_max[a >> REG_SHIFT] != g && __atomic_load_n(&dependency_watch, __ATOMIC_RELAXED)) {
        // The reference uses 32-bit ordered generations. A wrap/reset invalidates this experiment permanently for
        // the process; it must not turn a previously clean certificate into evidence across that discontinuity.
        if (g < reg_max[a >> REG_SHIFT]) __atomic_store_n(&dependency_reset, 1, __ATOMIC_RELEASE);
        dependency_changed((uint32_t)(a >> REG_SHIFT));
    }
    reg_max[a >> REG_SHIFT] = g;
    reg_last[a >> REG_SHIFT] = ww_ep;
    if (back) return;
    uint32_t *c = slot_in(cgen_chunk, a, 1);
    if (c) *c = g, creg_max[a >> REG_SHIFT] = g;
}

static void stamp_page(uint64_t a, uint32_t g) { stamp_page_as(a, g, 0); }

static void stamp_range(uint64_t lo, uint64_t hi, uint32_t g) {
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) stamp_page(a, g);
}

void vk_guest_written(uint64_t addr, uint64_t size) {
    if (!size || nvn_gpu_only(addr)) return;
    rc_registry_cpu_changed(rc_registry_runtime(), (RcSourceRange){1, addr, size});
    mem_backing_cpu_changed(addr, size);
    vk_mem_cpu_result_written(addr, size);
}

void vk_mem_cpu_result_written(uint64_t addr, uint64_t size) {
    if (!size || addr >= RT_ADDRESS_LIMIT || size > RT_ADDRESS_LIMIT - addr || nvn_gpu_only(addr)) return;
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    plat_mutex_lock(&gen_lock);
    stamp_range(lo, hi, generation_next());
    plat_mutex_unlock(&gen_lock);
}

int vk_mem_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now) {
    if (!size) { *now = since; return 0; }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    plat_mutex_lock(&gen_lock);
    uint32_t g = generation_next();
    int tracked = 1, frozen = 1;
    for (uint64_t a = lo & ~((1ull << REG_SHIFT) - 1); a < hi && frozen; a += 1ull << REG_SHIFT) frozen = mem_frozen(a);
    while (!frozen) {
        uint64_t n = ww_query(lo, hi - lo);
        if (n == UINT64_MAX) { tracked = 0; break; }
        for (uint64_t k = 0; k < n; k++) stamp_page(gen_pages[k], g);
        if (n < sizeof gen_pages / sizeof *gen_pages) break;   // a full answer may have left pages for the next call
    }
    int changed = 0;
    for (uint64_t a = lo; tracked && a < hi && !changed; a += 1ull << GEN_PAGE_SHIFT) {
        uint32_t *s = gen_slot(a, 0);
        if (s && *s > since) changed = 1;
    }
    plat_mutex_unlock(&gen_lock);
    *now = g;
    return tracked ? changed : -1;
}

// Write-watch snapshots per submission: the guest address space in 2 MiB regions, each queried at most once per epoch
// (a guest submission), at the first check that covers it. A query costs about 1.3 us plus 0.55 us per MiB, so one per
// region instead of one per texture. The snapshot only adds observations to the generation table: a page written
// before it carries a generation newer than any check before it, and a page written after it keeps its OS bit until
// the next query over it (at the latest the next epoch's snapshot).
#define SNAP_SHIFT 21
#define SNAP_REGIONS (RT_ADDRESS_LIMIT >> SNAP_SHIFT)
static uint32_t texture_empty_epoch[SNAP_REGIONS]; // previous costly clean texture observation under gen_lock
static uint8_t texture_queued[SNAP_REGIONS];
static uint64_t texture_offers, texture_tries, texture_protected;
// Schedule protection only after repeated authoritative scans have already cost this region about
// as much as protecting one. The estimate includes the follow-up write-watch drain. It is a scheduling
// hint under gen_lock. Only a successful freeze followed by observation can put a region to sleep.
typedef struct { uint32_t first, last, scan_ns; } TextureScanCost;
static TextureScanCost texture_cost[SNAP_REGIONS];
enum { TEXTURE_MIN_IDLE = 96, TEXTURE_HISTORY_GAP = 24, TEXTURE_MIN_COST_NS = 16000,
       TEXTURE_MAX_COST_NS = 128000, TEXTURE_MAX_SAMPLE_NS = 4000 };
static uint32_t texture_freeze_ns = 32000;
static uint64_t texture_cost_tries, texture_cost_protected;

static int texture_cost_ready(const TextureScanCost *cost) {
    return cost->first && ww_ep >= cost->last && ww_ep - cost->last <= TEXTURE_HISTORY_GAP &&
           cost->last - cost->first >= TEXTURE_HISTORY_GAP && cost->scan_ns >= texture_freeze_ns;
}

static void texture_cost_observe(uint32_t region, int continuity, uint32_t share_ns) {
    TextureScanCost *cost = &texture_cost[region];
    if (!continuity || ww_ep < cost->last || ww_ep - cost->last > TEXTURE_HISTORY_GAP ||
        reg_last[region] >= cost->first) {
        if (cost->first) *cost = (TextureScanCost){0};
    }
    if (!share_ns || !ww_ep || cost->last == ww_ep) return;
    if (!cost->first) cost->first = ww_ep;
    cost->last = ww_ep;
    if (cost->scan_ns < TEXTURE_MAX_COST_NS) cost->scan_ns += share_ns;
}
static uint32_t snap_at[SNAP_REGIONS];      // epoch of the region's last snapshot; 0 never
static uint8_t snap_untracked[SNAP_REGIONS];
static uint64_t snap_protection[SNAP_REGIONS], snap_mapping[SNAP_REGIONS];
static TextureSourceFreezeBridge source_freeze_bridge[SNAP_REGIONS]; // gen_lock: explicit, drained freeze transitions
static uint64_t n_snap_checks, n_snap_calls, n_snap_untracked;
static uint64_t freeze_snap_renewals;      // gen_lock: clean post-freeze observations made reusable


// Clean scans are hints about which regions to protect. They never satisfy a consumer's snapshot.
// All state below is private to gen_lock after mirror_mode initializes it.
#define CLEAN_SPANS 8
#define CLEAN_FRESH 24u
#define CLEAN_MIN_IDLE 48u
#define CLEAN_MIN_REGIONS 32u
#define CLEAN_MIN_NS 32000ull
typedef struct {
    uint32_t lo, hi;                 // unattempted intersection, exclusive end
    uint32_t first, last, repeats;
    uint64_t protection, mapping;    // oldest pre-query observations for this intersection
    int high;
} CleanSpan;
static CleanSpan clean_spans[CLEAN_SPANS];
static uint32_t *clean_attempt;      // exact last attempted freeze, including the normal queue
static unsigned clean_cursor, clean_live;
static int clean_normal_first;       // one pass reserved for the normal queue after priority goes first
typedef struct {
    uint64_t expensive, ready, full, examined, skipped, attempts, successes;
} CleanStats;
static CleanStats clean_stats;

static int clean_recent(uint32_t now, uint32_t before, uint32_t age) {
    return before && now >= before && now - before < age;
}

static void clean_drop(CleanSpan *span) {
    span->repeats = 0;
    clean_live--;
}

static void clean_observe(uint32_t lo, uint32_t hi, int zero, uint64_t ns, uint64_t protection, uint64_t mapping) {
    int expensive = zero && hi - lo >= CLEAN_MIN_REGIONS && ns >= CLEAN_MIN_NS;
    int tail = zero && hi - lo < CLEAN_MIN_REGIONS;
    if (!expensive && (!clean_live || (zero && !tail))) return;
    clean_stats.expensive += expensive;
    int vacant = -1, best = -1;
    uint32_t best_overlap = 0;
    for (int i = 0; i < CLEAN_SPANS; i++) {
        CleanSpan *s = &clean_spans[i];
        if (s->repeats && !clean_recent(ww_ep, s->last, CLEAN_FRESH + 1)) clean_drop(s);
        if (s->repeats && lo < s->hi && s->lo < hi) {
            if (!zero) {
                // A dirty, failed or saturated query cannot extend evidence, even after a clean final drain.
                clean_drop(s);
            } else {
                if (tail && (s->repeats != 3 || s->hi - s->lo >= CLEAN_MIN_REGIONS)) continue;
                uint32_t left = lo > s->lo ? lo : s->lo, right = hi < s->hi ? hi : s->hi;
                uint32_t overlap = right - left;
                if (overlap * 4 >= (s->hi - s->lo) * 3 && overlap > best_overlap) {
                    best = i;
                    best_overlap = overlap;
                }
            }
        }
        if (!s->repeats && vacant < 0) vacant = i;
    }
    if ((!expensive && best < 0) || !ww_ep || !protection || !mapping) return;
    if (best >= 0) {
        CleanSpan *s = &clean_spans[best];
        if (lo > s->lo) s->lo = lo;
        if (hi < s->hi) s->hi = hi;
        if (s->last != ww_ep && s->repeats < 3 && ++s->repeats == 3) clean_stats.ready++;
        s->last = ww_ep;
    } else if (vacant >= 0) {
        clean_spans[vacant] = (CleanSpan){lo, hi, ww_ep, ww_ep, 1, protection, mapping, 0};
        clean_live++;
    } else clean_stats.full++;   // leave current evidence intact and fall back to normal scanning
}

// Selection is bounded independently of the extent length and normal queue length.
// Progress through a confirmed span survives shrinking query endpoints until its recent demand expires.
static int clean_next(uint32_t *region) {
    unsigned rejected = 0;
    for (unsigned visited = 0; visited < CLEAN_SPANS && clean_live && rejected < 4; visited++) {
        CleanSpan *s = &clean_spans[clean_cursor++ % CLEAN_SPANS];
        if (!s->repeats) continue;
        if (!clean_recent(ww_ep, s->last, CLEAN_FRESH + 1)) { clean_drop(s); continue; }
        while (s->repeats == 3 && rejected < 4) {
            uint32_t j = s->high ? --s->hi : s->lo++;
            s->high = !s->high;
            uint32_t first = s->first;
            uint64_t p = s->protection, m = s->mapping;
            if (s->lo == s->hi) clean_drop(s);
            clean_stats.examined++;
            if (reg_last[j] < first && ww_ep - reg_last[j] >= CLEAN_MIN_IDLE &&
                !clean_recent(ww_ep, clean_attempt[j], (uint32_t)freeze_idle) &&
                snap_at[j] && !snap_untracked[j] && !mem_frozen((uint64_t)j << REG_SHIFT)) {
                uint64_t pv = mem_protection_region(j), mv = mem_mapping_region(j);
                if (pv && pv <= p && mv && mv <= m) { *region = j; return 1; }
            }
            clean_stats.skipped++;
            rejected++;
        }
    }
    return 0;
}

// A snapshot predating a freeze cannot prove that writes immediately before protection were collected. Retain the
// watermarks from before the OS query: a later thaw/refreeze revokes cross-submission reuse, while the first content
// observation inside one submission remains fixed for all of its consumers.
static int snap_lease_valid(uint32_t region) {
    uint64_t protection = mem_protection_region(region), mapping = mem_mapping_region(region);
    return protection && protection <= snap_protection[region] && mapping && mapping <= snap_mapping[region];
}

// Ordinary unchanged protection is the fast path. A changed region may cross only its recorded freeze: a later
// thaw, lazy commit or mapping change cannot be inferred compatible from its final permissions or empty write bits.
static int source_protection_compatible(uint64_t addr, uint64_t size, const TextureSourceStamp *prior) {
    if (mem_protection_range_unchanged(addr, size, prior->protection)) return 1;
    uint64_t serial = mem_protection_serial();
    if (!serial || !prior->protection || prior->protection > serial) return 0;
    uint64_t first = addr >> SNAP_SHIFT, last = (addr + size - 1) >> SNAP_SHIFT;
    for (uint64_t r = first; r <= last; r++) {
        uint64_t protection = mem_protection_region((uint32_t)r);
        if (!protection) return 0;
        if (protection <= prior->protection) continue;
        if (!texture_source_freeze_covers(&source_freeze_bridge[r], prior->protection, protection,
                                         mem_mapping_region((uint32_t)r), prior->lineage) ||
            !mem_frozen_stable(r << SNAP_SHIFT)) return 0;
    }
    return mem_protection_serial() == serial;
}

int vk_mem_dependency_sleep_locked(uint32_t region) {
    return region < SNAP_REGIONS && snap_at[region] && !snap_untracked[region] &&
           mem_frozen_stable((uint64_t)region << SNAP_SHIFT) && snap_lease_valid(region);
}

static int snap_current(uint32_t region, uint32_t epoch) {
    return epoch && snap_at[region] == epoch;
}

#ifdef _WIN32
typedef struct {
    uint64_t start[3];
    uint64_t found[2];
} SnapshotPeek;

static void peek_snapshot_half(void *arg, unsigned half) {
    SnapshotPeek *p = arg;
    uint64_t page;
    p->found[half] = mem_written_peek(p->start[half], p->start[half + 1] - p->start[half], &page, 1);
}

// A recent empty scan selects a range worth dividing. Only two successful, empty OS observations certify
// the current snapshot. A write after either observation stays armed for the next submission. A dirty or
// failed peek clears no bits, so the ordinary drain can still publish every page under the generation lock.
// Other pools can be queried between two visits to the same span. Collisions only lose a scheduling hint.
typedef struct { uint64_t lo, hi; } EmptySpan;
static EmptySpan empty_spans[16];   // gen_lock
static int snapshot_pair_enabled;
static uint64_t peek_pairs, peek_clean, peek_wall;
static int peek_large_snapshot(uint64_t lo, uint64_t hi) {
    uint64_t mid = lo + (((hi - lo) >> (SNAP_SHIFT + 1)) << SNAP_SHIFT);
    SnapshotPeek p = {.start = {lo, mid, hi}};
    uint64_t t0 = qpc();
    if (!vk_exec_chunk_pair(peek_snapshot_half, &p)) return 0;
    uint64_t dt = qpc() - t0;
    ww_calls += 2, ww_bytes += hi - lo, ww_ticks += dt;
    peek_pairs++, peek_wall += dt;
    if (p.found[0] || p.found[1]) return 0;
    peek_clean++;
    return 1;
}
#endif

int vk_mem_dependency_adopt_locked(uint32_t region, uint32_t epoch) {
    // Caller owns a sleeping membership and has consumed all three notification domains without a gap. Events
    // after that cutoff belong after this submission's logical snapshot; they wake the region for a later one.
    if (!epoch || region >= SNAP_REGIONS || !snap_at[region] || snap_untracked[region]) return 0;
    snap_at[region] = epoch;
    return 1;
}

int vk_mem_dependency_region_locked(uint32_t region, uint32_t epoch, uint32_t *version) {
    if (region >= SNAP_REGIONS) { *version = 0; return 0; }
    *version = reg_max[region];
    return snap_current(region, epoch) && !snap_untracked[region];
}

// gen_lock held exclusively. Queries the regions r0..r1 that were not snapped in epoch, one query per run of them;
// returns the number of queries.
static uint64_t snap_regions_for(uint64_t r0, uint64_t r1, uint32_t epoch, int texture_list) {
    uint32_t g = 0;
    uint64_t calls = 0;
    for (uint64_t r = r0; r <= r1;) {
        if (snap_current((uint32_t)r, epoch)) { r++; continue; }
        if (vk_mem_dependency_sleep_locked((uint32_t)r)) {
            snap_at[r++] = epoch;
            continue;
        }
        uint64_t k = r;   // one query over a run of regions not seen in this epoch
        while (k + 1 <= r1 && !snap_current((uint32_t)(k + 1), epoch) &&
               !vk_mem_dependency_sleep_locked((uint32_t)(k + 1))) k++;
        uint64_t a = r << SNAP_SHIFT, b = (k + 1) << SNAP_SHIFT;
        uint64_t protection = mem_protection_serial(), mapping = mem_mapping_serial();
        int tracked = 1, all_zero = 1;
        uint64_t ticks_before = ww_ticks;
        if (!g) g = generation_next();
        int observed_clean = 0;
#ifdef _WIN32
        int large_texture = snapshot_pair_enabled && texture_list && b - a >= (256ull << 20);
        EmptySpan *hint = &empty_spans[(r ^ k ^ (r >> 4)) % 16];
        if (large_texture && hint->lo == a && hint->hi == b) observed_clean = peek_large_snapshot(a, b);
#else
        (void)texture_list;
#endif
        while (!observed_clean) {
            uint64_t n = ww_query(a, b - a);
            if (n) all_zero = 0;
            if (n == UINT64_MAX) { tracked = 0; break; }
            for (uint64_t j = 0; j < n; j++) stamp_page(gen_pages[j], g);
            if (n < sizeof gen_pages / sizeof *gen_pages) break;
        }
#ifdef _WIN32
        if (large_texture) {
            *hint = (EmptySpan){a, all_zero && tracked ? b : 0};
        }
#endif
        if (clean_attempt)
            clean_observe((uint32_t)r, (uint32_t)(k + 1), all_zero && tracked, ww_ticks - ticks_before, protection, mapping);
        calls++;
        int costly_empty = texture_list && all_zero && tracked && b - a >= (64ull << 20) &&
                          ww_ticks - ticks_before >= 32000;
        uint64_t share = texture_list && all_zero && tracked ? (ww_ticks - ticks_before) / (k - r + 1) : 0;
        uint32_t share_ns = share < TEXTURE_MAX_SAMPLE_NS ? (uint32_t)share : TEXTURE_MAX_SAMPLE_NS;
        for (uint64_t j = r; j <= k; j++) {
            uint64_t pv = tracked ? mem_protection_region((uint32_t)j) : 0;
            uint64_t mv = tracked ? mem_mapping_region((uint32_t)j) : 0;
            int valid = tracked && pv && pv <= protection && mv && mv <= mapping;
            if ((share_ns || texture_cost[j].first) && freeze_idle > TEXTURE_MIN_IDLE) {
                int continuity = valid && all_zero && !snap_untracked[j] &&
                                 pv <= snap_protection[j] && mv <= snap_mapping[j];
                texture_cost_observe((uint32_t)j, continuity, valid ? share_ns : 0);
            }
            snap_protection[j] = protection, snap_mapping[j] = mapping;
            if (snap_untracked[j] != (uint8_t)!valid && __atomic_load_n(&dependency_watch, __ATOMIC_RELAXED))
                dependency_changed((uint32_t)j);
            snap_at[j] = epoch, snap_untracked[j] = (uint8_t)!valid;
            uint32_t idle = (uint32_t)freeze_idle;
            uint32_t previous = texture_empty_epoch[j];
            if (costly_empty && valid && previous && previous != ww_ep && ww_ep - previous <= 24 &&
                !texture_queued[j] && reg_last[j] < previous && idle > 600 &&
                ww_ep - reg_last[j] < idle) idle = 600;
            if (valid && share_ns && !texture_queued[j] && idle > TEXTURE_MIN_IDLE &&
                ww_ep - reg_last[j] < idle && texture_cost_ready(&texture_cost[j]))
                idle = TEXTURE_MIN_IDLE;
            texture_empty_epoch[j] = costly_empty && valid ? ww_ep : 0;
            if (valid && !mem_frozen_stable(j << SNAP_SHIFT) && idle && ww_ep - reg_last[j] >= idle &&
                (!reg_tried[j] || ww_ep - reg_tried[j] >= (uint32_t)freeze_idle)) {
                if (idle < (uint32_t)freeze_idle) texture_queued[j] = 1, texture_offers++;
                else reg_tried[j] = ww_ep;
                if (nfcand == cfcand) grow((void **)&fcand, &cfcand, sizeof *fcand);
                fcand[nfcand++] = (FreezeCandidate){(uint32_t)j, idle};
            }
        }
        r = k + 1;
    }
    return calls;
}

static uint64_t snap_regions(uint64_t r0, uint64_t r1, uint32_t epoch) {
    return snap_regions_for(r0, r1, epoch, 0);
}

int vk_mem_changed_snap(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch, uint32_t *now,
                        VkDirtyPages *dirty) {
    if (dirty) *dirty = (VkDirtyPages){0};
    if (!size) { *now = since; return 0; }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    if (!epoch || r1 >= SNAP_REGIONS) return vk_mem_changed(addr, size, since, now);
    int excl = 0;
    plat_mutex_lock_shared(&gen_lock);
    for (uint64_t r = r0; r <= r1; r++)
        if (!snap_current((uint32_t)r, epoch)) { excl = 1; break; }
    if (excl) {
        plat_mutex_unlock_shared(&gen_lock);
        plat_mutex_lock(&gen_lock);
        n_snap_calls += snap_regions(r0, r1, epoch);
    }
    int untracked = 0, changed = 0;
    for (uint64_t r = r0; r <= r1; r++) untracked |= snap_untracked[r] || !snap_current((uint32_t)r, epoch);
    // Pages only in regions stamped after since (reg_max; REG_SHIFT is SNAP_SHIFT): an unchanged texture of 4 MiB was
    // 1,024 page reads, at about 1,100 texture checks a frame.
    for (uint64_t r = r0; !untracked && r <= r1 && !changed; r++) {
        if (reg_max[r] <= since) continue;
        uint64_t a = r << REG_SHIFT > lo ? r << REG_SHIFT : lo, b = (r + 1) << REG_SHIFT < hi ? (r + 1) << REG_SHIFT : hi;
        for (; a < b && !changed; a += 1ull << GEN_PAGE_SHIFT) {
            uint32_t *s = gen_slot(a, 0);
            if (s && *s > since) {
                changed = 1;
                if (dirty) dirty->lo = a;
            }
        }
    }
    // A changed large resident image can refresh individual subresources. Find
    // the other end starting at the last page, skipping clean regions in bulk.
    if (changed && dirty && gen_now >= since) {
        uint64_t page = 1ull << GEN_PAGE_SHIFT;
        for (uint64_t end = hi; end > dirty->lo;) {
            uint64_t a = end - page, region = a >> REG_SHIFT;
            if (reg_max[region] <= since) { end = region << REG_SHIFT; continue; }
            uint32_t *s = gen_slot(a, 0);
            if (s && *s > since) { dirty->hi = end; break; }
            end = a;
        }
    }
    *now = gen_now;   // every observation so far is in the table
    __atomic_add_fetch(&n_snap_checks, 1, __ATOMIC_RELAXED);
    if (untracked) __atomic_add_fetch(&n_snap_untracked, 1, __ATOMIC_RELAXED);
    if (excl) plat_mutex_unlock(&gen_lock);
    else plat_mutex_unlock_shared(&gen_lock);
    return untracked ? -1 : changed;
}

int vk_mem_observe_subresources(uint64_t addr, uint64_t size, const TextureSourceStamp *prior, uint32_t epoch,
                                const TextureSourceSpan *spans, uint32_t count, TextureSourceObservation *out) {
    if (!out) return TEX_SOURCE_UNKNOWN;
    *out = (TextureSourceObservation){0};
    texture_source_all_dirty(out, count);
    if (!epoch || addr >= RT_ADDRESS_LIMIT || size > RT_ADDRESS_LIMIT - addr ||
        !texture_source_spans_valid(size, spans, count)) {
        out->tracking_failure = TEX_SOURCE_TRACKING_INPUT;
        return TEX_SOURCE_UNKNOWN;
    }
    uint64_t mapping = mem_mapping_serial(), protection = mem_protection_serial();
    if (!mapping || !protection) {
        out->tracking_failure = TEX_SOURCE_TRACKING_CLOCK;
        return TEX_SOURCE_UNKNOWN;
    }
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t first = lo >> SNAP_SHIFT, last = (hi - 1) >> SNAP_SHIFT;
    int exclusive = 0;
    plat_mutex_lock_shared(&gen_lock);
    for (uint64_t r = first; r <= last; r++)
        if (!snap_current((uint32_t)r, epoch) || !snap_lease_valid((uint32_t)r)) { exclusive = 1; break; }
    if (exclusive) {
        plat_mutex_unlock_shared(&gen_lock);
        plat_mutex_lock(&gen_lock);
        // A remap/thaw in an already observed submission invalidates that snapshot's tracking lease. Query it again
        // rather than reusing page observations from the previous backing or protection state.
        for (uint64_t r = first; r <= last; r++)
            if (snap_current((uint32_t)r, epoch) && !snap_lease_valid((uint32_t)r)) snap_at[r] = 0;
        n_snap_calls += snap_regions(first, last, epoch);
    }
    TextureSourceTrackingFailure failure;
    if (!gen_now || !source_generation_lineage) {
        failure = TEX_SOURCE_TRACKING_GENERATION;
        goto untracked;
    }
    for (uint64_t r = first; r <= last; r++) {
        if (!snap_current((uint32_t)r, epoch)) {
            failure = TEX_SOURCE_TRACKING_SNAPSHOT_EPOCH;
            goto untracked;
        }
        if (snap_untracked[r]) {
            failure = TEX_SOURCE_TRACKING_SNAPSHOT_UNTRACKED;
            goto untracked;
        }
        if (!snap_lease_valid((uint32_t)r)) {
            failure = TEX_SOURCE_TRACKING_SNAPSHOT_LEASE;
            goto untracked;
        }
    }
    if (!mem_mapping_range_unchanged(addr, size, mapping)) {
        failure = TEX_SOURCE_TRACKING_MAPPING_BEFORE;
        goto untracked;
    }
    if (!mem_protection_range_unchanged(addr, size, protection)) {
        failure = TEX_SOURCE_TRACKING_PROTECTION_BEFORE;
        goto untracked;
    }
    // This cutoff precedes the caller's source reads. A later write must retain a newer generation; advancing the
    // certificate after copying would falsely certify an intervening write that was not copied.
    out->stamp = (TextureSourceStamp){.mapping = mapping, .protection = protection,
        .lineage = source_generation_lineage, .generation = gen_now};
    __atomic_add_fetch(&n_snap_checks, 1, __ATOMIC_RELAXED);
    if (!texture_source_stamp_valid(prior) || prior->lineage != source_generation_lineage ||
        prior->generation > gen_now || !mem_mapping_range_unchanged(addr, size, prior->mapping) ||
        !source_protection_compatible(addr, size, prior)) {
        if (exclusive) plat_mutex_unlock(&gen_lock); else plat_mutex_unlock_shared(&gen_lock);
        return TEX_SOURCE_UNKNOWN;
    }
    memset(out->dirty, 0, sizeof out->dirty);
    int changed = 0;
    for (uint64_t r = first; r <= last; r++) {
        if (reg_max[r] <= prior->generation) continue;
        uint64_t begin = r << REG_SHIFT, end = (r + 1) << REG_SHIFT;
        if (begin < lo) begin = lo;
        if (end > hi) end = hi;
        for (uint64_t page = begin; page < end; page += 1ull << GEN_PAGE_SHIFT) {
            uint32_t *generation = gen_slot(page, 0);
            if (!generation || *generation <= prior->generation) continue;
            uint64_t dirty_lo = page > addr ? page - addr : 0;
            uint64_t dirty_hi = page + (1ull << GEN_PAGE_SHIFT) - addr;
            if (dirty_hi > size) dirty_hi = size;
            texture_source_mark_range(out, spans, count, dirty_lo, dirty_hi);
            changed = 1;
        }
    }
    if (!mem_mapping_range_unchanged(addr, size, mapping)) {
        failure = TEX_SOURCE_TRACKING_MAPPING_AFTER;
        goto changed_during_scan;
    }
    if (!mem_protection_range_unchanged(addr, size, protection)) {
        failure = TEX_SOURCE_TRACKING_PROTECTION_AFTER;
        goto changed_during_scan;
    }
    if (exclusive) plat_mutex_unlock(&gen_lock); else plat_mutex_unlock_shared(&gen_lock);
    return changed ? TEX_SOURCE_DIRTY : TEX_SOURCE_CLEAN;

untracked:
    __atomic_add_fetch(&n_snap_untracked, 1, __ATOMIC_RELAXED);
    goto refused;
changed_during_scan:
    out->stamp = (TextureSourceStamp){0};
    texture_source_all_dirty(out, count);
refused:
    out->tracking_failure = failure;
    if (exclusive) plat_mutex_unlock(&gen_lock); else plat_mutex_unlock_shared(&gen_lock);
    return TEX_SOURCE_UNKNOWN;
}

void vk_mem_gen_lock_shared(void) { plat_mutex_lock_shared(&gen_lock); }
void vk_mem_gen_unlock_shared(void) { plat_mutex_unlock_shared(&gen_lock); }

void vk_mem_gen_lock(void) { plat_mutex_lock(&gen_lock); }
void vk_mem_gen_unlock(void) { plat_mutex_unlock(&gen_lock); }

int vk_mem_unchanged_why_locked(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch) {
    if (!size || !epoch) return 4;
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    if (r1 >= SNAP_REGIONS) return 4;
    for (uint64_t r = r0; r <= r1; r++) {   // REG_SHIFT is SNAP_SHIFT: reg_max is the newest stamp in the region
        if (!snap_current((uint32_t)r, epoch)) return 1;
        if (snap_untracked[r]) return 2;
        if (reg_max[r] > since) return 3;
    }
    return 0;
}

int vk_mem_unchanged_locked(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch) {
    return !vk_mem_unchanged_why_locked(addr, size, since, epoch);
}

uint64_t vk_mem_snap_range_locked(uint64_t addr, uint64_t size, uint32_t epoch) {
    if (!size || !epoch) return 0;
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    if (r1 >= SNAP_REGIONS) return 0;
    for (uint64_t r = r0; r <= r1; r++)
        if (!snap_current((uint32_t)r, epoch)) {
            uint64_t q = snap_regions_for(r0, r1, epoch, 1);
            n_snap_calls += q;
            return q;
        }
    return 0;
}

void vk_mem_snap_stats(uint64_t *checks, uint64_t *calls, uint64_t *untracked) {
    *checks = __atomic_exchange_n(&n_snap_checks, 0, __ATOMIC_RELAXED);
    *calls = __atomic_exchange_n(&n_snap_calls, 0, __ATOMIC_RELAXED);
    *untracked = __atomic_exchange_n(&n_snap_untracked, 0, __ATOMIC_RELAXED);
}

// ---------------------------------------------------------------- dirty census (NX2NAVI_TRACE=perf)
// What a device-local mirror of the buffer-bound pools would upload. The replay notes every
// imported pool it binds as a vertex, index, uniform, storage or indirect buffer (vk_mem_bound_note). At the start of
// each submit one write-watch query per run of such pools stamps the pages written since with a new generation, in the
// shared generation table (no second consumer of the OS bits: texture checks see the stamps), and the pages of those
// pools are counted:
// - eager: pages stamped after the previous submit's scan, in every buffer-bound pool (upload at every submit);
// - lazy: pages stamped after the pool's previous count, in pools the previous submit bound (upload when bound);
// - first use: the size of a pool the first time it is bound (a mirror copies the whole pool then).
// Classes: CPU_CACHED, CPU_UNCACHED, other (CPU_NO_ACCESS). Logged as `mem: dirty` every 60 frames.
#define DIRTY_UNSYNCED UINT32_MAX
#define DIRTY_GAP 0x100000ull   // pools closer than this share one query (the gap's pages are stamped too)
#define DIRTY_WIN 60
enum { DC_CACHED, DC_UNC, DC_OTHER, DC_N };
typedef struct {
    uint64_t eager[DC_N], lazy[DC_N], first[DC_N];
    uint64_t submits, queries, scan_ns, sub_max, frames;
} DirtyAcc;
static uint32_t dirty_sub = 1, dirty_gen;   // submit lock: the current submit number, generation of the last scan
static uint64_t first_pending[DC_N], ever_n[DC_N], ever_bytes[DC_N];   // submit lock
static PoolRec **dlist;                     // submit lock: the buffer-bound pools of a scan
static int dcap;
static PlatMutex dirty_lock = PLAT_MUTEX_INIT;   // everything below
static DirtyAcc dfr, dwin;                  // this frame, this window
static uint32_t dwin_kib[DIRTY_WIN][DC_N];  // eager KiB of each frame of the window
static uint64_t live_n[DC_N], live_bytes[DC_N], live_untracked, ever_snap_n[DC_N], ever_snap_bytes[DC_N];

static int dc_class(const PoolRec *r) {
    int c = rec_class(r);
    return c == VK_MCLS_HOST_CACHED ? DC_CACHED : c == VK_MCLS_HOST_UNC ? DC_UNC : DC_OTHER;
}

void vk_mem_bound_note(void *pool) {
    PoolRec *r = pool;
    if (r->bound_sub == dirty_sub) return;
    r->bound_sub = dirty_sub;
    if (r->bufbound) return;
    r->bufbound = 1, r->synced_gen = DIRTY_UNSYNCED;
    int k = dc_class(r);
    ever_n[k]++, ever_bytes[k] += r->size, first_pending[k] += r->size;
}

static uint64_t page_lo(const PoolRec *r) { return r->lo & ~0xfffull; }
static uint64_t page_hi(const PoolRec *r) { return (r->lo + r->size + 0xfff) & ~0xfffull; }

enum StampResult { STAMP_UNTRACKED, STAMP_DIRTY, STAMP_CLEAN };

// gen_lock held exclusively. Only a complete drain with no written page is clean.
static enum StampResult query_stamp(uint64_t lo, uint64_t hi, uint32_t g) {
    enum StampResult result = STAMP_CLEAN;
    for (;;) {
        uint64_t n = ww_query(lo, hi - lo);
        if (n == UINT64_MAX) return STAMP_UNTRACKED;
        if (n) result = STAMP_DIRTY;
        for (uint64_t k = 0; k < n; k++) stamp_page(gen_pages[k], g);
        if (n < sizeof gen_pages / sizeof *gen_pages) return result;
    }
}

// Collect writes pending when the region became read-only. A clean drain under the same mapping and
// protection renews an existing tracked snapshot. Leave its epoch alone so that a thaw before the next
// submission's first use still forces that submission to observe the newly writable pages.
static void observe_after_freeze(uint32_t region, const MemFreezeTransition *transition) {
    uint64_t addr = (uint64_t)region << REG_SHIFT;
    uint64_t protection = mem_protection_region(region), mapping = mem_mapping_region(region);
    int eligible = protection && mapping && mapping <= snap_mapping[region] &&
                   snap_at[region] && !snap_untracked[region] && mem_frozen_stable(addr);
    uint64_t lineage = source_generation_lineage;
    int bridge = eligible && transition && transition->before && transition->after > transition->before &&
                 transition->before <= snap_protection[region] && transition->after == protection &&
                 transition->mapping == mapping && lineage;
    enum StampResult result = query_stamp(addr, addr + (1ull << REG_SHIFT), generation_next());
    int stable = mem_frozen_stable(addr) && mem_protection_region(region) == protection &&
                 mem_mapping_region(region) == mapping;
    if (result == STAMP_UNTRACKED || !stable || lineage != source_generation_lineage)
        source_freeze_bridge[region] = (TextureSourceFreezeBridge){0};
    // A dirty drain is still complete evidence: its writes were stamped above and remain dirty to old sources.
    // A failed drain or concurrent transition provides no bridge. Repeated requests on already frozen memory
    // produce no new transition and cannot extend a bridge across an intervening thaw.
    if (bridge && result != STAMP_UNTRACKED && stable && lineage == source_generation_lineage)
        source_freeze_bridge[region] = (TextureSourceFreezeBridge){transition->before, protection, mapping, lineage};
    if (eligible && result == STAMP_CLEAN && mem_frozen_stable(addr) &&
        mem_protection_region(region) == protection && mem_mapping_region(region) == mapping) {
        snap_protection[region] = protection;
        snap_mapping[region] = mapping;
        freeze_snap_renewals++;
    }
}

// ---------------------------------------------------------------- shadow storage
// Where importing guest memory costs too much (Linux: re-protecting pages that amdgpu imported as a userptr
// invalidates the buffer, about 1.3 ms per 16 MiB at the next submit, and the import's write pinning clears the
// write watch), a CPU-visible pool gets a host-visible buffer of its own instead, at the offsets the import would have:
// - CPU -> shadow: right before each submit (vk_mem_shadow_sync), the pools the replay resolved since the last one are
//   queried for written pages; pages stamped after a pool's sgen are copied in by the CPU.
// - The device address table (vk_gpa.c) names the shadows, and its readers reach pools the replay never resolves: the
//   instanced foliage's and Navi's data, through pointers read from other buffers. Without a sync those shadows held
//   what guest memory held when they were made, and Kokiri Forest's trees were bare trunks, without ferns or grass
//   tufts. So a guest submission whose record binds the table (vk_mem_shadow_table_read) syncs every shadow in the
//   table, once, before the first part recorded after the binding is submitted (vk_mem_shadow_sync), and those that
//   have a VRAM mirror, which the table then names, queue the same pages for upload. Once is enough: the copy-ins read
//   the guest submission's write-watch snapshots (snap_regions), which a second sync in the same guest submission
//   would find unchanged. A sync in every part that used the table ran about three times a frame outdoors. Measured
//   on Linux (RTX 4090, Kokiri Forest, by the change's author): a sync at every guest submission, about six a
//   frame, took 9.8 ms and 888 write-watch queries a frame; one in the guest submissions that read the table 3.9 ms
//   and 92.
// - GPU -> guest: the replay notes each GPU write into CPU-visible memory at record time (vk_mem_gpu_pending) and
//   reports it at the fence (vk_gpu_written), which copies the GPU's bytes back into guest memory.
// - Pages both write: a GPU write is noted with the whole range its binding names, and the CPU writes other fields of the
//   same records meanwhile. Each page such a range covers keeps a base copy from its first note on, the bytes guest
//   memory and the shadow last agreed on. A copy-in takes only the bytes where guest memory differs from the base (the
//   CPU's writes) and the write-back at the fence only those where the shadow differs from it (the GPU's), and either
//   moves the base along (merge_page). A byte both changed takes the CPU's value when a copy-in comes first and the
//   GPU's when the fence does. Whole-range copies both ways lost the CPU's bytes: Ocarina writes a lighting parameter
//   into a buffer whose binding its compute passes write every frame, the GPU only ever read zero there, the lighting
//   coefficients and then the volumetric fog came out NaN, and the fog pass drew the 3D scene black.
// - The game must not get ahead of the write-backs. With an import the GPU's bytes are in guest memory when the GPU
//   has stored them; here they get there in a completion callback, after the fence. A CPU write that should follow
//   them and lands before that callback is overtaken: the write-back puts the GPU's older bytes over it and moves the
//   base copy, and a rewrite with the base copy's value (a reset to zero) is no difference at any time. So the wait
//   after a frame's first submission covers the write-backs with shadow storage (vk_sync.c "the executed timeline",
//   which tells what the game rewrites there and what a frame looked like without it).
// - Pages a shader stores into through the device address table (vk_gpa.c): the store goes to the pool's VRAM mirror
//   and to its twin, which here is the shadow, not guest memory. No binding names the range, so nothing notes it
//   (vk_mem_gpu_pending): the page has no base copy and guest memory never gets the bytes. The game's culling passes
//   store counts and records this way into pages whose other words the CPU rewrites. For one such page P, a word X the
//   shaders store and a word Y the CPU writes:
//     guest submission A (with the pass that stores): the CPU wrote Y, so A's sync copies P in whole (shadow_copy)
//       and queues it for the mirror (mirror_sync). X in the shadow and in the mirror is now guest memory's X, which
//       no store ever reached. A runs and stores X = v into both.
//     guest submission B (a later one that reads X through the table), the CPU has not written P since A's
//       write-watch snapshot:
//       B's sync leaves P alone and its draws read v.
//     B, the CPU has written Y again between the two snapshots: B's sync copies P in whole and uploads it, X
//       is guest memory's stale word again in both copies, and the draws read that.
//   An import does not have the fault: its twin is guest memory, and a page copied up from there holds the store.
//   Measured (NX2NAVI_SHADOW_CHECK, a standing run outdoors, RTX 5090 with NX2NAVI_SHADOW=1, 2026-11-10): the stores
//   go into one page each of ten pools of 20 and 44 KiB, and whole copy-ins overwrite one exactly four times a frame
//   (240 in every 60 frames): A's case, the copy that precedes the frame's stores, which does no harm. B's second
//   case would add to the four and was not seen in 1,800 frames; nothing proves it cannot happen where the frame's
//   timing differs. (Counted when every part that used the table synced; a guest submission syncs once now, from the
//   same write-watch snapshots, and the counts were not taken again.)
//   NX2NAVI_SHADOW_TABLE_STORES=1 keeps the stores. The pages are found by their content: each whole copy-in leaves a
//   hash of the page it wrote (smark), and the next one hashes the shadow page first. A different hash means something
//   stored into the shadow in between, and with no base copy on the page that was a store through the table. That
//   copy-in still overwrites the store (nothing tells its bytes from the CPU's), and from then on the page keeps a
//   base copy: copy-ins take the CPU's bytes only, and the fence of every submission whose record bound the table
//   writes the stored bytes of all such pages back to guest memory (table_wb_cb), as the fence of a noted write does.
//   Guest memory needs them: a word the CPU resets each frame to the value it had before the stores would otherwise
//   never differ from its base, and the reset would never reach the GPU.
//   It is off by default, for three reasons. The fault it removes was not seen. Its hashes cost 1.8 ms a frame of
//   the replay there (copy_marked). And it puts these pages under the rule of the bullet above: a count the CPU
//   resets before that write-back has landed keeps counting from its old value, where the whole copy-in of today
//   resets it every time; whether the game resets any of these words from the CPU is not known.
// NX2NAVI_SHADOW_CHECK=1 counts, with the shadow line and at its rate every 60 frames (shadow_table_log): the whole
// copy-ins that overwrote a store through the table, per pool; the write-backs that moved bytes into a page the CPU
// had written since its copy-in (cpu_wrote_since); the bytes both sides changed since their base copy (taken from the
// CPU at a copy-in, from the GPU at a fence); what the copy-ins cost by kind.
// NX2NAVI_SHADOW=1/0 forces the mode on or off; the default is on for Linux and Wine, except for NVIDIA's own driver on
// Linux, where imports ran about twice as fast on 0.7.1 (mirror_mode).
#define SB_PAGE 4096ull
static PlatMutex shadow_lock = PLAT_MUTEX_INIT;   // after pool_lock; before gen_lock
static PoolRec **stouch;                          // shadow_lock: pools resolved since the last sync
static int nstouch, cstouch;
static uint64_t s_sb_pages;                       // shadow_lock: base copies kept (sbase_make)
static int64_t s_live_n, s_live_bytes;            // shadow_lock
static uint64_t s_syncs, s_copied, s_queries, s_untracked, s_wb_bytes, s_ticks;   // shadow_lock: since the last log
static uint64_t s_wb_seen, s_wb_ticks;            // shadow_lock: bytes the fence write-backs compared, their time
static uint64_t s_q_ticks;                        // shadow_lock: the copy-ins' write-watch queries
static int wb_no_reup(void);
// Pages a shader stores into through the device address table. The two switches are read in mirror_mode.
static int s_learn;                               // NX2NAVI_SHADOW_TABLE_STORES: such pages get a base copy and a write-back
static int s_check;                               // NX2NAVI_SHADOW_CHECK: the counts below marked "check"
static PoolRec **tpools;                          // shadow_lock: the pools that have such pages
static int ntpools, ctpools;
static uint64_t s_tab_pages;                      // shadow_lock: such pages, all pools
// shadow_lock, since the last log:
static uint64_t s_tab_over;                       // whole copy-ins that overwrote a store through the table
static uint64_t s_tab_wb, s_tab_wb_seen, s_tab_wb_ticks, s_tab_wb_jobs;   // table_wb_cb: bytes moved and compared, time, runs
static uint64_t s_marked, s_mark_ticks;           // pages copied in whole with a mark; check: the time of those copies
static uint64_t s_plain_ticks;                    // check: the time of the whole copies made without marks
static uint64_t s_both_cpu, s_both_gpu;           // check: bytes both sides changed, taken from the CPU / from the GPU
static uint64_t s_in_merge, s_in_nomirror;        // check: bytes copied in by merge_page; whole, into pools without a mirror
static uint64_t s_tab_ticks, s_tab_syncs;         // the sync for the table's readers: its time (part of s_ticks), its runs
static PoolRec **stab;                            // shadow_lock: the shadows in the device address table (PoolRec.stab_i)
static int nstab, cstab;
static uint32_t stab_want;                        // the last guest submission whose record bound the table
static uint64_t s_t_over_all;                     // shadow_lock: s_tab_over since the start
static uint64_t s_late_pages, s_late_bytes;       // shadow_lock, check: write-backs into pages the CPU wrote since their copy-in
static uint64_t s_stale_pages;                    // shadow_lock: write-backs left out, the shadow could not hold them (shadow_write_back)

static int shadow_pool(VkCtx *c, PoolRec *r, uint64_t addr, uint64_t size) {
    uint64_t n = r->import_hi - r->import_lo;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    // Cached where the device has it: the fence copies read it back on the CPU.
    VmaAllocationCreateInfo ai = {.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  .preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT};
    VmaAllocationInfo info;
    RcVmaRequest admission = {.producer = RC_BUDGET_BUFFER, .required = 1};
    VkResult res = resource_core_vma_create_buffer(&admission, c->vma, &bi, &ai, &r->buf, &r->salloc, &info, NULL);
    if (res != VK_SUCCESS) {
        rt_log("vk: pool %#llx+%#llx: shadow allocation failed (%d)", (unsigned long long)addr, (unsigned long long)size,
               (int)res);
        r->buf = VK_NULL_HANDLE, r->salloc = NULL;
        return 0;
    }
    r->smap = info.pMappedData;
    r->sborn = vk_sync_issued() + 1;   // the submission being recorded now, or the next one
    plat_mutex_lock(&gen_lock);
    r->sgen = gen_now;   // the copy below holds every write stamped so far; later stamps are copied again
    plat_mutex_unlock(&gen_lock);
    memcpy(r->smap, (const void *)(uintptr_t)r->import_lo, n);
    plat_mutex_lock(&shadow_lock);
    s_live_n++, s_live_bytes += (int64_t)n;
    plat_mutex_unlock(&shadow_lock);
    plat_inc32(&n_imports);
    return 1;
}

static void shadow_touch(PoolRec *r) {
    plat_mutex_lock(&shadow_lock);
    if (!r->stouched) {
        if (nstouch == cstouch) grow((void **)&stouch, &cstouch, sizeof *stouch);
        stouch[nstouch++] = r;
        r->stouched = 1;
    }
    plat_mutex_unlock(&shadow_lock);
}

// ---- base copies of the pages both write ("Pages both write" above). shadow_lock held throughout.
static uint32_t sbase_hash(uint64_t pg) { return (uint32_t)((pg >> 12) * 0x9E3779B97F4A7C15ull >> 32); }

// The base copy of guest page pg of r, NULL when it has none.
static uint8_t *sbase_find(const PoolRec *r, uint64_t pg) {
    if (!r->nsb) return NULL;
    for (uint32_t i = sbase_hash(pg) & (r->csb - 1);; i = (i + 1) & (r->csb - 1)) {
        if (r->sb[i].page == pg) return r->sb[i].base;
        if (!r->sb[i].page) return NULL;
    }
}

static void sbase_insert(struct SBase *t, uint32_t cap, uint64_t pg, uint8_t *base) {
    uint32_t i = sbase_hash(pg) & (cap - 1);
    while (t[i].page) i = (i + 1) & (cap - 1);
    t[i].page = pg, t[i].base = base;
}

// Page pg of r gets a base copy unless it has one. Taken from the shadow: no GPU write was noted on the page before, so
// the shadow holds what guest memory held at its last copy-in, and guest memory differs from it by the CPU's writes since.
static void sbase_make(PoolRec *r, uint64_t pg) {
    if (sbase_find(r, pg)) return;
    if ((r->nsb + 1) * 4 > r->csb * 3) {   // at most three quarters full
        uint32_t cap = r->csb ? r->csb * 2 : 16;
        struct SBase *t = calloc(cap, sizeof *t);
        if (!t) rt_fatal("vk: out of memory for the shadow's base copies");
        for (uint32_t i = 0; i < r->csb; i++)
            if (r->sb[i].page) sbase_insert(t, cap, r->sb[i].page, r->sb[i].base);
        free(r->sb);
        r->sb = t, r->csb = cap;
    }
    uint8_t *b = malloc(SB_PAGE);
    if (!b) rt_fatal("vk: out of memory for the shadow's base copies");
    memcpy(b, r->smap + (pg - r->import_lo), SB_PAGE);
    sbase_insert(r->sb, r->csb, pg, b);
    r->nsb++;
    s_sb_pages++;
}

static void sbase_free(PoolRec *r) {
    for (uint32_t i = 0; i < r->csb; i++) free(r->sb[i].base);
    free(r->sb);
    s_sb_pages -= r->nsb;
    r->sb = NULL, r->nsb = r->csb = 0;
}

// 0xff in each byte of x that is not zero, 0 in the others.
static inline uint64_t byte_mask(uint64_t x) {
    x |= x >> 4, x |= x >> 2, x |= x >> 1;
    return (x & 0x0101010101010101ull) * 0xff;
}

// The bytes of v that m selects into dst (little-endian), the other bytes of dst left as they are: the other side may
// be writing them at the same time.
static inline void store_bytes(uint8_t *dst, uint64_t v, uint64_t m) {
    if (m == ~0ull) {
        memcpy(dst, &v, 8);
        return;
    }
    for (int k = 0; k < 8; k++)
        if (m >> (8 * k) & 0xff) dst[k] = (uint8_t)(v >> (8 * k));
}

// n bytes of one page with base copy b: the bytes where src differs from the base go into dst and into the base. src
// and dst are guest memory and the shadow, one way or the other; each src byte that moves is read once, as its side
// may be writing it. Returns the bytes moved.
static uint64_t merge_page(uint8_t *dst, const uint8_t *src, uint8_t *b, uint64_t n) {
    uint64_t moved = 0, i = 0;
    for (; i < n && ((uintptr_t)(src + i) & 7); i++) {
        uint8_t v = ((const volatile uint8_t *)src)[i];
        if (v != b[i]) dst[i] = b[i] = v, moved++;
    }
    for (; i + 8 <= n; i += 8) {
        // Most of a page is unchanged: a 64-byte block equal to its base is passed over with one vector compare. A
        // byte its side writes after the compare is a later write, which the next copy-in or write-back takes.
        if (!((uintptr_t)(src + i) & 63) && i + 64 <= n) {
            uint64_t x = 0;
            for (int k = 0; k < 64; k += 8) {
                uint64_t sv, ov;
                memcpy(&sv, src + i + k, 8), memcpy(&ov, b + i + k, 8);
                x |= sv ^ ov;
            }
            if (!x) {
                i += 56;
                continue;
            }
        }
        uint64_t s = *(const volatile uint64_t *)(src + i), o;
        memcpy(&o, b + i, 8);
        uint64_t m = byte_mask(s ^ o);
        if (!m) continue;
        store_bytes(dst + i, s, m);
        memcpy(b + i, &s, 8);   // the bytes m leaves out are equal in both
        moved += (uint64_t)__builtin_popcountll(m) / 8;
    }
    for (; i < n; i++) {
        uint8_t v = ((const volatile uint8_t *)src)[i];
        if (v != b[i]) dst[i] = b[i] = v, moved++;
    }
    return moved;
}

// ---- pages a shader stores into through the device address table ("shadow storage" above). shadow_lock held.

// The mark of a 4 KiB page: a hash of its bytes, never 0. Four sums that do not wait for each other.
static uint32_t page_mark(const uint8_t *p) {
    uint64_t h[4] = {0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full, 0x165667B19E3779F9ull, 0x27D4EB2F165667C5ull};
    for (uint64_t i = 0; i < SB_PAGE; i += 32) {
        uint64_t w[4];
        memcpy(w, p + i, 32);
        for (int k = 0; k < 4; k++) {
            h[k] = (h[k] ^ w[k]) * 0xFF51AFD7ED558CCDull;
            h[k] = h[k] << 29 | h[k] >> 35;
        }
    }
    uint64_t x = (h[0] ^ h[1] << 13 ^ h[2] << 27 ^ h[3] << 41 ^ h[1] >> 51 ^ h[2] >> 37 ^ h[3] >> 23) * 0xC4CEB9FE1A85EC53ull;
    uint32_t m = (uint32_t)(x >> 32);
    return m ? m : 1;
}

// Guest page pg of r was stored into through the table: it keeps a base copy from here on, taken from the shadow as
// the whole copy-in just left it (guest memory's bytes), and table_wb_cb writes its stores back.
static void table_page_add(PoolRec *r, uint64_t pg) {
    sbase_make(r, pg);
    if (r->ntpg == r->ctpg) {
        uint32_t cap = r->ctpg ? r->ctpg * 2 : 16;
        uint64_t *t = realloc(r->tpg, cap * sizeof *t);
        if (!t) rt_fatal("vk: out of memory for the shadow's table pages");
        r->tpg = t, r->ctpg = cap;
    }
    r->tpg[r->ntpg++] = pg;
    s_tab_pages++;
    if (r->tlisted) return;
    if (ntpools == ctpools) grow((void **)&tpools, &ctpools, sizeof *tpools);
    tpools[ntpools++] = r;
    r->tlisted = 1;
}

// NX2NAVI_SHADOW_CHECK: the bytes of [0, n) that both sides changed since the base copy b and that now differ, which a
// merge of src into dst is about to settle in src's favour.
static uint64_t both_changed(const uint8_t *dst, const uint8_t *src, const uint8_t *b, uint64_t n) {
    uint64_t k = 0;
    for (uint64_t i = 0; i < n; i++) k += (src[i] != b[i]) & (dst[i] != b[i]) & (dst[i] != src[i]);
    return k;
}

// Copies the whole guest pages [a, e) of r into its shadow, none of which has a base copy. Each page's mark tells
// whether the shadow page still holds what the last such copy left: if not, a shader stored into it through the table
// meanwhile, and this copy overwrites that store ("Pages a shader stores into" above says why it cannot be kept, and
// what the page gets from here on).
// Only with NX2NAVI_SHADOW_TABLE_STORES or NX2NAVI_SHADOW_CHECK (shadow_copy). A mark is two hashes of the page, one
// of them of bytes the copy would not have read: outdoors, where 2,400 pages a frame are copied in whole, the marks
// cost 1.8 ms a frame of the replay (3.09 against 1.32 ms of copy-ins, 67 against 75 fps, RTX 5090 with
// NX2NAVI_SHADOW=1, 2026-11-10).
static void copy_marked(PoolRec *r, uint64_t a, uint64_t e) {
    if (!r->smark) {
        r->smark = calloc((r->import_hi - r->import_lo) / SB_PAGE, sizeof *r->smark);
        if (!r->smark) rt_fatal("vk: out of memory for the shadow's page marks");
    }
    for (uint64_t pg = a; pg < e; pg += SB_PAGE) {
        uint8_t *s = r->smap + (pg - r->import_lo);
        uint32_t *m = &r->smark[(pg - r->import_lo) / SB_PAGE];
        int stored = *m && page_mark(s) != *m;
        memcpy(s, (const void *)(uintptr_t)pg, SB_PAGE);
        if (stored) {
            s_tab_over++, s_t_over_all++, r->t_over++;
            if (s_learn) {   // merged from now on: its mark is not read again
                table_page_add(r, pg);
                continue;
            }
        }
        *m = page_mark(s);   // of the shadow, not of guest memory, which the CPU may be writing
    }
    s_marked += (e - a) / SB_PAGE;
}

// vk_pool_fini, after the pool left the index: off the touched list before its record goes.
static void shadow_forget(PoolRec *r) {
    plat_mutex_lock(&shadow_lock);
    for (int i = 0; i < nstouch; i++)
        if (stouch[i] == r) { stouch[i] = stouch[--nstouch]; break; }
    for (int i = 0; r->tlisted && i < ntpools; i++)
        if (tpools[i] == r) { tpools[i] = tpools[--ntpools]; break; }
    if (r->stab_i) {   // the last one takes its place
        PoolRec *last = stab[--nstab];
        stab[r->stab_i - 1] = last, last->stab_i = r->stab_i;
        r->stab_i = 0;
    }
    s_live_n--, s_live_bytes -= (int64_t)(r->import_hi - r->import_lo);
    s_tab_pages -= r->ntpg;
    r->tlisted = 0;
    plat_mutex_unlock(&shadow_lock);
}

static void shadow_metadata_destroy(PoolRec *r) {
    if (!r->sb && !r->tpg && !r->smark) return;
    plat_mutex_lock(&shadow_lock);
    free(r->tpg);
    free(r->smark);
    sbase_free(r);
    plat_mutex_unlock(&shadow_lock);
}

// shadow_lock held: copies guest [a, b) into the shadow; of a page with a base copy only the CPU's bytes (merge_page).
static void shadow_copy(PoolRec *r, uint64_t a, uint64_t b) {
    while (a < b) {
        uint64_t e = b;
        uint8_t *base = NULL;
        if (r->nsb) {   // the plain run up to the next page with a base copy
            for (e = a; e < b; e = (e & ~(SB_PAGE - 1)) + SB_PAGE)
                if ((base = sbase_find(r, e & ~(SB_PAGE - 1)))) break;
            if (e > b) e = b;
            if (e == a) {   // a starts on such a page: its part of [a, b)
                uint64_t pg = a & ~(SB_PAGE - 1), pe = pg + SB_PAGE < b ? pg + SB_PAGE : b;
                uint8_t *s = r->smap + (a - r->import_lo);
                if (s_check) s_both_cpu += both_changed(s, (const uint8_t *)(uintptr_t)a, base + (a - pg), pe - a);
                uint64_t n = merge_page(s, (const uint8_t *)(uintptr_t)a, base + (a - pg), pe - a);
                s_copied += n, s_in_merge += n;
                a = pe;
                continue;
            }
        }
        // The copies are of whole pages (shadow_sync_pool); the marks need the device address table, whose stores
        // they find, and one of the two switches.
        uint64_t t0 = s_check ? qpc() : 0;
        int marks = (s_learn || s_check) && r->gpa_rec && !((a | e) & (SB_PAGE - 1));
        if (marks) copy_marked(r, a, e);
        else memcpy(r->smap + (a - r->import_lo), (const void *)(uintptr_t)a, e - a);
        if (s_check) {
            *(marks ? &s_mark_ticks : &s_plain_ticks) += qpc() - t0;
            if (r->mstate != MS_ON) s_in_nomirror += e - a;
            r->s_in += e - a;
        }
        s_copied += e - a;
        a = e;
    }
}

// shadow_lock and gen_lock held: stamps the pool's written pages and copies those newer than its shadow. The pool's
// regions are queried through the write-watch snapshots of the guest submission (snap_regions, epoch), which the
// texture checks and the mirror syncs share: each 2 MiB region at most once per guest submission. A query of every
// touched pool at every part's submit was about 380 write-watch calls and 10 ms a frame of the submitting thread in
// Kokiri Forest, while the GPU waited for the next part (RTX 4090, 2026-11-09). A page the CPU writes after its
// region's snapshot goes in at the next guest submission's, as it does into a VRAM mirror.
// cpu (wb_no_reup): the pages are read from the tables without the stamps of a mirror's write-back (stamp_page_as),
// since that write-back went through the shadow, which holds those bytes already; with the other tables every storage
// buffer the game's compute shaders wrote was compared again at the next copy-in.
static void shadow_sync_pool(PoolRec *r, uint32_t epoch, int cpu) {
    uint64_t lo = r->import_lo, hi = r->import_hi;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    int tracked = 1;
    uint64_t t0 = qpc();
    if (epoch && r1 < SNAP_REGIONS) {
        s_queries += snap_regions(r0, r1, epoch);
        for (uint64_t k = r0; k <= r1 && tracked; k++)
            tracked = !snap_untracked[k] && snap_current((uint32_t)k, epoch);
    } else {   // outside the snapshot regions, or before the first guest submission: one query per unfrozen run
        uint32_t g = generation_next();
        for (uint64_t a = lo; a < hi;) {
            uint64_t b = (a | ((1ull << REG_SHIFT) - 1)) + 1;
            if (b > hi) b = hi;
            if (mem_frozen(a)) { a = b; continue; }
            while (b < hi && !mem_frozen(b)) b = b + (1ull << REG_SHIFT) < hi ? b + (1ull << REG_SHIFT) : hi;
            s_queries++;
            if (!query_stamp(a, b, g)) tracked = 0;
            a = b;
        }
    }
    s_q_ticks += qpc() - t0;
    uint32_t since = r->sgen;
    r->sgen = gen_now;
    if (!tracked) {   // nothing tells which pages changed: all of them
        s_untracked++;
        shadow_copy(r, lo, hi);
        return;
    }
    uint64_t run = 0;
    for (uint64_t k = lo >> REG_SHIFT; k <= (hi - 1) >> REG_SHIFT; k++) {
        uint64_t a = k << REG_SHIFT, b = a + (1ull << REG_SHIFT);
        if (a < lo) a = lo;
        if (b > hi) b = hi;
        const uint32_t *ch = (cpu ? cgen_chunk : gen_chunk)[a >> GEN_CHUNK_SHIFT];   // a 2 MiB region lies in one chunk
        if ((cpu ? creg_max : reg_max)[k] <= since || !ch) {
            if (run) shadow_copy(r, run, a), run = 0;
            continue;
        }
        for (uint64_t x = a; x < b; x += 1ull << GEN_PAGE_SHIFT) {
            int dirty = ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] > since;
            if (dirty && !run) run = x;
            else if (!dirty && run) shadow_copy(r, run, x), run = 0;
        }
    }
    if (run) shadow_copy(r, run, hi);
}

// Record time: the GPU may write [addr, addr+size) of CPU-visible memory. With shadow storage the pages it covers get
// their base copies before the submission's copy-in (vk_mem_shadow_sync) can reach them.
static void shadow_pending_pool(PoolRec *pool, uint64_t address, uint64_t size) {
    if (!pool || !pool->smap || address < pool->import_lo || address >= pool->import_hi ||
        size > pool->import_hi - address) return;
    plat_mutex_lock(&shadow_lock);
    for (uint64_t page = address & ~(SB_PAGE - 1); page < address + size; page += SB_PAGE) sbase_make(pool, page);
    plat_mutex_unlock(&shadow_lock);
}

void vk_mem_gpu_pending(uint64_t addr, uint64_t size) {
    if (!shadow_on || !size) return;
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *entry = pidx_find(&pool_index, addr, size);
    shadow_pending_pool(entry ? entry->buf : NULL, addr, size);
    plat_mutex_unlock_shared(&pool_lock);
}

void vk_mem_gpu_pending_span(const VkMemSpan *span, uint64_t address, uint64_t size) {
    // The actual command batch already owns this provider; there is no live VA lookup.
    if (shadow_on && size && span && span->pool) shadow_pending_pool(span->pool, address, size);
}

// NX2NAVI_WB_NO_REUP (default 1; 0 = off): the pages a VRAM mirror wrote back into guest memory are not copied up into
// it again at its next sync. vk_gpu_written stamps the pages the GPU wrote, at the fence, so the texture cache re-reads
// them; the same stamps made mirror_sync copy them back up into the mirror the write-back had read them from: the
// storage buffers the game's compute shaders write went down and came back up every frame, about 57 of the 60 MiB copied
// up a frame outdoors (RTX 4080 SUPER, 2026-11-08) and 20 MiB a frame in Kokiri Forest (RTX 4090, PCIe 3.0, 2026-11-09).
// A write into a mirror (vk_mem_mirrored, noted at record time) stamps only the texture tables (vk_gpu_written_back);
// mirror_sync reads the CPU-side ones (stamp_page_as), which a CPU write or a host path still stamps.
static int wb_reup_off = -1;
static int wb_no_reup(void) {
    if (wb_reup_off < 0) {
        int on = rt_knob("WB_NO_REUP", "1") == 1;
        rt_log("mem: pages a VRAM mirror wrote back %s (NX2NAVI_WB_NO_REUP)",
               on ? "are not copied up into it again" : "are copied up into it again");
        __atomic_store_n(&wb_reup_off, on, __ATOMIC_RELEASE);
    }
    return wb_reup_off;
}

// NX2NAVI_SHADOW_BULK_WRITE=1 (default 0): before a write-back stores into guest memory, the write watch's protection
// of the whole range is cleared in one call (plat_mem_prepare_write). Where the watch protects pages (Linux), the
// first store into each page otherwise takes a fault of its own; the pages count as written either way.
static int shadow_bulk_write(void) {
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("NX2NAVI_SHADOW_BULK_WRITE");
        if (on) rt_log("mem: shadow write-backs clear the write watch's protection of their range in one call "
                       "(NX2NAVI_SHADOW_BULK_WRITE)");
    }
    return on;
}

// NX2NAVI_SHADOW_CHECK: the CPU wrote guest page pg after the copy-in that sgen stands for: a write-watch snapshot
// stamped it later, or its write-watch bit is set and no snapshot has read it yet. A write-back that moves bytes into
// such a page may have come after a CPU write it should have come before ("The game must not get ahead of the
// write-backs" above); a page of which the CPU wrote other bytes counts too.
// No gen_lock, as lazy_clean: a stamp that lands meanwhile only blurs a count.
static int cpu_wrote_since(uint64_t pg, uint32_t sgen) {
    const uint32_t *ch = __atomic_load_n(&cgen_chunk[pg >> GEN_CHUNK_SHIFT], __ATOMIC_ACQUIRE);
    if (ch && __atomic_load_n(&ch[(pg >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)], __ATOMIC_RELAXED) > sgen) return 1;
    uint64_t p[1], n = mem_written_peek(pg, SB_PAGE, p, 1);
    return n && n != UINT64_MAX;
}

// Completion thread: the GPU's bytes of [addr, addr+size) from the shadow into guest memory (merge_page against the
// base copies; a page without one whole). Returns the bytes moved.
// A page without a base copy is one no write was noted on in this record (vk_mem_gpu_pending notes every write the
// replay records). It gets here when the pool the write was noted on was finalized before the write's fence and
// another was made at its address: shadow_forget dropped the base copies, and the new record's shadow is a copy of
// guest memory from its own start that holds no byte of that write. Copied whole, it put those older bytes over
// everything the CPU had written to the page since, for no gain. So the page is left alone when the record is younger
// than the submission whose fence this is (sborn), or when the CPU has written the page since the shadow's last
// copy-in (cpu_wrote_since: a pool made again while that submission was still being recorded has the same value).
// Any other such page goes whole as before: a write nothing noted, into a shadow that holds it.
static int pool_host_write_hold(const PoolRec *pool, uint64_t address, uint64_t size) {
    if (pool->host_backing) return mem_backing_read_hold_at(pool->host_backing, address, size);
    if (!pool->host_mapping_cutoff || !mem_read_hold(address, size)) return 0;
    if (mem_mapping_range_unchanged(address, size, pool->host_mapping_cutoff)) return 1;
    mem_read_release();
    return 0;
}

static uint64_t shadow_write_back_from(uint64_t addr, uint64_t size, PoolRec *recorded) {
    uint64_t moved = 0;
    int reconciled = 0;
    RcRegistryVisibility visibility = {0};
    rc_registry_visibility_begin(rc_registry_runtime(), RC_REGISTRY_VISIBILITY_COMPLETED, &visibility);
    int bulk = shadow_bulk_write();   // before the locks: its first call logs
    uint64_t value = vk_sync_cb_value();   // the fence being completed; a call from elsewhere reports submitted work
    if (!value) value = vk_sync_issued();
    PoolRec *r = recorded;
    if (!r) {
        // The standalone probe has no recorded provider. Normal work passes its retained provider.
        plat_mutex_lock_shared(&pool_lock);
        const pidx_ent *e = pidx_find(&pool_index, addr, size);
        r = e ? e->buf : NULL;
    }
    int held = r && r->smap && size <= UINT64_MAX - addr && addr >= r->import_lo &&
        addr + size <= r->import_hi && pool_host_write_hold(r, addr, size);
    plat_mutex_lock(&shadow_lock);
    if (held) {
        reconciled = r->sborn <= value;
        uint64_t t0 = qpc();
        mem_thaw(addr, size);
        if (bulk) plat_mem_prepare_write(addr, size);
        for (uint64_t a = addr, end = addr + size; a < end;) {
            uint64_t pg = a & ~(SB_PAGE - 1), pe = pg + SB_PAGE < end ? pg + SB_PAGE : end;
            uint8_t *base = sbase_find(r, pg), *s = r->smap + (a - r->import_lo);
            if (base && s_check) s_both_gpu += both_changed((const uint8_t *)(uintptr_t)a, s, base + (a - pg), pe - a);
            int late = base && s_check && cpu_wrote_since(pg, r->sgen);   // before the merge's own stores into the page
            uint64_t n = 0;
            if (base) moved += n = merge_page((uint8_t *)(uintptr_t)a, s, base + (a - pg), pe - a);
            else if (r->sborn > value || cpu_wrote_since(pg, r->sgen)) {
                s_stale_pages++;   // the shadow cannot hold the write
                reconciled = 0;
            }
            else memcpy((void *)(uintptr_t)a, s, pe - a), moved += pe - a;   // a write never noted as pending
            if (late && n) s_late_pages++, s_late_bytes += n;
            a = pe;
        }
        s_wb_bytes += moved, s_wb_seen += size, s_wb_ticks += qpc() - t0;
    }
    plat_mutex_unlock(&shadow_lock);
    // Zero changed bytes can still mean a complete merge. Conversely, copying some pages is not proof for a
    // range containing skipped replacement pages. Keep pool identity stable while publishing the visibility fact.
    if (reconciled) {
        rc_registry_visibility_apply(rc_registry_runtime(), &visibility, (RcSourceRange){1, addr, size});
        pool_host_visibility(&visibility, r, addr, size);
    }
    if (held) mem_read_release();
    if (!recorded) plat_mutex_unlock_shared(&pool_lock);
    return moved;
}

// Completion thread, at the fence of a submission whose record bound the device address table (vk_mem_shadow_sync):
// what the shaders stored into the pages found so far ("Pages a shader stores into" above) goes from the shadow into
// guest memory, the bytes that differ from the page's base copy. A page whose shadow equals its base is passed over
// before guest memory is touched, so a frozen region stays frozen. The pages that changed are stamped as a mirror's
// write-back is (swb_run): the stores are in the mirror already.
typedef struct {
    PoolRec *provider;
    uint64_t page;
} TableWritePage;
typedef struct {
    size_t count;
    TableWritePage pages[];
} TableWriteJob;

static uint64_t table_write_page(const TableWritePage *entry, const RcRegistryVisibility *proof, int bulk, int back) {
    PoolRec *r = entry->provider;
    uint64_t page = entry->page;
    if (!pool_host_write_hold(r, page, SB_PAGE)) return 0;
    uint64_t moved = 0;
    plat_mutex_lock(&shadow_lock);
    uint8_t *base = sbase_find(r, page), *source = r->smap + (page - r->import_lo);
    if (base && memcmp(base, source, SB_PAGE)) {
        // Freeze the value consumed by this merge. Only bytes different from its base are copied to guest storage;
        // neither an equal byte nor adjacent padding establishes CPU visibility.
        uint8_t previous[SB_PAGE], captured[SB_PAGE];
        memcpy(previous, base, sizeof previous);
        memcpy(captured, source, sizeof captured);
        if (mem_frozen(page)) mem_thaw(page, SB_PAGE);
        if (bulk) plat_mem_prepare_write(page, SB_PAGE);
        if (s_check) s_both_gpu += both_changed((const uint8_t *)(uintptr_t)page, captured, previous, SB_PAGE);
        moved = merge_page((uint8_t *)(uintptr_t)page, captured, base, SB_PAGE);
        if (moved && !nvn_gpu_only(page)) {
            plat_mutex_lock(&gen_lock);
            stamp_page_as(page, generation_next(), back);
            plat_mutex_unlock(&gen_lock);
        }
        for (uint64_t at = 0; at < SB_PAGE;) {
            while (at < SB_PAGE && previous[at] == captured[at]) ++at;
            uint64_t first = at;
            while (at < SB_PAGE && previous[at] != captured[at]) ++at;
            if (at != first) {
                rc_registry_visibility_apply(rc_registry_runtime(), proof, (RcSourceRange){1, page + first, at - first});
                pool_host_visibility(proof, r, page + first, at - first);
            }
        }
    }
    plat_mutex_unlock(&shadow_lock);
    mem_read_release();
    return moved;
}

static void table_wb_release(TableWriteJob *job) {
    for (size_t i = 0; i < job->count; i++) destroy_rec(vk_ctx(), job->pages[i].provider);
    free(job);
}

static void table_wb_cb(void *arg) {
    TableWriteJob *job = arg;
    int bulk = shadow_bulk_write(), back = wb_no_reup();
    uint64_t started = qpc(), moved = 0;
    RcRegistryVisibility visibility = {0};
    rc_registry_visibility_begin(rc_registry_runtime(), RC_REGISTRY_VISIBILITY_COMPLETED, &visibility);
    for (size_t i = 0; i < job->count; i++) moved += table_write_page(&job->pages[i], &visibility, bulk, back);
    plat_mutex_lock(&shadow_lock);
    s_tab_wb += moved;
    s_tab_wb_seen += job->count * SB_PAGE;
    s_tab_wb_jobs++;
    s_tab_wb_ticks += qpc() - started;
    plat_mutex_unlock(&shadow_lock);
    table_wb_release(job);
}

static void table_wb_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;
    table_wb_release(arg);
}

// shadow_lock held. Snapshot the existing learned pages before this submission's copy-ins mutate the learning set.
static void table_wb_queue(uint64_t fence) {
    size_t count = 0;
    for (int i = 0; i < ntpools; i++) count += tpools[i]->ntpg;
    if (!count) return;
    if (count > (SIZE_MAX - sizeof(TableWriteJob)) / sizeof(TableWritePage))
        rt_fatal("vk: GPA shadow writeback inventory is too large");
    TableWriteJob *job = malloc(sizeof *job + count * sizeof *job->pages);
    if (!job) rt_fatal("vk: GPA shadow writeback inventory allocation failed");
    job->count = count;
    size_t next = 0;
    for (int i = 0; i < ntpools; i++) {
        PoolRec *r = tpools[i];
        for (uint32_t j = 0; j < r->ntpg; j++) {
            pool_native_acquire(r);
            job->pages[next++] = (TableWritePage){r, r->tpg[j]};
        }
    }
    vk_completion_add_owned(fence, table_wb_cb, table_wb_cancel, job);
}

static void gpu_written(uint64_t addr, uint64_t size, int back, PoolRec *recorded) {
    // back: the bytes are in a VRAM mirror, and its write-back brings them into the shadow; that submission's
    // shadow_wb_cb, after the replay's own callbacks, then copies them into guest memory.
    if (shadow_on && size && !back) shadow_write_back_from(addr, size, recorded);
    if (!size || nvn_gpu_only(addr)) return;
    if (recorded && !pool_host_write_hold(recorded, addr, size)) return;
    uint64_t lo = addr & ~0xfffull, hi = (addr + size + 0xfff) & ~0xfffull;
    back = back && wb_no_reup();   // read before gen_lock: its first call logs
    plat_mutex_lock(&gen_lock);
    uint32_t g = generation_next();
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) stamp_page_as(a, g, back);
    plat_mutex_unlock(&gen_lock);
    if (recorded) mem_read_release();
    // Direct imported destinations carry an exact own-writer visibility fact from recording. Looking up today's
    // pool here cannot prove which backing an earlier command wrote, or reconcile other overlapping producers.
}

void vk_gpu_written(uint64_t addr, uint64_t size) { gpu_written(addr, size, 0, NULL); }
static int wb_lazy(void);
static int lazy_clean(uint64_t lo, uint64_t hi, void *ctx);

static void gpu_written_back(uint64_t addr, uint64_t size, PoolRec *recorded, VkBuffer buffer) {
    gpu_written(addr, size, 1, recorded);
    // NX2NAVI_WB_PROBE: the bytes are in guest memory now; note who reads them next. With shadow storage they get there
    // later, in the submission's shadow_wb_cb, whose thaw would end the probe at once: that job arms it instead.
    if (!size) return;
    int arm = wb_lazy();
    if (!recorded) {
        // The probe compatibility entry has no captured provider.
        if (!shadow_on) mem_probe(addr, size);
        return;
    }
    plat_mutex_lock_shared(&pool_lock);
    if (!recorded->unpublished && pool_host_write_hold(recorded, addr, size)) {
        if (!shadow_on) mem_probe(addr, size);
        if (arm && recorded->mstate == MS_ON && recorded->mbuf == buffer && buffer && recorded->buf &&
            addr >= recorded->import_lo && addr < recorded->import_hi && size <= recorded->import_hi - addr)
            mem_lazy_arm(addr, size, (uint64_t)(uintptr_t)recorded->mbuf, (uint64_t)(uintptr_t)recorded->buf,
                         addr - recorded->import_lo, recorded);
        mem_read_release();
    }
    plat_mutex_unlock_shared(&pool_lock);
}

void vk_gpu_written_back(uint64_t addr, uint64_t size) { gpu_written_back(addr, size, NULL, VK_NULL_HANDLE); }

int vk_mem_span_mirror_access_location(const VkMemSpan *span, RcRegistryBufferLocation *location) {
    if (!location) return 0;
    *location = (RcRegistryBufferLocation){0};
    if (!span || !span->pool || !span->buffer) return 0;
    plat_mutex_lock_shared(&pool_lock);
    const PoolRec *pool = span->pool;
    if (pool->kind == POOL_IMPORT && pool->mstate == MS_ON && !pool->unpublished &&
        pool->publication && pool->mirror_incarnation && span->buffer == pool->mbuf &&
        pool->import_hi > pool->import_lo && span->offset < pool->import_hi - pool->import_lo)
        *location = (RcRegistryBufferLocation){pool->publication, pool->mirror_incarnation, span->offset};
    plat_mutex_unlock_shared(&pool_lock);
    return location->allocation != 0;
}

int vk_mem_recorded_write_mirrored(const VkMemSpan *span) {
    const PoolRec *r = span ? span->pool : NULL;
    return r && r->mbuf && span->buffer == r->mbuf;
}

void *vk_mem_recorded_provider_acquire(const VkMemSpan *span) {
    PoolRec *pool = span ? span->pool : NULL;
    if (pool) pool_native_acquire(pool);
    return pool;
}

void vk_mem_recorded_provider_release(void *provider) {
    if (provider) destroy_rec(vk_ctx(), provider);
}

void vk_mem_buffer_write_completed(void *provider, VkBuffer buffer, uint64_t addr, uint64_t size, int mirror) {
    if (mirror) gpu_written_back(addr, size, provider, buffer);
    else gpu_written(addr, size, 0, provider);
}

int vk_mem_mirrored(uint64_t addr, uint64_t size) {
    VkMemSpan sp;
    if (!vk_mirror_on || !size || !vk_mem_lookup(addr, size, &sp) || !sp.pool) return 0;
    const PoolRec *r = sp.pool;
    return r->mstate == MS_ON && addr >= r->import_lo && addr + size <= r->import_hi;
}

// shadow_lock held, with the shadow line: the pages a shader stores into through the device address table ("shadow
// storage" above), and with NX2NAVI_SHADOW_CHECK (pool_lock held shared too, for the index) what the copy-ins did and
// cost and the pools whose stores were overwritten most since the last line.
#define TABLE_LOG_TOP 6
static void shadow_table_log(uint64_t frame) {
    rt_log("mem: shadow table stores at frame %llu (%s): %llu pages in %d pools keep a base copy; since the last: %llu whole "
           "copy-ins overwrote a store (%llu so far), %llu pages copied in with a mark; written back %.3f MiB of %.1f "
           "compared in %llu fence jobs, %.2f ms",
           (unsigned long long)frame, s_learn ? "NX2NAVI_SHADOW_TABLE_STORES=1: kept from the second copy-in on" : "overwritten",
           (unsigned long long)s_tab_pages, ntpools, (unsigned long long)s_tab_over, (unsigned long long)s_t_over_all,
           (unsigned long long)s_marked, s_tab_wb / 1048576.0, s_tab_wb_seen / 1048576.0, (unsigned long long)s_tab_wb_jobs,
           qpc_ms(s_tab_wb_ticks));
    if (s_check) {
        rt_log("mem: shadow check at frame %llu: copied in whole %.2f ms with marks, %.2f ms without; %.1f MiB by merge, "
               "%.1f MiB whole into pools without a mirror; %llu table syncs %.2f ms; bytes both sides changed: %llu taken "
               "from the CPU at a copy-in, %llu from the GPU at a fence; write-backs into %llu pages the CPU had written "
               "since their copy-in, %llu bytes",
               (unsigned long long)frame, qpc_ms(s_mark_ticks), qpc_ms(s_plain_ticks), s_in_merge / 1048576.0,
               s_in_nomirror / 1048576.0, (unsigned long long)s_tab_syncs, qpc_ms(s_tab_ticks),
               (unsigned long long)s_both_cpu, (unsigned long long)s_both_gpu, (unsigned long long)s_late_pages,
               (unsigned long long)s_late_bytes);
        PoolRec *top[TABLE_LOG_TOP];
        int nt = 0;
        for (int i = 0; i < pool_index.n; i++) {
            PoolRec *r = pool_index.e[i].buf;
            uint64_t d = r->kind == POOL_IMPORT ? r->t_over - r->t_over_log : 0;
            int k = nt;
            while (k > 0 && top[k - 1]->t_over - top[k - 1]->t_over_log < d) k--;
            if (!d || k >= TABLE_LOG_TOP) continue;
            int last = nt < TABLE_LOG_TOP ? nt : TABLE_LOG_TOP - 1;
            memmove(top + k + 1, top + k, (size_t)(last - k) * sizeof *top);
            top[k] = r;
            if (nt < TABLE_LOG_TOP) nt++;
        }
        for (int i = 0; i < nt; i++) {
            char f[160];
            flag_names(top[i]->flags, f, sizeof f);
            rt_log("mem: shadow check:   pool %#llx+%#llx (%s, %s): %llu whole copy-ins overwrote a store (%llu so far), "
                   "%u of its pages keep a base copy for it",
                   (unsigned long long)top[i]->lo, (unsigned long long)top[i]->size, f,
                   top[i]->mstate == MS_ON ? "mirrored" : "no mirror",
                   (unsigned long long)(top[i]->t_over - top[i]->t_over_log), (unsigned long long)top[i]->t_over, top[i]->ntpg);
        }
        // the pools that took the most whole pages, and what all pools of each flag set took
        struct { uint32_t flags; uint64_t bytes; int n; } cls[16];
        int ncls = 0;
        nt = 0;
        for (int i = 0; i < pool_index.n; i++) {
            PoolRec *r = pool_index.e[i].buf;
            if (r->kind != POOL_IMPORT || !r->s_in) continue;
            int c = 0, k = nt;
            while (c < ncls && cls[c].flags != r->flags) c++;
            if (c == ncls && ncls < 16) cls[ncls++] = (__typeof__(cls[0])){r->flags, 0, 0};
            if (c < ncls) cls[c].bytes += r->s_in, cls[c].n++;
            while (k > 0 && top[k - 1]->s_in < r->s_in) k--;
            if (k >= TABLE_LOG_TOP) continue;
            int last = nt < TABLE_LOG_TOP ? nt : TABLE_LOG_TOP - 1;
            memmove(top + k + 1, top + k, (size_t)(last - k) * sizeof *top);
            top[k] = r;
            if (nt < TABLE_LOG_TOP) nt++;
        }
        for (int i = 0; i < ncls; i++) {
            char f[160];
            flag_names(cls[i].flags, f, sizeof f);
            rt_log("mem: shadow check:   copied in whole into %d pools %s: %.2f MiB", cls[i].n, f, cls[i].bytes / 1048576.0);
        }
        for (int i = 0; i < nt; i++) {
            char f[160];
            flag_names(top[i]->flags, f, sizeof f);
            rt_log("mem: shadow check:   pool %#llx+%#llx (%s, %s, %s): %.2f MiB copied in whole",
                   (unsigned long long)top[i]->lo, (unsigned long long)top[i]->size, f,
                   top[i]->mstate == MS_ON ? "mirrored" : "no mirror", top[i]->used ? "resolved by the GPU" : "never resolved",
                   top[i]->s_in / 1048576.0);
        }
        for (int i = 0; i < pool_index.n; i++) {
            PoolRec *r = pool_index.e[i].buf;
            if (r->kind == POOL_IMPORT) r->t_over_log = r->t_over, r->s_in = 0;
        }
    }
    s_tab_over = s_tab_wb = s_tab_wb_seen = s_tab_wb_ticks = s_tab_wb_jobs = s_marked = s_mark_ticks = s_plain_ticks = 0;
    s_both_cpu = s_both_gpu = s_in_merge = s_in_nomirror = s_tab_ticks = s_tab_syncs = s_late_pages = s_late_bytes = 0;
}

static void shadow_log(uint64_t frame) {
    if (!shadow_on) return;
    if (s_check) plat_mutex_lock_shared(&pool_lock);   // before shadow_lock: shadow_table_log walks the index
    plat_mutex_lock(&shadow_lock);
    rt_log("mem: shadow at frame %llu: %lld pools %.1f MiB; since the last: %llu pool syncs, %llu queries, %.1f MiB "
           "copied in (%llu untracked), %.2f ms (queries %.2f); %.1f MiB written back of %.1f MiB the GPU may have written, %.2f ms; %llu "
           "pages the GPU writes keep a base copy; %llu pages not written back, their pool was made again or the CPU "
           "wrote them since",
           (unsigned long long)frame, (long long)s_live_n, s_live_bytes / 1048576.0, (unsigned long long)s_syncs,
           (unsigned long long)s_queries, s_copied / 1048576.0, (unsigned long long)s_untracked, qpc_ms(s_ticks),
           qpc_ms(s_q_ticks), s_wb_bytes / 1048576.0, s_wb_seen / 1048576.0, qpc_ms(s_wb_ticks), (unsigned long long)s_sb_pages,
           (unsigned long long)s_stale_pages);
    s_syncs = s_queries = s_copied = s_untracked = s_wb_bytes = s_ticks = s_wb_seen = s_wb_ticks = s_q_ticks = 0;
    s_stale_pages = 0;
    if (s_learn || s_check) shadow_table_log(frame);
    plat_mutex_unlock(&shadow_lock);
    if (s_check) plat_mutex_unlock_shared(&pool_lock);
}

void vk_mem_dirty_submit(void) {
    if (!(rt_trace & RT_TRACE_PERF) || vk_mirror_on) return;   // the mirror's own line replaces the census
    uint64_t t0 = kern_now_ns();
    uint32_t prev = dirty_sub;   // the submit whose replay noted the bindings
    dirty_sub = prev + 1 ? prev + 1 : 1;
    uint64_t eager[DC_N] = {0}, lazy[DC_N] = {0}, ln[DC_N] = {0}, lb[DC_N] = {0}, untracked = 0, queries = 0;
    plat_mutex_lock_shared(&pool_lock);
    if (dcap < pool_index.n) {
        dcap = pool_index.n * 2;
        dlist = realloc(dlist, (size_t)dcap * sizeof *dlist);
        if (!dlist) rt_fatal("vk: out of memory for the dirty census");
    }
    int n = 0;
    for (int i = 0; i < pool_index.n; i++) {
        PoolRec *r = pool_index.e[i].buf;
        if (r->kind == POOL_IMPORT && r->bufbound) dlist[n++] = r;
    }
    plat_mutex_lock(&gen_lock);
    uint32_t g = generation_next(), since = dirty_gen;
    dirty_gen = g;
    for (int i = 0; i < n;) {   // one query per run of pools (sorted by address)
        uint64_t lo = page_lo(dlist[i]), hi = page_hi(dlist[i]);
        int j = i + 1;
        for (; j < n && page_lo(dlist[j]) <= hi + DIRTY_GAP; j++)
            if (page_hi(dlist[j]) > hi) hi = page_hi(dlist[j]);
        queries++;
        int ok = query_stamp(lo, hi, g);
        for (int k = i; k < j; k++) {
            dlist[k]->untracked = 0;
            if (!ok) queries++, dlist[k]->untracked = (uint8_t)!query_stamp(page_lo(dlist[k]), page_hi(dlist[k]), g);
        }
        i = j;
    }
    uint64_t covered = 0;
    for (int i = 0; i < n; i++) {
        PoolRec *r = dlist[i];
        int k = dc_class(r);
        ln[k]++, lb[k] += r->size;
        int fresh = r->synced_gen == DIRTY_UNSYNCED, lz = !fresh && r->bound_sub == prev;
        uint32_t ls = r->synced_gen;
        if (fresh || r->bound_sub == prev) r->synced_gen = g;
        if (r->untracked) { untracked += r->size; continue; }
        if (fresh) continue;   // counted as first use
        uint64_t lo = page_lo(r), hi = page_hi(r), ne = 0, nl = 0;
        if (lo < covered) lo = covered;   // overlapping pools: each page once
        if (lo >= hi) continue;
        covered = hi;
        for (uint64_t a = lo; a < hi;) {
            uint64_t ci = a >> GEN_CHUNK_SHIFT, ce = (ci + 1) << GEN_CHUNK_SHIFT;
            if (ce > hi) ce = hi;
            const uint32_t *ch = ci < GEN_CHUNKS ? gen_chunk[ci] : NULL;
            if (ch)
                for (uint64_t x = a; x < ce; x += 1ull << GEN_PAGE_SHIFT) {
                    uint32_t v = ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)];
                    ne += v > since;
                    nl += lz && v > ls;
                }
            a = ce;
        }
        eager[k] += ne << GEN_PAGE_SHIFT, lazy[k] += nl << GEN_PAGE_SHIFT;
    }
    plat_mutex_unlock(&gen_lock);
    plat_mutex_unlock_shared(&pool_lock);
    uint64_t tot = 0;
    for (int k = 0; k < DC_N; k++) tot += eager[k];
    plat_mutex_lock(&dirty_lock);
    for (int k = 0; k < DC_N; k++) {
        dfr.eager[k] += eager[k], dfr.lazy[k] += lazy[k], dfr.first[k] += first_pending[k];
        first_pending[k] = 0;
        live_n[k] = ln[k], live_bytes[k] = lb[k], ever_snap_n[k] = ever_n[k], ever_snap_bytes[k] = ever_bytes[k];
    }
    live_untracked = untracked;
    dfr.submits++, dfr.queries += queries;
    if (tot > dfr.sub_max) dfr.sub_max = tot;
    dfr.scan_ns += kern_now_ns() - t0;
    plat_mutex_unlock(&dirty_lock);
}

static void dirty_frame(uint64_t frame) {
    plat_mutex_lock(&dirty_lock);
    if (dwin.frames < DIRTY_WIN)
        for (int k = 0; k < DC_N; k++) dwin_kib[dwin.frames][k] = (uint32_t)(dfr.eager[k] >> 10);
    for (int k = 0; k < DC_N; k++) dwin.eager[k] += dfr.eager[k], dwin.lazy[k] += dfr.lazy[k], dwin.first[k] += dfr.first[k];
    dwin.submits += dfr.submits, dwin.queries += dfr.queries, dwin.scan_ns += dfr.scan_ns, dwin.frames++;
    if (dfr.sub_max > dwin.sub_max) dwin.sub_max = dfr.sub_max;
    memset(&dfr, 0, sizeof dfr);
    if (frame % DIRTY_WIN) { plat_mutex_unlock(&dirty_lock); return; }
    DirtyAcc w = dwin;
    uint32_t kib[DIRTY_WIN][DC_N];
    memcpy(kib, dwin_kib, sizeof kib);
    uint64_t lv[DC_N], lvb[DC_N], ev[DC_N], evb[DC_N], unt = live_untracked;
    for (int k = 0; k < DC_N; k++) lv[k] = live_n[k], lvb[k] = live_bytes[k], ev[k] = ever_snap_n[k], evb[k] = ever_snap_bytes[k];
    memset(&dwin, 0, sizeof dwin);
    plat_mutex_unlock(&dirty_lock);
    if (!w.frames) return;
    uint64_t fl[2][3][2];
    nvn_flush_stats(fl);
    double f = (double)w.frames, M = 1048576.0;
    uint64_t sub_tot = 0;
    for (int k = 0; k < DC_N; k++) sub_tot += w.eager[k];
#define C3(a, d) (a)[0] / (d), (a)[1] / (d), (a)[2] / (d)
    rt_log("mem: dirty frame %llu: frames %llu submits_per_frame %.2f; per frame, cached/uncached/other: eager_mib "
           "%.3f/%.3f/%.3f lazy_mib %.3f/%.3f/%.3f first_mib %.3f/%.3f/%.3f flush_pool_calls %.1f/%.1f/%.1f "
           "flush_pool_mib %.3f/%.3f/%.3f flush_buf_calls %.1f/%.1f/%.1f flush_buf_mib %.3f/%.3f/%.3f; per submit: "
           "avg_mib %.3f max_mib %.3f; mirror: live_pools %llu/%llu/%llu live_mib %.1f/%.1f/%.1f ever_pools %llu/%llu/%llu "
           "ever_mib %.1f/%.1f/%.1f untracked_mib %.1f; cost: queries_per_frame %.1f scan_ms_per_frame %.3f",
           (unsigned long long)frame, (unsigned long long)w.frames, w.submits / f,
           C3(w.eager, f * M), C3(w.lazy, f * M), C3(w.first, f * M),
           fl[0][0][0] / f, fl[0][1][0] / f, fl[0][2][0] / f, fl[0][0][1] / f / M, fl[0][1][1] / f / M, fl[0][2][1] / f / M,
           fl[1][0][0] / f, fl[1][1][0] / f, fl[1][2][0] / f, fl[1][0][1] / f / M, fl[1][1][1] / f / M, fl[1][2][1] / f / M,
           w.submits ? sub_tot / (double)w.submits / M : 0.0, w.sub_max / M,
           (unsigned long long)lv[0], (unsigned long long)lv[1], (unsigned long long)lv[2], C3(lvb, M),
           (unsigned long long)ev[0], (unsigned long long)ev[1], (unsigned long long)ev[2], C3(evb, M), unt / M,
           w.queries / f, w.scan_ns / 1e6 / f);
#undef C3
    char line[DIRTY_WIN * 36 + 64];
    size_t k = (size_t)snprintf(line, sizeof line, "mem: dirty frames KiB cached/uncached/other:");
    for (uint64_t i = 0; i < w.frames && i < DIRTY_WIN; i++)
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, " %u/%u/%u", kib[i][0], kib[i][1], kib[i][2]);
    rt_log("%s", line);
}

// ---------------------------------------------------------------- VRAM mirror of buffer-bound pools
// An imported pool keeps vertices, indices, constants
// and storage buffers in system memory, so a discrete GPU reads them over PCIe on every draw (GTX 1050 Ti and RTX
// 3090 Ti: about 110 ms of GPU time per frame in the field). In mirror mode (device class discrete, NX2NAVI_MEMORY) an
// imported pool gets a device-local buffer over its imported range the first time the replay uses it as a vertex,
// index, uniform, storage or indirect buffer (vk_mem_use). From then on those bindings, and the buffer copies, fills
// and uniform updates that touch the pool (vk_mem_pick), use the mirror at the same offsets. Guest memory stays what
// the CPU reads and writes; textures and every host path that reads or writes pool bytes keep using it.
// - CPU -> mirror: the first access of a submission to a mirror syncs it (mirror_sync): the 2 MiB write-watch
//   snapshots of its regions for this epoch (snap_regions, shared with the texture checks: the OS bits keep one
//   consumer, the shared page generations) and every page the CPU-side tables (cgen_chunk) stamped after the mirror's
//   mgen goes into a copy. Where a region of the pool is not write-watched, so do the nvnMemoryPoolFlushMappedRange
//   ranges, each on its own: one span from the lowest to the highest flush of a 252 MiB pool copied up to the whole
//   pool for a few MiB flushed, 30-50 ms of GPU time over PCIe 3.0 x8 in crowded towns (NX2NAVI_FLUSH_SPAN=1 restores
//   it). Where the write watch covers the pool, a flushed page the CPU wrote is a written page already, and the game's
//   flushes of whole pools it had not written cost 47-57 MiB of copies a frame in Kokiri Forest (2026-11-09);
//   NX2NAVI_FLUSH_UPLOAD=1 copies them anyway. The first use copies the whole range. The copies run in a prologue command
//   buffer that vk_sync.c submits right before the part that accessed the mirror first, between full barriers.
//   vk_mem_mirror_begin snaps the regions of the pools the previous submission accessed in a few merged queries.
// - GPU -> guest: what the replay writes into a mirror (storage buffers its shaders write, buffer copies, fills,
//   uniform updates, texture-to-buffer copies: vk_mem_gpu_wrote) goes back into the imported buffer at the end of the
//   submission, exact byte ranges, before the fence signals (vk_mem_mirror_end). At the fence those pages get a new
//   generation for the texture checks (vk_gpu_written_back) but not in the CPU-side tables: the mirror holds the bytes.
// - NX2NAVI_MEMORY=verify: every submission also copies up to VERIFY_BYTES of mirror pages, round robin, into host memory;
//   at its fence the pages the CPU has not written since their sync are compared with guest memory.
// A pool stays imported when its mirror would not fit the VRAM budget, its range is not write-watched, it overlaps
// another pool or the allocation fails.
#define VERIFY_BYTES (4u << 20)
#define VERIFY_SAMPLES 64
#define VERIFY_LOG 200
enum { MR_BUDGET, MR_UNTRACKED, MR_OVERLAP, MR_ALLOC, MR_GPA_PROJECTION, MR_N };
static _Atomic(PoolRec *) mirror_pressure_first;
static PoolRec *mirror_pressure_last;

int vk_mirror_on;
static int mirror_verify;
static int mirror_gap = 2;             // pre-snap: hot pools at most this many 2 MiB regions apart share one query
static uint32_t pframe;                // replay: frames begun (vk_mem_mirror_begin: a place not after the last one's)
static int pplace = 99;                // replay: the place of the last submission vk_mem_mirror_begin saw

// NX2NAVI_PRESNAP_PLACE=1 (default), 0, ab: a submission's pre-snap (vk_mem_mirror_begin) takes the write-watch
// snapshots of the pools the submission at its place in the last frame synced, not those the submission before it
// synced. In Kokiri Forest the frame's first submission (#0) and its fourth (#3) use no pool, and their pre-snaps
// covered the pools of the submission before them, #5's and #2's: 36.8 queries over 282 MiB a frame, 0.23-0.31 ms; then
// #2 and #5, whose submissions before them use no pool either, queried their pools one at a time at their first use,
// 63.5 queries over 254 MiB.
static int presnap_place(void) {
    static int mode = -1;
    if (mode < 0) mode = rt_knob("PRESNAP_PLACE", "1");
    return RT_KNOB_ON(mode);
}

static int sub_place(void) { return vk_sub_in_frame >= 0 && vk_sub_in_frame < 7 ? vk_sub_in_frame : 7; }

static void pm_note(PoolRec *r) {   // replay: r synced in the submission being recorded
    if (r->pm_frame != pframe) {
        r->pm_prev = r->pm_frame + 1 == pframe ? r->pm_cur : 0;
        r->pm_cur = 0, r->pm_frame = pframe;
    }
    r->pm_cur |= (uint8_t)(1u << sub_place());
}

static uint8_t pm_last(const PoolRec *r) {   // the places that synced r in the last frame
    return r->pm_frame == pframe ? r->pm_prev : r->pm_frame + 1 == pframe ? r->pm_cur : 0;
}
static PoolRec **mlist;                // pool_lock: mirrored pools by import_lo
static int nm, mcap;
static uint32_t msub;                  // replay: number of the submission being recorded
static uint32_t mepoch;                // replay: its write-watch snapshot epoch (vk_texture_epoch)
typedef struct { PoolRec *r; uint64_t off, size; } MRange;
static MRange *pend, *wbl;             // replay: copies for the next prologue; GPU writes into mirrors this submission
static int npend, cpend, nwbl, cwbl;
static VkBufferCopy *mregions;         // replay: vkCmdCopyBuffer regions
static int cregions;
static PlatMutex flush_lock = PLAT_MUTEX_INIT;
static int flush_cap = FLUSH_RANGES;   // ranges a pool keeps; 1 with NX2NAVI_FLUSH_SPAN=1
static int flush_upload;               // NX2NAVI_FLUSH_UPLOAD=1: flush ranges are copied where the write watch sees them too

typedef struct {
    uint64_t subs, syncs, upload_bytes, upload_copies, first_n, first_bytes, wb_bytes, wb_copies, wb_ranges;
    uint64_t presnap_queries, sync_queries, presnap_t, sync_t, prologues, presnap_bytes;
    uint64_t snap_bytes, merge_bytes, merge_dispatches, snap_allocs;   // write-back merge
    uint64_t page_bytes, flush_n, flush_bytes, flush_calls, flush_sum;   // what mirror_sync queues, by source
    uint64_t flush_skipped;   // flush ranges not copied: the write watch covers them
    uint64_t cap_bytes, cap_ranges, cap_left, cap_t;   // mirror_capture: copied by the CPU, left to the GPU copy, its time
} MirrorAcc;
static MirrorAcc mrep;                 // replay: this submission
static MirrorAcc macc;                 // mirror_stat_lock: since the last summary
static PlatMutex mirror_stat_lock = PLAT_MUTEX_INIT;
static volatile int64_t m_live_n, m_live_bytes, m_refused_n[MR_N], m_refused_bytes[MR_N];
static volatile int64_t m_flush_calls, m_flush_bytes, v_pages, v_skipped, v_mismatch;

static void mirror_mode(int discrete, int nvidia_own) {
    const char *e = cfg_value("NX2NAVI_MEMORY");
    int mode = discrete ? 1 : 0;   // 0 import, 1 mirror, 2 verify
    const char *why = discrete ? "auto, discrete gpu" : "auto, uma";
    if (strcmp(e, "auto")) {
        if (!strcmp(e, "import")) mode = 0, why = "NX2NAVI_MEMORY=import";
        else if (!strcmp(e, "mirror")) mode = 1, why = "NX2NAVI_MEMORY=mirror";
        else if (!strcmp(e, "verify")) mode = 2, why = "NX2NAVI_MEMORY=verify";
        else rt_log("mem: NX2NAVI_MEMORY=%s is not import, mirror, verify or auto: auto", e);
    }
    vk_mirror_on = mode > 0;
    mirror_verify = mode == 2;
    const char *g = cfg_get("NX2NAVI_MIRROR_GAP");
    mirror_gap = g && *g ? atoi(g) : 2;
    if (mirror_gap < 0) mirror_gap = 0;
    if (rt_knob("FLUSH_SPAN", "0")) {
        flush_cap = 1;
        rt_log("mem: a pool's flushes copy one span from the lowest to the highest (NX2NAVI_FLUSH_SPAN=1)");
    }
    flush_upload = rt_knob("FLUSH_UPLOAD", "0") == 1;
    if (flush_upload) rt_log("mem: flushed ranges are copied into the mirrors as well as written pages (NX2NAVI_FLUSH_UPLOAD=1)");
    // Submissions a region stays unwritten before it is frozen (NX2NAVI_FREEZE_IDLE; 0: no freezing). 60 was chosen
    // when a thaw by a write cost about 60 us: in a fight 100 of the 257 MiB queried a frame lay in regions idle for
    // 30-299 submissions. While the game streams an area, though, it creates and finalizes up to 1,500 memory pools a
    // second, the driver pins and unpins the imported ones, and every VirtualQuery and VirtualProtect in the process
    // waits behind that: a thaw took 10-20 ms with frz_lock held, which the replay needs too. The streaming buffers,
    // idle for a few frames at a time, were frozen and thawed about five times a second, and walking from Kakariko
    // Village through the Bottom of the Well to the Graveyard had a frame of 25-90 ms in 18-24% of its seconds (11%
    // with freezing off, at 3-3.6 ms a frame more of write-watch queries over 3.5 GiB). At 1800 (about 3 s) a walk
    // through Hyrule Field and Ganon's Castle froze 0.1 regions a second, thawed none, queried 14 MiB in 0.03 ms a
    // frame, and had such a frame in 1 of 253 seconds.
    const char *fi = cfg_get("NX2NAVI_FREEZE_IDLE");
    freeze_idle = fi && *fi ? atoi(fi) : 1800;
    if (freeze_idle < 0) freeze_idle = 0;
#ifdef _WIN32
    const char *pair = cfg_get("NX2NAVI_TEXTURE_SNAPSHOT_PAIR");
    snapshot_pair_enabled = !pair || strcmp(pair, "0");
    const char *priority = cfg_get("NX2NAVI_FREEZE_CLEAN_PRIORITY");
    if (freeze_idle && priority && !strcmp(priority, "1") && !clean_attempt) {
        clean_attempt = calloc(SNAP_REGIONS, sizeof *clean_attempt);
        if (!clean_attempt) rt_log("mem: clean-scan freeze selection unavailable: no attempt history");
    }
#endif
    const char *sh = cfg_get("NX2NAVI_SHADOW");
#ifdef _WIN32
    shadow_on = sh && *sh ? strcmp(sh, "0") != 0 : plat_is_wine();
#else
    // NVIDIA's own driver imports guest memory without the cost shadow storage avoids on amdgpu (re-protecting imported
    // pages), so there the copies are only the slower path: in Kokiri Forest on 0.7.1 (RTX 4090, driver 615.78) imports
    // with VRAM mirrors ran at 51.9 frames a second and the copies at 27.4, and the copies hold about 5 GB. What a
    // shader stores through the device address table into a pool with a shadow copy also does not come back to guest
    // memory (vk_gpa.c). The pools stay imports unless NX2NAVI_SHADOW asks for the copies. The open driver for the same
    // cards keeps the copies: it is not measured. The AppImage's play.sh has set the same for this driver since 0.6.0.
    shadow_on = sh && *sh ? strcmp(sh, "0") != 0 : !nvidia_own;
#endif
    if (shadow_on) rt_log("mem: shadow storage: CPU-visible pools are host-visible copies of guest memory, not imports%s",
                          sh && *sh ? " (NX2NAVI_SHADOW)" : "");
    s_learn = shadow_on && rt_knob("SHADOW_TABLE_STORES", "0") == 1;
    s_check = shadow_on && cfg_bool("NX2NAVI_SHADOW_CHECK");
    if (shadow_on)
        rt_log("mem: what shaders store through the device address table into a shadow copy %s "
               "(NX2NAVI_SHADOW_TABLE_STORES)%s",
               s_learn ? "is kept through the copy-ins and written back to guest memory, from the second copy-in of its page on"
                       : "is overwritten by the next whole copy-in of its page",
               s_check ? "; NX2NAVI_SHADOW_CHECK counts the overwrites" : "");
    rt_log("mem: mode %s (%s): %s", mode == 2 ? "verify" : mode ? "mirror" : "import", why,
           mode ? "pools used as buffers get a device-local mirror" : "every CPU-visible pool is imported host memory");
    // Import is the integrated-GPU mode. On a discrete card every draw reads the pools over PCIe (the comment above
    // this section has the measurements), which is the difference between playable and 5 fps when it is not intended.
    if (!mode && discrete)
        rt_log("mem: warning: import on a discrete GPU: the GPU reads the game's pools over PCIe on every draw, far "
               "slower than mirror (the auto default on a discrete GPU); use memory = auto unless testing");
}

static void range_add(MRange **a, int *n, int *cap, PoolRec *r, uint64_t off, uint64_t size) {
    if (*n && (*a)[*n - 1].r == r && (*a)[*n - 1].off + (*a)[*n - 1].size == off) {
        (*a)[*n - 1].size += size;
        return;
    }
    if (*n == *cap) grow((void **)a, cap, sizeof **a);
    (*a)[(*n)++] = (MRange){r, off, size};
}

static int range_cmp(const void *x, const void *y) {
    const MRange *a = x, *b = y;
    if (a->r != b->r) return (uintptr_t)a->r < (uintptr_t)b->r ? -1 : 1;
    return a->off < b->off ? -1 : a->off > b->off;
}

static void mirror_queue_upload(PoolRec *r, uint64_t offset, uint64_t size, uint64_t scene) {
    r->mirror_origin_queued = scene ? scene : UINT64_MAX;
    range_add(&pend, &npend, &cpend, r, offset, size);
}

static void mirror_record_upload(PoolRec *r, const VkBufferCopy *regions, int count, uint64_t scene, int final) {
    // The renderer records these copies before it submits, so the scene is a receipt to wait on,
    // not a completion. Both the normal import and already captured staging take this route.
    if (count <= 0 || !scene) return;
    r->mirror_origin_last_copy = scene;
    if (count == 1 && !regions[0].dstOffset && regions[0].size == r->import_hi - r->import_lo &&
        !r->mirror_origin_full_copy)
        r->mirror_origin_full_copy = scene;
    if (final) {
        r->mirror_origin_full_queued = 0;
        r->mirror_origin_queued = 0;
    }
}

static int mirror_origin_initialized(const PoolRec *r, uint64_t scene, uint64_t completed,
                                     uint64_t offset, uint64_t size) {
    const uint64_t bytes = r->import_hi - r->import_lo;
    const uint64_t full = r->mirror_origin_full_copy, latest = r->mirror_origin_last_copy;
    if (!scene || r->mirror_origin_checked_scene != scene || !size || offset >= bytes || size > bytes - offset)
        return 0;
    // The complete first copy is already owned by this scene's ordinary mirror prologue. It is
    // executed before any body command, including a sparse writer, or both remain unhanded.
    const uint64_t queued = r->mirror_origin_queued;
    if (queued && queued != scene) return 0;
    if (r->mirror_origin_full_queued == scene) return 1;
    // Same-scene recorded prologues precede the part, and any queued refresh here is owned by
    // that part. Other scenes must have finished the full copy and every recorded refresh.
    return full && full <= latest && (latest == scene || (latest < scene && latest <= completed));
}

// Only the actual writeback spans become current. Descriptor extents can include lazy or deliberately skipped tails.
typedef struct {
    RcSourceRange range;
    VkMemSpanCapture *capture;
    VkMemSpan descriptor;
} CpuVisibilityRange;
typedef struct {
    RcRegistryVisibility proof;
    CpuVisibilityRange *ranges;
    uint32_t count, capacity;
    int incomplete;
} CpuVisibility;

static int cpu_visibility_reserve(CpuVisibility *job, uint32_t add) {
    if (add > UINT32_MAX - job->count) return 0;
    const uint32_t needed = job->count + add;
    if (needed <= job->capacity) return 1;
    uint32_t capacity = job->capacity ? job->capacity : 16;
    while (capacity < needed) {
        if (capacity > UINT32_MAX / 2) { capacity = needed; break; }
        capacity *= 2;
    }
    CpuVisibilityRange *ranges = realloc(job->ranges, (size_t)capacity * sizeof *ranges);
    if (!ranges) return 0;
    job->ranges = ranges;
    job->capacity = capacity;
    return 1;
}

static void cpu_visibility_add_guarded(CpuVisibility *job, const PoolRec *pool, uint64_t off, uint64_t size,
                                       VkMemSpanCapture *capture, const VkMemSpan *descriptor) {
    if (!job || job->incomplete || !job->proof.horizon || !pool || pool->kind != POOL_IMPORT || pool->smap ||
        pool->import_hi <= pool->import_lo || nvn_gpu_only(pool->import_lo) || (!!capture != !!descriptor)) return;
    uint64_t limit = pool->import_hi - pool->import_lo;
    if (off >= limit || !size) return;
    if (size > limit - off) size = limit - off;
    uint64_t addr = pool->import_lo + off;
    RcInterval *intervals;
    uint32_t count = copy_pool_host_intervals(pool, addr, size, &intervals);
    if (count == UINT32_MAX || !cpu_visibility_reserve(job, count + 1)) {
        free(intervals);
        job->incomplete = 1;
        return;
    }
    const VkMemSpan selected = descriptor ? *descriptor : (VkMemSpan){0};
    VkMemSpanCapture *held = capture ? vk_mem_span_capture_retain(capture) : NULL;
    if (capture && !held) {
        free(intervals);
        job->incomplete = 1;
        return;
    }
    job->ranges[job->count++] = (CpuVisibilityRange){
        .range = {1, addr, size}, .capture = held, .descriptor = selected};
    for (uint32_t i = 0; i < count; ++i) {
        VkMemSpanCapture *part = capture ? vk_mem_span_capture_retain(capture) : NULL;
        if (capture && !part) {
            free(intervals);
            job->incomplete = 1;
            return;
        }
        job->ranges[job->count++] = (CpuVisibilityRange){.range = intervals[i].range,
            .capture = part, .descriptor = selected};
    }
    free(intervals);
}

// The bitmap already owns an exact retained canonical source interval. Pair that interval with its guest
// compatibility address directly, so guarded visibility needs no late host-interval allocation. If the
// range journal cannot represent the pair, omit this scene's visibility proof and retain foreign authority.
static void cpu_visibility_add_guarded_pair(CpuVisibility *job, RcSourceRange guest, RcSourceRange canonical,
                                            VkMemSpanCapture *capture, const VkMemSpan *descriptor) {
    if (!job || job->incomplete || !job->proof.horizon || !capture || !descriptor ||
        guest.backing != RC_GUEST_COMPATIBILITY_BACKING || canonical.backing <= RC_GUEST_COMPATIBILITY_BACKING ||
        !rc_range_valid(guest) || !rc_range_valid(canonical) || guest.size != canonical.size) {
        if (job) job->incomplete = 1;
        return;
    }
    if (!cpu_visibility_reserve(job, 2)) {
        job->incomplete = 1;
        return;
    }
    VkMemSpanCapture *guest_hold = vk_mem_span_capture_retain(capture);
    VkMemSpanCapture *canonical_hold = vk_mem_span_capture_retain(capture);
    if (!guest_hold || !canonical_hold) {
        vk_mem_span_capture_release(&guest_hold);
        vk_mem_span_capture_release(&canonical_hold);
        job->incomplete = 1;
        return;
    }
    const VkMemSpan selected = *descriptor;
    job->ranges[job->count++] = (CpuVisibilityRange){.range = guest, .capture = guest_hold, .descriptor = selected};
    job->ranges[job->count++] = (CpuVisibilityRange){.range = canonical, .capture = canonical_hold, .descriptor = selected};
}

static void cpu_visibility_add(CpuVisibility *job, const PoolRec *pool, uint64_t off, uint64_t size) {
    cpu_visibility_add_guarded(job, pool, off, size, NULL, NULL);
}

static void cpu_visibility_ranges_release(CpuVisibility *job) {
    for (uint32_t i = 0; i < job->count; ++i)
        vk_mem_span_capture_release(&job->ranges[i].capture);
}
static void cpu_visibility_discard(CpuVisibility *job) {
    cpu_visibility_ranges_release(job);
    free(job->ranges);
    job->ranges = NULL;
    job->count = job->capacity = 0;
}

static void cpu_visibility_done(void *arg) {
    CpuVisibility *job = arg;
    for (uint32_t i = 0; i < job->count; i++) {
        CpuVisibilityRange *range = &job->ranges[i];
        if (!range->capture || vk_mem_span_capture_current(range->capture, &range->descriptor))
            rc_registry_visibility_apply(rc_registry_runtime(), &job->proof, range->range);
    }
    cpu_visibility_ranges_release(job);
    free(job->ranges);
    free(job);
}
static void cpu_visibility_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;
    CpuVisibility *job = arg;
    cpu_visibility_ranges_release(job);
    free(job->ranges);
    free(job);
}
static void cpu_visibility_queue(CpuVisibility *job, uint64_t fence) {
    if (job->incomplete) { cpu_visibility_discard(job); return; }
    if (!job->count) { free(job->ranges); job->ranges = NULL; job->capacity = 0; return; }
    CpuVisibility *owned = malloc(sizeof *owned);
    if (!owned) { cpu_visibility_discard(job); return; }
    *owned = *job;
    vk_completion_add_owned(fence, cpu_visibility_done, cpu_visibility_cancel, owned);
}
// Records the ranges as one vkCmdCopyBuffer per pool, overlapping and adjacent ranges merged; to_mirror: buf -> mbuf,
// else mbuf -> buf. Returns the commands recorded; adds what they move to *bytes and *copies.
static int copy_ranges(VkCommandBuffer cb, MRange *a, int n, int to_mirror, uint64_t *bytes, uint64_t *copies,
                       CpuVisibility *visibility, uint64_t upload_scene) {
    qsort(a, (size_t)n, sizeof *a, range_cmp);
    int rec = 0;
    for (int i = 0; i < n;) {
        PoolRec *r = a[i].r;
        uint64_t lim = r->import_hi - r->import_lo;
        int k = 0;
        for (; i < n && a[i].r == r; i++) {
            uint64_t lo = a[i].off, hi = a[i].off + a[i].size;
            if (hi > lim) hi = lim;
            if (lo >= hi) continue;
            if (k && lo <= mregions[k - 1].srcOffset + mregions[k - 1].size) {
                uint64_t end = mregions[k - 1].srcOffset + mregions[k - 1].size;
                if (hi > end) mregions[k - 1].size += hi - end;
                continue;
            }
            if (k == cregions) grow((void **)&mregions, &cregions, sizeof *mregions);
            mregions[k++] = (VkBufferCopy){lo, lo, hi - lo};
        }
        if (!k) continue;
        if (to_mirror) {
            vkCmdCopyBuffer(cb, r->buf, r->mbuf, (uint32_t)k, mregions);
            mirror_record_upload(r, mregions, k, upload_scene, 1);
        } else vkCmdCopyBuffer(cb, r->mbuf, r->buf, (uint32_t)k, mregions);
        rec++;
        for (int j = 0; j < k; j++) {
            *bytes += mregions[j].size;
            if (!to_mirror) cpu_visibility_add(visibility, r, mregions[j].dstOffset, mregions[j].size);
        }
        *copies += (uint64_t)k;
    }
    return rec;
}

static void mbarrier(VkCommandBuffer cb, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                     VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                           .dstStageMask = ds, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static int refuse(PoolRec *r, int why, const char *what) {
    r->refused_why = (uint8_t)why;
    uint64_t n = r->import_hi - r->import_lo;
    static volatile int32_t logged[MR_N];   // four lines per reason: the refusals asked again would repeat theirs
    plat_inc64(&m_refused_n[why]);
    if (plat_inc32(&logged[why]) <= 4) {
        char f[160];
        flag_names(r->flags, f, sizeof f);
        rt_log("mem: pool %#llx+%#llx (%s) stays imported: %s", (unsigned long long)r->lo, (unsigned long long)r->size,
               f, what);
    }
    plat_add64(&m_refused_bytes[why], (int64_t)n);
    return MS_REFUSED;
}

// ---- mirror reclamation
// Shared admission decides whether a mirror may grow memory. A refusal keeps the imported pool usable and may
// ask idle twins and textures to retire for a later retry. A mirror without a twin or with retained GPU-only
// writes is not a victim. Deferred destruction remains charged until its fence and actual allocator release;
// selecting a victim never creates speculative room for the replacement.
#define MIRROR_IDLE 120
static int mirror_evict_on = -1;
static int64_t m_evicted_n, m_evicted_bytes, m_retries;

static int evict_enabled(void) {
    if (mirror_evict_on < 0) {
        mirror_evict_on = rt_knob("MIRROR_EVICT", "1");
        rt_log("mem: idle mirrors %s (NX2NAVI_MIRROR_EVICT=%d)", mirror_evict_on ? "make room for new ones and for textures under VRAM pressure"
                                                                               : "stay; a pool over the VRAM budget stays imported", mirror_evict_on);
    }
    return mirror_evict_on;
}

// Replay thread: evicts mirrors idle for at least min_idle submissions, least recently synced first, until need bytes
// are freed; none when the candidates cannot free that much. Returns the bytes freed.
static uint64_t mirror_evict(uint32_t domain, uint64_t need, const char *why) {
    enum { MAXV = 64 };
    PoolRec *v[MAXV];
    int nv = 0;
    plat_mutex_lock(&pool_lock);
    for (int i = 0; i < nm; i++) {   // the MAXV least recently synced candidates, oldest first
        PoolRec *r = mlist[i];
        uint32_t age = msub - r->acc_sub;
        if (age < MIRROR_IDLE || r->wb_sub == msub || msub - r->wb_sub < 2) continue;
        if (__atomic_load_n(&r->mirror_input_refs, __ATOMIC_ACQUIRE)) continue;
        if (!r->twin && r->gpa_rec) continue;   // the table's writes are in the mirror alone
        if (r->wb_kept) continue;               // so are the bytes wb_skip kept from guest memory
        RcAdmissionVector cost;
        if (resource_core_vma_allocation_cost(r->malloc_, &cost) != RC_ADMISSION_OK || !cost.amount[domain]) continue;
        int at = nv;
        while (at > 0 && msub - v[at - 1]->acc_sub < age) at--;
        if (at == MAXV) continue;
        if (nv < MAXV) nv++;
        memmove(v + at + 1, v + at, sizeof *v * (size_t)(nv - 1 - at));
        v[at] = r;
    }
    uint64_t can = 0;   // what all the candidates would free: short of need, none goes (no mirror lost for nothing)
    for (int i = 0; i < nv; i++) can += v[i]->import_hi - v[i]->import_lo;
    if (can < need) nv = 0;
    uint64_t freed = 0;
    uint32_t idle_for[MAXV];
    VkBuffer dead[MAXV];
    VmaAllocation dead_a[MAXV];
    int n = 0;
    // All under the exclusive lock: vk_pool_fini, on another thread, takes a pool off the list under it, so a victim is
    // still a live pool here, and a finalize that follows finds it imported, its table pages on the imported buffer.
    for (int i = 0; i < nv && freed < need; i++) {
        PoolRec *r = v[i];
        mem_lazy_release(r->import_lo, r->import_hi - r->import_lo);   // make the import current before GPA can read it
        if (r->gpa_rec) {
            uint32_t projected = vk_gpa_remap_exact(r->import_lo, r->import_hi, r->gpa_rec, r->buf,
                                                    r->import_lo, 0, r->gpa_host_coherent, NULL);
            if (!projected) continue;
            r->gpa_rec = projected;
        }
        idle_for[n] = msub - r->acc_sub;
        mirror_forget(r);   // off the list, mirror_gen bumped: no buffer use gets the mirror from here on
        r->mstate = MS_NONE, r->acc_sub = 0;
        dead[n] = r->mbuf, dead_a[n] = r->malloc_;
        r->mbuf = VK_NULL_HANDLE, r->malloc_ = NULL;
        plat_mutex_lock(&flush_lock);   // flushes not copied yet: the next mirror starts from a whole copy
        __atomic_store_n(&r->fl_n, 0, __ATOMIC_RELEASE);
        r->fl_calls = r->fl_sum = 0;
        plat_mutex_unlock(&flush_lock);
        freed += r->import_hi - r->import_lo;
        if (m_evicted_n + n < 8)
            rt_log("mem: mirror of pool %#llx+%#llx (%.1f MiB) evicted, idle for %u submissions: %s", (unsigned long long)r->lo,
                   (unsigned long long)r->size, (r->import_hi - r->import_lo) / 1048576.0, idle_for[n], why);
        n++;
    }
    plat_mutex_unlock(&pool_lock);
    for (int i = 0; i < n; i++) vk_defer_buffer(dead[i], dead_a[i]);   // after the submission being recorded
    if (n) {
        plat_add64(&m_evicted_n, n);
        plat_add64(&m_evicted_bytes, (int64_t)freed);
    }
    return freed;
}

// ---- twins (NX2NAVI_GPA_TWIN_MAX_KB, default 0: every mirror has one)
// Shaders write through the device address table into both a mirror and its pool's imported buffer, guest memory
// (vk_gpa.c, "The twin"): such writes are not recorded as GPU writes, so with the mirror alone the CPU never saw
// them and a sync of a page the CPU wrote copied the stale guest page over them. On a discrete GPU the second write
// crosses PCIe: on PCIe 3.0 the twins cost about 3 ms of GPU time a frame in Kokiri Forest (NX2NAVI_GPA_TWIN=0 against
// on). The writes that went wrong without them were the foliage counts and records in pools of 20 and 44 KiB.
// GPA_TWIN_MAX_KB=n gives a twin only to mirrors of pools up to n KiB; a larger mirror keeps the table's writes to
// itself (its record names the mirror twice, which sends the second write to the table's scratch page) and its
// buffer fills stay in the mirror (vk_mem_guest_twin). Such a mirror is never evicted (mirror_evict), since guest
// memory lacks what the shaders wrote into it. For measuring which pools need a twin: a pool that does and is left
// without draws stretched or flickering foliage.
static int64_t twin_max = -1;          // bytes; 0 no limit
static int64_t m_untwinned_n, m_untwinned_bytes;

static int twin_wanted(const PoolRec *r) {
    if (twin_max < 0) {
        int kb = cfg_int("NX2NAVI_GPA_TWIN_MAX_KB");
        twin_max = kb > 0 ? (int64_t)kb << 10 : 0;
        if (twin_max)
            rt_log("mem: only mirrors of pools up to %d KiB send the device address table's writes to guest memory "
                   "too (NX2NAVI_GPA_TWIN_MAX_KB); larger ones keep them and are not evicted", kb);
    }
    return !twin_max || (int64_t)(r->import_hi - r->import_lo) <= twin_max;
}

static int mirror_pool_entry(const PoolRec *r) { // pool_lock held
    int i = pidx_upper(&pool_index, r->lo) - 1;
    while (i >= 0 && pool_index.e[i].buf != r) i--;
    return i;
}

static int mirror_pool_alone(int i) { // pool_lock held
    return i >= 0 && (i == 0 || pool_index.maxhi[i - 1] <= pool_index.e[i].lo) &&
           (i + 1 == pool_index.n || pool_index.e[i + 1].lo >= pool_index.e[i].hi);
}

static void mirror_pressure_remove(PoolRec *r) {
    if (r->mirror_pressure_prev) r->mirror_pressure_prev->mirror_pressure_next = r->mirror_pressure_next;
    else if (mirror_pressure_first == r) mirror_pressure_first = r->mirror_pressure_next;
    else return;
    if (r->mirror_pressure_next) r->mirror_pressure_next->mirror_pressure_prev = r->mirror_pressure_prev;
    else mirror_pressure_last = r->mirror_pressure_prev;
    r->mirror_pressure_prev = r->mirror_pressure_next = NULL;
}

static void mirror_pressure_add(PoolRec *r) { // pool_lock held, after publishing Pending
    if (!evict_enabled() || r->mirror_pressure_prev || mirror_pressure_first == r) return;
    RcAdmissionVector *peak = &r->mirror_dependency.peak;
    int gpu = 0;
    for (uint32_t d = RC_BUDGET_GPU_FIRST; d < RC_ADMISSION_MAX_DOMAINS; d++) gpu |= peak->amount[d] != 0;
    if (!gpu) return;
    r->mirror_pressure_prev = mirror_pressure_last;
    if (mirror_pressure_last) mirror_pressure_last->mirror_pressure_next = r;
    else mirror_pressure_first = r;
    mirror_pressure_last = r;
}

static _Thread_local PoolRec *mirror_pressure_demand; // retained by the thread asking the shared service

static unsigned mirror_reclaim(void *context, const RcAdmissionVector *deficit) {
    (void)context;
    PoolRec *demand = mirror_pressure_demand;
    if (!demand) return RC_BUDGET_RECLAIM_NONE;
    if (!vk_sync_try_provider_idle()) return RC_BUDGET_RECLAIM_BUSY;
    plat_mutex_lock_shared(&pool_lock);
    int live = !demand->unpublished && demand->mstate == MS_PENDING && !vk_sync_stopping();
    plat_mutex_unlock_shared(&pool_lock);
    unsigned result = RC_BUDGET_RECLAIM_NONE;
    if (live) {
        RcBudgetSnapshot budget;
        resource_core_budget_snapshot(&budget);
        for (uint32_t d = RC_BUDGET_GPU_FIRST; d < RC_ADMISSION_MAX_DOMAINS; d++) {
            uint64_t pending = budget.pending.amount[d];
            if (deficit->amount[d] <= pending) continue;
            uint64_t need = deficit->amount[d] - pending;
            // A freed compatible span can satisfy the native allocation without a new allocator block.
            uint64_t span = demand->import_hi - demand->import_lo;
            if (need > span) need = span;
            uint64_t selected = mirror_evict(d, need, "shared preparation pressure");
            if (selected) result |= RC_BUDGET_RECLAIM_PENDING;
        }
    }
    vk_sync_end_provider_idle();
    return result;
}

static void mirror_pressure_register(void *unused) {
    (void)unused;
    uint32_t gpu = ((1u << RC_BUDGET_GPU_HEAPS) - 1u) << RC_BUDGET_GPU_FIRST;
    if (resource_core_budget_register_reclaimer(RC_BUDGET_BUFFER, gpu, mirror_reclaim, NULL) != RC_ADMISSION_OK)
        rt_fatal("vk: could not register mirror preparation giveback");
}

int vk_mem_mirror_pressure_waiting(void) { return mirror_pressure_first != NULL; }

void vk_mem_mirror_service_pressure(void) {
    if (!vk_mirror_on || vk_sync_stopping() || vk_device_is_lost()) return;
    for (int i = 0; i < 4; i++) {
        plat_mutex_lock(&pool_lock);
        PoolRec *r = mirror_pressure_first;
        if (!r) { plat_mutex_unlock(&pool_lock); return; }
        mirror_pressure_remove(r);
        pool_native_acquire(r);
        RcAdmissionVector peak = r->mirror_dependency.peak;
        plat_mutex_unlock(&pool_lock);
        static PlatOnce registered = PLAT_ONCE_INIT;
        plat_once(&registered, mirror_pressure_register, NULL);
        RcBudgetProgress progress;
        mirror_pressure_demand = r;
        resource_core_budget_service(0, &peak, &progress);
        mirror_pressure_demand = NULL;
        int busy = (progress.reclaim & RC_BUDGET_RECLAIM_BUSY) != 0;
        if (busy) {
            plat_mutex_lock(&pool_lock);
            if (!r->unpublished && r->mstate == MS_PENDING && !vk_sync_stopping()) mirror_pressure_add(r);
            plat_mutex_unlock(&pool_lock);
        }
        destroy_rec(vk_ctx(), r);
        if (busy) return;
    }
}

static int mirror_finish_backing(PoolRec *r, VkBuffer buffer, VmaAllocation allocation, uint64_t n,
                                 int verify_import) {
    // Finalization and GPA projection share pool_lock (pool -> GPA, as in mirror eviction). The allocation
    // itself was prepared outside that lock, so a finalized or replaced pool can reject its private result.
    plat_mutex_lock(&pool_lock);
    if (!mirror_pool_alone(mirror_pool_entry(r)) || vk_sync_stopping() || vk_device_is_lost() ||
        (verify_import && (!r->host_backing || !mem_backing_read_hold(r->host_backing)))) {
        plat_mutex_unlock(&pool_lock);
        resource_core_vma_destroy_buffer(vk_ctx()->vma, buffer, allocation);
        return MS_REFUSED;
    }
    if (r->mirror_incarnation == UINT64_MAX) rt_fatal("vk: pool mirror identity exhausted");
    int twin = twin_wanted(r);
    if (r->gpa_rec) {
        uint32_t projected = vk_gpa_remap_exact(r->import_lo, r->import_hi, r->gpa_rec, buffer, r->import_lo,
                                                twin, twin && r->gpa_host_coherent, &twin);
        if (!projected) {
            if (verify_import) mem_read_release();
            plat_mutex_unlock(&pool_lock);
            resource_core_vma_destroy_buffer(vk_ctx()->vma, buffer, allocation);
            return refuse(r, MR_GPA_PROJECTION, "its complete device-address projection could not be installed");
        }
        r->gpa_rec = projected;
    }
    r->mirror_incarnation++;
    r->mirror_origin_checked_scene = vk_exec_resource_position().scene;
    r->mirror_origin_full_copy = 0;
    r->mirror_origin_last_copy = 0;
    r->mbuf = buffer;
    r->malloc_ = allocation;
    if (nm == mcap) grow((void **)&mlist, &mcap, sizeof *mlist);
    int at = nm;
    while (at > 0 && mlist[at - 1]->import_lo > r->import_lo) at--;
    memmove(mlist + at + 1, mlist + at, (size_t)(nm - at) * sizeof *mlist);
    mlist[at] = r;
    nm++;
    r->acc_sub = msub;
    // The address table and first full copy must be prepared before any worker binds the new buffer.
    r->twin = (uint8_t)twin;
    if (!r->twin) {
        plat_inc64(&m_untwinned_n);
        plat_add64(&m_untwinned_bytes, (int64_t)n);
    }
    mirror_queue_upload(r, 0, n, r->mirror_origin_checked_scene);
    r->mirror_origin_full_queued = r->mirror_origin_checked_scene;
    mrep.first_n++, mrep.first_bytes += n;
    plat_inc64(&m_live_n);
    plat_add64(&m_live_bytes, (int64_t)n);
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    r->mstate = MS_ON;
    if (verify_import) mem_read_release();
    plat_mutex_unlock(&pool_lock);
    return MS_ON;
}

// The creation owner prepares the complete backing while other workers still see MS_CREATING.
static int mirror_create_backing(PoolRec *r, RcVmaDependency *dependency, int verify_import) {
    VkCtx *c = vk_ctx();
    uint64_t n = r->import_hi - r->import_lo;
    if (r->kind != POOL_IMPORT || !r->buf || !n || vk_sync_stopping() || vk_device_is_lost()) return MS_REFUSED;
    plat_mutex_lock_shared(&pool_lock);
    int i = mirror_pool_entry(r);
    int alone = mirror_pool_alone(i);
    plat_mutex_unlock_shared(&pool_lock);
    if (i < 0) return MS_REFUSED;   // finalized meanwhile
    if (!alone) return refuse(r, MR_OVERLAP, "it overlaps another pool");
    uint64_t r0 = r->import_lo >> SNAP_SHIFT, r1 = (r->import_hi - 1) >> SNAP_SHIFT;
    if (r1 >= SNAP_REGIONS) return refuse(r, MR_UNTRACKED, "outside the write-watched address space");
    plat_mutex_lock(&gen_lock);
    mrep.sync_queries += snap_regions(r0, r1, mepoch);
    int untracked = 0;
    for (uint64_t k = r0; k <= r1; k++) untracked |= snap_untracked[k] || !snap_current((uint32_t)k, mepoch);
    r->mgen = gen_now;   // the whole-range copy below holds every write observed so far
    plat_mutex_unlock(&gen_lock);
    if (untracked) return refuse(r, MR_UNTRACKED, "its range is not write-watched");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = pool_usage(c),
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    RcVmaRequest request = {.producer = RC_BUDGET_BUFFER, .dependency = dependency};
    RcAdmissionStatus admission;
    VkBuffer buffer;
    VmaAllocation allocation;
    VkResult res = resource_core_vma_create_buffer(&request, c->vma, &bi, &ai, &buffer, &allocation, NULL, &admission);
    if (res == VK_SUCCESS) return mirror_finish_backing(r, buffer, allocation, n, verify_import);
    vk_device_lost_check(res, "preparation of a VRAM mirror");
    char why[128];
    int native_pressure = res == VK_ERROR_OUT_OF_DEVICE_MEMORY || res == VK_ERROR_OUT_OF_HOST_MEMORY;
    int placement = 0;
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) placement |= dependency->peak.amount[d] != 0;
    if (dependency->selected && (res == VK_NOT_READY || (native_pressure && (placement || dependency->can_reuse)) ||
                                 (admission == RC_ADMISSION_UNSUPPORTED && placement))) {
        snprintf(why, sizeof why, "its %.1f MiB mirror awaits shared capacity (result %d, admission %d)",
                 n / 1048576.0, (int)res, (int)admission);
        refuse(r, MR_BUDGET, why);
        return MS_PENDING;
    }
    snprintf(why, sizeof why, "device-local allocation of %.1f MiB failed (%d)", n / 1048576.0, (int)res);
    return refuse(r, MR_ALLOC, why);
}

static void mirror_create(PoolRec *r, int verify_import) {
    r->mstate = MS_CREATING;
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    RcVmaDependency dependency = {0};
    int state = mirror_create_backing(r, &dependency, verify_import);
    if (state == MS_ON) return;
    plat_mutex_lock(&pool_lock);
    mirror_pressure_remove(r);
    if (r->unpublished || vk_sync_stopping() || vk_device_is_lost()) state = MS_REFUSED;
    r->mirror_dependency = dependency;
    r->mirror_checked_sub = msub;
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    r->mstate = state;
    if (state == MS_PENDING) mirror_pressure_add(r);
    plat_mutex_unlock(&pool_lock);
}

static void mirror_forget(PoolRec *r) {
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    for (int i = 0; i < nm; i++)
        if (mlist[i] == r) {
            memmove(mlist + i, mlist + i + 1, (size_t)(nm - i - 1) * sizeof *mlist);
            nm--;
            plat_dec64(&m_live_n);
            plat_add64(&m_live_bytes, -(int64_t)(r->import_hi - r->import_lo));
            break;
        }
}

static void mirror_retire_check(PoolRec *pool, int completed) {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    plat_mutex_lock(&pool_lock);
    if (pool->mstate == MS_DORMANT && !pool->unpublished && !vk_device_is_lost()) {
        if (completed) pool->mirror_retire_completed = 1;
        if (pool->mirror_retire_completed && pool->mbuf &&
            !__atomic_load_n(&pool->mirror_input_refs, __ATOMIC_ACQUIRE)) {
            buffer = pool->mbuf;
            allocation = pool->malloc_;
            pool->mbuf = VK_NULL_HANDLE;
            pool->malloc_ = NULL;
            pool->twin = 0;
            pool->acc_sub = 0;
        }
    }
    plat_mutex_unlock(&pool_lock);
    if (buffer) {
        resource_core_vma_destroy_buffer(vk_ctx()->vma, buffer, allocation);
        plat_mutex_lock(&pool_lock);
        pool->mstate = MS_NONE;
        plat_mutex_unlock(&pool_lock);
    }
}

static void mirror_retire_completed(void *owner) {
    PoolRec *pool = owner;
    mirror_retire_check(pool, 1);
    destroy_rec(vk_ctx(), pool);
}

static void mirror_retire_canceled(void *owner, int may_have_executed) {
    (void)may_have_executed;
    // The live pool keeps the retired backing until actual scene proof or backend teardown.
    destroy_rec(vk_ctx(), owner);
}

// Finalized pools waiting for pool_retire (pool_fini says which); vk_mem_shutdown destroys those whose callback
// never ran.
static PoolRec **retiring;
static int nretiring, cretiring;
static PlatMutex retire_lock = PLAT_MUTEX_INIT;

static void pool_retiring(PoolRec *r) {
    if ((r->malloc_ && resource_core_vma_retire(r->malloc_) != RC_ADMISSION_OK) ||
        (r->salloc && resource_core_vma_retire(r->salloc) != RC_ADMISSION_OK) ||
        (r->alloc && resource_core_vma_retire(r->alloc) != RC_ADMISSION_OK))
        rt_fatal("vk: retiring pool lost its shared capacity owner");
    plat_mutex_lock(&retire_lock);
    if (nretiring == cretiring) grow((void **)&retiring, &cretiring, sizeof *retiring);
    retiring[nretiring++] = r;
    plat_mutex_unlock(&retire_lock);
}

static void pool_retire(void *arg) {
    plat_mutex_lock(&retire_lock);
    for (int i = 0; i < nretiring; i++)
        if (retiring[i] == arg) { retiring[i] = retiring[--nretiring]; break; }
    plat_mutex_unlock(&retire_lock);
    destroy_rec(vk_ctx(), arg);
}

// Replay thread, the first access of this submission: queues the copies of the pages written since the last sync.
static void mirror_sync(PoolRec *r) {
    uint64_t t0 = qpc(), scene = vk_exec_resource_position().scene;
    r->acc_sub = msub;
    pm_note(r);
    uint64_t lo = r->import_lo, hi = r->import_hi;
    uint64_t r0 = lo >> SNAP_SHIFT, r1 = (hi - 1) >> SNAP_SHIFT;
    int excl = 0;
    plat_mutex_lock_shared(&gen_lock);
    for (uint64_t k = r0; k <= r1; k++)
        if (!snap_current((uint32_t)k, mepoch)) { excl = 1; break; }
    if (excl) {
        plat_mutex_unlock_shared(&gen_lock);
        plat_mutex_lock(&gen_lock);
        mrep.sync_queries += snap_regions(r0, r1, mepoch);
    }
    uint32_t since = r->mgen;
    int untracked = 0;
    for (uint64_t k = r0; k <= r1; k++) {
        untracked |= snap_untracked[k] || !snap_current((uint32_t)k, mepoch);
        if (creg_max[k] <= since) continue;
        uint64_t a = k << SNAP_SHIFT, b = a + (1ull << SNAP_SHIFT);
        if (a < lo) a = lo;
        if (b > hi) b = hi;
        const uint32_t *ch = cgen_chunk[a >> GEN_CHUNK_SHIFT];   // a 2 MiB region lies in one chunk
        if (!ch) continue;
        for (uint64_t x = a; x < b; x += 1ull << GEN_PAGE_SHIFT)
            if (ch[(x >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)] > since) {
                mirror_queue_upload(r, x - lo, 1ull << GEN_PAGE_SHIFT, scene);
                mrep.page_bytes += 1ull << GEN_PAGE_SHIFT;
                r->st_pages += 1ull << GEN_PAGE_SHIFT;
            }
    }
    r->mgen = gen_now;
    if (excl) plat_mutex_unlock(&gen_lock);
    else plat_mutex_unlock_shared(&gen_lock);
    if (__atomic_load_n(&r->fl_n, __ATOMIC_ACQUIRE)) {
        FlushRange fl[FLUSH_RANGES + 1];
        plat_mutex_lock(&flush_lock);
        int n = r->fl_n;
        memcpy(fl, r->fl, (size_t)n * sizeof *fl);
        uint64_t calls = r->fl_calls, sum = r->fl_sum;
        r->fl_n = 0, r->fl_calls = r->fl_sum = 0;
        plat_mutex_unlock(&flush_lock);
        mrep.flush_calls += calls, mrep.flush_sum += sum;
        for (int i = 0; i < n; i++) {
            uint64_t a = fl[i].lo > lo ? fl[i].lo : lo, b = fl[i].hi < hi ? fl[i].hi : hi;
            if (a >= b) continue;
            if (!untracked && !flush_upload) {   // the write watch saw every write in them: the pages above
                mrep.flush_skipped += b - a;
                continue;
            }
            mirror_queue_upload(r, a - lo, b - a, scene);
            mrep.flush_n++, mrep.flush_bytes += b - a, r->st_flush += b - a;
            if ((rt_trace & RT_TRACE_PERF) && b - a >= 16u << 20)
                rt_log("mem: mirror frame %llu: flush range %.1f MiB at +%.1f MiB of a %.1f MiB pool (%d ranges from %llu "
                       "flushes of %.3f MiB)", (unsigned long long)mirror_frame, (b - a) / 1048576.0, (a - lo) / 1048576.0,
                       (hi - lo) / 1048576.0, n, (unsigned long long)calls, sum / 1048576.0);
        }
    }
    r->mirror_origin_checked_scene = untracked ? 0 : scene;
    mrep.syncs++, r->st_syncs++;
    mrep.sync_t += qpc() - t0;
}

// ---- shadow storage: the copy-ins before a part is submitted, and the device address table's readers
int vk_mem_shadow_storage(void) { return shadow_on; }
static int shadow_check_on(void) { return s_check; }
int vk_mem_shadow_table_stores(void) { return s_learn; }

static void shadow_table_add(PoolRec *r) {
    plat_mutex_lock(&shadow_lock);
    if (!r->stab_i) {
        if (nstab == cstab) grow((void **)&stab, &cstab, sizeof *stab);
        stab[nstab++] = r;
        r->stab_i = (uint32_t)nstab;
    }
    plat_mutex_unlock(&shadow_lock);
}

void vk_mem_shadow_table_read(void) {
    if (shadow_on) __atomic_store_n(&stab_want, vk_texture_epoch(), __ATOMIC_RELAXED);
}

// gen_lock held: nothing of the pool was stamped since its shadow's last copy-in. Its regions have this epoch's
// write-watch snapshot, none of them is untracked, and none holds a stamp newer than the shadow's sgen, which then
// stays as it is. With cpu the stamps are the CPU-side tables', as shadow_sync_pool reads them. The table's sync asks
// this of every shadow in the table before it runs shadow_sync_pool with its own timer reads, snapshot walk and
// region loop: outdoors about 3,800 pools are in the table and a few hundred were written.
static int shadow_pool_clean(const PoolRec *r, uint32_t epoch, int cpu) {
    uint64_t r0 = r->import_lo >> SNAP_SHIFT, r1 = (r->import_hi - 1) >> SNAP_SHIFT;
    if (!epoch || r1 >= SNAP_REGIONS) return 0;
    const uint32_t *mx = cpu ? creg_max : reg_max;
    for (uint64_t k = r0; k <= r1; k++)
        if (!snap_current((uint32_t)k, epoch) || snap_untracked[k] || mx[k] > r->sgen) return 0;
    return 1;
}

void vk_mem_shadow_sync(uint64_t fence) {
    if (!shadow_on) return;
    uint64_t t0 = qpc();
    uint32_t epoch = vk_texture_epoch();
    int cpu = vk_mirror_on && wb_no_reup();   // before the locks: its first call logs
    // The table's readers: the record of this guest submission bound the table, and no part of it has synced yet.
    // The callers are one at a time (vk_sync.c prologue under the submission lock, and the probe).
    static uint32_t tab_epoch;    // the guest submission the table's shadows were last synced for
    static uint64_t wb_fence;     // the last fence that got the write-back job
    int table = epoch && epoch != tab_epoch && __atomic_load_n(&stab_want, __ATOMIC_RELAXED) == epoch;
    plat_mutex_lock(&shadow_lock);
    if (table) {
        tab_epoch = epoch;
        if (s_learn && fence && fence != wb_fence && ntpools) {
            wb_fence = fence;
            table_wb_queue(fence);
        }
    }
    if (nstouch || (table && nstab)) {
        plat_mutex_lock(&gen_lock);
        for (int i = 0; i < nstouch; i++) shadow_sync_pool(stouch[i], epoch, cpu);
        uint64_t t1 = qpc();
        for (int i = 0; table && i < nstab; i++) {   // the shadows the replay did not resolve
            PoolRec *r = stab[i];
            if (r->stouched) continue;
            uint64_t r0 = r->import_lo >> SNAP_SHIFT, r1 = (r->import_hi - 1) >> SNAP_SHIFT;
            if (r1 < SNAP_REGIONS) s_queries += snap_regions(r0, r1, epoch);   // so that shadow_pool_clean can tell
            if (!shadow_pool_clean(r, epoch, cpu)) shadow_sync_pool(r, epoch, cpu), s_syncs++;
        }
        for (int i = 0; i < nstouch; i++) stouch[i]->stouched = 0;
        plat_mutex_unlock(&gen_lock);
        s_syncs += (uint64_t)nstouch;
        nstouch = 0;
        // A shadow with a VRAM mirror: the table names the mirror, which takes the same pages unless this guest
        // submission synced it already (a pool bound as a buffer). The uploads read the shadows just brought up to
        // date, and the prologue records them ahead of the part, after the write-backs of the submissions before it.
        // No mirror is made for a reader of the table.
        for (int i = 0; table && i < nstab; i++)
            if (stab[i]->mstate == MS_ON && stab[i]->acc_sub != msub) mirror_sync(stab[i]);
        if (table) s_tab_ticks += qpc() - t1, s_tab_syncs++;
    }
    s_ticks += qpc() - t0;
    plat_mutex_unlock(&shadow_lock);
}

// The replay workers (vk_exec.c) bind and write from several threads at once: while a chunk records, a mirror's first
// use in a submission (creation, sync) and the notes of GPU writes go under use_lock. A use the submission already
// synced takes no lock, nor does the replay outside a parallel group (it alone records then).
static PlatMutex use_lock = PLAT_MUTEX_INIT;

static int mirror_retry_due(PoolRec *r) { // serialized with mirror creation
    if (r->mirror_checked_sub == msub) return 0;
    r->mirror_checked_sub = msub;
    if (vk_sync_stopping() || vk_device_is_lost()) return 0;
    static uint32_t readiness_sub;
    static RcBudgetReadiness readiness;
    if (readiness_sub != msub) {
        resource_core_budget_readiness(0, &readiness);
        readiness_sub = msub;
    }
    return resource_core_vma_dependency_ready(&r->mirror_dependency, &readiness);
}

// Optional admission can leave the import established for the whole current mirror scene.
static uint64_t mirror_scene(void) { return (uint64_t)msub + 1; }

static uint64_t mirror_import_scene(const PoolRec *pool) {
    uint64_t scene = mirror_scene();
    int state = pool->mstate;
    if (state == MS_DORMANT) return scene;
    if (state != MS_NONE && state != MS_PENDING) return 0;
    const uint64_t writer = pool->import_writer_fence;
    return pool->mirror_import_only_scene == scene || (writer && writer > vk_sync_fence_done()) ||
           (pool->import_writer_refs && pool->import_writer_scene == scene) ||
           (state == MS_PENDING && pool->mirror_checked_sub == msub) ?
           scene : 0;
}

static void mem_use_locked(VkMemSpan *sp);
void vk_mem_use(VkMemSpan *sp) {
    PoolRec *r = sp->pool;
    if (r->mstate == MS_ON && r->acc_sub == msub) {
        r->vd_use_sub = msub;
        sp->buffer = r->mbuf;
        return;
    }
    if (r->mstate == MS_REFUSED || mirror_import_scene(r)) return;
    int lk = vk_exec_shared();
    if (lk) plat_mutex_lock(&use_lock);
    mem_use_locked(sp);
    if (lk) plat_mutex_unlock(&use_lock);
}

static void mirror_attempt_locked(PoolRec *r, int verify_import) {
    if (mirror_import_scene(r)) return;
    if (r->mstate == MS_PENDING && mirror_retry_due(r)) {
        plat_dec64(&m_refused_n[r->refused_why]);   // counted again if refused again
        plat_add64(&m_refused_bytes[r->refused_why], -(int64_t)(r->import_hi - r->import_lo));
        plat_inc64(&m_retries);
        plat_mutex_lock(&pool_lock);
        mirror_pressure_remove(r);
        plat_mutex_unlock(&pool_lock);
        mirror_create(r, verify_import);
    } else if (r->mstate == MS_NONE) mirror_create(r, verify_import);
}

static void mem_use_locked(VkMemSpan *sp) {
    PoolRec *r = sp->pool;
    mirror_attempt_locked(r, 0);
    if (r->mstate != MS_ON) return;
    if (r->acc_sub != msub) mirror_sync(r);
    r->vd_use_sub = msub;
    sp->buffer = r->mbuf;
}

static int mirror_prepare_input(const VkMemSpanCapture *source, const VkMemSpan *selected, VkMemMirrorInputPolicy policy) {
    // This caller is the ordered scene recorder. A speculative or still-shared pass cannot mutate provider choice.
    if (vk_exec_shared() || vk_sync_stopping() || vk_device_is_lost() ||
        !vk_mem_span_capture_current(source, selected)) return 0;
    PoolRec *pool = source->window;
    if (pool->kind != POOL_IMPORT || pool->mstate == MS_CREATING) return 0;
    if (pool->mstate == MS_ON || pool->mstate == MS_REFUSED || mirror_import_scene(pool)) return 1;
    if (policy == VK_MEM_MIRROR_PREPARE && pool->host_backing && mem_backing_current(pool->host_backing)) {
        // The initial upload covers the full native import, and GPA publication must follow the earlier GPU prefix.
        mirror_attempt_locked(pool, 1);
    }
    if (pool->mstate == MS_NONE || pool->mstate == MS_PENDING)
        pool->mirror_import_only_scene = (uint64_t)msub + 1;
    return 1;
}

// use_lock precedes the pool topology lock, as in mirror creation. The installed lease protects
// the selected mirror against eviction. Recording synchronizes this image of the pool without selecting anew.
static int span_record_mirror_input(const VkMemSpanCapture *capture, const VkMemSpan *selected) {
    int shared = vk_exec_shared();
    if (shared) plat_mutex_lock(&use_lock);
    plat_mutex_lock_shared(&pool_lock);
    int current = span_capture_match(capture, selected);
    if (current) {
        PoolRec *pool = capture->provider;
        if (pool->smap && !pool->stouched) shadow_touch(pool);
        if (pool->acc_sub != msub) mirror_sync(pool);
        pool->vd_use_sub = msub;
    }
    plat_mutex_unlock_shared(&pool_lock);
    if (shared) plat_mutex_unlock(&use_lock);
    return current;
}

void vk_mem_pick(VkMemSpan *sp) {
    PoolRec *r = sp->pool;
    if (!r) return;
    if (r->mstate == MS_CREATING && vk_exec_shared()) {
        // A pick does not create a mirror, but it must observe the completed choice of another worker.
        plat_mutex_lock(&use_lock);
        plat_mutex_unlock(&use_lock);
    }
    if (r->mstate != MS_ON) return;
    if (r->acc_sub != msub) {
        int lk = vk_exec_shared();
        if (lk) plat_mutex_lock(&use_lock);
        if (r->acc_sub != msub) mirror_sync(r);
        if (lk) plat_mutex_unlock(&use_lock);
    }
    sp->buffer = r->mbuf;
}

static int host_interval_vectors_match(const MemBackingCapture *source, const MemBackingCapture *provider,
                                       uint64_t address, uint64_t size, VkMemCanonicalSourceQualification *report) {
    uint32_t source_count, provider_count;
    const RcInterval *s = mem_backing_intervals(source, &source_count);
    const RcInterval *p = mem_backing_intervals(provider, &provider_count);
    uint32_t si = 0, pi = 0;
    uint64_t end = address + size;
    while (address < end) {
        while (si < source_count && s[si].address + s[si].range.size <= address) ++si;
        while (pi < provider_count && p[pi].address + p[pi].range.size <= address) ++pi;
        const RcInterval *a = si < source_count && s[si].address <= address ? &s[si] : NULL;
        const RcInterval *b = pi < provider_count && p[pi].address <= address ? &p[pi] : NULL;
        if (!a || !b || a->mapping != b->mapping || a->range.backing != b->range.backing ||
            a->range.offset + (address - a->address) != b->range.offset + (address - b->address)) {
            if (report) {
                report->mismatch_address = address;
                if (a) report->source_interval = *a;
                if (b) report->provider_interval = *b;
            }
            return 0;
        }
        uint64_t ae = a->address + a->range.size, be = b->address + b->range.size;
        address = ae < be ? ae : be;
    }
    return 1;
}

RcRegistryResult vk_mem_canonical_source_qualify_report(const MemBackingCapture *source, uint64_t address,
    uint64_t size, VkMemCanonicalSourceQualification *report) {
    if (report) {
        *report = (VkMemCanonicalSourceQualification){.stage = VK_MEM_CANONICAL_SOURCE_INVALID};
        mem_backing_intervals(source, &report->source_count);
    }
    if (!source || !size || size > UINT64_MAX - address) return RC_REGISTRY_INVALID;
    if (nvn_gpu_only(address)) {
        if (report) report->stage = VK_MEM_CANONICAL_SOURCE_GPU_ONLY;
        return RC_REGISTRY_UNSUPPORTED;
    }
    if (!mem_backing_current(source)) {
        if (report) report->stage = VK_MEM_CANONICAL_SOURCE_LEASE_BEFORE;
        return RC_REGISTRY_STALE;
    }
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *entry = pidx_find(&pool_index, address, size);
    const PoolRec *pool = entry ? entry->buf : NULL;
    if (report && pool) {
        report->provider_kind = pool->kind;
        report->provider_publication = pool->publication;
        report->provider_address = pool->lo;
        report->provider_bytes = pool->size;
        report->provider_import_lo = pool->import_lo;
        report->provider_import_hi = pool->import_hi;
        report->provider_mapping_cutoff = pool->host_mapping_cutoff;
        mem_backing_intervals(pool->host_backing, &report->provider_count);
    }
    RcRegistryResult result = RC_REGISTRY_UNSUPPORTED;
    VkMemCanonicalSourceQualificationStage stage;
    if (!pool) stage = VK_MEM_CANONICAL_SOURCE_PROVIDER_MISSING;
    else if (pool->unpublished) stage = VK_MEM_CANONICAL_SOURCE_PROVIDER_UNPUBLISHED;
    else if (pool->kind != POOL_IMPORT) stage = VK_MEM_CANONICAL_SOURCE_PROVIDER_KIND;
    else if (!pool->buf) stage = VK_MEM_CANONICAL_SOURCE_PROVIDER_BUFFER;
    else if (!pool->host_backing) stage = VK_MEM_CANONICAL_SOURCE_PROVIDER_HOST;
    else if (address < pool->import_lo || address >= pool->import_hi || size > pool->import_hi - address)
        stage = VK_MEM_CANONICAL_SOURCE_PROVIDER_EXTENT;
    else if (!host_interval_vectors_match(source, pool->host_backing, address, size, report)) {
        stage = VK_MEM_CANONICAL_SOURCE_VECTOR;
        result = RC_REGISTRY_STALE;
    } else {
        stage = VK_MEM_CANONICAL_SOURCE_EXACT;
        result = RC_REGISTRY_OK;
    }
    plat_mutex_unlock_shared(&pool_lock);
    if (result == RC_REGISTRY_OK && !mem_backing_current(source)) {
        stage = VK_MEM_CANONICAL_SOURCE_LEASE_AFTER;
        result = RC_REGISTRY_STALE;
    }
    if (report) report->stage = stage;
    return result;
}

RcRegistryResult vk_mem_canonical_source_qualify(const MemBackingCapture *source, uint64_t address, uint64_t size) {
    return vk_mem_canonical_source_qualify_report(source, address, size, NULL);
}

RcRegistryResult vk_mem_canonical_source_qualify_expected(const MemBackingCapture *source,
                                                         const VkMemSpanCapture *capture) {
    if (!source || !capture) return RC_REGISTRY_INVALID;
    const PoolRec *pool = capture->window;
    if (pool->kind != POOL_IMPORT) return RC_REGISTRY_UNSUPPORTED;
    if (nvn_gpu_only(capture->address)) return RC_REGISTRY_UNSUPPORTED;
    plat_mutex_lock_shared(&pool_lock);
    const MemBackingCapture *mapping = capture->host ? capture->host : pool->host_backing;
    int current = span_capture_match(capture, NULL) &&
                  host_interval_vectors_match(source, mapping, capture->address, capture->size, NULL);
    plat_mutex_unlock_shared(&pool_lock);
    return current ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}

enum { HOST_INPUT_FRAGMENT_LIMIT=64 };
typedef struct {
    VkMemSpan selected;
    RcRegistryBufferLocation location;
    RcRegistryBufferFragment fragments[HOST_INPUT_FRAGMENT_LIMIT];
    uint32_t count, fragment_count;
    RcRegistryBufferSpan demands[];
} HostInputOrdered;

typedef struct {
    const VkMemSpanCapture *span;
    RcRegistryPosition position;
    RcRegistryGpuCapture producer;
    HostInputOrdered *ordered;
    uint64_t address, size, watch_lineage, protection;
    uint32_t count, watch_generation;
    int mapped;
    RcSourceRange ranges[];
} HostInputRead;

// Use the shared page generations for the exact demanded pages. Each drain
// fits once in the existing write-watch vector, even if all its pages are hot.
static VkExecHostInputResult host_input_watch(HostInputRead *read, int finish) {
    const uint64_t page = UINT64_C(1) << GEN_PAGE_SHIFT;
    if (!read->size || read->address >= RT_ADDRESS_LIMIT ||
        read->size > RT_ADDRESS_LIMIT - read->address) return VK_EXEC_HOST_INPUT_UNTRACKED;
    const uint64_t start = read->address & ~(page - 1);
    const uint64_t end = (read->address + read->size + page - 1) & ~(page - 1);
    const uint64_t window = (sizeof gen_pages / sizeof *gen_pages - 1) * page;
    const VkExecHostInputResult unproved = finish ? VK_EXEC_HOST_INPUT_CHANGED : VK_EXEC_HOST_INPUT_UNTRACKED;

    plat_mutex_lock(&gen_lock);
    uint32_t generation = generation_next();
    uint64_t lineage = source_generation_lineage;
    VkExecHostInputResult result = unproved;
    if (generation && lineage && (!finish || (lineage == read->watch_lineage && generation > read->watch_generation))) {
        result = VK_EXEC_HOST_INPUT_READY;
        for (uint64_t lo = start; lo < end && result == VK_EXEC_HOST_INPUT_READY; ) {
            uint64_t hi = end - lo <= window ? end : lo + window;
            enum StampResult written = query_stamp(lo, hi, generation);
            if (written == STAMP_UNTRACKED || (finish && written == STAMP_DIRTY)) result = unproved;
            // Another observer may have consumed and stamped the same write-watch bits.
            for (uint64_t a = lo; finish && result == VK_EXEC_HOST_INPUT_READY && a < hi; a += page) {
                const uint32_t *stamp = gen_slot(a, 0);
                if (stamp && *stamp > read->watch_generation) result = unproved;
            }
            lo = hi;
        }
        if (!finish && result == VK_EXEC_HOST_INPUT_READY) {
            read->watch_generation = generation;
            read->watch_lineage = lineage;
        }
    }
    plat_mutex_unlock(&gen_lock);
    return result;
}

// Keep the established protection proof for watched mappings on platforms
// that cannot supply page write observations.
static int host_input_freeze(HostInputRead *read) {
    const uint64_t region = UINT64_C(1) << REG_SHIFT;
    const uint64_t first = read->address & ~(region - 1);
    const uint64_t end = read->address + read->size;
    for (uint64_t a = first; a < end; a += region)
        if (!mem_freeze(a)) return 0;
    const uint64_t protection = mem_protection_serial();
    if (!protection) return 0;
    for (uint64_t a = first; a < end; a += region)
        if (!mem_frozen_stable(a)) return 0;
    if (!mem_protection_range_unchanged(read->address, read->size, protection)) return 0;
    read->protection = protection;
    return 1;
}

static int host_input_range_compare(const void *left, const void *right) {
    const RcSourceRange *a = left, *b = right;
    if (a->backing != b->backing) return a->backing < b->backing ? -1 : 1;
    return a->offset < b->offset ? -1 : a->offset > b->offset;
}

static uint32_t host_input_range_normalize(RcSourceRange *ranges, uint32_t count) {
    if (count < 2) return count;
    qsort(ranges, count, sizeof *ranges, host_input_range_compare);
    uint32_t used = 1;
    for (uint32_t i = 1; i < count; i++) {
        RcSourceRange *previous = &ranges[used - 1], next = ranges[i];
        if (previous->backing == next.backing && next.offset <= previous->offset + previous->size) {
            uint64_t end = next.offset + next.size;
            if (end > previous->offset + previous->size) previous->size = end - previous->offset;
        } else ranges[used++] = next;
    }
    return used;
}

static VkExecHostInputResult host_input_result(RcRegistryResult result) {
    switch (result) {
    case RC_REGISTRY_OK: return VK_EXEC_HOST_INPUT_READY;
    case RC_REGISTRY_BLOCKED: return VK_EXEC_HOST_INPUT_BLOCKED;
    case RC_REGISTRY_STALE: return VK_EXEC_HOST_INPUT_STALE;
    case RC_REGISTRY_CAPACITY: case RC_REGISTRY_EXHAUSTED: return VK_EXEC_HOST_INPUT_CAPACITY;
    default: return VK_EXEC_HOST_INPUT_UNSUPPORTED;
    }
}

static VkExecHostInputResult host_input_ordered_select(HostInputRead *read, int remember) {
    HostInputOrdered *ordered=read->ordered;
    RcRegistryBufferLocation current;
    if (!vk_mem_span_buffer_location(read->span,&ordered->selected,&current) ||
        current.allocation!=ordered->location.allocation || current.incarnation || current.offset!=ordered->location.offset)
        return VK_EXEC_HOST_INPUT_STALE;
    RcRegistry *registry=rc_registry_runtime();
    if (!rc_registry_provisional_read_current(registry,read->position)) return VK_EXEC_HOST_INPUT_STALE;

    RcRegistryBufferFragment found[HOST_INPUT_FRAGMENT_LIMIT];
    uint32_t count=0;
    RcRegistryResult outcome=rc_registry_gpu_capture_buffer_fragments(registry,read->producer,
        ordered->demands,ordered->count,read->position,found,HOST_INPUT_FRAGMENT_LIMIT,&count);
    if (outcome!=RC_REGISTRY_OK) return host_input_result(outcome);
    if (!count) return VK_EXEC_HOST_INPUT_UNSUPPORTED;
    if (!remember && count!=ordered->fragment_count) return VK_EXEC_HOST_INPUT_CHANGED;
    for (uint32_t i=0;i<count;++i) {
        const RcRegistryBufferFragment *fragment=&found[i];
        // This is a live current-scene read. A historical raw import needs immutable producer material.
        if (fragment->output.scene!=read->position.scene ||
            rc_registry_position_compare(fragment->output,read->position)>=0) return VK_EXEC_HOST_INPUT_UNSUPPORTED;
        if (!remember) {
            const RcRegistryBufferFragment *prior=&ordered->fragments[i];
            if (prior->demand!=fragment->demand || prior->interval.address!=fragment->interval.address ||
                prior->interval.mapping!=fragment->interval.mapping ||
                !rc_range_equal(prior->interval.range,fragment->interval.range) ||
                rc_registry_position_compare(prior->output,fragment->output)) return VK_EXEC_HOST_INPUT_CHANGED;
        }
    }
    // Intervals without a proven native supplier must independently have ordinary CPU visibility.
    uint32_t i=0;
    for (uint32_t j=0;j<ordered->count;++j) {
        RcSourceRange gap=ordered->demands[j].interval.range;
        const uint64_t end=gap.offset+gap.size;
        while (i<count && found[i].demand==j) {
            const RcSourceRange supplied=found[i++].interval.range;
            gap.size=supplied.offset-gap.offset;
            if (gap.size && rc_registry_gpu_capture_read_at(registry,read->producer,&gap,1,read->position)!=RC_REGISTRY_OK)
                return VK_EXEC_HOST_INPUT_BLOCKED;
            gap.offset=supplied.offset+supplied.size;
        }
        gap.size=end-gap.offset;
        if (gap.size && rc_registry_gpu_capture_read_at(registry,read->producer,&gap,1,read->position)!=RC_REGISTRY_OK)
            return VK_EXEC_HOST_INPUT_BLOCKED;
    }
    if (remember) {
        memcpy(ordered->fragments,found,count*sizeof *found);
        ordered->fragment_count=count;
    }
    return VK_EXEC_HOST_INPUT_READY;
}

static VkExecHostInputResult host_input_ordered_prepare(HostInputRead *read,const VkMemSpan *selected,
                                                       const RcInterval *clipped,uint32_t count) {
    const VkMemSpanCapture *capture=read->span;
    RcRegistryBufferLocation location={0};
    plat_mutex_lock_shared(&pool_lock);
    const PoolRec *pool=capture->provider;
    const int coherent=capture->native_input && !capture->mirror_incarnation && pool==capture->window &&
        pool && pool->kind==POOL_IMPORT && pool->gpa_host_coherent && !pool->smap && pool->buf==selected->buffer &&
        pool->publication && span_capture_match(capture,selected);
    if (coherent) location=(RcRegistryBufferLocation){pool->publication,0,selected->offset};
    plat_mutex_unlock_shared(&pool_lock);
    if (!coherent) return VK_EXEC_HOST_INPUT_UNSUPPORTED;
    if ((uint64_t)count*sizeof(RcRegistryBufferSpan)>SIZE_MAX-sizeof(HostInputOrdered))
        return VK_EXEC_HOST_INPUT_CAPACITY;
    HostInputOrdered *ordered=calloc(1,sizeof *ordered+count*sizeof ordered->demands[0]);
    if (!ordered) return VK_EXEC_HOST_INPUT_CAPACITY;
    read->ordered=ordered;
    ordered->selected=*selected;
    ordered->location=location;
    ordered->count=count;
    uint64_t next=read->address;
    const uint64_t end=next+read->size;
    for (uint32_t i=0;i<count;++i) {
        const RcInterval part=clipped[i];
        if (part.address!=next || !part.mapping || !rc_range_valid(part.range) ||
            part.range.backing==RC_GUEST_COMPATIBILITY_BACKING || part.range.size>end-next ||
            part.address-capture->address>UINT64_MAX-location.offset ||
            part.range.size>UINT64_MAX-location.offset-(part.address-capture->address)) return VK_EXEC_HOST_INPUT_UNSUPPORTED;
        ordered->demands[i]=(RcRegistryBufferSpan){part,
            {location.allocation,0,location.offset+(part.address-capture->address)}};
        next+=part.range.size;
    }
    return next==end ? VK_EXEC_HOST_INPUT_READY : VK_EXEC_HOST_INPUT_UNSUPPORTED;
}

static VkExecHostInputResult host_input_current(HostInputRead *read, const char **refusal) {
    if (vk_sync_stopping() || vk_device_is_lost()) return VK_EXEC_HOST_INPUT_STOPPED;
    if (rc_registry_position_compare(read->position, vk_exec_resource_position())) {
        *refusal = "host-current-position";
        return VK_EXEC_HOST_INPUT_STALE;
    }
    if (!vk_mem_span_capture_current(read->span, NULL)) {
        *refusal = "host-current-span";
        return VK_EXEC_HOST_INPUT_STALE;
    }
    const VkExecHostInputResult result = read->ordered ? host_input_ordered_select(read,0) :
        host_input_result(rc_registry_gpu_capture_read_at(
            rc_registry_runtime(), read->producer, read->ranges, read->count, read->position));
    *refusal = result == VK_EXEC_HOST_INPUT_READY ? NULL :
        read->ordered ? "host-ordered-authority-changed" : "host-current-authority";
    return result;
}

static void host_input_release_mapping(HostInputRead *read) {
    if (!read->mapped) return;
    if (read->ordered) mem_backing_read_release_no_fault();
    else mem_read_release();
    read->mapped=0;
}

static void host_input_dispose(HostInputRead *read) {
    if (!read) return;
    host_input_release_mapping(read);
    if (read->producer.id && rc_registry_gpu_capture_end(rc_registry_runtime(), &read->producer) != RC_REGISTRY_OK)
        rt_fatal("vk: protected host input lost its producer observation");
    free(read->ordered);
    free(read);
}

static VkExecHostInputResult host_input_begin(const VkMemSpanCapture *span, const VkMemSpan *selected,
                                              uint64_t address, uint64_t size, RcRegistryPosition position,
                                              VkExecHostInputRead *out, int logical_pointer, VkExec *ordered) {
    if (!out || out->opaque || out->bytes) return VK_EXEC_HOST_INPUT_UNSUPPORTED;
    out->mapping = (RcInterval){0};
    out->refusal = "host-begin-range";
    if (!span || !selected || !size || size > UINT64_MAX - address ||
        address < span->address || address - span->address >= span->size ||
        size > span->size - (address - span->address) || !rc_registry_position_valid(position))
        return VK_EXEC_HOST_INPUT_UNSUPPORTED;
    if (vk_sync_stopping() || vk_device_is_lost()) return VK_EXEC_HOST_INPUT_STOPPED;
    const PoolRec *pool = span->window;
    out->refusal = "host-begin-provider";
    if (pool->kind != POOL_IMPORT || (!logical_pointer && (pool->smap || span->mirror_incarnation)) ||
        selected->alloc || nvn_gpu_only(address) ||
        (selected->cls != VK_MCLS_HOST_CACHED && selected->cls != VK_MCLS_HOST_UNC))
        return VK_EXEC_HOST_INPUT_UNSUPPORTED;
    out->refusal = "host-begin-span";
    if (!vk_mem_span_capture_current(span, selected)) return VK_EXEC_HOST_INPUT_STALE;
    out->refusal = "host-begin-position";
    if (rc_registry_position_compare(position, vk_exec_resource_position())) return VK_EXEC_HOST_INPUT_STALE;
    const MemBackingCapture *mapping = span->host ? span->host : pool->host_backing;
    uint32_t mapped;
    const RcInterval *intervals = mem_backing_intervals(mapping, &mapped);
    out->refusal = "host-begin-mapping";
    RcInterval *clipped;
    uint32_t count = resource_host_interval_slice(intervals, mapped, address, size, &clipped);
    if (!count || (uint64_t)count * sizeof(RcSourceRange) > SIZE_MAX - sizeof(HostInputRead)) {
        free(clipped);
        return VK_EXEC_HOST_INPUT_CAPACITY;
    }
    out->mapping = clipped[0];
    out->refusal = "host-begin-capacity";
    HostInputRead *read = calloc(1, sizeof *read + count * sizeof *read->ranges);
    if (!read) { free(clipped); return VK_EXEC_HOST_INPUT_CAPACITY; }
    read->span = span;
    read->position = position;
    read->address = address;
    read->size = size;
    VkExecHostInputResult result=VK_EXEC_HOST_INPUT_READY;
    if (ordered) {
        out->refusal="host-ordered-coherent-placement";
        result=host_input_ordered_prepare(read,selected,clipped,count);
    }
    for (uint32_t i = 0; i < count; i++) read->ranges[i] = clipped[i].range;
    free(clipped);
    if (result!=VK_EXEC_HOST_INPUT_READY) {host_input_dispose(read);return result;}
    read->count = host_input_range_normalize(read->ranges, count);
    out->refusal = "host-begin-authority";
    result=host_input_result(rc_registry_gpu_capture_begin(rc_registry_runtime(),read->ranges,
                                                           read->count,&read->producer));
    if (result==VK_EXEC_HOST_INPUT_READY && !ordered) result=host_input_current(read,&out->refusal);
    if (result==VK_EXEC_HOST_INPUT_READY) {
        out->refusal="host-begin-pages";
        result=host_input_watch(read,0);
        if (result==VK_EXEC_HOST_INPUT_UNTRACKED && host_input_freeze(read)) result=VK_EXEC_HOST_INPUT_READY;
    }
    if (result==VK_EXEC_HOST_INPUT_READY && ordered) {
        out->refusal="host-ordered-original-authority";
        result=host_input_ordered_select(read,1);
        if (result==VK_EXEC_HOST_INPUT_READY) {
            out->refusal="host-ordered-visible-prefix";
            const VkSyncProgressResult progress=vk_exec_ordered_host_input_prefix(ordered,position,selected->buffer,
                read->ordered->demands[0].native.offset,size);
            if (progress==VK_SYNC_PROGRESS_COMPLETE) result=host_input_current(read,&out->refusal);
            else result=progress==VK_SYNC_PROGRESS_STOPPED || progress==VK_SYNC_PROGRESS_STOPPED_AFTER_CUT ?
                VK_EXEC_HOST_INPUT_STOPPED : VK_EXEC_HOST_INPUT_BLOCKED;
        }
    }
    if (result == VK_EXEC_HOST_INPUT_READY && !(ordered ? mem_backing_read_hold_no_fault_at(mapping,address,size) :
                                                mem_backing_read_hold_at(mapping,address,size))) {
        out->refusal = "host-begin-cpu-hold";
        result = vk_mem_span_capture_current(span, selected) ? VK_EXEC_HOST_INPUT_BLOCKED : VK_EXEC_HOST_INPUT_STALE;
    }
    if (result != VK_EXEC_HOST_INPUT_READY) { host_input_dispose(read); return result; }
    read->mapped = 1;
    out->bytes = (const uint8_t *)(uintptr_t)address;
    out->opaque = read;
    out->refusal = NULL;
    return VK_EXEC_HOST_INPUT_READY;
}

VkExecHostInputResult vk_exec_host_input_begin(const VkMemSpanCapture *span, const VkMemSpan *selected,
                                              uint64_t address, uint64_t size, RcRegistryPosition position,
                                              VkExecHostInputRead *out) {
    return host_input_begin(span, selected, address, size, position, out, 0, NULL);
}

VkExecHostInputResult vk_exec_host_input_begin_ordered(VkExec *ex, const VkMemSpanCapture *span,
    const VkMemSpan *selected, uint64_t address, uint64_t size, RcRegistryPosition position, VkExecHostInputRead *out) {
    if (!ex || !rc_registry_provisional_read_current(rc_registry_runtime(),position)) return VK_EXEC_HOST_INPUT_STALE;
    return host_input_begin(span,selected,address,size,position,out,0,ex);
}

VkExecHostInputResult vk_exec_host_input_finish(VkExecHostInputRead *owned) {
    if (!owned || !owned->opaque) return VK_EXEC_HOST_INPUT_UNSUPPORTED;
    HostInputRead *read = owned->opaque;
    owned->bytes = NULL;
    owned->opaque = NULL;
    host_input_release_mapping(read);
    VkExecHostInputResult result = read->protection ?
        (mem_protection_range_unchanged(read->address, read->size, read->protection) ?
            VK_EXEC_HOST_INPUT_READY : VK_EXEC_HOST_INPUT_CHANGED) : host_input_watch(read, 1);
    if (result == VK_EXEC_HOST_INPUT_READY) result = host_input_current(read, &owned->refusal);
    else owned->refusal = "host-input-pages-changed";
    host_input_dispose(read);
    return result;
}

void vk_exec_host_input_cancel(VkExecHostInputRead *owned) {
    if (!owned) return;
    host_input_dispose(owned->opaque);
    *owned = (VkExecHostInputRead){0};
}


VkExecHostInputResult vk_exec_host_pointer_read(uint64_t address, RcRegistryPosition position, uint64_t *value) {
    if (!value || !rc_registry_position_valid(position) || sizeof *value > UINT64_MAX - address)
        return VK_EXEC_HOST_INPUT_UNSUPPORTED;
    if (vk_sync_stopping() || vk_device_is_lost()) return VK_EXEC_HOST_INPUT_STOPPED;
    VkMemSpan selected;
    if (!vk_mem_lookup(address, sizeof *value, &selected)) return VK_EXEC_HOST_INPUT_ABSENT;
    if (nvn_gpu_only(address) || selected.alloc) return VK_EXEC_HOST_INPUT_UNSUPPORTED;
    RcRegistryResult captured;
    VkMemSpanCapture *span = span_capture_create(address, sizeof *value, &selected, 0, &captured);
    if (!span) return host_input_result(captured);
    // Translation consumes the logical pointer, not the raw bytes of a selected mirror descriptor. The
    // retained import still needs exact CPU visibility of every earlier overlapping GPU producer.
    VkExecHostInputRead input = {0};
    VkExecHostInputResult result = host_input_begin(span, &selected, address, sizeof *value, position, &input, 1, NULL);
    if (result == VK_EXEC_HOST_INPUT_READY) {
        uint64_t sampled;
        memcpy(&sampled, input.bytes, sizeof sampled);
        result = vk_exec_host_input_finish(&input);
        if (result == VK_EXEC_HOST_INPUT_READY) *value = sampled;
    }
    vk_mem_span_capture_release(&span);
    return result;
}

int vk_mem_span_buffer_location(const VkMemSpanCapture *capture, const VkMemSpan *selected,
                                RcRegistryBufferLocation *location) {
    if (!location) return 0;
    *location = (RcRegistryBufferLocation){0};
    if (!capture || !capture->native_input || !selected) return 0;
    plat_mutex_lock_shared(&pool_lock);
    int current = span_capture_match(capture, selected);
    if (current)
        *location = (RcRegistryBufferLocation){capture->provider->publication, capture->mirror_incarnation,
                                              selected->offset};
    plat_mutex_unlock_shared(&pool_lock);
    return current && location->allocation;
}

int vk_mem_span_gpa_donor(const VkMemSpanCapture *capture, const VkMemSpan *selected,
                          RcRegistryBufferLocation *location, VkBuffer *twin) {
    *location = (RcRegistryBufferLocation){0};
    *twin = VK_NULL_HANDLE;
    if (!capture || !selected || !capture->native_input || !capture->mirror_incarnation) return 0;
    plat_mutex_lock_shared(&pool_lock);
    const PoolRec *owner = capture->provider;
    int current = owner->kind == POOL_IMPORT && !owner->smap && owner->buf && owner->publication &&
        owner->acc_sub == msub && span_capture_match(capture, selected);
    if (current) {
        *location = (RcRegistryBufferLocation){owner->publication, capture->mirror_incarnation, selected->offset};
        *twin = owner->buf;
    }
    plat_mutex_unlock_shared(&pool_lock);
    return current;
}

// The exact word is provenance for an unchanged descriptor. Its capture owner retains both native buffers.
typedef struct {
    VkBuffer mirror, import;
    RcRegistryBufferSpan native;
    uint64_t descriptor_size;
    RcRegistryPosition read;
} SelectedMirrorWord;

static RcRegistryResult selected_mirror_access_locked(const VkMemSpanCapture *capture, const VkMemSpan *selected,
    uint64_t address, uint64_t size, uint64_t descriptor_size, RcRegistryPosition read, SelectedMirrorWord *out) {
    if (!capture->host || !capture->native_input || !capture->mirror_incarnation || descriptor_size < 4 ||
        descriptor_size > capture->size || descriptor_size > selected->avail || !size || size > descriptor_size ||
        ((address | size) & 3) || size > UINT64_MAX - address || address < capture->address ||
        address - capture->address > descriptor_size - size || nvn_gpu_only(address)) return RC_REGISTRY_UNSUPPORTED;
    const PoolRec *pool = capture->provider;
    if (!span_capture_match(capture, selected)) return RC_REGISTRY_STALE;
    if (!pool || pool != capture->window || pool->kind != POOL_IMPORT || pool->smap ||
        !pool->gpa_host_coherent || !pool->buf || pool->buf == selected->buffer || !pool->publication ||
        pool->acc_sub != msub)
        return RC_REGISTRY_UNSUPPORTED;
    const uint64_t delta = address - capture->address, pool_size = pool->import_hi - pool->import_lo;
    if (delta > UINT64_MAX - selected->offset || selected->offset + delta > pool_size ||
        size > pool_size - selected->offset - delta || ((selected->offset + delta) & 3))
        return RC_REGISTRY_INVALID;

    uint32_t count;
    const RcInterval *mapping = mem_backing_intervals(capture->host, &count);
    for (uint32_t i = 0; i < count; ++i) {
        const RcInterval part = mapping[i];
        if (address < part.address || address - part.address >= part.range.size) continue;
        const uint64_t local = address - part.address;
        if (!part.mapping || part.range.backing == RC_GUEST_COMPATIBILITY_BACKING ||
            part.range.size - local < size || local > UINT64_MAX - part.range.offset ||
            part.range.offset + local > UINT64_MAX - size || ((part.range.offset + local) & 3))
            return RC_REGISTRY_UNSUPPORTED;
        *out = (SelectedMirrorWord){.mirror = selected->buffer, .import = pool->buf,
            .native = {{address, part.mapping, {part.range.backing, part.range.offset + local, size}},
                       {pool->publication, capture->mirror_incarnation, selected->offset + delta}},
            .descriptor_size = descriptor_size, .read = read};
        return RC_REGISTRY_OK;
    }
    return RC_REGISTRY_UNSUPPORTED;
}

static RcRegistryResult selected_mirror_word_locked(const VkMemSpanCapture *capture, const VkMemSpan *selected,
    uint64_t word, uint64_t descriptor_size, RcRegistryPosition read, SelectedMirrorWord *out) {
    return selected_mirror_access_locked(capture, selected, word, 4, descriptor_size, read, out);
}

static RcRegistryResult selected_mirror_access(const VkMemSpanCapture *capture, const VkMemSpan *selected,
    uint64_t address, uint64_t size, uint64_t descriptor_size, RcRegistryPosition read, SelectedMirrorWord *out) {
    if (!out) return RC_REGISTRY_INVALID;
    *out = (SelectedMirrorWord){0};
    if (!capture || !selected || !rc_registry_position_valid(read)) return RC_REGISTRY_INVALID;
    if (vk_exec_shared() || rc_registry_position_compare(read, vk_exec_resource_position()) ||
        !rc_registry_provisional_read_current(rc_registry_runtime(), read)) return RC_REGISTRY_STALE;
    plat_mutex_lock_shared(&pool_lock);
    RcRegistryResult result = selected_mirror_access_locked(capture, selected, address, size, descriptor_size, read, out);
    plat_mutex_unlock_shared(&pool_lock);
    return result;
}

static RcRegistryResult selected_mirror_word(const VkMemSpanCapture *capture, const VkMemSpan *selected,
    uint64_t word, uint64_t descriptor_size, RcRegistryPosition read, SelectedMirrorWord *out) {
    return selected_mirror_access(capture, selected, word, 4, descriptor_size, read, out);
}

RcRegistryResult vk_mem_span_buffer_fragments_with_payloads(const VkMemSpanCapture *capture, const VkMemSpan *selected,
    uint64_t address, uint64_t size, RcRegistryGpuCapture token, RcRegistryPosition read,
    const VkMemBufferInlineProof *inline_proofs, uint32_t inline_count,
    RcRegistryBufferFragment *out, RcRegistryBufferPayload **payloads, uint32_t capacity, uint32_t *fragment_count) {
    if (fragment_count) *fragment_count = 0;
    if (!fragment_count || !capture || !selected || (capacity && !out) || (inline_count && !inline_proofs) ||
        !size || size > UINT64_MAX - address ||
        address < capture->address || address - capture->address > capture->size ||
        size > capture->size - (address - capture->address) || rc_registry_position_compare(read, vk_exec_resource_position()))
        return RC_REGISTRY_INVALID;
    for (uint32_t i = 0; i < inline_count; ++i)
        if (!inline_proofs[i].size || inline_proofs[i].size > UINT64_MAX - inline_proofs[i].address ||
            inline_proofs[i].address < address || inline_proofs[i].address + inline_proofs[i].size > address + size ||
            inline_proofs[i].output.scene != read.scene ||
            rc_registry_position_compare(inline_proofs[i].output, read) >= 0) return RC_REGISTRY_INVALID;
    RcRegistryBufferLocation base;
    if (!vk_mem_span_buffer_location(capture, selected, &base)) return RC_REGISTRY_STALE;
    RcInterval *intervals = NULL;
    uint32_t count = vk_mem_span_input_intervals(capture, address, size, &intervals);
    RcRegistryBufferSpan *demand = count ? calloc(count, sizeof *demand) : NULL;
    if (!demand) {
        free(intervals);
        return count ? RC_REGISTRY_CAPACITY : RC_REGISTRY_UNSUPPORTED;
    }
    RcRegistryResult result = RC_REGISTRY_OK;
    uint64_t covered = address;
    const uint64_t end = address + size;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t delta = intervals[i].address - capture->address;
        if (intervals[i].address != covered || intervals[i].range.size > end - covered ||
            delta > UINT64_MAX - base.offset) { result = RC_REGISTRY_INVALID; break; }
        covered += intervals[i].range.size;
        demand[i] = (RcRegistryBufferSpan){intervals[i], {base.allocation, base.incarnation, base.offset + delta}};
    }
    if (covered != end) result = RC_REGISTRY_INVALID;
    uint32_t used = 0;
    RcRegistry *registry = rc_registry_runtime();
    if (result == RC_REGISTRY_OK)
        result = payloads ? rc_registry_gpu_capture_buffer_fragments_with_payloads(registry, token, demand,
            count, read, out, payloads, capacity, &used) :
            rc_registry_gpu_capture_buffer_fragments(registry, token, demand, count, read, out, capacity, &used);
    free(demand);
    free(intervals);
    if (result != RC_REGISTRY_OK) return result;
    if (!vk_mem_span_capture_current(capture, selected)) {
        for (uint32_t i = 0; payloads && i < used; i++) {
            if (payloads[i]) payloads[i]->release(payloads[i]);
            payloads[i] = NULL;
        }
        return RC_REGISTRY_STALE;
    }
    if (used && !(nvn_gpu_only(address) && capture->provider->kind == POOL_PHYS)) {
        // A mirrored producer and donor share the same synchronization scene. A host-writable import can
        // only be copied from earlier work recorded into this still-unhanded native part; later host stores
        // could otherwise change an already executed output before the copy.
        int local_mirror = 0;
        if (base.incarnation) {
            plat_mutex_lock_shared(&pool_lock);
            local_mirror = span_capture_match(capture, selected) && capture->provider->acc_sub == msub;
            plat_mutex_unlock_shared(&pool_lock);
            if (!local_mirror) {
                for (uint32_t i = 0; payloads && i < used; i++) {
                    if (payloads[i]) payloads[i]->release(payloads[i]);
                    payloads[i] = NULL;
                }
                return RC_REGISTRY_STALE;
            }
        }
        for (uint32_t i = 0; i < used && result == RC_REGISTRY_OK; i++) {
            const RcRegistryBufferFragment *fragment = &out[i];
            RcRegistryPosition output = fragment->output;
            if (local_mirror && output.scene == read.scene) continue;
            if (output.scene == read.scene && rc_registry_position_open(registry, output)) continue;
            if (local_mirror) {
                result = RC_REGISTRY_UNSUPPORTED;
                break;
            }
            uint64_t cursor = fragment->interval.address;
            uint64_t end = cursor + fragment->interval.range.size;
            while (cursor < end) {
                uint64_t covered = cursor;
                for (uint32_t j = 0; j < inline_count; ++j) {
                    const VkMemBufferInlineProof *proof = &inline_proofs[j];
                    if (proof->address <= cursor && proof->address + proof->size > covered &&
                        !rc_registry_position_compare(proof->output, output))
                        covered = proof->address + proof->size;
                }
                if (covered == cursor) {
                    result = RC_REGISTRY_UNSUPPORTED;
                    break;
                }
                cursor = covered < end ? covered : end;
            }
        }
    }
    // A transfer-policy refusal still exposes the validated winners for clipped inline proof.
    // A caller must obtain RC_REGISTRY_OK before using any of them as native copy sources.
    *fragment_count = used;
    return result;
}

RcRegistryResult vk_mem_span_buffer_fragments(const VkMemSpanCapture *capture, const VkMemSpan *selected,
    uint64_t address, uint64_t size, RcRegistryGpuCapture token, RcRegistryPosition read,
    const VkMemBufferInlineProof *inline_proofs, uint32_t inline_count,
    RcRegistryBufferFragment *out, uint32_t capacity, uint32_t *fragment_count) {
    return vk_mem_span_buffer_fragments_with_payloads(capture, selected, address, size, token, read,
        inline_proofs, inline_count, out, NULL, capacity, fragment_count);
}

// Recorded before publication. A successful bind cannot begin anywhere in the initialized range.
static int physical_initial_fill_covers(const PoolRec *provider, uint64_t offset, uint64_t size) {
    return provider->physical_origin_upload && offset < provider->physical_origin_bytes &&
        size <= provider->physical_origin_bytes - offset;
}

// pool_lock exclusive. Shared by the retained and compatibility raw span selections.
static int physical_use_preserve_locked(PoolRec *, uint64_t offset, uint64_t size);

uint32_t vk_mem_span_capture_device_intervals(const VkMemSpanCapture *capture, const VkMemSpan *selected,
                                              uint64_t address, uint64_t size, RcInterval **out) {
    *out = NULL;
    if (!capture || !capture->device_count || !vk_mem_span_capture_current(capture, selected)) return 0;
    return resource_host_interval_slice(capture->device_intervals, capture->device_count, address, size, out);
}

uint32_t vk_mem_span_input_intervals(const VkMemSpanCapture *span, uint64_t address, uint64_t size,
                                     RcInterval **out) {
    *out = NULL;
    if (!span || !span->native_input || !size || address < span->address ||
        address - span->address >= span->size || size > span->size - (address - span->address)) return 0;
    const PoolRec *pool = span->window;
    if (pool->kind != POOL_IMPORT) {
        uint32_t count = resource_host_interval_slice(span->device_intervals, span->device_count, address, size, out);
        if (!count) return 0;
        const uint64_t offset = span->selected.alloc_offset + (address - span->address);
        if (physical_initial_fill_covers(span->provider, offset, size)) return count;
        plat_mutex_lock(&pool_lock);
        const int ready = span_capture_match(span, &span->selected) &&
            physical_use_preserve_locked(span->provider, offset, size);
        plat_mutex_unlock(&pool_lock);
        if (ready) return count;
        free(*out);
        *out = NULL;
        return 0;
    }
    const MemBackingCapture *mapping = span->host ? span->host : pool->host_backing;
    uint32_t mapped;
    const RcInterval *intervals = mem_backing_intervals(mapping, &mapped);
    return resource_host_interval_slice(intervals, mapped, address, size, out);
}

int vk_mem_span_input_cpu_visible(const VkMemSpanCapture *span) {
    const PoolRec *pool = span->window;
    return pool->kind == POOL_IMPORT && !pool->smap && !span->mirror_incarnation && !nvn_gpu_only(span->address);
}

static uint32_t copy_pool_host_intervals(const PoolRec *pool, uint64_t addr, uint64_t size, RcInterval **out) {
    *out = NULL;
    if (!pool || !size || addr < pool->import_lo || addr >= pool->import_hi || size > pool->import_hi - addr) return 0;
    uint32_t count;
    const RcInterval *source = mem_backing_intervals(pool->host_backing, &count);
    return resource_host_interval_slice(source, count, addr, size, out);
}

uint32_t vk_mem_host_intervals(const VkMemSpan *span, uint64_t addr, uint64_t size, RcInterval **out) {
    *out = NULL;
    if (!span || !span->pool || !span->buffer || !size || size > span->avail) return 0;
    const PoolRec *pool = span->pool;
    if (pool->kind != POOL_IMPORT || addr < pool->import_lo || span->offset != addr - pool->import_lo) return 0;
    // The native span retains the selected pool via the recording's existing native-use obligation.
    // Use its immutable import-time host mapping, even if today's address index points elsewhere.
    return copy_pool_host_intervals(pool, addr, size, out);
}

static uint32_t device_intervals(const VkMemSpan *span, uint64_t addr, uint64_t size, RcInterval **out, int writer) {
    *out = NULL;
    if (!span || !span->alloc || !size || size > span->avail) return 0;
    RcBackingCapture *capture = NULL;
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *entry = pidx_find(&pool_index, addr, size);
    const PoolRec *pool = entry ? entry->buf : NULL;
    if (pool && pool->map.id)
        rc_backing_capture(pool_backings, pool->map, addr, size, RC_BACKING_REQUIRE_MAPPED, &capture);
    uint32_t count;
    const RcBackingSegment *segments = rc_backing_capture_segments(capture, &count);
    const PoolRec *first = count ? (const PoolRec *)segments[0].provider : NULL;
    for (uint32_t i = 0; i < count; i++) {
        const PoolRec *provider = (const PoolRec *)segments[i].provider;
        if (!provider || provider != first || provider->alloc != span->alloc || provider->buf != span->buffer ||
            segments[i].range.offset != span->alloc_offset + (segments[i].address - addr)) {
            count = 0;
            break;
        }
    }
    RcInterval *intervals = count ? calloc(count, sizeof *intervals) : NULL;
    if (intervals) {
        for (uint32_t i = 0; i < count; i++)
            intervals[i] = (RcInterval){segments[i].address, segments[i].mapping, segments[i].range};
        *out = intervals;
    }
    const int preserve = writer && intervals && !physical_initial_fill_covers(first, span->alloc_offset, size);
    rc_backing_capture_release(&capture);
    plat_mutex_unlock_shared(&pool_lock);
    if (preserve) {
        plat_mutex_lock(&pool_lock);
        const PoolRec *window = NULL;
        PoolRec *provider = pool_image_alias_provider_locked(addr, size, span, &window);
        const int ready = provider &&
            rc_backing_map_intervals_current(pool_backings, window->map, intervals, count, addr, size) &&
            physical_use_preserve_locked(provider, span->alloc_offset, size);
        plat_mutex_unlock(&pool_lock);
        if (!ready) {
            free(intervals);
            *out = NULL;
            return 0;
        }
    }
    return intervals ? count : 0;
}

uint32_t vk_mem_device_intervals(const VkMemSpan *span, uint64_t addr, uint64_t size, RcInterval **out) {
    return device_intervals(span, addr, size, out, 0);
}

uint32_t vk_mem_device_writer_intervals(const VkMemSpan *span, uint64_t addr, uint64_t size, RcInterval **out) {
    return device_intervals(span, addr, size, out, 1);
}

int vk_mem_cpu_visible_span(const VkMemSpan *sp, uint64_t addr, uint64_t size) {
    if (!sp || !sp->pool || !sp->buffer || sp->alloc || !size || size > sp->avail || nvn_gpu_only(addr)) return 0;
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *entry = pidx_find(&pool_index, addr, size);
    const PoolRec *r = entry ? entry->buf : NULL;
    int visible = r && r == sp->pool && r->kind == POOL_IMPORT && !r->smap && sp->buffer == r->buf &&
        addr >= r->import_lo && addr < r->import_hi && size <= r->import_hi - addr &&
        sp->offset == addr - r->import_lo;
    plat_mutex_unlock_shared(&pool_lock);
    return visible;
}

VkBuffer vk_mem_guest_twin(const VkMemSpan *sp) {
    const PoolRec *r = sp->pool;
    return r && r->mstate == MS_ON && r->twin && sp->buffer == r->mbuf ? r->buf : VK_NULL_HANDLE;
}

int vk_mem_mirror_probe(uint64_t addr) {
    VkMemSpan sp;
    if (!vk_mirror_on || !vk_mem_resolve(addr, 1, &sp) || !sp.pool) return 0;
    mepoch = vk_texture_epoch();
    vk_mem_use(&sp);
    return ((PoolRec *)sp.pool)->mstate == MS_ON;
}

// ---- write-back merge
// A storage buffer a shader writes whole records of does not
// go back to guest memory as a plain copy. The CPU may write fields of the same records while the GPU runs: NX2navi's
// particle emitters write a new particle's whole record into a free slot of a buffer that the simulation shaders update
// (the waterfall splashes and mist, 2026-11-02), and a plain copy puts the mirror's stale copy of those words back over
// them. Instead:
// - record time (gpu_wrote from vk_mem_gpu_wrote_shader): each distinct range of the submission gets room in a snapshot
//   chunk (snap_add);
// - each part's prologue, after the CPU -> mirror copies: the mirror's words of the ranges the part added are copied
//   into their snapshots (snap_copies), the state before the part's GPU work. Words of an earlier part's writes that a
//   later range overlaps are covered by the earlier range's own snapshot;
// - vk_mem_mirror_end: shaders/wb_merge.comp writes each 32-bit word that differs between the mirror and the snapshot
//   into the pool's imported buffer (wb_merge_record), one dispatch per pool and group of ranges.
// Words the GPU left unchanged keep the CPU's value; a word both changed takes the GPU's, as a plain copy would.
// NX2NAVI_WB_MERGE=0 restores the plain copies; so does a pipeline or allocation failure.
#define SNAP_CHUNK (1u << 20)
#define MERGE_SPAN (64ull << 20)   // the bytes of one pool a merge dispatch binds at most
#define MERGE_WORDS (65535u * 64u) // the words one merge dispatch covers at most
// Larger written ranges stay plain copies (NX2NAVI_WB_MERGE_MAX, MiB, default 32). A plain copy writes the whole range
// over PCIe at the end of each submission even when the shader changed a few words; a merge copies it within video
// memory and writes only the words that changed. With a 4 MiB limit the field ran about 46 MiB of plain write-back a
// frame in ranges of about 5.5 MiB, just above it; at 16 the same outdoor route went from a median 47.5 to 67.3 fps and
// from 11.3 to 7.2 ms of GPU time a frame (RTX 4080 SUPER, 120 fps limit, 2026-11-08). Kokiri Forest writes seven
// storage buffer ranges of 4-15 MiB a frame; at 32, merged in MERGE_PIECE parts so that no dispatch covers more than
// MERGE_WORDS, the plain copies of about 4 ms of GPU time a frame on PCIe 3.0 went (RTX 4090, 2026-11-09).
#define MERGE_PIECE (4ull << 20)
#define MERGE_MAX_LIMIT (256ull << 20)   // the snapshot pool's largest size (snap_pool_cap)
static uint64_t merge_max(void) {
    static uint64_t m;
    if (!m) {
        const char *e = cfg_get("NX2NAVI_WB_MERGE_MAX");
        uint64_t mib = e && *e ? (uint64_t)strtoull(e, NULL, 10) : 32;
        m = mib << 20;
        if (!m) m = 1;
        if (m > MERGE_MAX_LIMIT) m = MERGE_MAX_LIMIT;
        rt_log("mem: write-back merge covers ranges up to %.1f MiB (NX2NAVI_WB_MERGE_MAX)", m / 1048576.0);
    }
    return m;
}
typedef struct { PoolRec *r; uint64_t off, size, soff; int chunk; } SnapRange;
// A marked word is omitted from the ordinary tail and retains every original snapshot predicate.
typedef struct { PoolRec *r; uint64_t off; } SnapWord;
typedef struct { size_t word; SnapRange pre; } SnapWordPreimage;

static int snapshot_partition(const SnapRange *ranges, size_t count, const SnapWord *words, size_t nwords,
                              SnapRange *ordinary, size_t capacity, size_t *ordinary_count,
                              SnapWordPreimage *preimages, size_t pre_capacity, size_t *preimage_count) {
    *ordinary_count = *preimage_count = 0;
    for (size_t i = 0; i < nwords; ++i) {
        const SnapWord w = words[i];
        if (!w.r || (w.off & 3) || w.off > UINT64_MAX - 4) return 0;
        if (i && ((uintptr_t)w.r < (uintptr_t)words[i - 1].r ||
            (w.r == words[i - 1].r && w.off <= words[i - 1].off))) return 0;
    }
    for (int emit = 0; emit < 2; ++emit) {
        size_t np = 0, no = 0;
        for (size_t i = 0; i < count; ++i) {
            const SnapRange r = ranges[i];
            if (!r.r || !r.size || ((r.off | r.size | r.soff) & 3) ||
                r.size > UINT64_MAX - r.off || r.size > UINT64_MAX - r.soff || r.chunk < 0) return 0;
            uint64_t start = r.off, end = r.off + r.size;
            for (size_t j = 0; j < nwords; ++j) {
                const SnapWord w = words[j];
                if (w.r != r.r || w.off < r.off || w.off >= end) continue;
                if (no == SIZE_MAX || np == SIZE_MAX) return 0;
                if (start < w.off) {
                    if (emit) ordinary[no] = (SnapRange){r.r, start, w.off - start, r.soff + (start - r.off), r.chunk};
                    ++no;
                }
                if (emit) preimages[np] = (SnapWordPreimage){j, {r.r, w.off, 4, r.soff + (w.off - r.off), r.chunk}};
                ++np;
                start = w.off + 4;
            }
            if (start < end) {
                if (no == SIZE_MAX) return 0;
                if (emit) ordinary[no] = (SnapRange){r.r, start, end - start, r.soff + (start - r.off), r.chunk};
                ++no;
            }
        }
        if (no > capacity || np > pre_capacity || (no && !ordinary) || (np && !preimages)) return 0;
        *ordinary_count = no;
        *preimage_count = np;
    }
    return 1;
}
typedef struct { VkBuffer buf; VmaAllocation alloc; uint64_t size, used; int host; } SnapChunk;
typedef struct { int n; SnapChunk c[]; } SnapFree;   // a submission's chunks and range lists, pooled at its fence
static SnapRange *snl;              // replay: this submission's merged ranges; [0, snl_copied) have their snapshot copy
static int nsnl, csnl, snl_copied;
static SnapChunk *chunks;           // replay: this submission's snapshot chunks and merge range lists
static int nchunks, cchunks, nchunks_transferred;
static uint64_t snap_completion_fence;
// Chunks a finished submission used (snap_free) wait here for chunk_new, so a frame with merges creates no buffer once
// the pool holds the frames in flight: a VMA allocation that needs a new memory block costs 8-11 ms on the replay
// thread. Chunk sizes are powers of two so that one fits the next request; what the pool cannot hold is destroyed. The
// pool holds about three frames of Kokiri Forest's snapshots (65 MiB a frame); at 16 MiB it held less than one frame's
// and two buffers a frame were created and destroyed again (2026-11-09).
#define SNAP_LIST_MIN (64u << 10)
#define SNAP_POOL_N 128
// The pool holds up to 1/64 of the VRAM budget, between 64 and 256 MiB: room for merges of multi-MiB ranges across the
// submissions in flight. Its chunks prefer device memory and are not evictable, so a small card keeps less of them.
static uint64_t snap_pool_cap(void) {
    static uint64_t cap;
    if (!cap) {
        uint64_t used = 0, budget = 0;
        int exact = 0;
        uint64_t c = vk_mem_vram(&used, &budget, &exact) && budget ? budget / 64 : (64ull << 20);
        if (c < (64ull << 20)) c = 64ull << 20;
        if (c > (256ull << 20)) c = 256ull << 20;
        cap = c;
        rt_log("mem: write-back snapshot pool keeps up to %llu MiB", (unsigned long long)(cap >> 20));
    }
    return cap;
}
static PlatMutex snap_pool_lock = PLAT_MUTEX_INIT;
static SnapChunk snap_pool[SNAP_POOL_N];
static int nsnap_pool;
static uint64_t snap_pool_bytes;
static int merge_on = -1;
static struct {
    int tried, ok;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipe;
} WM;

static int merge_init(VkCtx *c) {
    WM.tried = 1;
    WM.push = (PFN_vkCmdPushDescriptorSetKHR)vk_dpa(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding b[5];
    for (int i = 0; i < 5; i++)
        b[i] = (VkDescriptorSetLayoutBinding){.binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 5, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof wb_merge_spv,
                                   .pCode = wb_merge_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    if (!WM.push || vkCreateDescriptorSetLayout(c->device, &dci, NULL, &WM.dsl) != VK_SUCCESS) goto fail;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &WM.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &WM.layout) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS)
        goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = WM.layout};
    VkResult res = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &WM.pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (res != VK_SUCCESS) goto fail;
    return WM.ok = 1;
fail:
    rt_log("mem: write-back merge off: its pipeline could not be created");
    return 0;
}

struct VkMemNativeRmwWord {
    struct VkMemNativeRmwWord *next;
    SelectedMirrorWord source;
    VkMemSpan descriptor;
    VkMemSpanCapture *provider;
    RcBudgetTicket host;
    VkBuffer marker;
    VmaAllocation allocation;
    volatile uint32_t *mapped;
    RcRegistryWrite tickets[2];
    VkCommandBuffer body, bound;
    struct NativeRmwTail *workspace;
    uint32_t references;
    RcRegistryPosition output;
    uint64_t marker_generation;
    int physical_recorded, tail_recorded;
    MemLazyWriteRange lazy_write;
};

enum { RMW_WORD_LIMIT = 16, RMW_SNAPSHOT_LIMIT = 4096, RMW_PREIMAGE_LIMIT = 256 };
typedef struct NativeRmwTail {
    size_t count, ordinary_count, preimage_count;
    SnapWord words[RMW_WORD_LIMIT];
    VkMemNativeRmwWord *owners[RMW_WORD_LIMIT];
    SnapRange ordinary[RMW_SNAPSHOT_LIMIT + RMW_PREIMAGE_LIMIT];
    SnapWordPreimage preimages[RMW_PREIMAGE_LIMIT];
} NativeRmwTail;

static int native_rmw_tail_supported(const PoolRec *, uint64_t);
static int native_rmw_word_bitmap_refuses(const PoolRec *, const SelectedMirrorWord *, VkCommandBuffer);

static struct {
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout set;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkMemNativeRmwWord *scene;
    NativeRmwTail *workspace;
    uint64_t generation;
    unsigned word_count;
} RW;

static void native_rmw_pipeline_free(VkCtx *c, VkDescriptorSetLayout *set, VkPipelineLayout *layout, VkPipeline *pipeline) {
    if (*pipeline) vkDestroyPipeline(c->device, *pipeline, NULL);
    if (*layout) vkDestroyPipelineLayout(c->device, *layout, NULL);
    if (*set) vkDestroyDescriptorSetLayout(c->device, *set, NULL);
    *pipeline = VK_NULL_HANDLE;
    *layout = VK_NULL_HANDLE;
    *set = VK_NULL_HANDLE;
}

static int native_rmw_pipeline_build(VkCtx *c, PFN_vkCmdPushDescriptorSetKHR *push, VkDescriptorSetLayout *set,
    VkPipelineLayout *layout, VkPipeline *pipeline, const uint32_t *code, size_t bytes, uint32_t push_bytes) {
    if (*pipeline) return 1;
    *push = (PFN_vkCmdPushDescriptorSetKHR)vk_dpa(c->device, "vkCmdPushDescriptorSetKHR");
    if (!*push) return 0;
    VkDescriptorSetLayoutBinding binding[4];
    for (uint32_t i = 0; i < 4; ++i)
        binding[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    const VkDescriptorSetLayoutCreateInfo set_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 4, .pBindings = binding};
    const VkPushConstantRange constants = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    const VkPipelineLayoutCreateInfo layout_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = set, .pushConstantRangeCount = 1, .pPushConstantRanges = &constants};
    const VkShaderModuleCreateInfo module_info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = bytes, .pCode = code};
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(c->device, &set_info, NULL, set) != VK_SUCCESS ||
        vkCreatePipelineLayout(c->device, &layout_info, NULL, layout) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &module_info, NULL, &module) != VK_SUCCESS) {
        native_rmw_pipeline_free(c, set, layout, pipeline);
        return 0;
    }
    const VkComputePipelineCreateInfo pipeline_info = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = module, .pName = "main"}, .layout = *layout};
    VkResult result = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, pipeline);
    vkDestroyShaderModule(c->device, module, NULL);
    if (result != VK_SUCCESS) native_rmw_pipeline_free(c, set, layout, pipeline);
    return result == VK_SUCCESS;
}

static void native_rmw_pipeline_destroy(VkCtx *c) {
    native_rmw_pipeline_free(c, &RW.set, &RW.layout, &RW.pipeline);
}

static int native_rmw_pipeline_ready(VkCtx *c) {
    return native_rmw_pipeline_build(c, &RW.push, &RW.set, &RW.layout, &RW.pipeline,
        native_rmw_word_spv, sizeof native_rmw_word_spv, 4 * sizeof(uint32_t));
}

static void native_rmw_word_destroy(VkMemNativeRmwWord *word) {
    mem_lazy_write_release(&word->lazy_write);
    if (word->marker) resource_core_vma_destroy_buffer(vk_ctx()->vma, word->marker, word->allocation);
    vk_mem_span_capture_release(&word->provider);
    RcBudgetTicket host = word->host;
    if (host.token.ledger && resource_core_budget_retire(&host) != RC_ADMISSION_OK)
        rt_fatal("vk: selected mesh word receipt lost its capacity charge");
    free(word);
    if (host.token.ledger && resource_core_budget_destroyed(&host) != RC_ADMISSION_OK)
        rt_fatal("vk: selected mesh word receipt lost its retired capacity charge");
}

static void native_rmw_word_release(void *opaque) {
    VkMemNativeRmwWord *word = opaque;
    if (__atomic_fetch_sub(&word->references, 1, __ATOMIC_ACQ_REL) == 1)
        native_rmw_word_destroy(word);
}

static void native_rmw_word_binding_canceled(void *opaque, int may_have_executed) {
    (void)may_have_executed;
    native_rmw_word_release(opaque);
}

static void native_rmw_word_record_physical(VkMemNativeRmwWord *word) {
    if (word->physical_recorded) return;
    word->physical_recorded = 1;
    __atomic_fetch_add(&word->references, 1, __ATOMIC_RELAXED);
    // A barrier or descriptor can expose native memory before the draw. Keep that owner separately.
    vk_completion_add_owned(word->source.read.scene, native_rmw_word_release, native_rmw_word_binding_canceled, word);
}

int vk_mem_native_rmw_word_bind(VkMemNativeRmwWord *word, VkCommandBuffer body) {
    if (!word || !body || word->body || !word->tickets[0].id || !word->tickets[1].id) return 0;
    if (word->bound) return word->bound == body;
    if (native_rmw_word_bitmap_refuses(word->descriptor.pool, &word->source, body)) return 0;
    word->bound = body;
    native_rmw_word_record_physical(word);
    return 1;
}

static RcSourceRange native_rmw_guest_range(const VkMemNativeRmwWord *word) {
    return (RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING, word->source.native.interval.address, 4};
}

static void native_rmw_word_cancel_tickets(VkMemNativeRmwWord *word, int possibly_executed) {
    RcRegistry *registry = rc_registry_runtime();
    for (unsigned i = 0; i < 2; ++i) {
        if (!word->tickets[i].id) continue;
        if (rc_registry_write_abandon(registry, word->tickets[i], possibly_executed) != RC_REGISTRY_OK)
            rc_registry_fail_closed(registry);
        word->tickets[i] = (RcRegistryWrite){0};
    }
}

void vk_mem_native_rmw_word_cancel(VkMemNativeRmwWord **owned) {
    if (!owned || !*owned) return;
    VkMemNativeRmwWord *word = *owned;
    *owned = NULL;
    native_rmw_word_cancel_tickets(word, 0);
    native_rmw_word_release(word);
}

static void native_rmw_word_cancel_submitted(void *opaque, int possibly_executed) {
    VkMemNativeRmwWord *word = opaque;
    // A native prefix may have been handed even when the enclosing scene tail was canceled.
    possibly_executed |= rc_registry_position_handed(rc_registry_runtime(), word->output);
    native_rmw_word_cancel_tickets(word, possibly_executed);
    native_rmw_word_release(word);
}

static void native_rmw_word_completed(void *opaque) {
    VkMemNativeRmwWord *word = opaque;
    RcRegistry *registry = rc_registry_runtime();
    if (!word->tail_recorded || vmaInvalidateAllocation(vk_ctx()->vma, word->allocation, 0, 2 * sizeof(uint32_t)) != VK_SUCCESS ||
        word->mapped[0] > 1 || !rc_registry_position_handed(registry, word->output)) {
        native_rmw_word_cancel_submitted(word, 1);
        return;
    }
    const RcSourceRange written[2] = {word->source.native.interval.range, native_rmw_guest_range(word)};
    const int executed = word->mapped[0] == 1;
    const int visible = vk_mem_span_capture_current(word->provider, &word->descriptor);
    RcRegistryWriteCompletion completion[2] = {0};
    for (unsigned i = 0; i < 2; ++i) {
        if (rc_registry_write_submit(registry, word->tickets[i], word->output.scene) != RC_REGISTRY_OK) {
            rc_registry_fail_closed(registry);
            native_rmw_word_cancel_submitted(word, 1);
            return;
        }
        completion[i] = (RcRegistryWriteCompletion){.ticket = word->tickets[i], .written = &written[i],
            .written_count = executed, .uncertain = &written[i], .uncertain_count = executed && !visible};
    }
    if (rc_registry_write_complete_extents(registry, completion, 2) != RC_REGISTRY_OK) {
        rc_registry_fail_closed(registry);
        native_rmw_word_cancel_submitted(word, 1);
        return;
    }
    if (executed && visible)
        vk_mem_buffer_write_completed(word->descriptor.pool, word->source.mirror,
            word->source.native.interval.address, 4, 1);
    if (rt_knob("GRAPHICS_PREP_TRACE", "0"))
        rt_log("vk: native mesh word complete output %llu/%llu address %#llx generation %llu executed %d visible %d",
            (unsigned long long)word->output.scene, (unsigned long long)word->output.ordinal,
            (unsigned long long)word->source.native.interval.address, (unsigned long long)word->marker_generation,
            executed, visible);
    native_rmw_word_release(word);
}

static VkResult native_rmw_marker_allocate(VkCtx *c, size_t host_bytes, VkDeviceSize marker_bytes, RcBudgetTicket *host,
    RcVmaDependency *dependency, VkBuffer *marker, VmaAllocation *allocation, VmaAllocationInfo *info) {
    const VkBufferCreateInfo buffer = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = marker_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    const VmaAllocationCreateInfo requested = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationCreateInfo placement = requested;
    if (dependency) {
        const RcVmaRequest query = {.producer = RC_BUDGET_BUFFER, .required = 1, .dependency = dependency};
        RcAdmissionVector peak;
        VkResult result = resource_core_vma_buffer_peak(&query, &buffer, &requested, &placement, &peak);
        if (result != VK_SUCCESS) return result;
        // Retry readiness must include the host receipt as well as native growth or reuse.
        RcAdmissionVector reuse = dependency->reuse;
        const uint32_t host_domains[] = {RC_BUDGET_RAM, RC_BUDGET_COMMIT};
        for (size_t i = 0; i < sizeof host_domains / sizeof host_domains[0]; ++i) {
            const uint32_t domain = host_domains[i];
            if (host_bytes > UINT64_MAX - peak.amount[domain] || host_bytes > UINT64_MAX - reuse.amount[domain])
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            peak.amount[domain] += host_bytes;
            reuse.amount[domain] += host_bytes;
        }
        dependency->peak = peak;
        dependency->reuse = reuse;
        resource_core_budget_readiness(1, &dependency->before);
    }

    // VMA accepts a host companion on the allocation request, with placement dependency queried separately.
    RcAdmissionVector attempted = {0};
    const RcVmaRequest request = {.producer = RC_BUDGET_BUFFER, .required = 1,
        .host_bytes = host_bytes, .host_ticket = host, .refused_peak = &attempted};
    RcAdmissionStatus admitted = RC_ADMISSION_OK;
    VkResult result = resource_core_vma_create_buffer(&request, c->vma, &buffer, &placement,
        marker, allocation, info, &admitted);
    if (result == VK_NOT_READY && dependency && dependency->selected) {
        if (attempted.amount[RC_BUDGET_COMMIT]) {
            dependency->peak = attempted;
            if (admitted == RC_ADMISSION_NO_RECORDS) dependency->before.records_available = 0;
        } else if (admitted == RC_ADMISSION_NO_RECORDS) {
            dependency->owner_full = 1;
        }
    }
    return result;
}

static MemLazyWriteResult native_rmw_word_acquire_lazy(VkMemNativeRmwWord *word, int parked) {
    const SelectedMirrorWord *source = &word->source;
    return mem_lazy_write_acquire(&word->lazy_write, source->native.interval.address, 4,
        (uint64_t)(uintptr_t)source->mirror, (uint64_t)(uintptr_t)source->import, source->native.native.offset,
        word->descriptor.pool, parked);
}

typedef struct {
    VkMemNativeRmwWord *word;
    MemLazyWriteResult result;
} NativeRmwWordLazyWork;

static int native_rmw_word_settle_lazy(void *opaque) {
    NativeRmwWordLazyWork *work = opaque;
    VkMemNativeRmwWord *word = work->word;
    if (!vk_mem_native_rmw_word_current(word, &word->descriptor, word->source.native.interval.address,
                                        word->source.read)) return 0;
    work->result = native_rmw_word_acquire_lazy(word, 1);
    return 1;
}

VkResult vk_mem_native_rmw_word_prepare(VkMemSpanCapture *provider, const VkMemSpan *descriptor,
    uint64_t descriptor_size, uint64_t word_address, RcRegistryPosition read,
    RcVmaDependency *dependency, VkMemNativeRmwWord **out) {
    if (!out || *out || vk_device_is_lost() || vk_sync_stopping()) return VK_ERROR_INITIALIZATION_FAILED;
    SelectedMirrorWord source;
    RcRegistryResult selected = selected_mirror_word(provider, descriptor, word_address, descriptor_size, read, &source);
    if (selected == RC_REGISTRY_OK)
        selected = rc_registry_buffer_access_check(rc_registry_runtime(), source.native, read);
    if (selected != RC_REGISTRY_OK || RW.word_count == RMW_WORD_LIMIT ||
        (RW.scene && RW.scene->output.scene != read.scene) ||
        !native_rmw_tail_supported(descriptor->pool, source.native.native.offset) ||
        native_rmw_word_bitmap_refuses(descriptor->pool, &source, VK_NULL_HANDLE)) {
        if (rt_knob("GRAPHICS_PREP_TRACE", "0"))
            rt_log("vk: native mesh word refused read %llu/%llu address %#llx authority %d",
                (unsigned long long)read.scene, (unsigned long long)read.ordinal,
                (unsigned long long)word_address, (int)selected);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    VkCtx *c = vk_ctx();
    if (!native_rmw_pipeline_ready(c)) return VK_ERROR_FEATURE_NOT_PRESENT;

    const size_t workspace_bytes = RW.scene ? 0 : sizeof(NativeRmwTail);
    const size_t host_bytes = sizeof(VkMemNativeRmwWord) + workspace_bytes;
    RcBudgetTicket host = {0};
    VkBuffer marker = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    VmaAllocationInfo info;
    VkResult result = native_rmw_marker_allocate(c, host_bytes, 2 * sizeof(uint32_t), &host, dependency, &marker, &allocation, &info);
    if (result != VK_SUCCESS) return result;
    VkMemNativeRmwWord *word = info.pMappedData ? calloc(1, host_bytes) : NULL;
    uint32_t *mapped = info.pMappedData;
    if (word) {
        mapped[0] = mapped[1] = 0;
        result = vmaFlushAllocation(c->vma, allocation, 0, 2 * sizeof *mapped);
    } else result = VK_ERROR_OUT_OF_HOST_MEMORY;
    const RcAdmissionVector retained = {.amount = {
        [RC_BUDGET_RAM] = host_bytes, [RC_BUDGET_COMMIT] = host_bytes}};
    if (result != VK_SUCCESS || !rc_admission_accounted(resource_core_budget_materialize(&host, &retained))) {
        resource_core_vma_destroy_buffer(c->vma, marker, allocation);
        if (resource_core_budget_cancel(&host) != RC_ADMISSION_OK)
            rt_fatal("vk: selected mesh word receipt lost its pending capacity charge");
        free(word);
        return result != VK_SUCCESS ? result : VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    const uint64_t generation = ++RW.generation;
    if (!generation) rt_fatal("vk: native mesh RMW marker generation exhausted");
    *word = (VkMemNativeRmwWord){.source = source, .descriptor = *descriptor, .host = host, .marker = marker,
        .allocation = allocation, .mapped = mapped, .provider = vk_mem_span_capture_retain(provider),
        .marker_generation = generation, .references = 1, .workspace = workspace_bytes ? (NativeRmwTail *)(word + 1) : NULL};

    MemLazyWriteResult lazy = native_rmw_word_acquire_lazy(word, 0);
    if ((lazy == MEM_LAZY_WRITE_BUSY || lazy == MEM_LAZY_WRITE_NEEDS_FETCH) && !vk_exec_shared() &&
        !rc_registry_position_compare(read, vk_exec_resource_position()) &&
        rc_registry_provisional_read_current(rc_registry_runtime(), read)) {
        NativeRmwWordLazyWork work = {.word = word, .result = lazy};
        const VkSyncProgressResult progress = vk_sync_progress_prepare_pipeline(native_rmw_word_settle_lazy, &work);
        if (progress == VK_SYNC_PROGRESS_COMPLETE) {
            lazy = work.result;
            if (lazy == MEM_LAZY_WRITE_READY && !vk_mem_native_rmw_word_current(word, descriptor, word_address, read)) {
                native_rmw_word_release(word);
                return VK_ERROR_INITIALIZATION_FAILED;
            }
        }
    }
    if (lazy != MEM_LAZY_WRITE_READY) {
        if (rt_knob("GRAPHICS_PREP_TRACE", "0"))
            rt_log("vk: native mesh word lazy custody refused read %llu/%llu address %#llx result %d",
                (unsigned long long)read.scene, (unsigned long long)read.ordinal,
                (unsigned long long)word_address, (int)lazy);
        native_rmw_word_release(word);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    *out = word;
    return VK_SUCCESS;
}

int vk_mem_native_rmw_word_current(const VkMemNativeRmwWord *word, const VkMemSpan *descriptor,
                                  uint64_t word_address, RcRegistryPosition read) {
    if (!word || word->body || !word->marker || !word->marker_generation || !descriptor || vk_exec_shared() ||
        rc_registry_position_compare(word->source.read, read) ||
        rc_registry_position_compare(vk_exec_resource_position(), read)) return 0;
    SelectedMirrorWord now = {0};
    plat_mutex_lock_shared(&pool_lock);
    RcRegistryResult result = selected_mirror_word_locked(word->provider, descriptor, word_address,
        word->source.descriptor_size, read, &now);
    plat_mutex_unlock_shared(&pool_lock);
    const RcRegistryBufferSpan a = word->source.native, b = now.native;
    return result == RC_REGISTRY_OK && word->source.mirror == now.mirror && word->source.import == now.import &&
        a.interval.address == b.interval.address && a.interval.mapping == b.interval.mapping &&
        rc_range_equal(a.interval.range, b.interval.range) && a.native.allocation == b.native.allocation &&
        a.native.incarnation == b.native.incarnation && a.native.offset == b.native.offset &&
        rc_registry_buffer_access_check(rc_registry_runtime(), now.native, read) == RC_REGISTRY_OK &&
        native_rmw_tail_supported(descriptor->pool, b.native.offset) &&
        !native_rmw_word_bitmap_refuses(descriptor->pool, &now, VK_NULL_HANDLE);
}

int vk_mem_native_rmw_word_order(VkMemNativeRmwWord *word, VkCommandBuffer body) {
    if (!word || !body || !word->source.mirror || word->body) return 0;
    native_rmw_word_record_physical(word);
    const VkBufferMemoryBarrier2 barrier = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = word->source.mirror, .offset = word->source.native.native.offset, .size = 4};
    const VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(body, &dependency);
    return 1;
}

VkDescriptorBufferInfo vk_mem_native_rmw_word_marker(const VkMemNativeRmwWord *word) {
    return word && !word->body ? (VkDescriptorBufferInfo){word->marker, 0, sizeof(uint32_t)} : (VkDescriptorBufferInfo){0};
}

RcRegistryResult vk_mem_native_rmw_word_declare(VkMemNativeRmwWord *word, RcRegistryPosition output) {
    if (!word || word->body || word->tickets[0].id || word->tickets[1].id ||
        output.scene != word->source.read.scene || !(output.ordinal & 1) ||
        output.ordinal - 1 != word->source.read.ordinal) return RC_REGISTRY_INVALID;
    RcRegistry *registry = rc_registry_runtime();
    RcRegistryResult result = rc_registry_write_begin_at(registry, word->source.native.interval.range,
        RC_REGISTRY_BUFFER, output, &word->tickets[0]);
    if (result == RC_REGISTRY_OK)
        result = rc_registry_write_begin_at(registry, native_rmw_guest_range(word), RC_REGISTRY_BUFFER, output, &word->tickets[1]);
    if (result != RC_REGISTRY_OK) native_rmw_word_cancel_tickets(word, 0);
    else word->output = output;
    return result;
}

void vk_mem_native_rmw_word_issued(VkMemNativeRmwWord **owned, VkCommandBuffer body) {
    VkMemNativeRmwWord *word = owned ? *owned : NULL;
    if (!word || !body || (!RW.scene && !word->workspace) || RW.word_count == RMW_WORD_LIMIT ||
        word->bound != body || word->body || !word->tickets[0].id || !word->tickets[1].id)
        rt_fatal("vk: issued native mesh RMW lost its exact receipt");
    *owned = NULL;
    word->body = body;
    if (!RW.scene) RW.workspace = word->workspace;
    word->next = RW.scene;
    RW.scene = word;
    RW.word_count++;
    RcRegistry *registry = rc_registry_runtime();
    if (rc_registry_buffer_access_record(registry, word->tickets[0], word->source.native, word->output) != RC_REGISTRY_OK ||
        rc_registry_buffer_access_record(registry, word->tickets[1], word->source.native, word->output) != RC_REGISTRY_OK)
        rc_registry_fail_closed(registry);
    vk_completion_add_owned(word->output.scene, native_rmw_word_completed, native_rmw_word_cancel_submitted, word);
}

typedef struct {
    PoolRec *pool;
    uint64_t off;
} NativeRmwBitmapEdge;

typedef struct {
    VkMemNativeRmwBitmap *owner;
    uint64_t off, size;
} NativeRmwBitmapSlice;

struct VkMemNativeRmwBitmap {
    struct VkMemNativeRmwBitmap *next;
    MemLazyWriteRange lazy_write;
    SelectedMirrorWord source;
    VkMemSpan descriptor;
    VkMemSpanCapture *provider;
    RcBudgetTicket host;
    VkBuffer marker;
    VmaAllocation allocation;
    volatile uint32_t *mapped;
    RcRegistryWrite tickets[2];
    VkCommandBuffer body, bound;
    RcRegistryPosition output;
    VkPipelineStageFlags2 writer_stage;
    uint32_t references, words, map_dwords;
    int physical_recorded, tail_recorded;
    RcSourceRange extents[];
};

// The shader set sees only the execution plane. A disjoint internal plane in the
// same allocation collects earlier snapshot differences and overlapping draw masks.
static struct {
    PFN_vkCmdPushDescriptorSetKHR push;
    VkDescriptorSetLayout set;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkMemNativeRmwBitmap *scene;
    size_t count, capacity;
    NativeRmwBitmapEdge *edge;
    NativeRmwBitmapSlice *slice;
    RcBudgetTicket shape;
} RB;

static void native_rmw_bitmap_free_shapes(void) {
    if (!RB.edge) return;
    if (resource_core_budget_retire(&RB.shape) != RC_ADMISSION_OK)
        rt_fatal("vk: fragment atomic shape storage lost its host receipt");
    free(RB.edge);
    if (resource_core_budget_destroyed(&RB.shape) != RC_ADMISSION_OK)
        rt_fatal("vk: fragment atomic shape storage lost its retired host receipt");
    RB.edge = NULL;
    RB.slice = NULL;
    RB.capacity = 0;
}

static int native_rmw_bitmap_reserve_shapes(size_t needed) {
    if (needed <= RB.capacity) return 1;
    const size_t stride = 2 * (sizeof *RB.edge + sizeof *RB.slice);
    if (needed > SIZE_MAX / stride) return 0;
    size_t capacity = RB.capacity > 0 && RB.capacity <= SIZE_MAX / (2 * stride) ? 2 * RB.capacity : needed;
    if (capacity < needed) capacity = needed;
    const size_t bytes = stride * capacity;
    const RcAdmissionVector cost = {.amount = {[RC_BUDGET_RAM] = bytes, [RC_BUDGET_COMMIT] = bytes}};
    RcBudgetTicket ticket = {0};
    const RcAdmissionStatus admitted = resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, &cost, &ticket);
    if (admitted != RC_ADMISSION_OK) return admitted == RC_ADMISSION_REFUSED ? -1 : 0;
    NativeRmwBitmapEdge *edge = malloc(bytes);
    if (edge) {
        volatile uint8_t *pages = (volatile uint8_t *)edge;
        for (size_t off = 0;; off += 4096) {
            pages[off] = 0;
            if (bytes - off <= 4096) break;
        }
        pages[bytes - 1] = 0;
    }
    if (!edge || !rc_admission_accounted(resource_core_budget_materialize(&ticket, &cost))) {
        free(edge);
        if (resource_core_budget_cancel(&ticket) != RC_ADMISSION_OK)
            rt_fatal("vk: fragment atomic shape storage lost its pending host receipt");
        return 0;
    }
    native_rmw_bitmap_free_shapes();
    RB.edge = edge;
    RB.slice = (NativeRmwBitmapSlice *)(void *)(edge + 2 * capacity);
    RB.capacity = capacity;
    RB.shape = ticket;
    return 1;
}

static uint64_t native_rmw_bitmap_marker_bytes(const VkMemNativeRmwBitmap *bitmap) {
    return ((uint64_t)bitmap->map_dwords + 1) * sizeof(uint32_t);
}

static uint64_t native_rmw_bitmap_allocation_bytes(uint32_t map_dwords) {
    return (2 * (uint64_t)map_dwords + 1) * sizeof(uint32_t);
}

static void native_rmw_bitmap_pipeline_destroy(VkCtx *c) {
    native_rmw_pipeline_free(c, &RB.set, &RB.layout, &RB.pipeline);
}

static int native_rmw_bitmap_pipeline_ready(VkCtx *c) {
    return native_rmw_pipeline_build(c, &RB.push, &RB.set, &RB.layout, &RB.pipeline,
        native_rmw_bitmap_spv, sizeof native_rmw_bitmap_spv, 6 * sizeof(uint32_t));
}

static void native_rmw_bitmap_release(void *opaque) {
    VkMemNativeRmwBitmap *bitmap = opaque;
    if (__atomic_fetch_sub(&bitmap->references, 1, __ATOMIC_ACQ_REL) != 1) return;
    resource_core_vma_destroy_buffer(vk_ctx()->vma, bitmap->marker, bitmap->allocation);
    mem_lazy_write_release(&bitmap->lazy_write);
    vk_mem_span_capture_release(&bitmap->provider);
    RcBudgetTicket host = bitmap->host;
    if (resource_core_budget_retire(&host) != RC_ADMISSION_OK)
        rt_fatal("vk: fragment atomic bitmap lost its host receipt");
    free(bitmap);
    if (resource_core_budget_destroyed(&host) != RC_ADMISSION_OK)
        rt_fatal("vk: fragment atomic bitmap lost its retired host receipt");
}

static void native_rmw_bitmap_binding_canceled(void *opaque, int possibly_executed) {
    (void)possibly_executed;
    native_rmw_bitmap_release(opaque);
}

static void native_rmw_bitmap_retain_recording(VkMemNativeRmwBitmap *bitmap) {
    if (bitmap->physical_recorded) return;
    bitmap->physical_recorded = 1;
    __atomic_fetch_add(&bitmap->references, 1, __ATOMIC_RELAXED);
    vk_completion_add_owned(bitmap->source.read.scene, native_rmw_bitmap_release, native_rmw_bitmap_binding_canceled, bitmap);
}

static void native_rmw_bitmap_cancel_tickets(VkMemNativeRmwBitmap *bitmap, int possibly_executed) {
    RcRegistry *registry = rc_registry_runtime();
    if (bitmap->tickets[0].id && bitmap->tickets[1].id) {
        RcRegistryResult result = possibly_executed
            ? rc_registry_write_abandon_extents(registry, bitmap->tickets, 2)
            : rc_registry_write_cancel_extents(registry, bitmap->tickets, 2);
        if (result != RC_REGISTRY_OK) rc_registry_fail_closed(registry);
        bitmap->tickets[0] = bitmap->tickets[1] = (RcRegistryWrite){0};
        return;
    }
    for (unsigned i = 0; i < 2; ++i) {
        if (!bitmap->tickets[i].id) continue;
        if (rc_registry_write_abandon(registry, bitmap->tickets[i], possibly_executed) != RC_REGISTRY_OK)
            rc_registry_fail_closed(registry);
        bitmap->tickets[i] = (RcRegistryWrite){0};
    }
}

void vk_mem_native_rmw_bitmap_cancel(VkMemNativeRmwBitmap **owned) {
    if (!owned || !*owned) return;
    VkMemNativeRmwBitmap *bitmap = *owned;
    *owned = NULL;
    native_rmw_bitmap_cancel_tickets(bitmap, 0);
    native_rmw_bitmap_release(bitmap);
}

static void native_rmw_bitmap_cancel_submitted(void *opaque, int possibly_executed) {
    VkMemNativeRmwBitmap *bitmap = opaque;
    possibly_executed |= rc_registry_position_handed(rc_registry_runtime(), bitmap->output);
    native_rmw_bitmap_cancel_tickets(bitmap, possibly_executed);
    native_rmw_bitmap_release(bitmap);
}

static int native_rmw_bitmap_test(const volatile uint32_t *marks, uint32_t word) {
    return (marks[1 + (word >> 5)] >> (word & 31)) & 1;
}

// Alternating executed/skipped words are the maximum possible number of output intervals.
static uint32_t native_rmw_bitmap_extents(const volatile uint32_t *marks, uint32_t words, RcSourceRange base,
                                         RcSourceRange *out, uint32_t *executed) {
    uint32_t count = 0;
    *executed = 0;
    for (uint32_t i = 0; i < words;) {
        if (!native_rmw_bitmap_test(marks, i)) {
            ++i;
            continue;
        }
        const uint32_t first = i++;
        while (i < words && native_rmw_bitmap_test(marks, i)) ++i;
        out[count++] = (RcSourceRange){base.backing, base.offset + 4ull * first, 4ull * (i - first)};
        *executed += i - first;
    }
    return count;
}

static RcSourceRange native_rmw_bitmap_guest_range(const VkMemNativeRmwBitmap *bitmap) {
    return (RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING, bitmap->source.native.interval.address,
                           bitmap->source.descriptor_size};
}

static void native_rmw_bitmap_completed(void *opaque) {
    VkMemNativeRmwBitmap *bitmap = opaque;
    RcRegistry *registry = rc_registry_runtime();
    if (!bitmap->tail_recorded || !rc_registry_position_handed(registry, bitmap->output) ||
        vmaInvalidateAllocation(vk_ctx()->vma, bitmap->allocation, 0, native_rmw_bitmap_marker_bytes(bitmap)) != VK_SUCCESS) {
        native_rmw_bitmap_cancel_submitted(bitmap, 1);
        return;
    }
    const uint32_t used = bitmap->words & 31;
    if (used && (bitmap->mapped[bitmap->map_dwords] >> used)) {
        native_rmw_bitmap_cancel_submitted(bitmap, 1);
        return;
    }
    const RcSourceRange canonical = bitmap->source.native.interval.range;
    const RcSourceRange guest = native_rmw_bitmap_guest_range(bitmap);
    const size_t capacity = ((size_t)bitmap->words + 1) / 2;
    RcSourceRange *range[2] = {bitmap->extents, bitmap->extents + capacity};
    uint32_t executed;
    uint32_t count = native_rmw_bitmap_extents(bitmap->mapped, bitmap->words, canonical, range[0], &executed);
    for (uint32_t i = 0; i < count; ++i)
        range[1][i] = (RcSourceRange){guest.backing, guest.offset + (range[0][i].offset - canonical.offset), range[0][i].size};
    const int visible = vk_mem_span_capture_current(bitmap->provider, &bitmap->descriptor);
    RcRegistryWriteCompletion completion[2];
    if (rc_registry_write_submit_extents(registry, bitmap->tickets, 2, bitmap->output.scene) != RC_REGISTRY_OK) {
        rc_registry_fail_closed(registry);
        native_rmw_bitmap_cancel_submitted(bitmap, 1);
        return;
    }
    if (executed && !visible) {
        // A later SNL visibility proof must not reauthorize this stale alias. Abandon both original
        // exclusions in place, without allocating per-run authority or asserting a definite whole write.
        native_rmw_bitmap_cancel_tickets(bitmap, 1);
        native_rmw_bitmap_release(bitmap);
        return;
    }
    for (unsigned i = 0; i < 2; ++i) {
        completion[i] = (RcRegistryWriteCompletion){.ticket = bitmap->tickets[i], .written = range[i],
            .written_count = count, .uncertain = NULL, .uncertain_count = 0};
    }
    RcRegistryResult completed = rc_registry_write_complete_extents(registry, completion, 2);
    if (completed == RC_REGISTRY_CAPACITY) {
        // Exact sparse uncertainty can require more writer records than remain. Keep broad unknown
        // authority in the original pair; never fail-close globally or claim every byte was written.
        native_rmw_bitmap_cancel_tickets(bitmap, 1);
        native_rmw_bitmap_release(bitmap);
        return;
    }
    if (completed != RC_REGISTRY_OK) {
        rc_registry_fail_closed(registry);
        native_rmw_bitmap_cancel_submitted(bitmap, 1);
        return;
    }
    if (visible) for (uint32_t i = 0; i < count; ++i)
        vk_mem_buffer_write_completed(bitmap->descriptor.pool, bitmap->source.mirror, range[1][i].offset, range[1][i].size, 1);
    if (rt_knob("GRAPHICS_PREP_TRACE", "0"))
        rt_log("vk: native fragment bitmap complete output %llu/%llu address %#llx words %u executed %u extents %u visible %d",
            (unsigned long long)bitmap->output.scene, (unsigned long long)bitmap->output.ordinal,
            (unsigned long long)guest.offset, bitmap->words, executed, count, visible);
    native_rmw_bitmap_release(bitmap);
}

static const char *native_rmw_bitmap_tail_refusal(const PoolRec *, uint64_t, uint64_t, uint64_t,
    VkMemNativeRmwBitmapFailure *);
static int native_rmw_bitmap_tail_supported(const PoolRec *, uint64_t, uint64_t, uint64_t);
static const char *native_rmw_bitmap_word_refusal(const PoolRec *, const SelectedMirrorWord *, VkCommandBuffer,
                                                 VkMemNativeRmwBitmapFailure *);

static VkResult native_rmw_bitmap_failure(VkMemNativeRmwBitmapFailure *out, const char *gate,
    VkResult result, RcRegistryResult registry, const SelectedMirrorWord *source) {
    if (out) {
        out->gate = gate;
        out->result = result;
        out->registry = registry;
        if (source) {
            out->range = source->native.interval;
            out->mirror_offset = source->native.native.offset;
        }
    }
    return result;
}

static VkResult native_rmw_bitmap_source_failure(VkMemNativeRmwBitmapFailure *failure, const char *gate,
    RcRegistryResult registry, const SelectedMirrorWord *source, RcRegistryPosition read,
    uint64_t address, uint64_t size) {
    if (rt_knob("GRAPHICS_PREP_TRACE", "0"))
        rt_log("vk: native fragment bitmap refused read %llu/%llu address %#llx size %#llx authority %d",
            (unsigned long long)read.scene, (unsigned long long)read.ordinal,
            (unsigned long long)address, (unsigned long long)size, registry);
    return native_rmw_bitmap_failure(failure, gate, VK_ERROR_FEATURE_NOT_PRESENT, registry, source);
}

typedef struct {
    VkMemNativeRmwBitmap *bitmap;
    MemLazyWriteResult result;
} NativeRmwLazyWork;

static MemLazyWriteResult native_rmw_bitmap_acquire_lazy(VkMemNativeRmwBitmap *bitmap, int parked) {
    const SelectedMirrorWord *source = &bitmap->source;
    return mem_lazy_write_acquire(&bitmap->lazy_write, source->native.interval.address, source->descriptor_size,
        (uint64_t)(uintptr_t)source->mirror, (uint64_t)(uintptr_t)source->import, source->native.native.offset,
        bitmap->descriptor.pool, parked);
}

static int native_rmw_bitmap_settle_lazy(void *opaque) {
    NativeRmwLazyWork *work = opaque;
    VkMemNativeRmwBitmap *bitmap = work->bitmap;
    if (!vk_mem_native_rmw_bitmap_current_report(bitmap, &bitmap->descriptor, bitmap->source.read, NULL))
        work->result = MEM_LAZY_WRITE_INVALID;
    else work->result = native_rmw_bitmap_acquire_lazy(bitmap, 1);
    return 1;
}

VkResult vk_mem_native_rmw_bitmap_prepare_report(VkMemSpanCapture *provider, const VkMemSpan *descriptor,
    uint64_t descriptor_size, VkPipelineStageFlags2 writer_stage, RcRegistryPosition read,
    RcVmaDependency *dependency, VkMemNativeRmwBitmap **out, VkMemNativeRmwBitmapFailure *failure) {
    if (failure) *failure = (VkMemNativeRmwBitmapFailure){0};
    if (!out)
        return native_rmw_bitmap_failure(failure, "bitmap-no-output", VK_ERROR_INITIALIZATION_FAILED, RC_REGISTRY_OK, NULL);
    if (*out)
        return native_rmw_bitmap_failure(failure, "bitmap-output-owned", VK_ERROR_INITIALIZATION_FAILED,
                                         RC_REGISTRY_OK, NULL);
    if (!provider || !descriptor)
        return native_rmw_bitmap_failure(failure, "bitmap-missing-provider", VK_ERROR_INITIALIZATION_FAILED, RC_REGISTRY_OK, NULL);
    if (vk_device_is_lost())
        return native_rmw_bitmap_failure(failure, "bitmap-device-lost", VK_ERROR_INITIALIZATION_FAILED, RC_REGISTRY_OK, NULL);
    if (vk_sync_stopping())
        return native_rmw_bitmap_failure(failure, "bitmap-stopping", VK_ERROR_INITIALIZATION_FAILED, RC_REGISTRY_OK, NULL);
    if (writer_stage != VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT && writer_stage != VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT)
        return native_rmw_bitmap_failure(failure, "bitmap-writer-stage", VK_ERROR_FEATURE_NOT_PRESENT, RC_REGISTRY_OK, NULL);
    const VkPhysicalDeviceLimits *limits = &vk_ctx()->props.limits;
    const uint64_t words = descriptor_size / 4;
    const uint64_t device_words = (uint64_t)limits->maxComputeWorkGroupCount[0] * 64;
    const uint64_t max_range = limits->maxStorageBufferRange;
    const uint64_t align = limits->minStorageBufferOffsetAlignment;
    if (!align || !descriptor_size || (descriptor_size & 3) || words > UINT32_MAX ||
        words > device_words || align > max_range || descriptor_size > max_range - (align - 1))
        return native_rmw_bitmap_failure(failure, "bitmap-device-limits", VK_ERROR_FEATURE_NOT_PRESENT, RC_REGISTRY_OK, NULL);
    SelectedMirrorWord source = {0};
    RcRegistryResult selected = selected_mirror_access(provider, descriptor, provider->address, descriptor_size,
                                                       descriptor_size, read, &source);
    if (selected != RC_REGISTRY_OK)
        return native_rmw_bitmap_source_failure(failure, "bitmap-mirror-access", selected, NULL,
                                                read, provider->address, descriptor_size);
    selected = rc_registry_buffer_access_check(rc_registry_runtime(), source.native, read);
    if (selected != RC_REGISTRY_OK)
        return native_rmw_bitmap_source_failure(failure, "bitmap-authority", selected, &source,
                                                read, provider->address, descriptor_size);
    if (RB.scene && RB.scene->output.scene != read.scene) {
        if (failure) failure->blocker_read = RB.scene->output;
        return native_rmw_bitmap_source_failure(failure, "bitmap-scene", selected, &source,
                                                read, provider->address, descriptor_size);
    }
    const char *tail = native_rmw_bitmap_tail_refusal(descriptor->pool, source.native.native.offset,
                                                     descriptor_size, read.scene, failure);
    if (!tail) tail = native_rmw_bitmap_word_refusal(descriptor->pool, &source, VK_NULL_HANDLE, failure);
    if (tail)
        return native_rmw_bitmap_source_failure(failure, tail, selected, &source, read, provider->address, descriptor_size);
    const PoolRec *pool = descriptor->pool;
    if (source.native.interval.address < pool->import_lo ||
        source.native.interval.address - pool->import_lo != source.native.native.offset)
        return native_rmw_bitmap_source_failure(failure, "bitmap-lazy-affine", RC_REGISTRY_INVALID, &source,
                                                read, provider->address, descriptor_size);
    VkCtx *c = vk_ctx();
    if (!native_rmw_bitmap_pipeline_ready(c))
        return native_rmw_bitmap_failure(failure, "bitmap-tail-pipeline", VK_ERROR_FEATURE_NOT_PRESENT, selected, &source);
    if (RB.count == SIZE_MAX)
        return native_rmw_bitmap_failure(failure, "bitmap-scene-count", VK_ERROR_TOO_MANY_OBJECTS, selected, &source);
    const int shapes = native_rmw_bitmap_reserve_shapes(RB.count + 1);
    if (shapes <= 0)
        return native_rmw_bitmap_failure(failure, "bitmap-tail-shapes", shapes < 0 ? VK_NOT_READY : VK_ERROR_OUT_OF_HOST_MEMORY,
                                         selected, &source);

    const uint64_t capacity = (words + 1) / 2;
    if (capacity > (SIZE_MAX - sizeof(VkMemNativeRmwBitmap)) / (2 * sizeof(RcSourceRange)))
        return native_rmw_bitmap_failure(failure, "bitmap-range-count", VK_ERROR_OUT_OF_HOST_MEMORY, selected, &source);
    const size_t host_bytes = sizeof(VkMemNativeRmwBitmap) + (size_t)capacity * 2 * sizeof(RcSourceRange);
    const uint32_t map_dwords = (uint32_t)((words + 31) / 32);
    const uint64_t marker_bytes = native_rmw_bitmap_allocation_bytes(map_dwords);
    RcBudgetTicket host = {0};
    VkBuffer marker = VK_NULL_HANDLE;
    VmaAllocation allocation = NULL;
    VmaAllocationInfo info;
    VkResult result = native_rmw_marker_allocate(c, host_bytes, marker_bytes, &host, dependency, &marker, &allocation, &info);
    if (result != VK_SUCCESS)
        return native_rmw_bitmap_failure(failure, "bitmap-marker-allocation", result, selected, &source);
    VkMemNativeRmwBitmap *bitmap = info.pMappedData ? calloc(1, host_bytes) : NULL;
    const char *marker_gate = NULL;
    if (bitmap) {
        memset(info.pMappedData, 0, (size_t)marker_bytes);
        result = vmaFlushAllocation(c->vma, allocation, 0, marker_bytes);
        if (result != VK_SUCCESS) marker_gate = "bitmap-marker-flush";
    } else {
        result = VK_ERROR_OUT_OF_HOST_MEMORY;
        marker_gate = info.pMappedData ? "bitmap-host-receipt" : "bitmap-marker-unmapped";
    }
    const RcAdmissionVector retained = {.amount = {[RC_BUDGET_RAM] = host_bytes, [RC_BUDGET_COMMIT] = host_bytes}};
    if (result != VK_SUCCESS || !rc_admission_accounted(resource_core_budget_materialize(&host, &retained))) {
        resource_core_vma_destroy_buffer(c->vma, marker, allocation);
        if (resource_core_budget_cancel(&host) != RC_ADMISSION_OK)
            rt_fatal("vk: fragment atomic bitmap lost its pending host receipt");
        free(bitmap);
        return native_rmw_bitmap_failure(failure, marker_gate ? marker_gate : "bitmap-host-settlement",
            result != VK_SUCCESS ? result : VK_ERROR_OUT_OF_HOST_MEMORY, selected, &source);
    }
    *bitmap = (VkMemNativeRmwBitmap){.source = source, .descriptor = *descriptor,
        .provider = vk_mem_span_capture_retain(provider), .host = host, .marker = marker, .allocation = allocation,
        .mapped = info.pMappedData, .writer_stage = writer_stage, .references = 1,
        .words = (uint32_t)words, .map_dwords = map_dwords};
    MemLazyWriteResult lazy = native_rmw_bitmap_acquire_lazy(bitmap, 0);
    if ((lazy == MEM_LAZY_WRITE_BUSY || lazy == MEM_LAZY_WRITE_NEEDS_FETCH) && !vk_exec_shared() &&
        !rc_registry_position_compare(read, vk_exec_resource_position()) &&
        rc_registry_provisional_read_current(rc_registry_runtime(), read)) {
        NativeRmwLazyWork work = {bitmap, lazy};
        const VkSyncProgressResult progress = vk_sync_progress_prepare_pipeline(native_rmw_bitmap_settle_lazy, &work);
        if (progress == VK_SYNC_PROGRESS_COMPLETE) lazy = work.result;
        else if (progress == VK_SYNC_PROGRESS_STOPPED || progress == VK_SYNC_PROGRESS_STOPPED_AFTER_CUT) {
            native_rmw_bitmap_release(bitmap);
            return native_rmw_bitmap_failure(failure, "bitmap-lazy-stopped", VK_ERROR_INITIALIZATION_FAILED,
                                             RC_REGISTRY_OK, &source);
        }
    }
    if (lazy != MEM_LAZY_WRITE_READY) {
        const char *gate = lazy == MEM_LAZY_WRITE_FOREIGN ? "bitmap-lazy-foreign" :
            lazy == MEM_LAZY_WRITE_BUSY ? "bitmap-lazy-busy" :
            lazy == MEM_LAZY_WRITE_NEEDS_FETCH ? "bitmap-lazy-fetch" :
            lazy == MEM_LAZY_WRITE_FETCH_FAILED ? "bitmap-lazy-fetch-failed" : "bitmap-lazy-invalid";
        native_rmw_bitmap_release(bitmap);
        return native_rmw_bitmap_source_failure(failure, gate, selected, &source,
                                                read, provider->address, descriptor_size);
    }
    if (!vk_mem_native_rmw_bitmap_current_report(bitmap, descriptor, read, failure)) {
        native_rmw_bitmap_release(bitmap);
        if (failure) failure->result = VK_ERROR_INITIALIZATION_FAILED;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *out = bitmap;
    return VK_SUCCESS;
}

VkResult vk_mem_native_rmw_bitmap_prepare(VkMemSpanCapture *provider, const VkMemSpan *descriptor,
    uint64_t descriptor_size, VkPipelineStageFlags2 writer_stage, RcRegistryPosition read,
    RcVmaDependency *dependency, VkMemNativeRmwBitmap **out) {
    return vk_mem_native_rmw_bitmap_prepare_report(provider, descriptor, descriptor_size, writer_stage,
                                                    read, dependency, out, NULL);
}

static int native_rmw_bitmap_not_current(VkMemNativeRmwBitmapFailure *failure, const char *gate,
    RcRegistryResult registry, const VkMemNativeRmwBitmap *bitmap) {
    native_rmw_bitmap_failure(failure, gate, VK_SUCCESS, registry, bitmap ? &bitmap->source : NULL);
    return 0;
}

int vk_mem_native_rmw_bitmap_current_report(const VkMemNativeRmwBitmap *bitmap, const VkMemSpan *descriptor,
    RcRegistryPosition read, VkMemNativeRmwBitmapFailure *failure) {
    if (failure) *failure = (VkMemNativeRmwBitmapFailure){0};
    if (!bitmap) return native_rmw_bitmap_not_current(failure, "bitmap-missing-receipt", RC_REGISTRY_OK, NULL);
    if (bitmap->body) {
        if (failure) failure->blocker_read = bitmap->output;
        return native_rmw_bitmap_not_current(failure, "bitmap-already-issued", RC_REGISTRY_OK, bitmap);
    }
    if (!bitmap->marker) return native_rmw_bitmap_not_current(failure, "bitmap-missing-marker", RC_REGISTRY_OK, bitmap);
    if (!descriptor) return native_rmw_bitmap_not_current(failure, "bitmap-missing-descriptor", RC_REGISTRY_OK, bitmap);
    if (vk_exec_shared()) return native_rmw_bitmap_not_current(failure, "bitmap-shared-recorder", RC_REGISTRY_OK, bitmap);
    if (rc_registry_position_compare(bitmap->source.read, read)) {
        if (failure) failure->blocker_read = bitmap->source.read;
        return native_rmw_bitmap_not_current(failure, "bitmap-receipt-read", RC_REGISTRY_OK, bitmap);
    }
    RcRegistryPosition actual = vk_exec_resource_position();
    if (rc_registry_position_compare(actual, read)) {
        if (failure) failure->blocker_read = actual;
        return native_rmw_bitmap_not_current(failure, "bitmap-recorder-read", RC_REGISTRY_OK, bitmap);
    }
    const uint64_t size = bitmap->source.descriptor_size;
    SelectedMirrorWord now = {0};
    plat_mutex_lock_shared(&pool_lock);
    RcRegistryResult selected = selected_mirror_access_locked(bitmap->provider, descriptor,
        bitmap->source.native.interval.address, size, size, read, &now);
    plat_mutex_unlock_shared(&pool_lock);
    if (selected != RC_REGISTRY_OK)
        return native_rmw_bitmap_not_current(failure, "bitmap-mirror-access", selected, bitmap);
    const RcRegistryBufferSpan a = bitmap->source.native, b = now.native;
    if (bitmap->source.mirror != now.mirror || bitmap->source.import != now.import ||
        a.interval.address != b.interval.address || a.interval.mapping != b.interval.mapping ||
        !rc_range_equal(a.interval.range, b.interval.range) || a.native.allocation != b.native.allocation ||
        a.native.incarnation != b.native.incarnation || a.native.offset != b.native.offset)
        return native_rmw_bitmap_not_current(failure, "bitmap-retained-span", RC_REGISTRY_OK, bitmap);
    selected = rc_registry_buffer_access_check(rc_registry_runtime(), now.native, read);
    if (selected != RC_REGISTRY_OK)
        return native_rmw_bitmap_not_current(failure, "bitmap-authority", selected, bitmap);
    const char *tail = native_rmw_bitmap_tail_refusal(descriptor->pool, b.native.offset, size, read.scene, failure);
    if (!tail) tail = native_rmw_bitmap_word_refusal(descriptor->pool, &now, VK_NULL_HANDLE, failure);
    if (tail) return native_rmw_bitmap_not_current(failure, tail, RC_REGISTRY_OK, bitmap);
    return 1;
}

int vk_mem_native_rmw_bitmap_current(const VkMemNativeRmwBitmap *bitmap, const VkMemSpan *descriptor,
    RcRegistryPosition read) {
    return vk_mem_native_rmw_bitmap_current_report(bitmap, descriptor, read, NULL);
}

int vk_mem_native_rmw_bitmap_order(VkMemNativeRmwBitmap *bitmap, VkCommandBuffer body) {
    if (!bitmap || !body || !bitmap->source.mirror || bitmap->body) return 0;
    native_rmw_bitmap_retain_recording(bitmap);
    const VkBufferMemoryBarrier2 barrier = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = bitmap->writer_stage,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = bitmap->source.mirror, .offset = bitmap->source.native.native.offset, .size = bitmap->source.descriptor_size};
    const VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(body, &dependency);
    return 1;
}

VkDescriptorBufferInfo vk_mem_native_rmw_bitmap_marker(const VkMemNativeRmwBitmap *bitmap) {
    return bitmap && !bitmap->body ? (VkDescriptorBufferInfo){bitmap->marker, 0, native_rmw_bitmap_marker_bytes(bitmap)} :
        (VkDescriptorBufferInfo){0};
}

int vk_mem_native_rmw_bitmap_bind(VkMemNativeRmwBitmap *bitmap, VkCommandBuffer body) {
    if (!bitmap || !body || bitmap->body || !bitmap->tickets[0].id || !bitmap->tickets[1].id) return 0;
    if (bitmap->bound) return bitmap->bound == body;
    if (native_rmw_bitmap_word_refusal(bitmap->descriptor.pool, &bitmap->source, body, NULL)) return 0;
    bitmap->bound = body;
    native_rmw_bitmap_retain_recording(bitmap);
    return 1;
}

RcRegistryResult vk_mem_native_rmw_bitmap_declare(VkMemNativeRmwBitmap *bitmap, RcRegistryPosition output) {
    if (!bitmap || bitmap->body || bitmap->tickets[0].id || bitmap->tickets[1].id ||
        output.scene != bitmap->source.read.scene || !(output.ordinal & 1) ||
        output.ordinal - 1 != bitmap->source.read.ordinal) return RC_REGISTRY_INVALID;
    RcRegistry *registry = rc_registry_runtime();
    const RcRegistryWriteExtent ranges[2] = {
        {.range = bitmap->source.native.interval.range, .kind = RC_REGISTRY_BUFFER},
        {.range = native_rmw_bitmap_guest_range(bitmap), .kind = RC_REGISTRY_BUFFER},
    };
    RcRegistryResult result = rc_registry_write_begin_extents(rc_registry_runtime(), ranges, 2, output, bitmap->tickets);
    if (result == RC_REGISTRY_OK) bitmap->output = output;
    return result;
}

void vk_mem_native_rmw_bitmap_issued(VkMemNativeRmwBitmap **owned, VkCommandBuffer body) {
    VkMemNativeRmwBitmap *bitmap = owned ? *owned : NULL;
    if (!bitmap || !body || RB.count >= RB.capacity || bitmap->bound != body || bitmap->body ||
        !bitmap->tickets[0].id || !bitmap->tickets[1].id)
        rt_fatal("vk: issued fragment atomic lost its bitmap receipt");
    *owned = NULL;
    bitmap->body = body;
    bitmap->next = RB.scene;
    RB.scene = bitmap;
    ++RB.count;
    RcRegistry *registry = rc_registry_runtime();
    if (rc_registry_buffer_access_record(registry, bitmap->tickets[0], bitmap->source.native, bitmap->output) != RC_REGISTRY_OK ||
        rc_registry_buffer_access_record(registry, bitmap->tickets[1], bitmap->source.native, bitmap->output) != RC_REGISTRY_OK)
        rc_registry_fail_closed(registry);
    vk_completion_add_owned(bitmap->output.scene, native_rmw_bitmap_completed, native_rmw_bitmap_cancel_submitted, bitmap);
}

static int merging(void) {
    if (merge_on < 0) {
        merge_on = cfg_bool("NX2NAVI_WB_MERGE");
        rt_log("mem: write-back merge %s (NX2NAVI_WB_MERGE)", merge_on ? "on" : "off");
    }
    if (!merge_on) return 0;
    if (!WM.tried) merge_init(vk_ctx());
    return WM.ok;
}

// A chunk of at least size bytes in chunks[]: the smallest fitting one of the pool, else a new buffer; -1 when the
// allocation fails. host: 0 device memory (snapshots), CHUNK_LIST host memory the CPU writes (range lists),
// CHUNK_MARKS host memory the CPU reads back (the merge's page marks, shadow_wb_cb).
enum { CHUNK_LIST = 1, CHUNK_MARKS = 2 };
static int chunk_new(uint64_t size, int host) {
    uint64_t want = host ? SNAP_LIST_MIN : SNAP_CHUNK;
    while (want < size) want <<= 1;
    if (nchunks == cchunks) grow((void **)&chunks, &cchunks, sizeof *chunks);
    SnapChunk *ch = &chunks[nchunks];
    int best = -1;
    plat_mutex_lock(&snap_pool_lock);
    for (int i = 0; i < nsnap_pool; i++)
        if (snap_pool[i].host == host && snap_pool[i].size >= want && (best < 0 || snap_pool[i].size < snap_pool[best].size))
            best = i;
    if (best >= 0) {
        *ch = snap_pool[best];
        snap_pool[best] = snap_pool[--nsnap_pool];
        snap_pool_bytes -= ch->size;
    }
    plat_mutex_unlock(&snap_pool_lock);
    if (best < 0) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = want,
                                 .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,   // the source of mirror_capture's copies
                                 .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
        VmaAllocationCreateInfo ai = {.usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                      .flags = host == CHUNK_MARKS ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                                                         VMA_ALLOCATION_CREATE_MAPPED_BIT
                                             : host ? VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                                          VMA_ALLOCATION_CREATE_MAPPED_BIT : 0};
        RcVmaRequest request = {.producer = RC_BUDGET_SNAPSHOT, .required = 0};
        if (resource_core_vma_create_buffer(&request, vk_ctx()->vma, &bi, &ai, &ch->buf, &ch->alloc, NULL, NULL) != VK_SUCCESS)
            return -1; // the caller records its plain-copy path without snapshot storage
        ch->size = want, ch->host = host;
        mrep.snap_allocs++;
    }
    ch->used = 0;
    return nchunks++;
}

// Record time: room in a snapshot chunk for [off, off+size) of r's mirror, once per submission. 0: plain copy instead.
static int snap_add(PoolRec *r, uint64_t off, uint64_t size) {
    for (int i = nsnl; i-- > 0;)
        if (snl[i].r == r && snl[i].off == off && snl[i].size == size) return 1;
    int k = -1;
    for (int i = nchunks; i-- > 0;)   // the newest snapshot chunk, if the range fits
        if (chunks[i].used != UINT64_MAX && chunks[i].size - chunks[i].used >= size) { k = i; break; }
    if (k < 0 && (k = chunk_new(size > SNAP_CHUNK ? size : SNAP_CHUNK, 0)) < 0) return 0;
    if (nsnl == csnl) grow((void **)&snl, &csnl, sizeof *snl);
    snl[nsnl++] = (SnapRange){r, off, size, chunks[k].used, k};
    chunks[k].used += (size + 15) & ~15ull;
    return 1;
}

// Prologue: the snapshot copies of the ranges added since the last prologue.
static void snap_copies(VkCommandBuffer cb) {
    for (int i = snl_copied; i < nsnl; i++) {
        VkBufferCopy cp = {snl[i].off, snl[i].soff, snl[i].size};
        vkCmdCopyBuffer(cb, snl[i].r->mbuf, chunks[snl[i].chunk].buf, 1, &cp);
        mrep.snap_bytes += snl[i].size;
    }
    snl_copied = nsnl;
}

static int snap_cmp(const void *x, const void *y) {
    const SnapRange *a = x, *b = y;
    if (a->r != b->r) return (uintptr_t)a->r < (uintptr_t)b->r ? -1 : 1;
    if (a->chunk != b->chunk) return a->chunk < b->chunk ? -1 : 1;
    return a->off < b->off ? -1 : a->off > b->off;
}

// One dispatch: ranges s[0..n) share a pool and a chunk and lie in [base, base + span) of the pool. marks: the chunk
// of the page marks, its first at marks_off, or -1 when the dispatch marks none (the range list stands in for the
// binding, which the shader then does not write).
static void merge_dispatch(VkCommandBuffer cb, const SnapRange *s, int n, uint64_t base, uint64_t span, int list,
                           uint64_t list_off, uint32_t total, int marks, uint64_t marks_off) {
    uint64_t npg = ((base & (SB_PAGE - 1)) + span + SB_PAGE - 1) / SB_PAGE;
    VkDescriptorBufferInfo bi[5] = {{s->r->mbuf, base, span}, {chunks[s->chunk].buf, 0, chunks[s->chunk].size},
                                    {s->r->buf, base, span}, {chunks[list].buf, list_off, (uint64_t)n * 16},
                                    marks >= 0 ? (VkDescriptorBufferInfo){chunks[marks].buf, marks_off, npg * 4}
                                               : (VkDescriptorBufferInfo){chunks[list].buf, list_off, (uint64_t)n * 16}};
    VkWriteDescriptorSet w[5];
    for (int i = 0; i < 5; i++)
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = (uint32_t)i,
                                      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      .pBufferInfo = &bi[i]};
    rec_push(WM.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, WM.layout, 0, 5, w);   // in call order (vk_rec.h)
    uint32_t pc[4] = {(uint32_t)n, total, (uint32_t)(base & (SB_PAGE - 1)), marks >= 0};
    vkCmdPushConstants(cb, WM.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (total + 63) / 64, 1, 1);
}

// ---- shadow storage: the GPU's writes into VRAM mirrors back into guest memory
// The mirror's write-back (vk_mem_mirror_end) merges or copies them into the shadow, and the submission's job, added to
// its fence after the replay's own callbacks, copies them on into guest memory: of a merge dispatch only the pages its
// shader marked (wb_merge.comp), of a copy every page, compared with its base copy (merge_page). The fence callback of
// each noted range (gpu_written) compared all of it: 65 MiB of storage buffers a frame in Kokiri Forest, 9 ms a frame
// of the completion thread, behind which the game's fence waits sat (RTX 4090, 2026-11-09).
typedef struct { PoolRec *provider; uint64_t addr, size; uint32_t mark0; } SwbRange;   // guest range; its first page's mark, or SWB_ALL
#define SWB_ALL UINT32_MAX
typedef struct { VmaAllocation marks_alloc; const uint32_t *marks; int n; SwbRange g[]; } SwbJob;
static SwbRange *swl;              // replay: this submission's merge dispatches
static int nswl, cswl;
static int swl_marks = -1;         // replay: their page marks' chunk, -1 none

static void swl_add(PoolRec *provider, uint64_t addr, uint64_t size, uint32_t mark0) {
    if (nswl == cswl) grow((void **)&swl, &cswl, sizeof *swl);
    pool_native_acquire(provider);
    swl[nswl++] = (SwbRange){provider, addr, size, mark0};
}

// Completion thread: [a, b) into guest memory, and the pages that changed stamped as the mirror's write-back.
static void swb_run(PoolRec *provider, uint64_t a, uint64_t b) {
    if (b <= a || !shadow_write_back_from(a, b - a, provider) || nvn_gpu_only(a) ||
        !pool_host_write_hold(provider, a, b - a)) return;
    int back = wb_no_reup();
    plat_mutex_lock(&gen_lock);
    uint32_t g = generation_next();
    for (uint64_t x = a & ~0xfffull; x < b; x += 1ull << GEN_PAGE_SHIFT) stamp_page_as(x, g, back);
    plat_mutex_unlock(&gen_lock);
    mem_read_release();
}

static void shadow_wb_cb(void *arg) {   // completion thread
    SwbJob *j = arg;
    if (j->marks_alloc) vmaInvalidateAllocation(vk_ctx()->vma, j->marks_alloc, 0, VK_WHOLE_SIZE);
    for (int i = 0; i < j->n; i++) {
        const SwbRange *g = &j->g[i];
        uint64_t end = g->addr + g->size;
        if (g->mark0 == SWB_ALL) {
            swb_run(g->provider, g->addr, end);
            continue;
        }
        uint64_t run = 0;   // start of the run of marked pages; guest address 0 is never a pool's
        uint32_t k = g->mark0;
        for (uint64_t pg = g->addr & ~(SB_PAGE - 1); pg < end; pg += SB_PAGE, k++) {
            uint64_t a = pg > g->addr ? pg : g->addr;
            if (j->marks[k]) {
                if (!run) run = a;
            } else if (run) {
                swb_run(g->provider, run, a);
                run = 0;
            }
        }
        if (run) swb_run(g->provider, run, end);
    }
    // NX2NAVI_WB_PROBE (vk_gpu_written_back): every range is in guest memory now; a later range's thaw would end the
    // probe of a region an earlier one armed, so the probes come last
    for (int i = 0; i < j->n; i++) {
        SwbRange *range = &j->g[i];
        if (pool_host_write_hold(range->provider, range->addr, range->size)) {
            mem_probe(range->addr, range->size);
            mem_read_release();
        }
        destroy_rec(vk_ctx(), range->provider);
    }
    free(j);
}

// vk_mem_mirror_end, before the snapshot chunks go back at the fence (snap_free reuses the marks' chunk): the job of
// this submission's merge dispatches and copies.
static void shadow_wb_cancel(void *arg, int may_have_executed) {
    (void)may_have_executed;
    SwbJob *job = arg;
    for (int i = 0; i < job->n; i++) destroy_rec(vk_ctx(), job->g[i].provider);
    free(job);
}

static void swb_job(uint64_t fence) {
    SwbJob *j = malloc(sizeof *j + sizeof(SwbRange) * (size_t)(nswl + nwbl));
    if (!j) rt_fatal("vk: out of memory");
    j->marks_alloc = NULL, j->marks = NULL;
    if (swl_marks >= 0) {
        VmaAllocationInfo info;
        vmaGetAllocationInfo(vk_ctx()->vma, chunks[swl_marks].alloc, &info);
        j->marks_alloc = chunks[swl_marks].alloc, j->marks = info.pMappedData;
    }
    memcpy(j->g, swl, sizeof *swl * (size_t)nswl);
    j->n = nswl;
    for (int i = 0; i < nwbl; i++) {
        pool_native_acquire(wbl[i].r);
        j->g[j->n++] = (SwbRange){wbl[i].r, wbl[i].r->import_lo + wbl[i].off, wbl[i].size, SWB_ALL};
    }
    nswl = 0, swl_marks = -1;
    vk_completion_add_owned(fence, shadow_wb_cb, shadow_wb_cancel, j);
}

// vk_mem_mirror_end: the merge dispatches of this submission's ranges. Returns the commands recorded; 0 with nsnl
// left as it was when the range list cannot be allocated (the caller copies them plainly). With shadow storage each
// dispatch also marks the pages it changed, for swb_job.
static int native_rmw_bitmap_overlaps(const PoolRec *, uint64_t, uint64_t);

static int wb_merge_record(VkCommandBuffer cb, const SnapRange *selected, int count) {
    VkCtx *c = vk_ctx();
    uint64_t align = c->props.limits.minStorageBufferOffsetAlignment;
    if (align < 16) align = 16;
    SnapRange *s = malloc(sizeof *s * (size_t)count);
    struct MGroup { int i, j; uint64_t base, end; uint32_t total; } *gr = malloc(sizeof *gr * (size_t)count);
    if (!s || !gr) rt_fatal("vk: out of memory");
    memcpy(s, selected, sizeof *s * (size_t)count);
    if (RB.scene) {
        int kept = 0;
        for (int i = 0; i < count; ++i)
            if (!native_rmw_bitmap_overlaps(s[i].r, s[i].off, s[i].size)) s[kept++] = s[i];
        count = kept;
        if (!count) { free(s); free(gr); return 0; }
    }
    qsort(s, (size_t)count, sizeof *s, snap_cmp);
    // the dispatches: ranges of one pool and snapshot chunk, within MERGE_SPAN bytes and MERGE_WORDS words
    int ng = 0;
    uint64_t nmarks = 0;   // bytes of page marks, each dispatch's at an aligned offset
    for (int i = 0; i < count;) {
        uint64_t base = s[i].off & ~(align - 1), end = s[i].off + s[i].size;
        uint32_t total = 0;
        int j = i;
        for (; j < count && s[j].r == s[i].r && s[j].chunk == s[i].chunk; j++) {
            uint64_t e = s[j].off + s[j].size > end ? s[j].off + s[j].size : end;
            if (j > i && (e - base > MERGE_SPAN || total + s[j].size / 4 > MERGE_WORDS)) break;
            end = e;
            total += (uint32_t)(s[j].size / 4);
        }
        gr[ng++] = (struct MGroup){i, j, base, end, total};
        nmarks += (((base & (SB_PAGE - 1)) + end - base + SB_PAGE - 1) / SB_PAGE * 4 + align - 1) & ~(align - 1);
        i = j;
    }
    // the range lists: at most one 16-byte entry per range, each dispatch's list at an aligned offset
    int list = chunk_new((uint64_t)count * 16 + (uint64_t)count * align, CHUNK_LIST);
    if (list < 0) {
        free(s), free(gr);
        return 0;
    }
    chunks[list].used = UINT64_MAX;   // not a snapshot chunk
    int marks = shadow_on ? chunk_new(nmarks, CHUNK_MARKS) : -1;
    uint32_t *mm = NULL;
    if (marks >= 0) {
        chunks[marks].used = UINT64_MAX;
        VmaAllocationInfo mi;
        vmaGetAllocationInfo(c->vma, chunks[marks].alloc, &mi);
        mm = mi.pMappedData;
        memset(mm, 0, nmarks);
        vmaFlushAllocation(c->vma, chunks[marks].alloc, 0, VK_WHOLE_SIZE);
        swl_marks = marks;
    }
    VmaAllocationInfo li;
    vmaGetAllocationInfo(c->vma, chunks[list].alloc, &li);
    uint32_t *lm = li.pMappedData;
    uint64_t lo = 0, mo = 0;
    int rec = 0;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, WM.pipe);
    for (int k = 0; k < ng; k++) {
        const struct MGroup *d = &gr[k];
        lo = (lo + align - 1) & ~(align - 1);
        uint64_t l0 = lo;
        uint32_t first = 0;   // the range's first invocation
        for (int j = d->i; j < d->j; j++) {
            uint32_t *q = lm + lo / 4;
            q[0] = (uint32_t)((s[j].off - d->base) / 4), q[1] = (uint32_t)(s[j].soff / 4), q[2] = (uint32_t)(s[j].size / 4);
            q[3] = first;
            first += (uint32_t)(s[j].size / 4);
            lo += 16;
            mrep.merge_bytes += s[j].size;
        }
        merge_dispatch(cb, s + d->i, d->j - d->i, d->base, d->end - d->base, list, l0, d->total, marks, mo);
        if (shadow_on) {
            uint64_t ga = s[d->i].r->import_lo + d->base;
            if (marks >= 0) swl_add(s[d->i].r, ga, d->end - d->base, (uint32_t)(mo / 4));
            else
                for (int j = d->i; j < d->j; j++)
                    swl_add(s[j].r, s[j].r->import_lo + s[j].off, s[j].size, SWB_ALL);
        }
        mo += (((d->base & (SB_PAGE - 1)) + d->end - d->base + SB_PAGE - 1) / SB_PAGE * 4 + align - 1) & ~(align - 1);
        mrep.merge_dispatches++;
        rec += 3;
    }
    vmaFlushAllocation(c->vma, chunks[list].alloc, 0, VK_WHOLE_SIZE);
    for (int i = 0; i < count; i++) selected[i].r->wb_sub = msub;
    free(s), free(gr);
    return rec;
}

static void snap_chunks_destroy(VkCtx *c, const SnapChunk *owned, int count) {
    for (int i = 0; i < count; i++)
        if (owned[i].buf) resource_core_vma_destroy_buffer(c->vma, owned[i].buf, owned[i].alloc);
}

static void snap_free(void *arg) {   // completion thread: the submission is done with its chunks
    SnapFree *f = arg;
    plat_mutex_lock(&snap_pool_lock);
    for (int i = 0; i < f->n; i++) {
        SnapChunk *ch = &f->c[i];
        if (nsnap_pool == SNAP_POOL_N || snap_pool_bytes + ch->size > snap_pool_cap()) continue;
        snap_pool[nsnap_pool++] = *ch;
        snap_pool_bytes += ch->size;
        ch->buf = VK_NULL_HANDLE;
    }
    plat_mutex_unlock(&snap_pool_lock);
    snap_chunks_destroy(vk_ctx(), f->c, f->n);
    free(f);
}

static void snap_free_canceled(void *arg, int may_have_executed) {
    // The completion queue calls cancellation only for unhanded work or after a proven drain.
    (void)may_have_executed;
    SnapFree *f = arg;
    snap_chunks_destroy(vk_ctx(), f->c, f->n);
    free(f);
}

static void snap_transfer(uint64_t fence) {
    if (!fence || nchunks_transferred == nchunks) return;
    const int count = nchunks - nchunks_transferred;
    SnapFree *f = malloc(sizeof *f + sizeof(SnapChunk) * (size_t)count);
    if (!f) rt_fatal("vk: out of memory");
    f->n = count;
    memcpy(f->c, chunks + nchunks_transferred, sizeof(SnapChunk) * (size_t)count);
    // The final prologue can still read the borrowed descriptors until this fence runs.
    nchunks_transferred = nchunks;
    vk_completion_add_owned(fence, snap_free, snap_free_canceled, f);
}

static void snap_chunks_shutdown(VkCtx *c) {
    // A stopped recorder can skip the scene tail. Sync shutdown and the device drain precede this call.
    if (nchunks_transferred < nchunks)
        snap_chunks_destroy(c, chunks + nchunks_transferred, nchunks - nchunks_transferred);
    free(chunks);
    chunks = NULL;
    nchunks = cchunks = nchunks_transferred = 0;
    snap_completion_fence = 0;
    plat_mutex_lock(&snap_pool_lock);
    snap_chunks_destroy(c, snap_pool, nsnap_pool);
    nsnap_pool = 0;
    snap_pool_bytes = 0;
    plat_mutex_unlock(&snap_pool_lock);
}

// ---- lazy write-back (NX2NAVI_WB_LAZY; the default by platform and driver, wb_lazy_choice.h; memory.c "Lazy write-back")
// A range a shader wrote goes back to guest memory once and is then armed (vk_gpu_written_back, mem_lazy_arm). While it
// stays armed, gpu_wrote leaves the later writes of its pages in the mirror: no snapshot, no merge, no copy. When the
// CPU touches a page, memory.c calls lazy_fetch with the mirror and the imported buffer of the range: a copy of its
// own, sent after every part handed over so far (the held ones first), and waited for. The part that is still being
// recorded when that happens has already left its writes out: at its end (lazy_part_check, from vk_mem_mirror_end or
// from the prologue of a part split off earlier) it writes back, as ever, what it skipped and finds no longer armed;
// the snapshot for that merge is taken in the part's own prologue, before its GPU work. From that check to the part's
// hand-over (vk_mem_lazy_handed) memory.c's window keeps a fault from releasing a range unseen.
static VkCommandPool lz_pool;
static VkCommandBuffer lz_cb;
static VkFence lz_fence;
static volatile int32_t lz_state;   // 0 not asked, 1 on, 2 off
static PlatMutex lz_init_lock = PLAT_MUTEX_INIT;
typedef struct { VkMemSpan sp; uint64_t size; int shader; } LzSkip;
static LzSkip *lz_skip;             // replay: what the part being recorded left in the mirror
static int lz_nskip, lz_cskip;
static int lz_win;                  // replay: the part's window is open (lazy_part_check)
static uint64_t lz_skipped;         // replay: bytes left in the mirror since the last log line

static void mbarrier(VkCommandBuffer cb, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                     VkAccessFlags2 da);

// memory.c's fetch (frz_lock held there, so one at a time): [off, off + size) of the mirror h0 into the imported
// buffer h1, after all the GPU work handed over so far.
static int lazy_fetch_verified(uint64_t h0, uint64_t h1, uint64_t off, uint64_t size, uint64_t guest_addr, void *provider) {
    if (vk_device_is_lost()) return 0;   // the run is ending. No CPU visibility was published.
    VkCtx *c = vk_ctx();
    VkBufferCopy reg = {off, off, size};
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkResetCommandPool(c->device, lz_pool, 0);
    vkBeginCommandBuffer(lz_cb, &bi);
    mbarrier(lz_cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdCopyBuffer(lz_cb, (VkBuffer)(uintptr_t)h0, (VkBuffer)(uintptr_t)h1, 1, &reg);
    mbarrier(lz_cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
             VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
             VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    vkEndCommandBuffer(lz_cb);
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &lz_cb};
    vk_queue_lock();
    vk_sync_flush_held();   // the held submissions wrote first
    // Current recording can still contain unsubmitted writers. Certify only authority already completed here;
    // newer pending/foreign producers retain their exclusions even though this exact copy succeeds.
    RcRegistryVisibility visibility = {0};
    rc_registry_visibility_begin(rc_registry_runtime(), RC_REGISTRY_VISIBILITY_COMPLETED, &visibility);
    VkResult r = vkQueueSubmit(c->gfx_queue, 1, &si, lz_fence);
    vk_queue_unlock();
    if (r == VK_SUCCESS) r = vkWaitForFences(c->device, 1, &lz_fence, VK_TRUE, 10000000000ull);
    if (r != VK_SUCCESS) {
        vk_device_lost_check(r, "the submit or the wait that brings GPU writes down to guest memory");
        rt_fatal("vk: bringing %llu bytes the GPU wrote down to guest memory failed (%d)", (unsigned long long)size, (int)r);
    }
    rc_registry_visibility_apply(rc_registry_runtime(), &visibility, (RcSourceRange){1, guest_addr, size});
    // frz_lock protects the segment from pool retirement or native host mapping mutation. No current lookup.
    pool_host_visibility(&visibility, provider, guest_addr, size);
    vkResetFences(c->device, 1, &lz_fence);
    return 1;
}

static void lazy_fetch(uint64_t h0, uint64_t h1, uint64_t off, uint64_t size, uint64_t guest_addr, void *provider) {
    (void)lazy_fetch_verified(h0, h1, off, size, guest_addr, provider);
}

// memory.c's test before it arms [lo, hi) of the pool ctx (frz_lock held there, the pages already inaccessible): the
// mirror holds every CPU write to them. A fetch copies the mirror over the whole range, not the merge of the bytes the
// GPU changed that a write-back does, so a CPU write that came after the range's write-back reached guest memory and
// before this arm, and that no sync of the mirror has carried up yet, would be overwritten by the mirror's older bytes:
// a page stamped after the mirror's last sync (the CPU-write table, mgen), or a write-watch bit nobody has read yet.
// No lock: the pages are inaccessible, so nothing stamps them meanwhile; the mirror's mgen only grows.
static int lazy_clean(uint64_t lo, uint64_t hi, void *ctx) {
    const PoolRec *r = ctx;
    uint32_t mgen = __atomic_load_n(&r->mgen, __ATOMIC_ACQUIRE);
    for (uint64_t a = lo; a < hi; a += 1ull << GEN_PAGE_SHIFT) {
        const uint32_t *ch = __atomic_load_n(&cgen_chunk[a >> GEN_CHUNK_SHIFT], __ATOMIC_ACQUIRE);
        if (ch && __atomic_load_n(&ch[(a >> GEN_PAGE_SHIFT) & (GEN_PER_CHUNK - 1)], __ATOMIC_RELAXED) > mgen) return 0;
    }
    uint64_t p[1];
    return mem_written_peek(lo, hi - lo, p, 1) == 0;   // UINT64_MAX (not watched) counts as written
}

// Whether the lazy write-back is wanted on this device, decided and logged once, when the driver is known
// (vk_device.c create_device, after vk_caps_init): the player's setting, else the default of the platform and driver
// (wb_lazy_choice.h). Never in import mode, where no pool has a mirror; with shadow storage, where a fetch would copy
// the mirror into the pool's buffer, which is guest memory only when the pool is imported (a shadow's bytes reach
// guest memory in the submission's shadow_wb_cb); or with the write-back probe, which watches the same pages.
_Static_assert(WBL_DRIVER_AMD == VK_DRIVER_ID_AMD_PROPRIETARY, "wb_lazy_choice.h names AMD's own driver");
static int lz_wanted = -1;          // -1 not decided, 0 off, 1 on
static const char *lz_why = "";     // the reason the log gave

void vk_mem_lazy_decide(void) {
    const char *set = cfg_is_set("NX2NAVI_WB_LAZY") ? cfg_get("NX2NAVI_WB_LAZY") : NULL;
    WblChoice ch = wb_lazy_choice(set, WBL_ON_WINDOWS, vk_caps.driver_id);
    if (!vk_mirror_on) ch = (WblChoice){0, "no pool has a mirror in import mode"};
    else if (shadow_on) ch = (WblChoice){0, "shadow storage keeps it off"};
    else if (ch.on && cfg_bool("NX2NAVI_WB_PROBE")) ch = (WblChoice){0, "the write-back probe watches the same pages"};
    lz_why = ch.why;
    __atomic_store_n(&lz_wanted, ch.on, __ATOMIC_RELEASE);
    rt_log("mem: lazy write-back %s: %s", ch.on ? "on" : "off", ch.why);
}

static int wb_lazy(void) {
    int s = __atomic_load_n(&lz_state, __ATOMIC_ACQUIRE);
    if (s) return s == 1;
    plat_mutex_lock(&lz_init_lock);
    if (!lz_state) {
        VkCtx *c = vk_ctx();
        int on = c && __atomic_load_n(&lz_wanted, __ATOMIC_ACQUIRE) == 1;
        if (on) {
            VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                          .queueFamilyIndex = c->gfx_family};
            VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            on = vkCreateCommandPool(c->device, &pi, NULL, &lz_pool) == VK_SUCCESS;
            VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = lz_pool,
                                              .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
            on = on && vkAllocateCommandBuffers(c->device, &ai, &lz_cb) == VK_SUCCESS &&
                 vkCreateFence(c->device, &fi, NULL, &lz_fence) == VK_SUCCESS;
            on = on && mem_lazy_enable(lazy_fetch, lazy_clean, lazy_fetch_verified);
        }
        rt_log("mem: what shaders write into mirrored pools %s (lazy write-back: %s)",
               on ? "goes back to guest memory once and then only when the CPU touches it"
                  : "goes back to guest memory at the end of every submission",
               on || lz_wanted != 1 ? lz_why : "its command buffer and fence could not be made");
        __atomic_store_n(&lz_state, on ? 1 : 2, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&lz_init_lock);
    return lz_state == 1;
}

static void gpu_wrote_range(const VkMemSpan *sp, uint64_t size, int shader);

// NX2NAVI_WB_SKIP_MB (default 0, off), a test switch for drivers that run without the lazy write-back (AMD's, and
// outside Windows: wb_lazy_choice.h): there, what a binding written by a shader or a copy holds goes back to guest
// memory at the end of every submission, up to 66 MiB a frame in Hyrule Field. With it, a written range of at least
// this many MiB goes back only for its first WB_SKIP_HEAD bytes and the rest stays in the mirror. The CPU reads a few
// words at the start of such buffers (12 bytes into ranges of 6.3 and 15.2 MiB, see "Lazy write-back" in memory.c),
// which the head still carries; a CPU read further in would see older bytes, and a CPU write further in would have the
// next sync copy that page's older guest bytes over what the GPU wrote there. The pool's mirror is never evicted from
// then on, since guest memory lacks those bytes.
#define WB_SKIP_HEAD (64u << 10)
static uint64_t wb_skip(void) {
    static int64_t at = -1;
    if (at < 0) {
        int mb = cfg_int("NX2NAVI_WB_SKIP_MB");
        at = mb > 0 ? (int64_t)mb << 20 : 0;
        if (at)
            rt_log("mem: without the lazy write-back, written ranges of %d MiB or more go back to guest memory only for "
                   "their first %u KiB (NX2NAVI_WB_SKIP_MB)", mb, WB_SKIP_HEAD >> 10);
    }
    return (uint64_t)at;
}
static int64_t m_wb_skipped;    // bytes wb_skip kept in mirrors since the last summary

static void gpu_wrote(const VkMemSpan *sp, uint64_t size, int shader) {
    PoolRec *r = sp->pool;
    if (!r || r->mstate != MS_ON || sp->buffer != r->mbuf || !size || !wb_lazy()) {
        uint64_t skip = r && r->mstate == MS_ON && sp->buffer == r->mbuf ? wb_skip() : 0;
        if (skip && size >= skip && !wb_lazy()) {
            gpu_wrote_range(sp, WB_SKIP_HEAD, shader);
            r->wb_kept = 1;
            plat_add64(&m_wb_skipped, (int64_t)(size - WB_SKIP_HEAD));
            return;
        }
        gpu_wrote_range(sp, size, shader);
        return;
    }
    uint64_t g0 = r->import_lo + sp->offset, g1 = g0 + size, run = g0, out[16][2];
    int k = mem_lazy_skip(g0, size, (uint64_t)(uintptr_t)r->mbuf, -(int64_t)r->import_lo, r, out, 16);
    for (int i = 0; i <= k; i++) {
        uint64_t lo = i < k ? out[i][0] : g1, hi = i < k ? out[i][1] : g1;
        if (lo > run) {   // not armed: written back as ever
            VkMemSpan part = *sp;
            part.offset = sp->offset + (run - g0);
            gpu_wrote_range(&part, lo - run, shader);
        }
        if (i < k) {
            if (lz_nskip == lz_cskip) grow((void **)&lz_skip, &lz_cskip, sizeof *lz_skip);
            LzSkip k = {*sp, hi - lo, shader};
            k.sp.offset = sp->offset + (lo - g0);
            lz_skip[lz_nskip++] = k;
            lz_skipped += hi - lo;
        }
        run = hi;
    }
}

// The end of the part being recorded: what it left in the mirror for a range that is no longer armed (the CPU touched
// it meanwhile) is written back after all. Opens memory.c's window, which vk_mem_lazy_handed closes.
static void lazy_part_check(void) {
    if (lz_win || !lz_nskip) return;
    mem_lazy_window(1);
    lz_win = 1;
    for (int i = 0; i < lz_nskip; i++) {
        const PoolRec *r = lz_skip[i].sp.pool;
        if (!mem_lazy_armed(r->import_lo + lz_skip[i].sp.offset, lz_skip[i].size))
            gpu_wrote_range(&lz_skip[i].sp, lz_skip[i].size, lz_skip[i].shader);
    }
}

// vk_sync.c, before a part's prologue is recorded: the check for a part that was split off (the last part's ran in
// vk_mem_mirror_end already).
void vk_mem_lazy_part_end(void) { lazy_part_check(); }

// vk_sync.c, once the part is held or handed to the queue: a fetch from here on has its writes.
void vk_mem_lazy_handed(void) {
    lz_nskip = 0;
    if (lz_win) mem_lazy_window(0), lz_win = 0;
}

static void gpu_wrote_range(const VkMemSpan *sp, uint64_t size, int shader) {
    PoolRec *r = sp->pool;
    if (!r || r->mstate != MS_ON || sp->buffer != r->mbuf || !size) return;
    r->vd_wrote_sub = msub, r->vd_wrote_n++;
    // A binding to the end of its pool (no size) can span hundreds of MiB: those stay copies.
    uint64_t o = 0;
    if (shader && !(sp->offset & 3) && !(size & 3) && size <= merge_max() && merging())
        for (uint64_t n; o < size; o += n)
            if (!snap_add(r, sp->offset + o, n = size - o < MERGE_PIECE ? size - o : MERGE_PIECE)) break;
    r->st_merge += o;
    if (o == size) return;
    r->st_wb += size - o;
    range_add(&wbl, &nwbl, &cwbl, r, sp->offset + o, size - o);   // what found no snapshot room
}

// The current import can replace a mirror only when it already contains the authoritative bytes.
// A waited shader prefix alone says nothing about its eventual mirror-to-import tail.
static int mirror_transition_ready_locked(const PoolRec *pool, RcRegistryPosition read) {
    if (!pool->mbuf || !pool->malloc_ || !pool->mirror_incarnation || !pool->host_backing ||
        !pool->gpa_host_coherent || pool->wb_kept || pool->import_hi <= pool->import_lo ||
        pool->wb_fence > vk_sync_fence_done() ||
        (pool->gpa_rec && !vk_sync_mirror_publication_safe(read.scene))) return 0;
    for (int i = 0; i < nwbl; ++i) if (wbl[i].r == pool) return 0;
    for (int i = 0; i < nsnl; ++i) if (snl[i].r == pool) return 0;
    for (int i = 0; i < lz_nskip; ++i) if (lz_skip[i].sp.pool == pool) return 0;
    const uint64_t extent = pool->import_hi - pool->import_lo;
    if (!mem_backing_current_range(pool->host_backing, pool->import_lo, extent)) return 0;
    RcRegistry *registry = rc_registry_runtime();
    const RcSourceRange guest = {RC_GUEST_COMPATIBILITY_BACKING, pool->import_lo, extent};
    if (rc_registry_read_range_at(registry, guest, read) != RC_REGISTRY_OK) return 0;
    uint32_t count;
    const RcInterval *parts = mem_backing_intervals(pool->host_backing, &count);
    uint64_t cursor = pool->import_lo;
    for (uint32_t i = 0; i < count; ++i) {
        const RcInterval *piece = &parts[i];
        if (!piece->mapping || piece->address != cursor || !piece->range.size ||
            piece->range.size > pool->import_hi - cursor ||
            piece->range.backing == RC_GUEST_COMPATIBILITY_BACKING ||
            rc_registry_read_range_at(registry, piece->range, read) != RC_REGISTRY_OK) return 0;
        cursor += piece->range.size;
    }
    return cursor == pool->import_hi;
}
void vk_mem_gpu_wrote(const VkMemSpan *sp, uint64_t size) {
    int lk = vk_exec_shared();
    if (lk) plat_mutex_lock(&use_lock);
    gpu_wrote(sp, size, 0);
    if (lk) plat_mutex_unlock(&use_lock);
}
void vk_mem_gpu_wrote_exact(const VkMemSpan *sp, uint64_t size) {
    int locked = vk_exec_shared();
    if (locked) plat_mutex_lock(&use_lock);
    gpu_wrote_range(sp, size, 0);
    if (locked) plat_mutex_unlock(&use_lock);
}

void vk_mem_gpu_wrote_shader(const VkMemSpan *sp, uint64_t size) {
    int lk = vk_exec_shared();
    if (lk) plat_mutex_lock(&use_lock);
    gpu_wrote(sp, size, 1);
    if (lk) plat_mutex_unlock(&use_lock);
}

// Adds [a, b) to l, n sorted ranges that do not touch; returns the new count. Above cap ranges (l holds cap + 1) the two
// nearest merge, so a full list copies the fewest bytes that were not flushed; cap 1 keeps one span over all flushes.
static int flush_add(FlushRange *l, int n, uint64_t a, uint64_t b, int cap) {
    int i = 0;
    while (i < n && l[i].hi < a) i++;
    int j = i;
    for (; j < n && l[j].lo <= b; j++) {
        if (l[j].lo < a) a = l[j].lo;
        if (l[j].hi > b) b = l[j].hi;
    }
    memmove(l + i + 1, l + j, (size_t)(n - j) * sizeof *l);
    l[i] = (FlushRange){a, b};
    n += 1 - (j - i);
    if (n > cap) {
        int k = 0;
        for (int x = 1; x + 1 < n; x++)
            if (l[x + 1].lo - l[x].hi < l[k + 1].lo - l[k].hi) k = x;
        l[k].hi = l[k + 1].hi;
        memmove(l + k + 1, l + k + 2, (size_t)(n - k - 2) * sizeof *l);
        n--;
    }
    return n;
}

int vk_mem_flush_add_probe(void) {
    FlushRange l[5];
    int n = 0;
    n = flush_add(l, n, 0x8000, 0x9000, 4);
    n = flush_add(l, n, 0x1000, 0x2000, 4);
    n = flush_add(l, n, 0x4000, 0x5000, 4);
    if (n != 3 || l[0].lo != 0x1000 || l[1].lo != 0x4000 || l[2].hi != 0x9000) return 1;   // kept apart, sorted
    n = flush_add(l, n, 0x2000, 0x3000, 4);
    if (n != 3 || l[0].hi != 0x3000) return 2;                                             // touching ranges join
    n = flush_add(l, n, 0x800, 0x8800, 4);
    if (n != 1 || l[0].lo != 0x800 || l[0].hi != 0x9000) return 3;                         // one covers several
    n = 0;
    n = flush_add(l, n, 0x10000, 0x11000, 2);
    n = flush_add(l, n, 0x40000, 0x41000, 2);
    n = flush_add(l, n, 0x12000, 0x13000, 2);                                              // over the cap: nearest merge
    if (n != 2 || l[0].lo != 0x10000 || l[0].hi != 0x13000 || l[1].lo != 0x40000) return 4;
    n = 0;
    n = flush_add(l, n, 0x10000, 0x11000, 1);
    n = flush_add(l, n, 0x40000, 0x41000, 1);
    if (n != 1 || l[0].lo != 0x10000 || l[0].hi != 0x41000) return 5;                      // NX2NAVI_FLUSH_SPAN=1
    return 0;
}

uint64_t vk_mem_mirror_uploaded(void) {
    plat_mutex_lock(&mirror_stat_lock);
    uint64_t n = macc.upload_bytes;
    plat_mutex_unlock(&mirror_stat_lock);
    return n;
}

void nvn_pool_flushed(const pool_t *p, uint64_t off, uint64_t size) {
    if (p && p->mem && size && off <= UINT64_MAX - (uintptr_t)p->mem) {
        uint64_t addr = (uintptr_t)p->mem + off;
        rc_registry_cpu_changed(rc_registry_runtime(), (RcSourceRange){1, addr, size});
        mem_backing_cpu_changed(addr, size);
    }
    if (!vk_mirror_on || !size || !p || !p->mem) return;
    plat_inc64(&m_flush_calls);
    plat_add64(&m_flush_bytes, (int64_t)size);
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *e = pidx_find(&pool_index, (uintptr_t)p->mem, 0);
    PoolRec *r = e && e->key == p ? e->buf : NULL;
    if (r && r->mstate == MS_ON) {
        uint64_t a = ((uintptr_t)p->mem + off) & ~0xfffull, b = ((uintptr_t)p->mem + off + size + 0xfff) & ~0xfffull;
        plat_mutex_lock(&flush_lock);
        __atomic_store_n(&r->fl_n, flush_add(r->fl, r->fl_n, a, b, flush_cap), __ATOMIC_RELEASE);
        r->fl_calls++, r->fl_sum += size;
        plat_mutex_unlock(&flush_lock);
    }
    plat_mutex_unlock_shared(&pool_lock);
}

// gen_lock exclusive: freezes up to FREEZE_PER_SUB candidate regions still idle. The query after the
// protection collects the writes made before it took effect.
// A freeze is a VirtualQuery and a VirtualProtect of 2 MiB, some microseconds as a rule; in the second after an area
// load, while the loaders write and the driver imports and locks new memory, each took 8 to 17 ms. The newly loaded
// regions all go idle together, so 32 of them came up in one submission: its mirror begin took 70.5 ms and the frame
// 78 ms, and one freeze of 9 ms followed in each of the next 20 frames. A freeze only saves later write-watch queries
// (about a microsecond per MiB per submission), so it can wait. Submissions share a budget between presents, and a
// freeze over FREEZE_SLOW_NS holds the passes for FREEZE_BACKOFF_NS, doubled for each slow one in a row up to 2 s.
// NX2NAVI_FREEZE_BUDGET_US (default 1000; 0: no budget and no backoff, as before).
#define FREEZE_SLOW_NS 2000000ull
#define FREEZE_BACKOFF_NS 250000000ull
#define FREEZE_BACKOFF_MAX_NS 2000000000ull
static uint64_t freeze_budget_ns = ~0ull;   // ~0: not read yet
static uint64_t frz_hold_until, frz_backoff, frz_held_passes, frz_slow;   // gen_lock
// All freeze passes run under gen_lock. A frame can submit several command
// buffers. Charge their maintenance work to the same present interval.
static uint64_t freeze_frame, freeze_spent_ns;

static void freeze_pass(void) {
    if (freeze_budget_ns == ~0ull) {
        long long us = cfg_int("NX2NAVI_FREEZE_BUDGET_US");
        freeze_budget_ns = us > 0 ? (uint64_t)us * 1000 : 0;
    }
    uint64_t start = qpc();
    if (freeze_budget_ns && start < frz_hold_until) {
        if (nfcand) frz_held_passes++;
        return;
    }
    uint64_t available = freeze_budget_ns;
    if (available) {
        uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (frame != freeze_frame) {
            freeze_frame = frame;
            freeze_spent_ns = 0;
        }
        if (freeze_spent_ns >= available) return;
        available -= freeze_spent_ns;
    }
    uint32_t preferred = 0;
    int priority = clean_attempt && freeze_budget_ns && clean_live && (!nfcand || !clean_normal_first) &&
                   clean_next(&preferred);
    if (clean_attempt) clean_normal_first = priority && nfcand;
    for (int done = 0, inspected = 0; (priority || nfcand) && done < FREEZE_PER_SUB;) {
        // Expired offers also consume the present interval's maintenance budget.
        if ((++inspected % 16) == 0 && available && qpc() - start >= available) break;
        int selected = priority;
        FreezeCandidate candidate = selected ? (FreezeCandidate){preferred, 0} : fcand[--nfcand];
        uint64_t j = candidate.region, a = j << REG_SHIFT;
        if (!selected && candidate.idle < (uint32_t)freeze_idle) texture_queued[j] = 0;
        priority = 0;
        if (!selected && (ww_ep - reg_last[j] < candidate.idle || mem_frozen(a) ||
                          (candidate.idle < (uint32_t)freeze_idle && (!snap_lease_valid((uint32_t)j) ||
                           (candidate.idle == TEXTURE_MIN_IDLE ? !texture_cost_ready(&texture_cost[j]) :
                            (!texture_empty_epoch[j] || ww_ep - texture_empty_epoch[j] > 24)))) ||
                          (clean_attempt && clean_recent(ww_ep, clean_attempt[j], (uint32_t)freeze_idle)))) continue;
        if (!selected && candidate.idle < (uint32_t)freeze_idle) reg_tried[j] = ww_ep;
        if (clean_attempt) clean_attempt[j] = ww_ep;
        if (selected) clean_stats.attempts++;
        done++;
        uint64_t t0 = qpc();
        MemFreezeTransition transition;
        int ok = mem_freeze_observed(a, &transition);
        uint64_t t1 = qpc();
        if (selected && ok) clean_stats.successes++;
        if (!selected && candidate.idle < (uint32_t)freeze_idle) {
            texture_tries++;
            texture_protected += ok != 0;
        }
        slow_mem_call("freeze", a, 1ull << REG_SHIFT, 0, t1 - t0);
        if (ok) observe_after_freeze((uint32_t)j, &transition);
        else n_frz_refused++;
        if (!selected && candidate.idle < (uint32_t)freeze_idle) {
            if (candidate.idle == TEXTURE_MIN_IDLE) {
                texture_cost_tries++;
                texture_cost_protected += ok != 0;
            }
            if (ok) {
                uint64_t ns = qpc() - t0;
                if (ns < TEXTURE_MIN_COST_NS) ns = TEXTURE_MIN_COST_NS;
                if (ns > TEXTURE_MAX_COST_NS) ns = TEXTURE_MAX_COST_NS;
                texture_freeze_ns = (uint32_t)((texture_freeze_ns * 15 + ns) / 16);
            }
        }
        if (!freeze_budget_ns) continue;
        if (t1 - t0 >= FREEZE_SLOW_NS) {
            frz_slow++;
            frz_backoff = frz_backoff ? frz_backoff * 2 : FREEZE_BACKOFF_NS;
            if (frz_backoff > FREEZE_BACKOFF_MAX_NS) frz_backoff = FREEZE_BACKOFF_MAX_NS;
            frz_hold_until = t1 + frz_backoff;
            static uint32_t logged;
            if (++logged <= 40)
                rt_log("mem: freezing an idle region took %.1f ms: no freezes for %.0f ms (%llu slow freezes so far, %llu "
                       "passes held, %d regions waiting)", (t1 - t0) / 1e6, frz_backoff / 1e6, (unsigned long long)frz_slow,
                       (unsigned long long)frz_held_passes, nfcand);
            break;
        }
        frz_backoff = 0;
        if (qpc() - start >= available) break;
    }
    if (freeze_budget_ns) freeze_spent_ns += qpc() - start;
}

void vk_mem_mirror_begin(void) {
    // Any previous recorder that skipped its writeback tail has already retained its exact owners
    // in completion callbacks. Drop only the scene traversal links before starting a new scene.
    RW.scene = NULL;
    RW.workspace = NULL;
    RW.word_count = 0;
    RB.scene = NULL;
    RB.count = 0;
    RP_SCOPE(rp_mirror, RP_MIRROR_BEGIN);
    if (!vk_mirror_on) {   // import mode: the texture checks' snapshots still skip frozen regions
        RP_VALUE(RPK_OUTCOME, RPO_IMPORTED);
        if (!freeze_idle) return;
        RP_CALL(RP_MIRROR_LOCK, plat_mutex_lock(&gen_lock));
        ww_ep++;
        RP_CALL(RP_MIRROR_FREEZE, freeze_pass());
        plat_mutex_unlock(&gen_lock);
        return;
    }
    uint64_t t0 = qpc();
    nsnl = snl_copied = nchunks = nchunks_transferred = 0;
    snap_completion_fence = 0;
    uint32_t prev = msub;
    msub = msub + 1 ? msub + 1 : 1;
    __atomic_add_fetch(&mirror_gen, 1, __ATOMIC_RELEASE);
    mepoch = vk_texture_epoch();
    uint64_t q = 0, run0 = 0, run1 = 0;
    int have = 0;
    RP_SCOPE(rp_mirror_locks, RP_MIRROR_LOCK);
    plat_mutex_lock_shared(&pool_lock);
    plat_mutex_lock(&gen_lock);
    RP_END(rp_mirror_locks);
    ww_ep++;
    uint64_t b0 = ww_bytes;
    int pl = sub_place(), pp = presnap_place();
    if (pl <= pplace) pframe++;   // a new frame, though its first submissions may have had nothing to record
    pplace = pl;
    RP_SCOPE(rp_queries, RP_MIRROR_QUERY);
    for (int i = 0; i < nm; i++) {
        const PoolRec *r = mlist[i];
        if (pp ? !(pm_last(r) >> pl & 1) : r->acc_sub != prev) continue;
        uint64_t a = r->import_lo >> SNAP_SHIFT, b = (r->import_hi - 1) >> SNAP_SHIFT;
        if (have && a <= run1 + 1 + (uint64_t)mirror_gap) {
            if (b > run1) run1 = b;
            continue;
        }
        if (have) q += snap_regions(run0, run1, mepoch);
        run0 = a, run1 = b, have = 1;
    }
    if (have) q += snap_regions(run0, run1, mepoch);
    RP_VALUE(RPK_QUERIES, q);
    RP_VALUE(RPK_WW_BYTES, ww_bytes - b0);
    RP_END(rp_queries);
    mrep.presnap_bytes += ww_bytes - b0;
    if (freeze_idle) RP_CALL(RP_MIRROR_FREEZE, freeze_pass());
    plat_mutex_unlock(&gen_lock);
    plat_mutex_unlock_shared(&pool_lock);
    RP_VALUE(RPK_OUTCOME, RPO_MIRRORED);
    mrep.subs++;
    mrep.presnap_queries += q;
    mrep.presnap_t += qpc() - t0;
    slow_mem_call("mirror begin (its queries and freezes)", 0, ww_bytes - b0, q, qpc() - t0);
}

int vk_mem_mirror_pending(void) { return npend || snl_copied < nsnl; }

// NX2NAVI_MIRROR_CAPTURE_KB (default 0, off): the pages the CPU wrote since a VRAM mirror's last sync are copied out of
// guest memory by the CPU when the part that needs them is handed to the queue, up to this many KiB a part, into a
// host buffer the submission owns (a chunk of the snapshot pool, given back at its fence); the prologue copies them
// from there into the mirror. Without it the prologue copies them from guest memory when the GPU runs it, a frame or
// more after the game's submit when the GPU is behind: the GPU then reads whatever the game wrote there meanwhile,
// values meant for a later frame. A pool stays with the GPU-time copy while a write-back of its mirror into guest
// memory is still on the GPU (the CPU would read guest bytes that write-back has not reached yet and copy them over
// the mirror's newer ones), and when its device address table writes reach guest memory as well (a twin: what they
// write is not recorded); so does a range with a page that cannot be read without a fault (mem_copy_readable: a
// lazily armed range). The whole-pool copy of a mirror's first use is larger than the limit and stays with the GPU.
static uint64_t capture_cap(void) {
    static int64_t cap = -1;
    if (cap < 0) {
        int kb = cfg_int("NX2NAVI_MIRROR_CAPTURE_KB");
        cap = kb > 0 ? (int64_t)kb << 10 : 0;
        if (cap)
            rt_log("mem: mirror pages the CPU wrote are copied out of guest memory when their part is handed over, up to "
                   "%d KiB a part (NX2NAVI_MIRROR_CAPTURE_KB)", kb);
    }
    return (uint64_t)cap;
}

static MRange *capm;   // replay: mirror_capture's merged ranges
static uint64_t *caps; // replay: the chunk offset of each, UINT64_MAX for one left to the GPU-time copy
static uint8_t **capd, *capok;   // replay: mem_copy_readable's destinations and results
static uint64_t (*capsrc)[2];    // replay: and its sources
static int ccapm, ccaps, ccapd, ccapsrc, ccapok;

// Prologue, before copy_ranges: takes the ranges of pend the CPU copies (capture_cap) out of it and records their
// copies from the chunk into the mirrors; what is left in pend goes the GPU-time way.
static void mirror_capture(VkCommandBuffer cb, uint64_t scene) {
    uint64_t cap = capture_cap();
    if (!cap || shadow_on || !npend) return;
    uint64_t t0 = qpc(), done = vk_sync_fence_done(), total = 0;
    qsort(pend, (size_t)npend, sizeof *pend, range_cmp);
    int n = 0;
    for (int i = 0; i < npend; i++) {   // merged as copy_ranges merges them
        PoolRec *r = pend[i].r;
        uint64_t lim = r->import_hi - r->import_lo, lo = pend[i].off, hi = pend[i].off + pend[i].size;
        if (hi > lim) hi = lim;
        if (lo >= hi) continue;
        if (n && capm[n - 1].r == r && lo <= capm[n - 1].off + capm[n - 1].size) {
            if (hi > capm[n - 1].off + capm[n - 1].size) capm[n - 1].size = hi - capm[n - 1].off;
            continue;
        }
        if (n == ccapm) grow((void **)&capm, &ccapm, sizeof *capm);
        capm[n++] = (MRange){r, lo, hi - lo};
    }
    while (ccaps < ccapm) grow((void **)&caps, &ccaps, sizeof *caps);
    for (int i = 0; i < n; i++) {
        const PoolRec *r = capm[i].r;
        uint64_t sz = (capm[i].size + 15) & ~15ull;
        caps[i] = UINT64_MAX;
        if (r->twin || r->wb_fence > done || total + sz > cap) continue;
        caps[i] = total;
        total += sz;
    }
    int k = total ? chunk_new(total, CHUNK_LIST) : -1;
    uint8_t *map = NULL;
    if (k >= 0) {
        chunks[k].used = UINT64_MAX;   // not a snapshot chunk
        VmaAllocationInfo info;
        vmaGetAllocationInfo(vk_ctx()->vma, chunks[k].alloc, &info);
        map = info.pMappedData;
    }
    // the copies out of guest memory, all under one hold of memory.c's freeze lock
    while (ccapd < ccapm) grow((void **)&capd, &ccapd, sizeof *capd);
    while (ccapsrc < ccapm) grow((void **)&capsrc, &ccapsrc, sizeof *capsrc);
    while (ccapok < ccapm) grow((void **)&capok, &ccapok, sizeof *capok);
    int nc = 0;
    for (int i = 0; map && i < n; i++)
        if (caps[i] != UINT64_MAX) {
            capd[nc] = map + caps[i];
            capsrc[nc][0] = capm[i].r->import_lo + capm[i].off, capsrc[nc][1] = capm[i].size;
            nc++;
        }
    if (nc) mem_copy_readable(capd, (const uint64_t (*)[2])capsrc, nc, capok);
    // Without a successful flush these host bytes cannot supply an upload. The original
    // import copy remains available for every range, as when capture admission is refused.
    if (map && vmaFlushAllocation(vk_ctx()->vma, chunks[k].alloc, 0, VK_WHOLE_SIZE) != VK_SUCCESS)
        map = NULL;
    npend = 0;
    for (int i = 0, c = 0; i < n;) {
        PoolRec *r = capm[i].r;
        int nr = 0, left = 0;
        for (; i < n && capm[i].r == r; i++) {
            int got = map && caps[i] != UINT64_MAX && capok[c++];
            if (got) {
                if (nr == cregions) grow((void **)&mregions, &cregions, sizeof *mregions);
                mregions[nr++] = (VkBufferCopy){caps[i], capm[i].off, capm[i].size};
                mrep.cap_bytes += capm[i].size, mrep.cap_ranges++;
                continue;
            }
            range_add(&pend, &npend, &cpend, r, capm[i].off, capm[i].size);
            left = 1;
            mrep.cap_left += capm[i].size;
        }
        if (nr) {
            vkCmdCopyBuffer(cb, chunks[k].buf, r->mbuf, (uint32_t)nr, mregions);
            mirror_record_upload(r, mregions, nr, scene, !left);
        }
    }
    mrep.cap_t += qpc() - t0;
}

void vk_mem_mirror_prologue(VkCommandBuffer cb, uint64_t scene) {
    const uint64_t completion_fence = snap_completion_fence;
    snap_completion_fence = 0;
    if (!npend && snl_copied == nsnl) return;
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    mirror_capture(cb, scene);
    copy_ranges(cb, pend, npend, 1, &mrep.upload_bytes, &mrep.upload_copies, NULL, scene);
    if (snl_copied < nsnl) {   // the snapshots read what the uploads above wrote
        if (npend) mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                            VK_ACCESS_2_TRANSFER_READ_BIT);
        snap_copies(cb);
    }
    mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
             VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    npend = 0;
    mrep.prologues++;
    // The final prologue can allocate a host capture chunk after mirror_end transferred the earlier chunks.
    snap_transfer(completion_fence);
}

// ---- verify mode
typedef struct { PoolRec *r; uint64_t off, len, dst; uint32_t mgen; } VSample;
typedef struct { uint32_t sub; int n; VkBuffer buf; VmaAllocation alloc; const uint8_t *map; VSample s[VERIFY_SAMPLES]; } VJob;
static int ver_i;
static uint64_t ver_off;

// The CPU wrote the page since the mirror's sync that mgen records: stamped later, or its write-watch bit is set.
static int page_dirty(uint64_t a, uint32_t mgen) {
    uint32_t *s = gen_slot(a, 0);
    if (s && __atomic_load_n(s, __ATOMIC_RELAXED) > mgen) return 1;
    uint64_t p[1];
    return mem_written_peek(a, 1ull << GEN_PAGE_SHIFT, p, 1) != 0;   // UINT64_MAX (not watched) counts as written
}

static void verify_cb(void *arg) {   // completion thread
    VJob *j = arg;
    static volatile int32_t logged;
    uint64_t pages = 0, skipped = 0, bad = 0;
    for (int i = 0; i < j->n; i++) {
        const VSample *s = &j->s[i];
        PoolRec *r = s->r;
        uint32_t wb = __atomic_load_n(&r->wb_sub, __ATOMIC_RELAXED);
        if (wb != j->sub && wb - j->sub < 0x80000000u) {   // a later submission wrote back into it
            skipped += s->len >> GEN_PAGE_SHIFT;
            continue;
        }
        for (uint64_t o = 0; o < s->len; o += 1ull << GEN_PAGE_SHIFT) {
            uint64_t a = r->import_lo + s->off + o;
            const uint8_t *m = j->map + s->dst + o, *g = (const uint8_t *)(uintptr_t)a;
            pages++;
            if (!memcmp(m, g, 1ull << GEN_PAGE_SHIFT)) continue;
            // gen_lock: a write-watch query clears the page's bit and stamps it in one hold (vk_mem_changed, mirror_sync),
            // so the check sees one or the other; without it a page between the two counted as a mismatch.
            plat_mutex_lock_shared(&gen_lock);
            int dirty = page_dirty(a, s->mgen);
            plat_mutex_unlock_shared(&gen_lock);
            if (dirty) { skipped++; continue; }
            bad++;
            if (plat_inc32(&logged) <= VERIFY_LOG) {
                int k = 0, e = (1 << GEN_PAGE_SHIFT) - 1;
                while (k < e && m[k] == g[k]) k++;
                while (e > k && m[e] == g[e]) e--;
                char f[160];
                flag_names(r->flags, f, sizeof f);
                rt_log("mem: verify mismatch: pool %#llx+%#llx (%s) offset %#llx: bytes %#x..%#x of the page differ, "
                       "mirror %02x guest %02x at the first (submission %u)", (unsigned long long)r->lo,
                       (unsigned long long)r->size, f, (unsigned long long)(a - r->lo), k, e, m[k], g[k], j->sub);
                const uint32_t *gs = gen_slot(a & ~((1ull << GEN_PAGE_SHIFT) - 1), 0);
                char mh[100], gh[100];
                for (int b = 0; b < 16; b++) snprintf(mh + 3 * b, 4, "%02x ", m[b]), snprintf(gh + 3 * b, 4, "%02x ", g[b]);
                rt_log("mem: verify mismatch detail: written binds %llu, last %u; used %u; written back %u; gpa record %u; "
                       "page cpu stamp %u (synced to %u, now %u); mirror %s| guest %s",
                       (unsigned long long)r->vd_wrote_n, r->vd_wrote_sub, r->vd_use_sub, wb, r->gpa_rec, gs ? *gs : 0,
                       s->mgen, gen_now, mh, gh);
            }
        }
    }
    plat_add64(&v_pages, (int64_t)pages);
    plat_add64(&v_skipped, (int64_t)skipped);
    plat_add64(&v_mismatch, (int64_t)bad);
    resource_core_vma_destroy_buffer(vk_ctx()->vma, j->buf, j->alloc);
    free(j);
}

static int verify_record(VkCommandBuffer cb, uint64_t fence) {
    VJob *j = calloc(1, sizeof *j);
    if (!j) rt_fatal("vk: out of memory");
    uint64_t left = VERIFY_BYTES, dst = 0;
    plat_mutex_lock_shared(&pool_lock);
    for (int tries = 0; nm && left && j->n < VERIFY_SAMPLES && tries <= nm; tries++) {
        if (ver_i >= nm) ver_i = 0, ver_off = 0;
        PoolRec *r = mlist[ver_i];
        uint64_t size = r->import_hi - r->import_lo;
        if (ver_off >= size) { ver_i++, ver_off = 0; continue; }
        uint64_t len = size - ver_off < left ? size - ver_off : left;
        j->s[j->n++] = (VSample){r, ver_off, len, dst, r->mgen};
        dst += len, left -= len, ver_off += len;
        if (ver_off >= size) ver_i++, ver_off = 0;
    }
    plat_mutex_unlock_shared(&pool_lock);
    if (!j->n) { free(j); return 0; }
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = dst,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    RcVmaRequest request = {.producer = RC_BUDGET_SNAPSHOT, .required = 0};
    if (resource_core_vma_create_buffer(&request, vk_ctx()->vma, &bi, &ai, &j->buf, &j->alloc, &info, NULL) != VK_SUCCESS) {
        free(j);
        return 0; // diagnostic readback is optional
    }
    j->map = info.pMappedData;
    j->sub = msub;
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
             VK_ACCESS_2_TRANSFER_READ_BIT);
    for (int i = 0; i < j->n; i++) {
        VkBufferCopy c = {j->s[i].off, j->s[i].dst, j->s[i].len};
        vkCmdCopyBuffer(cb, j->s[i].r->mbuf, j->buf, 1, &c);
    }
    mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
             VK_ACCESS_2_HOST_READ_BIT);
    vk_completion_add(fence, verify_cb, j);
    return j->n + 2;
}

typedef struct {
    size_t count;
    const SnapRange *ordinary;
    size_t ordinary_count;
    uint64_t dispatch_bytes;
} NativeRmwBitmapTail;

static int native_rmw_spans_overlap(uint64_t a, uint64_t a_size, uint64_t b, uint64_t b_size) {
    return a_size && b_size && (a < b ? b - a < a_size : a - b < b_size);
}

static int native_rmw_bitmap_overlaps(const PoolRec *pool, uint64_t off, uint64_t size) {
    for (const VkMemNativeRmwBitmap *r = RB.scene; r; r = r->next)
        if (r->descriptor.pool == pool &&
            native_rmw_spans_overlap(off, size, r->source.native.native.offset, r->source.descriptor_size)) return 1;
    return 0;
}

static int native_rmw_bitmap_word_matches(const SelectedMirrorWord *word, const SelectedMirrorWord *bitmap) {
    const RcRegistryBufferSpan w = word->native, b = bitmap->native;
    if (w.native.offset < b.native.offset) return 0;
    const uint64_t delta = w.native.offset - b.native.offset;
    return delta <= bitmap->descriptor_size && 4 <= bitmap->descriptor_size - delta &&
        b.interval.range.size == bitmap->descriptor_size && w.interval.range.size == 4 && !(delta & 3) &&
        word->mirror == bitmap->mirror && word->import == bitmap->import &&
        w.native.allocation == b.native.allocation && w.native.incarnation == b.native.incarnation &&
        w.interval.mapping == b.interval.mapping && w.interval.range.backing == b.interval.range.backing &&
        w.interval.range.offset >= b.interval.range.offset && w.interval.range.offset - b.interval.range.offset == delta &&
        w.interval.address >= b.interval.address && w.interval.address - b.interval.address == delta;
}

// The source has already passed the registry's native access check. The tail can share a
// commit only with an issued scalar in the same recording or an ordered, handed prefix.
static const char *native_rmw_bitmap_word_blocker(const VkMemNativeRmwWord *word, const SelectedMirrorWord *bitmap,
                                                VkCommandBuffer body) {
    // A registry write occupies the odd position immediately after its operation read.
    if (!word->body || !word->marker || !word->marker_generation || !word->tickets[0].id || !word->tickets[1].id ||
        word->tail_recorded || !rc_registry_position_valid(word->output) || !(word->output.ordinal & 1) ||
        word->output.scene != bitmap->read.scene ||
        rc_registry_position_compare(word->output, bitmap->read) >= 0 ||
        (body && word->body != body && !rc_registry_position_handed(rc_registry_runtime(), word->output)))
        return "bitmap-word-order";

    return native_rmw_bitmap_word_matches(&word->source, bitmap) ? NULL : "bitmap-word-mirror";
}

// Reverse overlap is proved only in the same body. Preparation checks the receipt and
// ordering first; binding and the tail also require the actual command buffer identity.
static int native_rmw_word_follows_bitmap(const SelectedMirrorWord *word, VkCommandBuffer body,
                                           const VkMemNativeRmwBitmap *bitmap) {
    return bitmap->body && bitmap->marker && bitmap->tickets[0].id && bitmap->tickets[1].id &&
        !bitmap->tail_recorded && rc_registry_position_valid(bitmap->output) && (bitmap->output.ordinal & 1) &&
        bitmap->output.scene == word->read.scene && rc_registry_position_compare(bitmap->output, word->read) < 0 &&
        (!body || bitmap->body == body) && native_rmw_bitmap_word_matches(word, &bitmap->source);
}

static int native_rmw_word_bitmap_refuses(const PoolRec *pool, const SelectedMirrorWord *word, VkCommandBuffer body) {
    for (const VkMemNativeRmwBitmap *bitmap = RB.scene; bitmap; bitmap = bitmap->next)
        if (bitmap->descriptor.pool == pool &&
            native_rmw_spans_overlap(word->native.native.offset, 4, bitmap->source.native.native.offset,
                                     bitmap->source.descriptor_size) &&
            !native_rmw_word_follows_bitmap(word, body, bitmap)) return 1;
    return 0;
}

static const char *native_rmw_bitmap_word_refusal(const PoolRec *pool, const SelectedMirrorWord *bitmap,
                                                 VkCommandBuffer body, VkMemNativeRmwBitmapFailure *failure) {
    for (const VkMemNativeRmwWord *word = RW.scene; word; word = word->next) {
        const uint64_t off = word->source.native.native.offset;
        if (word->descriptor.pool != pool ||
            !native_rmw_spans_overlap(bitmap->native.native.offset, bitmap->descriptor_size, off, 4)) continue;
        const char *blocker = native_rmw_bitmap_word_blocker(word, bitmap, body);
        if (!blocker) continue;
        if (failure) {
            failure->blocker_read = word->output;
            failure->blocker_offset = off;
            failure->blocker_bytes = 4;
        }
        return blocker;
    }
    return NULL;
}

static const char *native_rmw_bitmap_tail_refusal(const PoolRec *pool, uint64_t off, uint64_t size, uint64_t scene,
    VkMemNativeRmwBitmapFailure *failure) {
    if (RW.scene && RW.scene->output.scene != scene) {
        if (failure) failure->blocker_read = RW.scene->output;
        return "bitmap-word-scene";
    }
    for (int i = 0; i < nwbl; ++i) {
        if (wbl[i].r != pool || !native_rmw_spans_overlap(off, size, wbl[i].off, wbl[i].size)) continue;
        if (failure) {
            failure->blocker_offset = wbl[i].off;
            failure->blocker_bytes = wbl[i].size;
        }
        return "bitmap-plain-writeback-overlap";
    }
    for (int i = 0; i < lz_nskip; ++i) {
        if (lz_skip[i].sp.pool != pool || !native_rmw_spans_overlap(off, size, lz_skip[i].sp.offset, lz_skip[i].size))
            continue;
        if (failure) {
            failure->blocker_offset = lz_skip[i].sp.offset;
            failure->blocker_bytes = lz_skip[i].size;
        }
        return "bitmap-lazy-writeback-overlap";
    }
    return NULL;
}

static int native_rmw_bitmap_tail_supported(const PoolRec *pool, uint64_t off, uint64_t size, uint64_t scene) {
    return !native_rmw_bitmap_tail_refusal(pool, off, size, scene, NULL);
}

static int native_rmw_bitmap_compatible(const VkMemNativeRmwBitmap *a, const VkMemNativeRmwBitmap *b,
                                        uint64_t shared) {
    const RcRegistryBufferSpan x = a->source.native, y = b->source.native;
    const uint64_t x_relative = shared - x.native.offset, y_relative = shared - y.native.offset;
    return a->source.mirror == b->source.mirror && a->source.import == b->source.import &&
        x.native.allocation == y.native.allocation && x.native.incarnation == y.native.incarnation &&
        x.interval.mapping == y.interval.mapping && x.interval.range.backing == y.interval.range.backing &&
        x.interval.range.offset + x_relative == y.interval.range.offset + y_relative &&
        x.interval.address + x_relative == y.interval.address + y_relative;
}

static int native_rmw_bitmap_edge_compare(const void *left, const void *right) {
    const NativeRmwBitmapEdge *a = left, *b = right;
    if (a->pool != b->pool) return (uintptr_t)a->pool < (uintptr_t)b->pool ? -1 : 1;
    return a->off < b->off ? -1 : a->off > b->off;
}

static int native_rmw_bitmap_slices(uint64_t scene, NativeRmwBitmapTail *tail) {
    NativeRmwBitmapEdge *edges = RB.edge;
    size_t count = 0;
    for (VkMemNativeRmwBitmap *r = RB.scene; r; r = r->next) {
        const uint64_t off = r->source.native.native.offset, size = r->source.descriptor_size;
        if (count == 2 * RB.capacity || !size || ((off | size) & 3) || size > UINT64_MAX - off ||
            r->output.scene != scene || !r->body || r->tail_recorded ||
            !native_rmw_bitmap_tail_supported(r->descriptor.pool, off, size, scene)) return 0;
        edges[count++] = (NativeRmwBitmapEdge){r->descriptor.pool, off};
        edges[count++] = (NativeRmwBitmapEdge){r->descriptor.pool, off + size};
    }
    qsort(edges, count, sizeof edges[0], native_rmw_bitmap_edge_compare);
    for (size_t i = 1; i < count; ++i) {
        const NativeRmwBitmapEdge lo = edges[i - 1], hi = edges[i];
        if (lo.pool != hi.pool || lo.off == hi.off) continue;
        VkMemNativeRmwBitmap *owner = NULL;
        for (VkMemNativeRmwBitmap *r = RB.scene; r; r = r->next) {
            const uint64_t off = r->source.native.native.offset;
            if (lo.pool != r->descriptor.pool || lo.off < off || hi.off - off > r->source.descriptor_size) continue;
            if (!owner) owner = r;
            else if (!native_rmw_bitmap_compatible(owner, r, lo.off)) return 0;
        }
        if (owner) RB.slice[tail->count++] = (NativeRmwBitmapSlice){owner, lo.off, hi.off - lo.off};
    }
    return tail->count != 0;
}

static size_t native_rmw_bitmap_first_clip(const NativeRmwBitmapTail *tail, const SnapRange *range) {
    for (size_t i = 0; i < tail->count; ++i) {
        const NativeRmwBitmapSlice s = RB.slice[i];
        if (s.owner->descriptor.pool == range->r &&
            native_rmw_spans_overlap(range->off, range->size, s.off, s.size)) return i;
    }
    return tail->count;
}

static int native_rmw_bitmap_snapshot_clipped(const NativeRmwBitmapTail *tail, const SnapRange *range) {
    return native_rmw_bitmap_first_clip(tail, range) != tail->count;
}

// Yield one uncovered piece at a time. The original SNL still owns the whole preimage
// and can need its copy in the final prologue. The supplied ordinary range already excludes mesh words.
static int native_rmw_bitmap_next_complement(const NativeRmwBitmapTail *tail, const SnapRange *range,
    size_t *next, uint64_t *cursor, SnapRange *out) {
    const uint64_t end = range->off + range->size;
    while (*next < tail->count && *cursor < end) {
        const NativeRmwBitmapSlice s = RB.slice[(*next)++];
        if (s.owner->descriptor.pool != range->r || s.off >= end) break;
        const uint64_t mask_end = s.off + s.size;
        if (mask_end <= *cursor) continue;
        const uint64_t lo = *cursor;
        *cursor = mask_end < end ? mask_end : end;
        if (lo < s.off) {
            *out = (SnapRange){range->r, lo, s.off - lo, range->soff + (lo - range->off), range->chunk};
            return 1;
        }
    }
    if (*cursor == end) return 0;
    const uint64_t lo = *cursor;
    *cursor = end;
    *out = (SnapRange){range->r, lo, end - lo, range->soff + (lo - range->off), range->chunk};
    return 1;
}

static int native_rmw_bitmap_tail_plan(NativeRmwBitmapTail *tail, uint64_t scene,
                                       const SnapRange *ordinary, size_t count) {
    if (!native_rmw_bitmap_slices(scene, tail)) return 0;
    const VkPhysicalDeviceLimits *limits = &vk_ctx()->props.limits;
    const uint64_t align = limits->minStorageBufferOffsetAlignment;
    const uint64_t range = limits->maxStorageBufferRange;
    if (!align || (align & (align - 1)) || align > range) return 0;
    const uint64_t groups_bytes = (uint64_t)limits->maxComputeWorkGroupCount[0] * 64 * 4;
    const uint64_t window_bytes = range - (align - 1);
    tail->dispatch_bytes = (groups_bytes < window_bytes ? groups_bytes : window_bytes) & ~UINT64_C(3);
    if (!tail->dispatch_bytes) return 0;
    for (int i = 0; i < nsnl; ++i) {
        const SnapRange r = snl[i];
        if (!r.r || !r.size || ((r.off | r.size | r.soff) & 3) ||
            r.size > UINT64_MAX - r.off || r.size > UINT64_MAX - r.soff ||
            r.chunk < 0 || r.chunk >= nchunks ||
            r.soff > chunks[r.chunk].size || r.size > chunks[r.chunk].size - r.soff) return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        const SnapRange *r = &ordinary[i];
        const size_t clip = native_rmw_bitmap_first_clip(tail, r);
        if (clip == tail->count) continue;
        const VkMemNativeRmwBitmap *owner = RB.slice[clip].owner;
        const PoolRec *pool = r->r;
        if (pool->import_hi <= pool->import_lo || r->off + r->size > pool->import_hi - pool->import_lo ||
            owner->source.mirror != pool->mbuf || owner->source.import != pool->buf) return 0;
    }
    tail->ordinary = ordinary;
    tail->ordinary_count = count;
    return 1;
}

static VkDescriptorBufferInfo native_rmw_bitmap_window(VkBuffer buffer, uint64_t off, uint64_t size,
                                                        uint32_t *first_word) {
    const uint64_t align = vk_ctx()->props.limits.minStorageBufferOffsetAlignment;
    const uint64_t base = off & ~(align - 1);
    *first_word = (uint32_t)((off - base) / 4);
    return (VkDescriptorBufferInfo){buffer, base, off - base + size};
}

enum { NATIVE_RMW_BITMAP_UNION, NATIVE_RMW_BITMAP_COMPARE, NATIVE_RMW_BITMAP_COMMIT, NATIVE_RMW_BITMAP_DIRECT };
static void native_rmw_bitmap_dispatch(VkCommandBuffer cb, const VkMemNativeRmwBitmap *owner,
    uint64_t off, uint64_t size, uint32_t mode, VkDescriptorBufferInfo source, uint32_t first_word) {
    uint32_t pc[6] = {mode, (uint32_t)(size / 4), first_word, 0, 0, 1 + owner->map_dwords};
    if (mode != NATIVE_RMW_BITMAP_DIRECT) pc[4] = (uint32_t)((off - owner->source.native.native.offset) / 4);
    VkDescriptorBufferInfo dst = native_rmw_bitmap_window(owner->source.import, off, size, &pc[3]);
    VkDescriptorBufferInfo mirror = native_rmw_bitmap_window(owner->source.mirror, off, size, &pc[3]);
    VkDescriptorBufferInfo buffers[4] = {source, mirror,
        {owner->marker, 0, native_rmw_bitmap_allocation_bytes(owner->map_dwords)}, dst};
    VkWriteDescriptorSet writes[4];
    for (uint32_t i = 0; i < 4; ++i)
        writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffers[i]};
    rec_push(RB.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, RB.layout, 0, 4, writes);
    vkCmdPushConstants(cb, RB.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (pc[1] - 1) / 64 + 1, 1, 1);
}

static void native_rmw_bitmap_dispatch_barrier(VkCommandBuffer cb) {
    mbarrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
}

static int native_rmw_bitmap_snapshot_record(VkCommandBuffer cb, const NativeRmwBitmapTail *tail,
    const VkMemNativeRmwBitmap *owner, const SnapRange *part, uint32_t mode) {
    int recorded = 0;
    for (uint64_t i = 0; i < part->size;) {
        const uint64_t remaining = part->size - i;
        const uint64_t bytes = remaining > tail->dispatch_bytes ? tail->dispatch_bytes : remaining;
        uint32_t source_word;
        VkDescriptorBufferInfo source = native_rmw_bitmap_window(chunks[part->chunk].buf,
            part->soff + i, bytes, &source_word);
        native_rmw_bitmap_dispatch(cb, owner, part->off + i, bytes, mode, source, source_word);
        // Later overlapping preimages can write the same import word in direct mode.
        native_rmw_bitmap_dispatch_barrier(cb);
        i += bytes;
        recorded += 4;
    }
    return recorded;
}

static int native_rmw_bitmap_visibility_ranges(const VkMemNativeRmwBitmap *owner, const PoolRec *pool,
    uint64_t off, uint64_t size, RcSourceRange *guest, RcSourceRange *canonical) {
    if (!owner || !pool || !guest || !canonical || !size || pool != owner->descriptor.pool ||
        pool->kind != POOL_IMPORT || pool->smap || pool->import_hi <= pool->import_lo ||
        nvn_gpu_only(pool->import_lo)) return 0;
    const SelectedMirrorWord *source = &owner->source;
    const uint64_t source_offset = source->native.native.offset;
    if (off < source_offset) return 0;
    const uint64_t delta = off - source_offset;
    if (!rc_range_valid(source->native.interval.range) || delta > source->native.interval.range.size ||
        size > source->native.interval.range.size - delta || delta > UINT64_MAX - source->native.interval.address ||
        delta > UINT64_MAX - source->native.interval.range.offset || pool->import_lo > UINT64_MAX - off ||
        pool->import_lo + off != source->native.interval.address + delta ||
        source->native.interval.address + delta < pool->import_lo ||
        source->native.interval.address + delta >= pool->import_hi ||
        size > pool->import_hi - (source->native.interval.address + delta)) return 0;
    *guest = (RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING, source->native.interval.address + delta, size};
    *canonical = (RcSourceRange){source->native.interval.range.backing,
        source->native.interval.range.offset + delta, size};
    return rc_range_valid(*guest) && rc_range_valid(*canonical);
}

// The bitmap union shader reads a bit from source word 1. Use the scalar marker's
// private conditions word for that bit so both operation execution flags survive intact.
static int native_rmw_bitmap_stage_words(VkCommandBuffer cb) {
    int recorded = 0;
    for (const VkMemNativeRmwWord *word = RW.scene; word; word = word->next) {
        if (!native_rmw_bitmap_overlaps(word->descriptor.pool, word->source.native.native.offset, 4)) continue;
        if (!recorded) {
            mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
            ++recorded;
        }
        const VkBufferCopy copy = {0, sizeof(uint32_t), sizeof(uint32_t)};
        vkCmdCopyBuffer(cb, word->marker, word->marker, 1, &copy);
        ++recorded;
    }
    return recorded;
}

static int native_rmw_bitmap_tail_record(VkCommandBuffer cb, NativeRmwBitmapTail *tail, CpuVisibility *visibility) {
    const int staged = native_rmw_bitmap_stage_words(cb);
    // Include the transfer writes to scalar scratch along with the graphics marker writes.
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, RB.pipeline);
    int recorded = staged + 2;
    for (size_t i = 0; i < tail->ordinary_count; ++i) {
        const SnapRange *r = &tail->ordinary[i];
        size_t next = native_rmw_bitmap_first_clip(tail, r);
        if (next == tail->count) continue;
        const VkMemNativeRmwBitmap *owner = RB.slice[next].owner;
        uint64_t cursor = r->off;
        SnapRange part;
        while (native_rmw_bitmap_next_complement(tail, r, &next, &cursor, &part)) {
            recorded += native_rmw_bitmap_snapshot_record(cb, tail, owner, &part, NATIVE_RMW_BITMAP_DIRECT);
            cpu_visibility_add(visibility, part.r, part.off, part.size);
        }
    }
    for (size_t i = 0; i < tail->count; ++i) {
        const NativeRmwBitmapSlice s = RB.slice[i];
        PoolRec *pool = s.owner->descriptor.pool;
        const uint64_t end = s.off + s.size;
        for (const VkMemNativeRmwBitmap *r = RB.scene; r; r = r->next) {
            uint64_t off = r->source.native.native.offset;
            if (r->descriptor.pool != pool || s.off < off || end - off > r->source.descriptor_size) continue;
            native_rmw_bitmap_dispatch(cb, s.owner, s.off, s.size, NATIVE_RMW_BITMAP_UNION,
                (VkDescriptorBufferInfo){r->marker, 0, native_rmw_bitmap_marker_bytes(r)}, (uint32_t)((s.off - off) / 4));
            native_rmw_bitmap_dispatch_barrier(cb);
            recorded += 4;
        }
        for (const VkMemNativeRmwWord *word = RW.scene; word; word = word->next) {
            const uint64_t off = word->source.native.native.offset;
            if (word->descriptor.pool != pool || off < s.off || off >= end) continue;
            native_rmw_bitmap_dispatch(cb, s.owner, off, 4, NATIVE_RMW_BITMAP_UNION,
                (VkDescriptorBufferInfo){word->marker, 0, 2 * sizeof(uint32_t)}, 0);
            native_rmw_bitmap_dispatch_barrier(cb);
            recorded += 4;
        }
        for (int j = 0; j < nsnl; ++j) {
            const SnapRange r = snl[j];
            if (r.r != pool || !native_rmw_spans_overlap(r.off, r.size, s.off, s.size)) continue;
            const uint64_t lo = r.off > s.off ? r.off : s.off;
            const uint64_t hi = r.off + r.size < end ? r.off + r.size : end;
            const SnapRange part = {r.r, lo, hi - lo, r.soff + (lo - r.off), r.chunk};
            recorded += native_rmw_bitmap_snapshot_record(cb, tail, s.owner, &part, NATIVE_RMW_BITMAP_COMPARE);
            RcSourceRange guest, canonical;
            if (!native_rmw_bitmap_visibility_ranges(s.owner, pool, lo, part.size, &guest, &canonical)) {
                visibility->incomplete = 1;
                continue;
            }
            cpu_visibility_add_guarded_pair(visibility, guest, canonical, s.owner->provider, &s.owner->descriptor);
        }
        native_rmw_bitmap_dispatch(cb, s.owner, s.off, s.size, NATIVE_RMW_BITMAP_COMMIT,
            (VkDescriptorBufferInfo){s.owner->marker, 0, native_rmw_bitmap_marker_bytes(s.owner)}, 0);
        native_rmw_bitmap_dispatch_barrier(cb);
        pool->wb_fence = s.owner->output.scene;
        pool->wb_sub = msub;
        recorded += 4;
    }
    VkPipelineStageFlags2 source_stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    for (const VkMemNativeRmwBitmap *r = RB.scene; r; r = r->next) source_stages |= r->writer_stage;
    mbarrier(cb, source_stages,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    for (VkMemNativeRmwBitmap *r = RB.scene; r; r = r->next) r->tail_recorded = 1;
    RB.scene = NULL;
    RB.count = 0;
    return recorded + 1;
}

static int native_rmw_word_overlap(uint64_t offset, uint64_t size, uint64_t word) {
    return size && (word < offset ? offset - word < 4 : word - offset < size);
}

static int native_rmw_tail_supported(const PoolRec *pool, uint64_t word) {
    if (nsnl > RMW_SNAPSHOT_LIMIT) return 0;
    for (int i = 0; i < nwbl; ++i)
        if (wbl[i].r == pool && native_rmw_word_overlap(wbl[i].off, wbl[i].size, word)) return 0;
    for (int i = 0; i < lz_nskip; ++i)
        if (lz_skip[i].sp.pool == pool && native_rmw_word_overlap(lz_skip[i].sp.offset, lz_skip[i].size, word)) return 0;
    return 1;
}

static int native_rmw_tail_plan(NativeRmwTail *tail, uint64_t scene) {
    if (!tail || !RW.scene || RW.word_count > RMW_WORD_LIMIT || nsnl > RMW_SNAPSHOT_LIMIT) return 0;
    for (VkMemNativeRmwWord *r = RW.scene; r; r = r->next) {
        if (r->output.scene != scene || !r->body || !r->marker_generation || r->tail_recorded) return 0;
        const SnapWord target = {r->descriptor.pool, r->source.native.native.offset};
        int bitmap_owned = 0;
        for (const VkMemNativeRmwBitmap *bitmap = RB.scene; bitmap; bitmap = bitmap->next) {
            if (bitmap->descriptor.pool != target.r ||
                !native_rmw_spans_overlap(target.off, 4, bitmap->source.native.native.offset,
                                          bitmap->source.descriptor_size)) continue;
            if (!bitmap->body) return 0;
            if (rc_registry_position_compare(r->output, bitmap->source.read) < 0) {
                if (native_rmw_bitmap_word_blocker(r, &bitmap->source, bitmap->body)) return 0;
            } else if (!native_rmw_word_follows_bitmap(&r->source, r->body, bitmap)) return 0;
            bitmap_owned = 1;
        }
        // Bitmap ownership covers the scalar marker and every snapshot comparison at this word.
        if (bitmap_owned) continue;
        if (!native_rmw_tail_supported(target.r, target.off)) return 0;
        size_t slot = 0;
        for (; slot < tail->count; ++slot) {
            const SnapWord prior = tail->words[slot];
            if (prior.r == target.r && prior.off >= target.off) break;
            if ((uintptr_t)prior.r > (uintptr_t)target.r) break;
        }
        if (slot < tail->count && tail->words[slot].r == target.r && tail->words[slot].off == target.off) {
            const VkMemNativeRmwWord *first = tail->owners[slot];
            if (first->source.mirror != r->source.mirror || first->source.import != r->source.import ||
                !rc_range_equal(first->source.native.interval.range, r->source.native.interval.range) ||
                first->source.native.interval.mapping != r->source.native.interval.mapping ||
                first->source.native.native.allocation != r->source.native.native.allocation ||
                first->source.native.native.incarnation != r->source.native.native.incarnation)
                return 0;
            continue;
        }
        if (tail->count == RMW_WORD_LIMIT) return 0;
        memmove(tail->words + slot + 1, tail->words + slot, (tail->count - slot) * sizeof *tail->words);
        memmove(tail->owners + slot + 1, tail->owners + slot, (tail->count - slot) * sizeof *tail->owners);
        tail->words[slot] = target;
        tail->owners[slot] = r;
        tail->count++;
    }
    return snapshot_partition(snl, (size_t)nsnl, tail->words, tail->count, tail->ordinary,
        sizeof tail->ordinary / sizeof tail->ordinary[0], &tail->ordinary_count, tail->preimages,
        sizeof tail->preimages / sizeof tail->preimages[0], &tail->preimage_count);
}

static VkDescriptorBufferInfo native_rmw_window(VkBuffer buffer, uint64_t offset, uint32_t *word) {
    const uint64_t align = vk_ctx()->props.limits.minStorageBufferOffsetAlignment;
    const uint64_t base = offset & ~(align - 1);
    *word = (uint32_t)((offset - base) / 4);
    return (VkDescriptorBufferInfo){buffer, base, offset - base + 4};
}

static void native_rmw_dispatch(VkCommandBuffer cb, const VkMemNativeRmwWord *owner, VkBuffer source,
                                uint64_t source_offset, uint32_t mode) {
    const uint64_t offset = owner->source.native.native.offset;
    uint32_t pc[4] = {mode};
    VkDescriptorBufferInfo buffers[4] = {
        native_rmw_window(source, source_offset, &pc[1]),
        native_rmw_window(owner->source.mirror, offset, &pc[2]),
        {owner->marker, 0, 2 * sizeof(uint32_t)},
        native_rmw_window(owner->source.import, offset, &pc[3]),
    };
    VkWriteDescriptorSet writes[4];
    for (uint32_t i = 0; i < 4; ++i)
        writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffers[i]};
    rec_push(RW.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, RW.layout, 0, 4, writes);
    vkCmdPushConstants(cb, RW.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, 1, 1, 1);
}

static int native_rmw_tail_record(VkCommandBuffer cb, const NativeRmwTail *tail, CpuVisibility *visibility) {
    int recorded = 2;
    mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, RW.pipeline);
    for (size_t i = 0; i < tail->count; ++i) {
        const VkMemNativeRmwWord *owner = tail->owners[i];
        for (VkMemNativeRmwWord *r = RW.scene; r; r = r->next) {
            if (r->descriptor.pool != tail->words[i].r || r->source.native.native.offset != tail->words[i].off) continue;
            native_rmw_dispatch(cb, owner, r->marker, 0, 0);
            recorded += 3;
        }
        int had_snapshot = 0;
        for (size_t j = 0; j < tail->preimage_count; ++j) {
            const SnapWordPreimage *p = &tail->preimages[j];
            if (p->word != i) continue;
            native_rmw_dispatch(cb, owner, chunks[p->pre.chunk].buf, p->pre.soff, 1);
            had_snapshot = 1;
            recorded += 3;
        }
        mbarrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        native_rmw_dispatch(cb, owner, owner->marker, 0, 2);
        recorded += 4;
        if (had_snapshot)
            cpu_visibility_add(visibility, tail->words[i].r, tail->words[i].off, 4);
        tail->words[i].r->wb_fence = owner->output.scene;
    }
    mbarrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
             VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    for (VkMemNativeRmwWord *r = RW.scene; r; r = r->next) r->tail_recorded = 1;
    RW.scene = NULL;
    RW.word_count = 0;
    RW.workspace = NULL;
    return recorded + 1;
}

int vk_mem_mirror_wb_pending(void) { return vk_mirror_on && !RW.scene && !RB.scene && (nsnl || nwbl); }

int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t fence) {
    if (!vk_mirror_on) return 0;
    int rec = 0;
    lazy_part_check();   // before the merges are recorded: what the last part skipped and has to write back after all
    NativeRmwTail *tail = RW.scene ? RW.workspace : NULL;
    if (RW.scene && !native_rmw_tail_plan(tail, fence)) {
        // The body may already be in a handed prefix. The exact declarations remain owned by
        // their cancellation callbacks. An unsupported tail must not fall back to a whole copy.
        RW.scene = NULL;
        RW.workspace = NULL;
        RW.word_count = 0;
        RB.scene = NULL;
        RB.count = 0;
        return -1;
    }
    const SnapRange *selected_snl = tail ? tail->ordinary : snl;
    int selected_count = tail ? (int)tail->ordinary_count : nsnl;
    NativeRmwBitmapTail bitmap_tail = {0};
    if (RB.scene) {
        if (!native_rmw_bitmap_tail_plan(&bitmap_tail, fence, selected_snl, (size_t)selected_count)) {
            RW.scene = NULL;
            RW.workspace = NULL;
            RW.word_count = 0;
            RB.scene = NULL;
            RB.count = 0;
            return -1;
        }
        selected_snl = bitmap_tail.ordinary;
        selected_count = (int)bitmap_tail.ordinary_count;
    }
    CpuVisibility visibility = {0};
    if (!shadow_on && (nsnl || nwbl)) {
        rc_registry_visibility_begin(rc_registry_runtime(), RC_REGISTRY_VISIBILITY_ORDERED, &visibility.proof);
        // vk_exec_record has drained workers and ended all guest passes before this final scene writeback.
        // The registry also requires each completed producer's real submission to be no later than this scene.
        visibility.proof.submission = fence;
    }
    int timing = vk_gpu_passes_capturing();
    if (selected_count) {
        if (timing) {
            vk_gpu_gap_boundary(cb);
            vk_gpu_gap_note(VK_GAP_WB_BEFORE, 0);
        }
        // the merge reads the mirror the submission's shaders wrote and writes the imported buffers
        mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        uint64_t before = mrep.merge_bytes;
        if (timing) vk_gpu_gap_boundary(cb);
        int m = wb_merge_record(cb, selected_snl, selected_count);
        if (m)
            for (int i = 0; i < selected_count; i++)
                if (!RB.scene || !native_rmw_bitmap_overlaps(selected_snl[i].r, selected_snl[i].off, selected_snl[i].size))
                    cpu_visibility_add(&visibility, selected_snl[i].r, selected_snl[i].off, selected_snl[i].size);
        if (timing) {
            vk_gpu_gap_note(VK_GAP_WB_MERGE, mrep.merge_bytes - before);
            vk_gpu_gap_boundary(cb);
            vk_gpu_gap_note(VK_GAP_WB_AFTER, 0);
        }
        if (!m) {   // no range list: copy unclipped ranges plainly; bitmap tail owns clipped ranges
            for (int i = 0; i < selected_count; i++)
                if (!RB.scene || !native_rmw_bitmap_overlaps(selected_snl[i].r, selected_snl[i].off, selected_snl[i].size))
                    range_add(&wbl, &nwbl, &cwbl, selected_snl[i].r, selected_snl[i].off, selected_snl[i].size);
        }
        for (int i = 0; i < selected_count; i++)
            if (!RB.scene || !native_rmw_bitmap_overlaps(selected_snl[i].r, selected_snl[i].off, selected_snl[i].size))
                selected_snl[i].r->wb_fence = fence;
        rec += m + 1;
        mbarrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        if (timing) vk_gpu_gap_boundary(cb);
    }
    if (RB.scene) rec += native_rmw_bitmap_tail_record(cb, &bitmap_tail, &visibility);
    if (tail) rec += native_rmw_tail_record(cb, tail, &visibility);
    if (shadow_on && (nswl || nwbl)) swb_job(fence);   // before snap_free takes back the marks' chunk
    snap_completion_fence = vk_mem_mirror_pending() ? fence : 0;
    snap_transfer(fence);
    if (nwbl) {
        if (timing) {
            vk_gpu_gap_boundary(cb);
            vk_gpu_gap_note(VK_GAP_WB_BEFORE, 0);
        }
        mbarrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
        for (int i = 0; i < nwbl; i++) wbl[i].r->wb_sub = msub, wbl[i].r->wb_fence = fence;
        uint64_t before = mrep.wb_bytes;
        if (timing) vk_gpu_gap_boundary(cb);
        mrep.wb_ranges += (uint64_t)nwbl;
        rec += copy_ranges(cb, wbl, nwbl, 0, &mrep.wb_bytes, &mrep.wb_copies, &visibility, 0) + 2;
        if (timing) {
            vk_gpu_gap_note(VK_GAP_WB_COPY, mrep.wb_bytes - before);
            vk_gpu_gap_boundary(cb);
            vk_gpu_gap_note(VK_GAP_WB_AFTER, 0);
        }
        mbarrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        nwbl = 0;
        if (timing) vk_gpu_gap_boundary(cb);
    }
    // Registered after replay owners: their completion first converts written ranges to foreign authority.
    // Only this successful scene completion then proves the collected copies/merges reached guest memory.
    cpu_visibility_queue(&visibility, fence);
    if (mirror_verify) rec += verify_record(cb, fence);
    plat_mutex_lock(&mirror_stat_lock);
    uint64_t *d = (uint64_t *)&macc, *s = (uint64_t *)&mrep;
    for (size_t i = 0; i < sizeof macc / sizeof(uint64_t); i++) d[i] += s[i];
    plat_mutex_unlock(&mirror_stat_lock);
    memset(&mrep, 0, sizeof mrep);
    return rec;
}

// NX2NAVI_TRACE=perf, every 600 frames: the mirrored pools that moved the most bytes since the last list, per frame: what
// their CPU writes and flushes copied up, how often a submission synced them, and what the GPU's writes sent back
// (merged or copied). A pool whose upload is close to its size every frame is rewritten whole each frame.
#define POOL_LOG_TOP 12
static uint64_t pool_score(const PoolRec *r) { return r->st_pages + r->st_flush + r->st_wb + r->st_merge; }
static void mirror_pool_log(uint64_t frame) {
    static uint64_t last;
    double fr = frame > last ? (double)(frame - last) : 1.0;
    last = frame;
    PoolRec *top[POOL_LOG_TOP];
    int nt = 0;
    uint64_t all_up = 0;
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < nm; i++) {
        PoolRec *r = mlist[i];
        all_up += r->st_pages + r->st_flush;
        uint64_t sc = pool_score(r);
        if (!sc || (nt == POOL_LOG_TOP && sc <= pool_score(top[nt - 1]))) continue;
        int k = nt < POOL_LOG_TOP ? nt++ : nt - 1;   // insertion into top[], largest first
        for (; k > 0 && pool_score(top[k - 1]) < sc; k--) top[k] = top[k - 1];
        top[k] = r;
    }
    rt_log("mem: mirror pools frame %llu: %d mirrored, %.1f MiB a frame copied up in all; the top %d (MiB a frame: written pages, "
           "flush ranges (overlap with the pages counted twice), syncs a frame, GPU writes back merged / copied):",
           (unsigned long long)frame, nm, all_up / fr / 1048576.0, nt);
    for (int i = 0; i < nt; i++) {
        const PoolRec *r = top[i];
        char f[160];
        flag_names(r->flags, f, sizeof f);
        rt_log("mem:   pool %#llx %.1f MiB %s: up %.2f + %.2f, %.1f syncs, back %.2f / %.2f", (unsigned long long)r->lo,
               r->size / 1048576.0, f, r->st_pages / fr / 1048576.0, r->st_flush / fr / 1048576.0, r->st_syncs / fr,
               r->st_merge / fr / 1048576.0, r->st_wb / fr / 1048576.0);
    }
    for (int i = 0; i < nm; i++) {
        PoolRec *r = mlist[i];
        r->st_pages = r->st_flush = r->st_wb = r->st_merge = 0, r->st_syncs = 0;
    }
    plat_mutex_unlock_shared(&pool_lock);
}

// One summary line: with NX2NAVI_TRACE=perf every 60 frames, else at frames 600 and 1800.
static void mirror_log(uint64_t frame) {
    static uint64_t last, ww_last;
    plat_mutex_lock(&gen_lock);
    uint64_t wc = ww_calls, wb = ww_bytes, wt = ww_ticks, dc = ww_dirty_calls, dt = ww_dirty_ticks, wp = ww_pages;
    uint64_t renewals = freeze_snap_renewals;
    freeze_snap_renewals = 0;
    ww_pages = 0;
    uint64_t nr = n_frz_refused;
    uint64_t offers = texture_offers, tries = texture_tries, protected_count = texture_protected;
    uint64_t cost_tries = texture_cost_tries, cost_ok = texture_cost_protected;
    texture_cost_tries = texture_cost_protected = 0;
    texture_offers = texture_tries = texture_protected = 0;
    CleanStats clean = clean_stats;
    clean_stats = (CleanStats){0};
#ifdef _WIN32
    uint64_t pairs = peek_pairs, empty = peek_clean, pair_ns = peek_wall;
    peek_pairs = peek_clean = peek_wall = 0;
#endif
    ww_calls = ww_bytes = ww_ticks = ww_dirty_calls = ww_dirty_ticks = n_frz_refused = 0;
    plat_mutex_unlock(&gen_lock);
    double wf = frame > ww_last ? (double)(frame - ww_last) : 1.0;
    ww_last = frame;
#ifdef _WIN32
    if (pairs && (rt_trace & RT_TRACE_PERF))
        rt_log("mem: texture snapshot pairs %llu, empty %llu, elapsed %.3f ms/frame",
               (unsigned long long)pairs, (unsigned long long)empty, qpc_ms(pair_ns) / wf);
#endif
    uint64_t fz, ff, ft, fo, fh;
    mem_freeze_stats(&fz, &ff, &ft, &fo, &fh);
    if (renewals && (rt_trace & RT_TRACE_PERF))
        rt_log("mem: clean snapshots renewed after freeze at frame %llu: %llu", (unsigned long long)frame,
               (unsigned long long)renewals);
    if (clean_attempt && (rt_trace & RT_TRACE_PERF))
        rt_log("mem: clean-scan frame %llu: expensive %llu, newly confirmed %llu, full %llu, edges examined %llu, "
               "rejected %llu, priority attempts %llu, freeze successes %llu",
               (unsigned long long)frame, (unsigned long long)clean.expensive, (unsigned long long)clean.ready,
               (unsigned long long)clean.full, (unsigned long long)clean.examined, (unsigned long long)clean.skipped,
               (unsigned long long)clean.attempts, (unsigned long long)clean.successes);
    if (offers || tries) rt_log("mem: earlier texture protection frame %llu: offered %llu, tried %llu, protected %llu",
        (unsigned long long)frame, (unsigned long long)offers, (unsigned long long)tries, (unsigned long long)protected_count);
    if (cost_tries && (rt_trace & RT_TRACE_PERF))
        rt_log("mem: repeat scan protection frame %llu: tried %llu, protected %llu",
               (unsigned long long)frame, (unsigned long long)cost_tries, (unsigned long long)cost_ok);
    mem_probe_report(frame);
    mem_lazy_report(frame, lz_skipped);
    lz_skipped = 0;
    // NX2NAVI_WB_LAZY_TOUCH=<n> (test, 0 off): at each of these lines this thread reads a byte of n armed ranges, as
    // the game's CPU might, so the fault, the fetch and the hold run without waiting for the game to do it.
    static int touch = -1;
    if (touch < 0) touch = cfg_int("NX2NAVI_WB_LAZY_TOUCH");
    if (touch > 0) {
        uint64_t at[16];
        int k = mem_lazy_pick(at, touch > 16 ? 16 : touch);
        for (int i = 0; i < k; i++) {
            volatile uint8_t v = *(volatile const uint8_t *)(uintptr_t)at[i];
            (void)v;
        }
    }
    if (rt_trace & RT_TRACE_PERF)
        rt_log("mem: write-watch frame %llu: per frame, every consumer: %.1f queries over %.1f MiB, %.3f ms (%.1f found "
               "%.0f written pages, %.3f ms); frozen %.0f MiB; since the last line: freezes %llu, thaws by a write %llu, by a file read "
               "or mapping change %llu, refused %llu; quick thaws that held a hot region thawed %llu",
               (unsigned long long)frame, wc / wf, wb / wf / 1048576.0, qpc_ms(wt) / wf, dc / wf, wp / wf, qpc_ms(dt) / wf,
               fz * 2.0, (unsigned long long)ff, (unsigned long long)ft, (unsigned long long)fo, (unsigned long long)nr,
               (unsigned long long)fh);
    if (!vk_mirror_on) return;
    plat_mutex_lock(&mirror_stat_lock);
    MirrorAcc a = macc;
    memset(&macc, 0, sizeof macc);
    plat_mutex_unlock(&mirror_stat_lock);
    double f = frame > last ? (double)(frame - last) : 1.0, M = 1048576.0;
    last = frame;
    uint64_t fc = (uint64_t)plat_xchg64(&m_flush_calls, 0), fb = (uint64_t)plat_xchg64(&m_flush_bytes, 0);
    char ver[160] = "";
    if (mirror_verify)
        snprintf(ver, sizeof ver, "; verify: pages compared %lld, skipped %lld (written since), mismatches %lld",
                 (long long)plat_xchg64(&v_pages, 0), (long long)plat_xchg64(&v_skipped, 0),
                 (long long)plat_xchg64(&v_mismatch, 0));
    vk_gpa_report();
    rt_log("mem: mirror frame %llu: pools %lld, %.1f MiB device-local; stay imported: budget %lld (%.1f MiB), untracked "
           "%lld, overlapping %lld, failed %lld, GPA projection %lld; per frame: submits %.2f, syncs %.1f, upload %.3f MiB in %.1f ranges, "
           "first use %.2f pools %.3f MiB, write-back %.3f MiB in %.1f ranges, merge %.3f MiB in %.1f dispatches "
           "(snapshots %.3f MiB, new buffers %.2f), flush calls %.1f (%.3f MiB), queries "
           "%.1f pre-snap (%.1f MiB) + %.1f sync, prologues %.2f, tracking cpu %.3f ms (pre-snap %.3f, syncs %.3f); so far "
           "evicted %lld (%.1f MiB), budget refusals asked again %lld, made without a twin %lld (%.1f MiB)%s",
           (unsigned long long)frame, (long long)m_live_n, m_live_bytes / M, (long long)m_refused_n[MR_BUDGET],
           m_refused_bytes[MR_BUDGET] / M, (long long)m_refused_n[MR_UNTRACKED], (long long)m_refused_n[MR_OVERLAP],
           (long long)m_refused_n[MR_ALLOC], (long long)m_refused_n[MR_GPA_PROJECTION], a.subs / f, a.syncs / f,
           a.upload_bytes / f / M, a.upload_copies / f,
           a.first_n / f, a.first_bytes / f / M, a.wb_bytes / f / M, a.wb_copies / f, a.merge_bytes / f / M,
           a.merge_dispatches / f, a.snap_bytes / f / M, a.snap_allocs / f, fc / f, fb / f / M,
           a.presnap_queries / f, a.presnap_bytes / f / M, a.sync_queries / f, a.prologues / f, qpc_ms(a.presnap_t + a.sync_t) / f,
           qpc_ms(a.presnap_t) / f, qpc_ms(a.sync_t) / f, (long long)m_evicted_n, m_evicted_bytes / M, (long long)m_retries,
           (long long)m_untwinned_n, m_untwinned_bytes / M, ver);
    char capl[240] = "";
    if (wb_skip()) {
        snprintf(capl, sizeof capl, "; written ranges kept in the mirror alone %.3f MiB",
                 (uint64_t)plat_xchg64(&m_wb_skipped, 0) / f / M);
    }
    if (capture_cap())
        snprintf(capl + strlen(capl), sizeof capl - strlen(capl), "; copied by the CPU at hand-over %.3f MiB in %.1f "
                 "ranges, %.3f ms; left to the GPU copy %.3f MiB", a.cap_bytes / f / M, a.cap_ranges / f,
                 qpc_ms(a.cap_t) / f, a.cap_left / f / M);
    rt_log("mem: mirror frame %llu: upload sources per frame, overlaps counted twice: written pages %.3f MiB; flush ranges "
           "%.2f, %.3f MiB, from %.1f flushes of %.3f MiB; flushed and not copied (write-watched) %.3f MiB%s",
           (unsigned long long)frame, a.page_bytes / f / M, a.flush_n / f,
           a.flush_bytes / f / M, a.flush_calls / f, a.flush_sum / f / M, a.flush_skipped / f / M, capl);
}

// After vram_log: every heap's usage as the driver reports it (VK_EXT_memory_budget) beside VMA's share of it, VMA by
// memory type, and the process's private commit with its write-combined part (host-visible memory the driver maps
// uncached), to tell which memory a long session grows. When the write-combined total (VMA's blocks where that is not
// known) grew by 2 GiB since the last look, VMA's detailed map goes to vma-frame<N>.json beside the log, 4 at most.
// The process's part walks the whole address space (plat_mem_census), 100-240 ms in a long session, so it runs on a
// thread of its own (census_main) and logs a line of its own; a look while the last walk still runs goes without it.
typedef struct { VkCtx *c; uint64_t frame; } CensusJob;
static volatile int32_t census_busy;

static void census_main(void *arg) {
    CensusJob j = *(CensusJob *)arg;
    free(arg);
    const double M = 1048576.0;
    uint64_t priv = 0, wc = 0, wcn = 0, now;
    int have = plat_mem_census(&priv, &wc, &wcn);
    if (have) {
        rt_log("mem: process frame %llu: private %.0f MiB, write-combined %.0f MiB in %llu allocations",
               (unsigned long long)j.frame, priv / M, wc / M, (unsigned long long)wcn);
        now = wc;
    } else {
        VmaBudget vb[VK_MAX_MEMORY_HEAPS];
        vmaGetHeapBudgets(j.c->vma, vb);
        now = 0;
        for (uint32_t h = 0; h < j.c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++) now += vb[h].statistics.blockBytes;
    }
    static uint64_t base;
    static int dumps = -1;   // -1: no baseline yet
    if (dumps < 0) base = now, dumps = 0;
    else if (dumps < 4 && now >= base + (2ull << 30) && rt_log_path()) {
        base = now, dumps++;
        const char *lp = rt_log_path(), *sl = strrchr(lp, '/'), *bs = strrchr(lp, '\\');
        if (bs && (!sl || bs > sl)) sl = bs;
        char path[1024];
        snprintf(path, sizeof path, "%.*svma-frame%llu.json", sl ? (int)(sl - lp) + 1 : 0, lp, (unsigned long long)j.frame);
        char *json = NULL;
        vmaBuildStatsString(j.c->vma, &json, VK_TRUE);
        FILE *f = json ? fopen(path, "wb") : NULL;
        int ok = f && fputs(json, f) >= 0;
        if (f && fclose(f)) ok = 0;
        rt_log("mem: vma detailed map %s %s", ok ? "written to" : "not written:", path);
        if (json) vmaFreeStatsString(j.c->vma, json);
    }
    plat_xchg32(&census_busy, 0);
}

static void heap_log(VkCtx *c, uint64_t frame, const VmaBudget *vb) {
    const double M = 1048576.0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    if (c->has_memory_budget) {
        VkPhysicalDeviceMemoryProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bp};
        vkGetPhysicalDeviceMemoryProperties2(c->phys, &p2);
    }
    char line[1536];
    int o = snprintf(line, sizeof line, "mem: heaps frame %llu:", (unsigned long long)frame);
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS && o < (int)sizeof line - 160; h++)
        o += snprintf(line + o, sizeof line - (size_t)o, " heap %u (%s) used %.0f of %.0f MiB, vma %u blocks %.0f MiB, %u allocations %.0f MiB;",
                      h, c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? "device" : "host",
                      (c->has_memory_budget ? bp.heapUsage[h] : vb[h].usage) / M,
                      (c->has_memory_budget ? bp.heapBudget[h] : vb[h].budget) / M, vb[h].statistics.blockCount,
                      vb[h].statistics.blockBytes / M, vb[h].statistics.allocationCount, vb[h].statistics.allocationBytes / M);
    rt_log("%s", line);
    VmaTotalStatistics ts;
    vmaCalculateStatistics(c->vma, &ts);
    o = snprintf(line, sizeof line, "mem: vma by type frame %llu:", (unsigned long long)frame);
    for (uint32_t t = 0; t < c->mem_props.memoryTypeCount && t < VK_MAX_MEMORY_TYPES && o < (int)sizeof line - 120; t++) {
        const VmaDetailedStatistics *d = &ts.memoryType[t];
        if (!d->statistics.blockCount) continue;
        o += snprintf(line + o, sizeof line - (size_t)o,
                      " type %u: %u blocks %.0f MiB, %u allocations %.0f MiB, largest %.1f MiB, %u free ranges (largest %.1f MiB);",
                      t, d->statistics.blockCount, d->statistics.blockBytes / M, d->statistics.allocationCount,
                      d->statistics.allocationBytes / M, d->allocationSizeMax / M, d->unusedRangeCount,
                      d->unusedRangeCount ? d->unusedRangeSizeMax / M : 0.0);
    }
    rt_log("%s", line);
    // The address-space walk of the census (plat_mem_census, 100-240 ms of VirtualQuery) runs on its own thread, but
    // it holds the process's address-space lock, which the game's threads take at every page fault and every
    // VirtualAlloc: the present after each walk arrived 65-120 ms late. NX2NAVI_MEM_CENSUS=1 asks for it.
    static int census_on = -1;
    if (census_on < 0) census_on = cfg_bool("NX2NAVI_MEM_CENSUS");
    if (!census_on || plat_xchg32(&census_busy, 1)) return;
    CensusJob *j = malloc(sizeof *j);
    if (j) *j = (CensusJob){c, frame};
    if (!j || !plat_thread_spawn(census_main, j, "mem census", 0)) free(j), plat_xchg32(&census_busy, 0);
}

// After the vram line: the imported pools the GPU reached through a resolve (a binding, copy or texture placement) and
// those it never did, which hold host memory the driver counts in the host heap all the same (the 4.3 GiB imported
// at the title screen exceed the 3.9 GiB host heap of an integrated GPU with 16 GB of RAM); then the VRAM mirrors no
// submission synced for MIRROR_IDLE submissions, which a scene that needs the room could take back.
static void pool_census_log(uint64_t frame) {
    uint64_t un = 0, ub = 0, nn = 0, nb = 0, mn = 0, mb = 0, in = 0, ib = 0;
    uint32_t sub = __atomic_load_n(&msub, __ATOMIC_RELAXED);   // the replay's; a log line may read it a submission late
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) {
        const PoolRec *r = pool_index.e[i].buf;
        if (r->kind != POOL_IMPORT || !r->buf) continue;
        uint64_t n = r->import_hi - r->import_lo;
        if (__atomic_load_n(&r->used, __ATOMIC_RELAXED)) un++, ub += n;
        else nn++, nb += n;
        if (r->mstate != MS_ON) continue;
        mn++, mb += n;
        if (sub - r->acc_sub >= MIRROR_IDLE) in++, ib += n;
    }
    plat_mutex_unlock_shared(&pool_lock);
    const double M = 1048576.0;
    rt_log("mem: pools frame %llu: imported %llu %.0f MiB resolved by the GPU, %llu %.0f MiB never; mirrors %llu %.0f MiB, "
           "idle for %d submissions %llu %.0f MiB", (unsigned long long)frame, (unsigned long long)un, ub / M,
           (unsigned long long)nn, nb / M, (unsigned long long)mn, mb / M, MIRROR_IDLE, (unsigned long long)in, ib / M);
}

// After the heap lines: the driver's figure for the device-local heaps split by what holds it. VMA's blocks hold the
// textures with their own memory (live and released), the VRAM mirrors, the physical pools, the ASTC decoder's scratch,
// the headroom's spare blocks (their anchors), everything else VMA placed, and the free space inside the blocks, which
// the driver counts as used until a block empties. The rest of the driver's figure was never allocated through VMA:
// the swapchain, DLSS and frame generation (Streamline), compiled pipelines and the driver's own memory; DLSS's share
// as its statistics report it, when known.
static void vram_use_log(VkCtx *c, uint64_t frame, const VmaBudget *vb, const VkTexCensus *t) {
    uint64_t usage = 0, blocks = 0, allocs = 0, phys = 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    if (c->has_memory_budget) {
        VkPhysicalDeviceMemoryProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bp};
        vkGetPhysicalDeviceMemoryProperties2(c->phys, &p2);
    }
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            usage += c->has_memory_budget ? bp.heapUsage[h] : vb[h].statistics.blockBytes,
                blocks += vb[h].statistics.blockBytes, allocs += vb[h].statistics.allocationBytes;
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < nphys; i++)
        if (phys_list[i]->alloc) phys += phys_list[i]->size;
    plat_mutex_unlock_shared(&pool_lock);
    uint64_t mir = (uint64_t)__atomic_load_n(&m_live_bytes, __ATOMIC_RELAXED), held = vk_headroom_held(),
             scratch = vk_astc_scratch_bytes();
    uint64_t known = t->bytes_live + t->bytes_released + mir + phys + held + scratch;
    int64_t dlss = vk_dlaa_vram();
    char ds[48] = "n/a";
    if (dlss >= 0) snprintf(ds, sizeof ds, "%.0f MiB", dlss / 1048576.0);
    const double M = 1048576.0;
    rt_log("mem: vram by use frame %llu: in use %.0f MiB = VMA blocks %.0f MiB (textures %.0f, released textures %.0f, "
           "VRAM mirrors %.0f, physical pools %.0f, ASTC scratch %.0f, headroom spare %.0f, other %.0f, free inside the "
           "blocks %.0f) + outside VMA %.0f MiB (swapchain, DLSS and frame generation, pipelines, driver; DLSS reports %s)",
           (unsigned long long)frame, usage / M, blocks / M, t->bytes_live / M, t->bytes_released / M, mir / M, phys / M,
           scratch / M, held / M, allocs > known ? (allocs - known) / M : 0.0, blocks > allocs ? (blocks - allocs) / M : 0.0,
           usage > blocks ? (usage - blocks) / M : 0.0, ds);
}

// With NX2NAVI_TRACE=perf every 60 frames: a line when the device-local memory VMA did not allocate (vram_use_log's
// "outside VMA") moved by OUTSIDE_STEP or more since the last line, so the log places it beside what did it. In one
// session it grew from 0.6 to 2.6 GiB at the title screen, in two steps of about 1 GiB while the boot's compute
// pipelines were being built (vk_pipeline.c, stage library prebuild), and DLSS and frame generation add theirs when
// they start.
#define OUTSIDE_STEP (256ull << 20)
static void outside_watch(uint64_t frame) {
    VkCtx *c = vk_ctx();
    if (!c || !c->has_memory_budget) return;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bp};
    vkGetPhysicalDeviceMemoryProperties2(c->phys, &p2);
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    uint64_t usage = 0, blocks = 0;
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            usage += bp.heapUsage[h], blocks += vb[h].statistics.blockBytes;
    uint64_t out = usage > blocks ? usage - blocks : 0;
    static uint64_t last;
    static int seen;
    if (seen && (out > last ? out - last : last - out) < OUTSIDE_STEP) return;
    const double M = 1048576.0;
    rt_log("mem: frame %llu: device memory outside VMA (swapchain, DLSS and frame generation, pipelines, driver) %s%.0f "
           "MiB (%s%.0f MiB)", (unsigned long long)frame, seen ? "now " : "", out / M, out >= last ? "+" : "-",
           (out >= last ? out - last : last - out) / M);
    last = out, seen = 1;
}

// The overlay's "game" figure beside the card's memory in use (perfmon.c, its sampling thread): what the driver
// counts for this process on the device-local heaps, the same figure vram_use_log splits by use.
// With the device lost report (vk_device.c): the video memory in use and the budget the driver gives the game. A lost
// device with the memory at or over its budget points at the memory; one far below it does not, and the report of a
// card with 8 GB that ended this way held neither number.
void vk_mem_lost_report(void) {
    uint64_t used, budget;
    int exact;
    if (!vram_query(&used, &budget, &exact)) return;
    rt_log("vk: video memory when the device was lost: %llu MiB in use of a budget of %llu MiB (%s)",
           (unsigned long long)(used >> 20), (unsigned long long)(budget >> 20),
           exact ? "the driver's figures, every program counted in the budget" : "this game's own allocations only");
}

// The settings that change how the game's memory reaches the GPU, as they are in effect, for a fatal error's summary
// line (vk_device.c fatal_note): the memory mode and the lazy write-back with its reason and state. No lock, any thread.
void vk_mem_fatal_note(char *out, size_t n) {
    char lazy[160];
    mem_lazy_note(lazy, sizeof lazy);
    int st = __atomic_load_n(&lz_state, __ATOMIC_ACQUIRE), want = __atomic_load_n(&lz_wanted, __ATOMIC_ACQUIRE);
    snprintf(out, n, "memory mode %s%s, lazy write-back %s (%s%s%s)", mirror_verify ? "verify" : vk_mirror_on ? "mirror" : "import",
             shadow_on ? " with shadow storage" : "", want < 0 ? "not decided" : st == 2 && want ? "failed to start" : want ? "on" : "off",
             lz_why, *lazy ? "; " : "", lazy);
}

float vk_mem_device_used_mb(void) {
    VkCtx *c = vk_ctx();
    if (!c || !c->has_memory_budget) return -1;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bp};
    vkGetPhysicalDeviceMemoryProperties2(c->phys, &p2);
    uint64_t usage = 0;
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) usage += bp.heapUsage[h];
    return (float)(usage / 1048576.0);
}

// One line per boot stage (vk_pipeline.c calls it after the pipeline cache load, the shader prewarm, the recorded
// pipelines and the compute pipeline prebuild): the device-local memory in use, the part of it VMA did not allocate
// (outside_watch's "outside VMA": pipelines and the driver's own, before the swapchain and the upscaler exist), and the
// process's commit and working set. Subtracting two lines gives what a stage left behind: on an installed copy the
// compute pipeline prebuild alone left 2.1 GiB of device memory on an RTX 5090, held by twenty very large compute
// shaders no run dispatches (vk_pipeline.c, "compute pipelines the boot keeps"), which this line was added to place.
void vk_mem_stage_log(const char *stage) {
    VkCtx *c = vk_ctx();
    double proc, used, limit;
    float ram_used, ram_total, ws;
    plat_commit_info(&proc, &used, &limit);
    plat_mem_info(&ram_used, &ram_total, &ws);
    const double M = 1048576.0;
    uint64_t usage = 0, blocks = 0;
    int budget = c && c->has_memory_budget;
    if (budget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bp};
        vkGetPhysicalDeviceMemoryProperties2(c->phys, &p2);
        VmaBudget vb[VK_MAX_MEMORY_HEAPS] = {0};
        if (c->vma) vmaGetHeapBudgets(c->vma, vb);
        for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
            if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                usage += bp.heapUsage[h], blocks += vb[h].statistics.blockBytes;
    }
    char dev[96] = "device memory unknown (no VK_EXT_memory_budget)";
    if (budget)
        snprintf(dev, sizeof dev, "device memory in use %.0f MiB, outside VMA %.0f MiB", usage / M,
                 usage > blocks ? (usage - blocks) / M : 0.0);
    rt_log("mem: boot stage, after %s: %s; process commit %.0f MB, working set %.0f MB; system commit %.0f of %.0f MB",
           stage, dev, proc, (double)ws, used, limit);
}

// With NX2NAVI_TRACE=perf every 600 frames: what holds device-local memory, to see what grows over a long session.
// VMA's blocks and allocations on the device-local heaps, texture entries with their own memory, the VRAM mirrors and
// the objects waiting for their last submission to retire.
static void vram_log(uint64_t frame) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    VmaBudget vb[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(c->vma, vb);
    uint64_t blocks = 0, block_bytes = 0, allocs = 0, alloc_bytes = 0;
    for (uint32_t h = 0; h < c->mem_props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; h++)
        if (c->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            blocks += vb[h].statistics.blockCount, block_bytes += vb[h].statistics.blockBytes,
                allocs += vb[h].statistics.allocationCount, alloc_bytes += vb[h].statistics.allocationBytes;
    VkTexCensus t;
    vk_texture_census(&t);
    const double M = 1048576.0;
    rt_log("mem: vram frame %llu: vma %llu blocks %.0f MiB, %llu allocations %.0f MiB; textures live %llu (own memory %llu, "
           "%.0f MiB), released %llu (own memory %llu, %.0f MiB); mirrors %lld %.0f MiB; releases pending %d",
           (unsigned long long)frame, (unsigned long long)blocks, block_bytes / M, (unsigned long long)allocs,
           alloc_bytes / M, (unsigned long long)t.live, (unsigned long long)t.own_live, t.bytes_live / M,
           (unsigned long long)t.released, (unsigned long long)t.own_released, t.bytes_released / M, (long long)m_live_n,
           m_live_bytes / M, dead_pending());
    vk_texture_census_log(frame);
    pool_census_log(frame);
    heap_log(c, frame, vb);
    vram_use_log(c, frame, vb, &t);
}

// Device class: INTEGRATED -> uma; a discrete GPU with a DEVICE_LOCAL heap of 1 GiB or more beside a
// heap that is not device-local (host memory) -> discrete; otherwise uma (import everywhere). Sets the mirror mode.
void vk_mem_class_log(const VkPhysicalDeviceProperties *p, const VkPhysicalDeviceMemoryProperties *m) {
    int big_local = -1, host_heap = -1;
    for (uint32_t h = 0; h < m->memoryHeapCount; h++) {
        if (m->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (m->memoryHeaps[h].size >= (1ull << 30) && big_local < 0) big_local = (int)h;
        } else if (host_heap < 0) host_heap = (int)h;
    }
    static const char *const types[] = {"other", "integrated", "discrete", "virtual", "cpu"};
    int t = (int)p->deviceType;
    const char *tn = t >= 0 && t <= 4 ? types[t] : "unknown";
    const char *cls, *why;
    if (p->deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) cls = "uma", why = "integrated gpu";
    else if (p->deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) cls = "uma", why = "not a discrete gpu";
    else if (big_local < 0) cls = "uma", why = "no device-local heap of 1 GiB or more";
    else if (host_heap < 0) cls = "uma", why = "every heap is device-local";
    else cls = "discrete", why = "device-local heap separate from host memory";
    int discrete = cls[0] == 'd';
    char line[2048];
    size_t k = (size_t)snprintf(line, sizeof line, "mem: class %s (%s; device type %d %s, %s)", cls, why, t, tn, p->deviceName);
    for (uint32_t h = 0; h < m->memoryHeapCount; h++)
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, "; heap %u %.0f MiB flags %#x%s%s", h,
                              m->memoryHeaps[h].size / 1048576.0, m->memoryHeaps[h].flags,
                              m->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "",
                              m->memoryHeaps[h].flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT ? " MULTI_INSTANCE" : "");
    rt_log("%s", line);
    k = (size_t)snprintf(line, sizeof line, "mem: types");
    for (uint32_t i = 0; i < m->memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = m->memoryTypes[i].propertyFlags;
        k += (size_t)snprintf(line + k, k < sizeof line ? sizeof line - k : 0, "; %u heap %u flags %#x%s%s%s%s%s", i,
                              m->memoryTypes[i].heapIndex, f, f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "",
                              f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? " HOST_VISIBLE" : "",
                              f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? " HOST_COHERENT" : "",
                              f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? " HOST_CACHED" : "",
                              f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT ? " LAZY" : "");
    }
    rt_log("%s", line);
    mirror_mode(discrete, vk_caps.driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY);
}

// ---------------------------------------------------------------- upload batches and deferred destruction
typedef struct { VkBuffer buf; VmaAllocation alloc; VkDeviceSize size; } Staging;   // size: the VkBuffer's own
typedef struct { VkBuffer buf; VmaAllocation alloc; uint8_t *map; } StgChunk;          // a staging arena chunk
typedef struct {
    VkCommandBuffer cb;
    VkFence fence;
    uint64_t seq;
    Staging *stg;
    int nstg, cap;
    StgChunk *chunk;          // arena chunks this batch's uploads took ranges of (vk_staging_at), the last one open
    int nchunk, cap_chunk;
    VkDeviceSize chunk_used;  // bytes of the last chunk handed out
    VkDeviceSize staging_bytes; // complete buffer/chunk spans owned until this batch's actual fence completion
} Batch;

enum { DEAD_IMAGE = 1, DEAD_IMAGE_VMA, DEAD_VIEW, DEAD_BUFFER_VMA, DEAD_HOST, DEAD_UPLOAD_CALLBACK };
typedef struct OwnedImageView {
    struct OwnedImageView *next;
    VkImageView view;
} OwnedImageView;
struct VkMemoryImageHold {
    VkImage image;
    PoolRec *pool;
    VkDeviceMemory pool_memory;
    VkDeviceSize pool_begin, pool_end;
    struct VkMemoryImageHold *placement_previous, *placement_next;
    OwnedImageView *views;
    uint32_t readers;
    int detached, binding_pending;
};
typedef struct {
    int kind;
    uint64_t handle;
    VmaAllocation alloc;
    uint64_t seq, tl, reclaim_bytes;
    void (*destroyed)(void *);
    void *context;
    VkMemoryImageHold *image_hold;
} Dead;

// Allocation bytes selected for pressure reclamation but not yet destroyed. This includes objects handed to
// vk_mem_collect's out-of-lock destruction loop; fence completion alone does not release their allocation.
static uint64_t reclaim_pending_bytes;

uint64_t vk_mem_reclaim_pending(void) {
    return __atomic_load_n(&reclaim_pending_bytes, __ATOMIC_ACQUIRE);
}

static PlatRMutex cmd_lock;
static PlatOnce cmd_once = PLAT_ONCE_INIT;
// Only images with an exact pool binding or a present reader need a record. On retirement the record moves to Dead.
static nvn_hmap image_holds = {.klen = sizeof(VkImage)};
// Actual pool-bound placements are indexed by the native device memory, including images pending
// destruction. Separate guest addresses can refer to the same interval of one native allocation.
static nvn_hmap image_placements = {.klen = sizeof(VkDeviceMemory)};
static uint32_t image_readers;
static VkCommandPool cmd_pool;
static Batch open_b;
static int open_active;
static Batch *inflight;
static int ninflight, cap_inflight;
static Batch *spare;                  // completed batches whose command buffer and fence are reused
static int nspare, cap_spare;
static uint64_t next_seq, done_seq;   // batches handed out / completed (all batches up to done_seq)
static uint64_t submitted_seq;
static uint32_t upload_depth;
static uint64_t inflight_staging_bytes, upload_checkpoints, upload_pressure_waits;
// A long front preparation can record thousands of uploads before its scene is ready. Submit completed upload
// operations in bounded batches so their staging can return while preparation continues. One indivisible operation
// may exceed either target; never end a command buffer while an upload caller still holds it.
#define UPLOAD_BATCH_MIN (16ull << 20)
#define UPLOAD_BATCH_MAX (64ull << 20)
#define UPLOAD_INFLIGHT_MIN (64ull << 20)
#define UPLOAD_INFLIGHT_MAX (256ull << 20)
static uint64_t upload_batch_window = UPLOAD_BATCH_MAX, upload_inflight_window = UPLOAD_INFLIGHT_MAX;
static Dead *dead;
static int ndead, cap_dead;
static uint64_t tl_noted;             // newest timeline value vk_sync.c reported submitted

static unsigned staging_budget_reclaim(void *unused, const RcAdmissionVector *deficit);

static void cmd_init(void *p) {
    (void)p;
    const uint32_t gpu = ((1u << RC_BUDGET_GPU_HEAPS) - 1u) << RC_BUDGET_GPU_FIRST;
    const uint32_t host = (1u << RC_BUDGET_RAM) | (1u << RC_BUDGET_COMMIT) | (1u << RC_BUDGET_STAGING);
    if (resource_core_budget_register_required_reclaimer(RC_BUDGET_STAGE, host | gpu, staging_budget_reclaim, NULL) !=
        RC_ADMISSION_OK) rt_fatal("vk: could not register reusable staging giveback");
}

static void grow(void **arr, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 16;
    void *n = realloc(*arr, (size_t)*cap * elem);
    if (!n) rt_fatal("vk: out of memory (upload bookkeeping)");
    *arr = n;
}

// Staging buffers of a finished batch go back to a cache instead of the allocator: an upload-heavy scene creates and
// retires the same few sizes thousands of times a frame (a texture re-upload, a texel copy, an ASTC decode), and each
// vmaCreateBuffer/vmaDestroyBuffer pair walks the pool's block list under its lock. Bounded in count and bytes; a buffer
// that does not fit is destroyed as before. The batch is complete when this runs, so nothing reads the bytes again.
// Sizes are the VkBuffers' own: an allocation can be larger than its buffer, and a copy reads only the buffer's range.
// cmd_lock held (retire, and vk_staging between vk_upload_begin and vk_upload_end).
#define STG_CACHE 24
#define STG_CACHE_BYTES (192ull << 20)
static Staging stg_cache[STG_CACHE];
static int stg_ncache;
static VkDeviceSize stg_cache_bytes;

// The staging arena: an upload of at most ARENA_MAX bytes takes an ARENA_ALIGN-aligned range of a persistent mapped
// chunk (vk_staging_at) instead of a buffer of its own. The chunks belong to the open batch while it records and go
// back to the free list when its fence signals, so a frame's few hundred small texture uploads cost a pointer bump
// each instead of a vmaCreateBuffer and vmaDestroyBuffer under the allocator's lock. ARENA_KEEP free chunks stay; the
// rest are destroyed. 256-byte alignment covers every bufferOffset rule of a buffer-to-image copy. NX2NAVI_STG_ARENA=0
// gives every upload its own buffer again.
#define ARENA_CHUNK (8ull << 20)
#define ARENA_MAX (ARENA_CHUNK / 4)
#define ARENA_ALIGN 256ull
#define ARENA_KEEP 8
static StgChunk arena_free[ARENA_KEEP];
static int narena_free;
static uint64_t arena_made, arena_ranges;

static int staging_cache_room(int count, VkDeviceSize bytes, VkDeviceSize size) {
    return count < STG_CACHE && size <= STG_CACHE_BYTES - bytes;
}

static void free_staging(VkCtx *c, Batch *b) {
    const int reusable = stg_ncache + narena_free;
    for (int i = 0; i < b->nstg; i++) {
        if (staging_cache_room(stg_ncache, stg_cache_bytes, b->stg[i].size)) {
            stg_cache[stg_ncache++] = b->stg[i];
            stg_cache_bytes += b->stg[i].size;
            continue;
        }
        resource_core_vma_destroy_buffer(c->vma, b->stg[i].buf, b->stg[i].alloc);
    }
    b->nstg = 0;
    for (int i = 0; i < b->nchunk; i++) {
        if (narena_free < ARENA_KEEP) arena_free[narena_free++] = b->chunk[i];
        else resource_core_vma_destroy_buffer(c->vma, b->chunk[i].buf, b->chunk[i].alloc);
    }
    b->nchunk = 0, b->chunk_used = 0;
    b->staging_bytes = 0;
    if (stg_ncache + narena_free != reusable) resource_core_budget_progress();
}

// The cached staging buffer that fits n with the least waste: at most four times the request, so a small upload does not
// take a block-sized buffer another upload wanted.
static int staging_size_fits(VkDeviceSize size, VkDeviceSize need) {
    return size >= need && size <= need * 4 + 65536;
}

static int staging_cache_take(VkCtx *c, VkDeviceSize n, Staging *out, void **map) {
    int best = -1;
    for (int i = 0; i < stg_ncache; i++) {
        if (!staging_size_fits(stg_cache[i].size, n)) continue;
        if (best < 0 || stg_cache[i].size < stg_cache[best].size) best = i;
    }
    if (best < 0) return 0;
    VmaAllocationInfo ai;
    vmaGetAllocationInfo(c->vma, stg_cache[best].alloc, &ai);
    *out = stg_cache[best], *map = ai.pMappedData;
    stg_cache_bytes -= stg_cache[best].size;
    stg_cache[best] = stg_cache[--stg_ncache];
    return 1;
}

// Batches complete in submission order (one queue): retire from the front while their fences are signalled.
static void retire(VkCtx *c, int wait_all) {
    int k = 0;
    for (; k < ninflight; k++) {
        Batch *b = &inflight[k];
        if (wait_all) vkWaitForFences(c->device, 1, &b->fence, VK_TRUE, UINT64_MAX);
        else if (vkGetFenceStatus(c->device, b->fence) != VK_SUCCESS) break;
        inflight_staging_bytes -= b->staging_bytes;
        free_staging(c, b);
        done_seq = b->seq;
        if (nspare == cap_spare) grow((void **)&spare, &cap_spare, sizeof *spare);
        spare[nspare++] = *b;
    }
    memmove(inflight, inflight + k, (size_t)(ninflight - k) * sizeof *inflight);
    ninflight -= k;
}

// cmd_lock held. Only the texture owner's locked selection may start a new read or publish a pool alias.
static VkMemoryImageHold *image_hold_find_or_add(VkImage image) {
    VkMemoryImageHold *hold = nvn_hm_get(&image_holds, &image);
    if (!hold) {
        hold = calloc(1, sizeof *hold);
        if (!hold) rt_fatal("vk: native image hold allocation failed");
        hold->image = image;
        nvn_hm_put(&image_holds, &image, hold);
    }
    return hold;
}

// The placement list for one device memory is sorted and disjoint. A distinct image has its own
// optimal tiling/layout state even when its guest bytes share the same backing.
static int image_placement_available(VkDeviceMemory memory, VkDeviceSize begin, VkDeviceSize end,
                                      VkMemoryImageHold **previous, VkMemoryImageHold **next) {
    *previous = NULL;
    *next = nvn_hm_get(&image_placements, &memory);
    while (*next && (*next)->pool_end <= begin) {
        *previous = *next;
        *next = (*next)->placement_next;
    }
    return !*next || end <= (*next)->pool_begin;
}

static void image_placement_insert(VkMemoryImageHold *hold, VkMemoryImageHold *previous, VkMemoryImageHold *next) {
    hold->placement_previous = previous;
    hold->placement_next = next;
    if (previous) previous->placement_next = hold;
    else nvn_hm_put(&image_placements, &hold->pool_memory, hold);
    if (next) next->placement_previous = hold;
}

static void image_placement_remove(VkMemoryImageHold *hold) {
    VkMemoryImageHold *previous = hold->placement_previous, *next = hold->placement_next;
    if (previous) previous->placement_next = next;
    else if (next) nvn_hm_put(&image_placements, &hold->pool_memory, next);
    else nvn_hm_del(&image_placements, &hold->pool_memory);
    if (next) next->placement_previous = previous;
}

#define IMAGE_ALIAS_BASELINE_LIMIT 128

typedef enum {
    IMAGE_ALIAS_BASELINE_OPEN,
    IMAGE_ALIAS_BASELINE_HANDED,
    IMAGE_ALIAS_BASELINE_SUBMITTED,
} ImageAliasBaselineMode;

struct VkMemImageAliasBaseline {
    VkMemImageAliasBaseline *next;
    PoolRec *window, *provider;
    VkMemSpan native;
    RcBudgetTicket budget;
    RcRegistryPosition read;
    uint64_t address, bytes, fill_upload;
    uint32_t count;
    ImageAliasBaselineMode mode;
    RcRegistryBufferSpan spans[];
};

// pool_lock held. Track exact raw ranges so unrelated bound images can still use the physical pool.
static int physical_origin_image_allowed(const PoolRec *pool, uint64_t lo, uint64_t hi) {
    for (const VkMemImageAliasBaseline *borrow = pool->physical_origin_borrows; borrow; borrow = borrow->next)
        for (uint32_t i = 0; i < borrow->count; ++i) {
            const RcRegistryBufferSpan *part = &borrow->spans[i];
            if (part->native.offset < hi && lo < part->native.offset + part->interval.range.size) return 0;
        }
    return 1;
}

// pool_lock held. Merge overlaps and adjacency without allocating or dropping older intervals.
static int physical_range_add(FlushRange *ranges, uint32_t *used, uint64_t lo, uint64_t hi) {
    const uint32_t count = *used;
    uint32_t first = 0;
    while (first < count && ranges[first].hi < lo) first++;
    uint32_t last = first;
    while (last < count && ranges[last].lo <= hi) {
        if (ranges[last].lo < lo) lo = ranges[last].lo;
        if (ranges[last].hi > hi) hi = ranges[last].hi;
        last++;
    }
    if (first == last && count == PHYSICAL_ORIGIN_POISON_LIMIT) return 0;
    memmove(ranges + first + 1, ranges + last, (count - last) * sizeof *ranges);
    ranges[first] = (FlushRange){lo, hi};
    *used = count - (last - first) + 1;
    return 1;
}

// pool_lock held. The exact span stays poisoned after image discard; destroying an image does not restore raw bytes.
static void physical_origin_poison(PoolRec *pool, uint64_t lo, uint64_t hi) {
    if (!pool->physical_origin_upload || pool->physical_origin_unknown || lo >= hi) return;
    if (!physical_range_add(pool->physical_origin_poison, &pool->physical_origin_poison_count, lo, hi))
        pool->physical_origin_unknown = 1;
}

static int physical_use_preserve_locked(PoolRec *pool, uint64_t lo, uint64_t size) {
    const uint64_t hi = lo + size;
    for (const PhysicalOriginBinding *bind = pool->physical_origin_pending; bind; bind = bind->next)
        if (bind->lo < hi && lo < bind->hi) return 0;
    if (pool->physical_raw_use_unknown || physical_initial_fill_covers(pool, lo, size)) return 1;
    if (!pool->physical_raw_uses)
        pool->physical_raw_uses = calloc(PHYSICAL_ORIGIN_POISON_LIMIT, sizeof *pool->physical_raw_uses);
    if (!pool->physical_raw_uses || !physical_range_add(pool->physical_raw_uses, &pool->physical_raw_use_count, lo, hi))
        pool->physical_raw_use_unknown = 1;
    return 1;
}

static int physical_use_overlaps(const PoolRec *pool, uint64_t lo, uint64_t hi) {
    if (pool->physical_raw_use_unknown) return 1;
    for (uint32_t i = 0; i < pool->physical_raw_use_count; ++i) {
        const FlushRange part = pool->physical_raw_uses[i];
        if (part.lo >= hi) return 0;
        if (lo < part.hi) return 1;
    }
    return 0;
}

// pool_lock held. Keep potential initialization before upload completion as well as completed raw contents.
// A known earlier optimal alias removes only its exact bytes from the original initialization proof.
static int physical_origin_fill_overlaps(const PoolRec *provider, uint64_t lo, uint64_t hi) {
    if (!provider->physical_origin_upload || lo >= provider->physical_origin_bytes) return 0;
    if (provider->physical_origin_unknown) return 1;
    if (hi > provider->physical_origin_bytes) hi = provider->physical_origin_bytes;
    for (uint32_t i = 0; i < provider->physical_origin_poison_count; ++i) {
        const FlushRange poisoned = provider->physical_origin_poison[i];
        if (poisoned.hi <= lo) continue;
        if (lo < poisoned.lo) return 1;
        lo = poisoned.hi;
        if (lo >= hi) return 0;
    }
    return lo < hi;
}

static int image_alias_placement_inputs_valid(uint64_t address, const VkMemoryRequirements *requirements,
    const VkMemSpan *selected, const RcInterval *mapping, uint32_t mapping_count) {
    return requirements && requirements->size && requirements->alignment && requirements->memoryTypeBits &&
        requirements->size <= UINT64_MAX - address && selected && selected->alloc && selected->buffer &&
        requirements->size <= selected->avail && mapping && mapping_count;
}

static PoolRec *image_alias_placement_provider_locked(uint64_t address, uint64_t size, const VkMemSpan *selected,
    const RcInterval *mapping, uint32_t mapping_count) {
    const PoolRec *window = NULL;
    PoolRec *provider = pool_image_alias_provider_locked(address, size, selected, &window);
    if (!provider || !provider->publication ||
        !rc_backing_map_intervals_current(pool_backings, window->map, mapping, mapping_count, address, size))
        return NULL;
    return provider;
}

static int physical_image_alias_range(const PoolRec *pool, uint64_t offset, uint64_t size,
                                      uint64_t granularity, FlushRange *range) {
    const uint64_t base = pool->physical_memory_offset;
    if (!granularity || (granularity & (granularity - 1)) || pool->size > UINT64_MAX - base ||
        offset >= pool->size || !size || size > pool->size - offset) return 0;
    // A linear buffer and optimal image can alias at bufferImageGranularity even when their
    // unpadded ranges are disjoint. Round device-memory offsets, then clip to this raw buffer.
    const uint64_t mask = granularity - 1;
    const uint64_t lower = (base + offset) & ~mask;
    const uint64_t end = offset + size;
    const uint64_t end_in_page = (base + end) & mask;
    const uint64_t tail = end_in_page ? granularity - end_in_page : 0;
    *range = (FlushRange){lower > base ? lower - base : 0,
        tail < pool->size - end ? end + tail : pool->size};
    return 1;
}

static VkMemImageAliasPlacement image_alias_preplacement_locked(uint64_t address, uint64_t size,
    const VkMemSpan *selected, const RcInterval *mapping, uint32_t mapping_count, PoolRec **provider, FlushRange *range) {
    *provider = image_alias_placement_provider_locked(address, size, selected, mapping, mapping_count);
    if (!*provider) return VK_MEM_IMAGE_ALIAS_STALE;
    PoolRec *physical = *provider;
    if (!physical_image_alias_range(physical, selected->alloc_offset, size,
                                    vk_ctx()->props.limits.bufferImageGranularity, range)) return VK_MEM_IMAGE_ALIAS_FAILED;
    const int fill = physical_origin_fill_overlaps(physical, range->lo, range->hi);
    const int use = !fill && physical_use_overlaps(physical, range->lo, range->hi);
    const int borrow = !fill && !use && !physical_origin_image_allowed(physical, range->lo, range->hi);
    if (rt_knob("IMAGE_BASELINE_TRACE", "0")) {
        static uint32_t entries;
        if (__atomic_fetch_add(&entries, 1, __ATOMIC_RELAXED) < 128) {
            RcRegistryPosition read = vk_exec_resource_position();
            rt_log("vk: image placement raw guard read %llu/%llu address %#llx publication %llu backing %llu "
                   "offset %#llx size %#llx fill %d upload %llu selected-raw-use %d active-borrow %d decision %s "
                   "hazard %#llx+%#llx",
                   (unsigned long long)read.scene, (unsigned long long)read.ordinal, (unsigned long long)address,
                   (unsigned long long)physical->publication, (unsigned long long)physical->backing.id,
                   (unsigned long long)selected->alloc_offset, (unsigned long long)size, fill,
                   (unsigned long long)physical->physical_origin_upload, use, borrow,
                   fill || use || borrow ? "private" : "direct",
                   (unsigned long long)range->lo, (unsigned long long)(range->hi - range->lo));
        }
    }
    if (fill || use || borrow) return VK_MEM_IMAGE_ALIAS_PRIVATE_PRESERVE;
    return VK_MEM_IMAGE_ALIAS_DIRECT;
}

VkMemImageAliasPlacement vk_mem_image_alias_preplacement(uint64_t address, const VkMemoryRequirements *requirements,
    const VkMemSpan *selected, const RcInterval *mapping, uint32_t mapping_count) {
    if (!image_alias_placement_inputs_valid(address, requirements, selected, mapping, mapping_count))
        return VK_MEM_IMAGE_ALIAS_FAILED;
    PoolRec *provider;
    FlushRange range;
    plat_mutex_lock_shared(&pool_lock);
    const VkMemImageAliasPlacement result =
        image_alias_preplacement_locked(address, requirements->size, selected, mapping, mapping_count, &provider, &range);
    plat_mutex_unlock_shared(&pool_lock);
    return result;
}

static VkMemImageAliasPlacement physical_origin_begin_image(uint64_t address, uint64_t size, const VkMemSpan *selected,
    const RcInterval *mapping, uint32_t mapping_count, PoolRec *provider, PhysicalOriginBinding *binding) {
    plat_mutex_lock(&pool_lock);
    PoolRec *current;
    FlushRange range;
    VkMemImageAliasPlacement result = image_alias_preplacement_locked(address, size, selected, mapping, mapping_count,
                                                                       &current, &range);
    if (current != provider) result = VK_MEM_IMAGE_ALIAS_STALE;
    if (result == VK_MEM_IMAGE_ALIAS_DIRECT) {
        *binding = (PhysicalOriginBinding){provider->physical_origin_pending, range.lo, range.hi};
        provider->physical_origin_pending = binding;
    }
    plat_mutex_unlock(&pool_lock);
    return result;
}

static void physical_origin_end_image(PoolRec *provider, PhysicalOriginBinding *binding, int bound) {
    plat_mutex_lock(&pool_lock);
    PhysicalOriginBinding **link = &provider->physical_origin_pending;
    while (*link && *link != binding) link = &(*link)->next;
    if (!*link) rt_fatal("vk: lost a provisional physical image placement");
    *link = binding->next;
    if (bound) physical_origin_poison(provider, binding->lo, binding->hi);
    plat_mutex_unlock(&pool_lock);
}

// A provisional bind excludes its raw bytes even while its result is unknown.
static int physical_origin_pending_overlap(const PoolRec *provider, const RcRegistryBufferSpan *spans, uint32_t count) {
    for (const PhysicalOriginBinding *binding = provider->physical_origin_pending; binding; binding = binding->next)
        for (uint32_t i = 0; i < count; ++i) {
            const uint64_t start = spans[i].native.offset;
            if (binding->lo < start + spans[i].interval.range.size && start < binding->hi) return 1;
        }
    return 0;
}

static int image_alias_baseline_read_current(RcRegistryPosition read, ImageAliasBaselineMode mode) {
    if (!rc_registry_position_valid(read)) return 0;
    if (mode == IMAGE_ALIAS_BASELINE_HANDED) return !rc_registry_position_compare(read, rc_registry_handed_cutoff(rc_registry_runtime()));
    return !(read.ordinal & 1) && !rc_registry_position_compare(read, vk_exec_resource_position()) &&
        rc_registry_position_open(rc_registry_runtime(), read);
}

static void image_alias_baseline_trace(RcRegistryPosition read, uint64_t address, uint64_t bytes,
    const RcInterval *mapping, const PoolRec *pool, const VkMemSpan *native, uint64_t submitted, uint64_t completed,
    uint32_t count, RcRegistryResult result, const char *reason) {
    if (!rt_knob("IMAGE_BASELINE_TRACE", "0")) return;
    static uint32_t entries;
    if (__atomic_fetch_add(&entries, 1, __ATOMIC_RELAXED) >= 128) return;
    rt_log("vk: image baseline read %llu/%llu guest %#llx+%#llx backing %llu mapping %llu "
           "publication %llu buffer %#llx+%#llx init %#llx seq %llu submitted %llu completed %llu "
           "poison %u unknown %d spans %u status %d %s",
        (unsigned long long)read.scene, (unsigned long long)read.ordinal, (unsigned long long)address,
        (unsigned long long)bytes, (unsigned long long)mapping->range.backing, (unsigned long long)mapping->mapping,
        (unsigned long long)(pool ? pool->publication : 0), (unsigned long long)(uintptr_t)native->buffer,
        (unsigned long long)native->offset, (unsigned long long)(pool ? pool->physical_origin_bytes : 0),
        (unsigned long long)(pool ? pool->physical_origin_upload : 0),
        (unsigned long long)submitted, (unsigned long long)completed,
        pool ? pool->physical_origin_poison_count : 0, pool ? pool->physical_origin_unknown : 0,
        count, (int)result, reason);
}

// pool_lock held. Native offsets are affine for this one captured physical run.
static RcRegistryResult image_alias_baseline_partition(const PoolRec *pool, const VkMemSpan *native,
    uint64_t address, uint64_t bytes, const RcInterval *mapping, uint32_t count,
    RcRegistryBufferSpan *out, uint32_t *written) {
    const uint64_t limit = address + bytes;
    uint64_t guest = address;
    uint32_t used = 0;
    for (uint32_t i = 0; i < count && guest < limit; ++i) {
        const RcInterval part = mapping[i];
        const uint64_t end = part.address + part.range.size < limit ? part.address + part.range.size : limit;
        if (end <= guest) continue;
        if (!rc_range_valid(part.range) || part.address > guest || !part.mapping) return RC_REGISTRY_STALE;
        // The backing store has already proved this mapping current. A different physical shape supplies
        // no raw baseline to this receiver, leaving exact image-only coverage available.
        if (part.range.backing != pool->backing.id ||
            part.range.offset + (guest - part.address) != native->offset + (guest - address)) {
            *written = 0;
            return RC_REGISTRY_OK;
        }
        uint64_t lo = native->offset + (guest - address);
        uint64_t hi = native->offset + (end - address);
        if (hi > pool->physical_origin_bytes) hi = pool->physical_origin_bytes;
        for (uint32_t j = 0; lo < hi && j <= pool->physical_origin_poison_count; ++j) {
            const FlushRange gap = j < pool->physical_origin_poison_count ?
                pool->physical_origin_poison[j] : (FlushRange){hi, hi};
            if (gap.hi <= lo) continue;
            const uint64_t until = gap.lo < hi ? gap.lo : hi;
            if (lo < until) {
                if (used == IMAGE_ALIAS_BASELINE_LIMIT) return RC_REGISTRY_CAPACITY;
                out[used++] = (RcRegistryBufferSpan){
                    {address + (lo - native->offset), part.mapping, {part.range.backing, lo, until - lo}},
                    {pool->publication, 0, lo}};
            }
            lo = gap.hi;
        }
        guest = end;
    }
    if (guest != limit) return RC_REGISTRY_STALE;
    *written = used;
    return RC_REGISTRY_OK;
}
// Find the next change in the already-recorded physical source envelope. Being below the initial
// fill is insufficient inside an optimal-image alias. This is an effect partition, not a read proof.
static uint64_t fixed_buffer_part_end(const PoolRec *provider, uint64_t begin, uint64_t end) {
    if (begin >= provider->physical_origin_bytes) return end;
    if (end > provider->physical_origin_bytes) end = provider->physical_origin_bytes;
    for (uint32_t i = 0; i < provider->physical_origin_poison_count; i++) {
        const FlushRange poison = provider->physical_origin_poison[i];
        if (begin >= poison.hi) continue;
        const uint64_t edge = begin < poison.lo ? poison.lo : poison.hi;
        return edge < end ? edge : end;
    }
    return end;
}

uint32_t vk_mem_span_buffer_fixed_partitions(const VkMemSpanCapture *capture, const VkMemSpan *selected,
    const RcInterval *spans, uint32_t count, RcInterval *parts, uint32_t capacity) {
    if (!capture || !capture->native_input || !selected || !selected->alloc || !spans || !count ||
        !parts || count >= capacity || !nvn_gpu_only(capture->address)) return 0;
    plat_mutex_lock_shared(&pool_lock);
    const PoolRec *provider = capture->provider;
    uint32_t used = 0;
    int current = provider->kind == POOL_PHYS && provider->physical_origin_upload &&
        !provider->physical_origin_unknown && span_capture_match(capture, selected);
    for (uint32_t i = 0; current && i < count; i++) {
        const RcInterval span = spans[i];
        if (!span.mapping || !rc_range_valid(span.range) || span.address < capture->address ||
            span.address - capture->address >= capture->size ||
            span.range.size > capture->size - (span.address - capture->address)) {
            current = 0;
            break;
        }
        const uint64_t within = span.address - capture->address;
        if (within > UINT64_MAX - selected->offset ||
            span.range.size > UINT64_MAX - selected->offset - within) {
            current = 0;
            break;
        }
        const uint64_t start = selected->offset + within;
        const uint64_t end = start + span.range.size;
        for (uint64_t begin = start; begin < end;) {
            if (used == capacity) { current = 0; break; }
            const uint64_t edge = fixed_buffer_part_end(provider, begin, end);
            const uint64_t delta = begin - start;
            parts[used++] = (RcInterval){span.address + delta, span.mapping,
                {span.range.backing, span.range.offset + delta, edge - begin}};
            begin = edge;
        }
    }
    plat_mutex_unlock_shared(&pool_lock);
    return current && used > count ? used : 0;
}

int vk_mem_span_buffer_preserve_current(void *opaque, RcRegistryBufferSpan result, RcRegistryPosition read,
                                        int *initialized) {
    VkMemSpanCapture *capture = opaque;
    *initialized = 0;
    if (!capture || !capture->native_input || !result.interval.mapping || !rc_range_valid(result.interval.range) ||
        rc_registry_position_compare(read, vk_exec_resource_position())) return 0;
    const RcInterval interval = result.interval;
    if (interval.address < capture->address || interval.address - capture->address >= capture->size ||
        interval.range.size > capture->size - (interval.address - capture->address)) return 0;
    const uint64_t delta = interval.address - capture->address;
    const VkMemSpan *selected = &capture->selected;
    const PoolRec *provider = capture->provider;
    uint64_t submitted, completed;
    vk_upload_progress(&submitted, &completed);
    plat_mutex_lock_shared(&pool_lock);
    const int fixed = provider->kind == POOL_PHYS && nvn_gpu_only(interval.address);
    const int coherent_import = provider->kind == POOL_IMPORT && !provider->smap && provider->gpa_host_coherent &&
        !nvn_gpu_only(interval.address);
    const int mirror = coherent_import && capture->mirror_incarnation && provider->acc_sub == msub;
    // Only the selected writable import transition proves that earlier mirror and lazy writes are
    // already reconciled to this coherent import. An ordinary host lookup is not that proof.
    const int imported = coherent_import && !capture->mirror_incarnation && capture->host &&
        capture->cold_writer && capture->input_recorded && capture->writer_issued_scene &&
        capture->writer_issued_scene == read.scene && selected->buffer == provider->buf;
    int current = (fixed || mirror || imported) && span_capture_match(capture, selected) &&
        provider->publication == result.native.allocation && capture->mirror_incarnation == result.native.incarnation &&
        delta <= UINT64_MAX - selected->offset && selected->offset + delta == result.native.offset;
    if (current) {
        uint32_t count = capture->device_count;
        const RcInterval *mapping = capture->device_intervals;
        if (mirror || imported)
            mapping = mem_backing_intervals(capture->host ? capture->host : provider->host_backing, &count);
        current = 0;
        for (uint32_t i = 0; i < count; i++) {
            const RcInterval part = mapping[i];
            if (part.mapping != interval.mapping || part.range.backing != interval.range.backing ||
                interval.address < part.address || interval.address - part.address >= part.range.size) continue;
            const uint64_t within = interval.address - part.address;
            if (interval.range.size <= part.range.size - within && interval.range.offset == part.range.offset + within) {
                current = 1;
                break;
            }
        }
    }
    if (current && fixed) {
        const uint64_t begin = result.native.offset, end = begin + interval.range.size;
        // Binding an optimal image can invalidate raw bytes without a positioned buffer writer. Neither
        // an initial fill nor an earlier result can certify that placement across unknown alias history.
        current = !provider->physical_origin_unknown && !physical_origin_pending_overlap(provider, &result, 1);
        for (uint32_t i = 0; current && i < provider->physical_origin_poison_count; i++) {
            const FlushRange range = provider->physical_origin_poison[i];
            if (range.lo < end && begin < range.hi) current = 0;
        }
        *initialized = current && provider->physical_origin_upload && provider->physical_origin_upload <= submitted &&
            provider->physical_origin_upload <= completed &&
            physical_initial_fill_covers(provider, result.native.offset, interval.range.size);
    }
    if (current && mirror)
        *initialized = mirror_origin_initialized(provider, read.scene, vk_sync_fence_done(),
                                                result.native.offset, interval.range.size);
    if (current && imported) *initialized = 1;
    plat_mutex_unlock_shared(&pool_lock);
    return current;
}

// Neither a new address winner nor a recreated physical buffer may stand in for this receipt.
static int image_alias_baseline_current_locked(const VkMemImageAliasBaseline *owner) {
    const pidx_ent *entry = pidx_find(&pool_index, owner->address, owner->bytes);
    VkMemSpan now = {0};
    PoolRec *provider = NULL;
    if (!entry || entry->buf != owner->window || owner->window->unpublished ||
        !pool_device_span(owner->window, owner->address, owner->bytes, &now, &provider) ||
        provider != owner->provider || provider->unpublished || provider->physical_origin_unknown ||
        now.buffer != owner->native.buffer || now.alloc != owner->native.alloc ||
        now.offset != owner->native.offset || owner->spans[0].native.allocation != provider->publication ||
        (owner->mode != IMAGE_ALIAS_BASELINE_OPEN && owner->fill_upload != provider->physical_origin_upload))
        return 0;
    if (physical_origin_pending_overlap(provider, owner->spans, owner->count)) return 0;
    for (uint32_t i = 0; i < owner->count; ++i) {
        const RcInterval *part = &owner->spans[i].interval;
        if (!rc_backing_map_intervals_current(pool_backings, owner->window->map, part, 1, part->address, part->range.size))
            return 0;
        const uint64_t begin = owner->spans[i].native.offset;
        const uint64_t end = begin + part->range.size;
        if (end > provider->physical_origin_bytes) return 0;
        for (uint32_t j = 0; j < provider->physical_origin_poison_count; ++j) {
            const FlushRange poison = provider->physical_origin_poison[j];
            if (poison.lo < end && begin < poison.hi) return 0;
        }
    }
    return 1;
}

static void image_alias_baseline_drop_pools(PoolRec *window, PoolRec *provider) {
    VkCtx *c = vk_ctx();
    if (window != provider) destroy_rec(c, window);
    destroy_rec(c, provider);
}

static void image_alias_baseline_destroy(VkMemImageAliasBaseline *owner) {
    RcBudgetTicket budget = owner->budget;
    image_alias_baseline_drop_pools(owner->window, owner->provider);
    if (resource_core_budget_retire(&budget) != RC_ADMISSION_OK)
        rt_fatal("vk: native image baseline lost its capacity charge");
    free(owner);
    if (resource_core_budget_destroyed(&budget) != RC_ADMISSION_OK)
        rt_fatal("vk: native image baseline lost its retired capacity charge");
}

static RcRegistryResult image_alias_baseline_capture(uint64_t address, uint64_t bytes, const RcInterval *mapping,
    uint32_t count, RcRegistryPosition read, ImageAliasBaselineMode mode, VkMemImageAliasBaseline **out) {
    if (!out || *out || !address || !bytes || bytes > UINT64_MAX - address || !mapping || !count ||
        !rc_registry_position_valid(read) || (mode != IMAGE_ALIAS_BASELINE_HANDED && (read.ordinal & 1)))
        return RC_REGISTRY_INVALID;
    if (!image_alias_baseline_read_current(read, mode) || vk_device_is_lost()) return RC_REGISTRY_STALE;

    // This is a read-only watermark query. It never submits or waits for an initialization.
    uint64_t submitted, completed, fill_upload = 0;
    vk_upload_progress(&submitted, &completed);
    RcRegistryBufferSpan spans[IMAGE_ALIAS_BASELINE_LIMIT];
    uint32_t used = 0;
    RcRegistryResult result = RC_REGISTRY_OK;
    VkMemSpan native = {0};
    PoolRec *window = NULL, *provider = NULL;
    const char *reason = "no-physical-run";
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *entry = pidx_find(&pool_index, address, bytes);
    window = entry ? entry->buf : NULL;
    if (!window || window->unpublished) {
        result = RC_REGISTRY_STALE;
        reason = "no-current-window";
    } else if (window->kind == POOL_PHYS || window->kind == POOL_VIRT) {
        if (!rc_backing_map_intervals_current(pool_backings, window->map, mapping, count, address, bytes)) {
            result = RC_REGISTRY_STALE;
            reason = "mapping-changed";
        } else if (pool_device_span(window, address, bytes, &native, &provider) && provider &&
                   provider->kind == POOL_PHYS && !provider->unpublished && native.buffer && native.alloc) {
            if (!provider->physical_origin_upload || !provider->physical_origin_bytes)
                reason = "no-recorded-fill";
            else if (provider->physical_origin_upload > submitted ||
                (mode == IMAGE_ALIAS_BASELINE_OPEN && provider->physical_origin_upload > completed)) reason = "fill-incomplete";
            else if (provider->physical_origin_unknown)
                reason = "raw-layout-history-lost";
            else {
                result = image_alias_baseline_partition(provider, &native, address, bytes, mapping, count, spans, &used);
                if (result == RC_REGISTRY_OK && physical_origin_pending_overlap(provider, spans, used)) {
                    used = 0;
                    reason = "image-bind-pending";
                } else if (result != RC_REGISTRY_OK) reason = "coverage-refused";
                else if (used) {
                    fill_upload = provider->physical_origin_upload;
                    reason = fill_upload <= completed ? "initialized" : "fill-submitted";
                } else reason = "no-raw-slice";
            }
        }
    }
    image_alias_baseline_trace(read, address, bytes, mapping, provider, &native, submitted, completed, used, result, reason);
    if (result == RC_REGISTRY_OK && used) {
        pool_native_acquire(window);
        if (window != provider) pool_native_acquire(provider);
    }
    plat_mutex_unlock_shared(&pool_lock);
    if (result != RC_REGISTRY_OK || !used) return result;

    const size_t host_bytes = sizeof(VkMemImageAliasBaseline) + (size_t)used * sizeof *spans;
    const RcAdmissionVector cost = {.amount = {[RC_BUDGET_RAM] = host_bytes, [RC_BUDGET_COMMIT] = host_bytes}};
    RcBudgetTicket budget = {0};
    VkMemImageAliasBaseline *owner = NULL;
    if (resource_core_budget_reserve(RC_BUDGET_SNAPSHOT, 1, &cost, &budget) == RC_ADMISSION_OK)
        owner = malloc(host_bytes);
    if (!owner || !rc_admission_accounted(resource_core_budget_materialize(&budget, &cost))) {
        if (budget.token.ledger && resource_core_budget_cancel(&budget) != RC_ADMISSION_OK)
            rt_fatal("vk: native image baseline lost its pending capacity charge");
        free(owner);
        image_alias_baseline_drop_pools(window, provider);
        return RC_REGISTRY_CAPACITY;
    }
    *owner = (VkMemImageAliasBaseline){.window = window, .provider = provider, .native = native, .budget = budget,
        .read = read, .address = address, .bytes = bytes, .fill_upload = fill_upload, .count = used, .mode = mode};
    memcpy(owner->spans, spans, (size_t)used * sizeof *spans);
    const int read_current = image_alias_baseline_read_current(read, mode);
    plat_mutex_lock(&pool_lock);
    int current = read_current && image_alias_baseline_current_locked(owner);
    if (current) {
        owner->next = provider->physical_origin_borrows;
        provider->physical_origin_borrows = owner;
    }
    plat_mutex_unlock(&pool_lock);
    if (!current) {
        image_alias_baseline_destroy(owner);
        return RC_REGISTRY_STALE;
    }
    *out = owner;
    return RC_REGISTRY_OK;
}

RcRegistryResult vk_mem_image_alias_baseline_capture(uint64_t address, uint64_t bytes, const RcInterval *mapping,
    uint32_t count, RcRegistryPosition read, VkMemImageAliasBaseline **out) {
    return image_alias_baseline_capture(address, bytes, mapping, count, read, IMAGE_ALIAS_BASELINE_OPEN, out);
}

RcRegistryResult vk_mem_image_alias_baseline_capture_handed(uint64_t address, uint64_t bytes, const RcInterval *mapping,
    uint32_t count, RcRegistryPosition cut, VkMemImageAliasBaseline **out, uint64_t *fill_upload) {
    if (!out || !fill_upload) return RC_REGISTRY_INVALID;
    const RcRegistryResult result = image_alias_baseline_capture(address, bytes, mapping, count, cut,
                                                               IMAGE_ALIAS_BASELINE_HANDED, out);
    *fill_upload = result == RC_REGISTRY_OK && *out ? (*out)->fill_upload : 0;
    return result;
}

static int image_alias_baseline_fill_completed(const VkMemImageAliasBaseline *owner) {
    uint64_t completed = 0;
    vk_upload_progress(NULL, &completed);
    return owner->fill_upload <= completed;
}

RcRegistryResult vk_mem_image_alias_baseline_capture_submitted(uint64_t address, uint64_t bytes,
    const RcInterval *mapping, uint32_t count, RcRegistryPosition read, VkMemImageAliasBaseline **out,
    uint64_t *fill_upload) {
    if (!out || !fill_upload) return RC_REGISTRY_INVALID;
    *fill_upload = 0;
    const RcRegistryResult result = image_alias_baseline_capture(address, bytes, mapping, count, read,
                                                               IMAGE_ALIAS_BASELINE_SUBMITTED, out);
    if (result == RC_REGISTRY_OK && *out && !image_alias_baseline_fill_completed(*out))
        *fill_upload = (*out)->fill_upload;
    return result;
}

static int image_alias_baseline_lease_current(const VkMemImageAliasBaseline *owner) {
    if (!owner || !image_alias_baseline_read_current(owner->read, owner->mode) || vk_device_is_lost()) return 0;
    plat_mutex_lock_shared(&pool_lock);
    const int current = image_alias_baseline_current_locked(owner);
    plat_mutex_unlock_shared(&pool_lock);
    return current;
}

const RcRegistryBufferSpan *vk_mem_image_alias_baseline_submitted_spans(const VkMemImageAliasBaseline *owner,
    uint64_t fill_upload, uint32_t *count) {
    if (!count) return NULL;
    *count = 0;
    if (!owner || owner->mode != IMAGE_ALIAS_BASELINE_SUBMITTED || !fill_upload || fill_upload != owner->fill_upload)
        return NULL;
    uint64_t submitted = 0;
    vk_upload_progress(&submitted, NULL);
    if (fill_upload > submitted || !image_alias_baseline_lease_current(owner)) return NULL;
    *count = owner->count;
    return owner->spans;
}

const RcRegistryBufferSpan *vk_mem_image_alias_baseline_spans(const VkMemImageAliasBaseline *owner, uint32_t *count) {
    if (!count) return NULL;
    *count = 0;
    if (!owner || (owner->mode != IMAGE_ALIAS_BASELINE_OPEN && !image_alias_baseline_fill_completed(owner))) return NULL;
    *count = owner->count;
    return owner->spans;
}

int vk_mem_image_alias_baseline_current(const VkMemImageAliasBaseline *owner) {
    return owner && (owner->mode != IMAGE_ALIAS_BASELINE_SUBMITTED || image_alias_baseline_fill_completed(owner)) &&
        image_alias_baseline_lease_current(owner);
}

int vk_mem_image_alias_baseline_source(const VkMemImageAliasBaseline *owner, uint32_t baseline,
    RcRegistryBufferLocation native, uint64_t size, VkBuffer *buffer, VkDeviceSize *offset) {
    if (!buffer || !offset) return 0;
    *buffer = VK_NULL_HANDLE;
    *offset = 0;
    uint32_t count;
    const RcRegistryBufferSpan *spans = vk_mem_image_alias_baseline_spans(owner, &count);
    if (baseline >= count || !size) return 0;
    const RcRegistryBufferSpan *span = &spans[baseline];
    if (native.allocation != span->native.allocation || native.incarnation != span->native.incarnation ||
        native.offset < span->native.offset || size > span->interval.range.size ||
        native.offset - span->native.offset > span->interval.range.size - size ||
        !vk_mem_image_alias_baseline_current(owner)) return 0;
    *buffer = owner->native.buffer;
    *offset = native.offset;
    return 1;
}

void vk_mem_image_alias_baseline_release(VkMemImageAliasBaseline **owned) {
    if (!owned || !*owned) return;
    VkMemImageAliasBaseline *owner = *owned;
    *owned = NULL;
    plat_mutex_lock(&pool_lock);
    VkMemImageAliasBaseline **link = &owner->provider->physical_origin_borrows;
    while (*link && *link != owner) link = &(*link)->next;
    if (!*link) rt_fatal("vk: native image baseline lost its raw lease");
    *link = owner->next;
    plat_mutex_unlock(&pool_lock);
    image_alias_baseline_destroy(owner);
}

VkMemImageAliasPlacement vk_mem_bind_image_alias(VkImage image, uint64_t address, const VkMemoryRequirements *requirements,
    const VkMemSpan *selected, const RcInterval *mapping, uint32_t mapping_count, VkResult *native_failure) {
    if (native_failure) *native_failure = VK_SUCCESS;
    if (!image || !image_alias_placement_inputs_valid(address, requirements, selected, mapping, mapping_count))
        return VK_MEM_IMAGE_ALIAS_FAILED;
    const uint64_t size = requirements->size;
    PoolRec *provider;
    FlushRange range;
    plat_mutex_lock_shared(&pool_lock);
    VkMemImageAliasPlacement result = image_alias_preplacement_locked(address, size, selected, mapping, mapping_count,
                                                                       &provider, &range);
    if (result == VK_MEM_IMAGE_ALIAS_DIRECT) pool_native_acquire(provider);
    plat_mutex_unlock_shared(&pool_lock);
    if (result != VK_MEM_IMAGE_ALIAS_DIRECT) return result;

    VkCtx *c = vk_ctx();
    VmaAllocationInfo allocation;
    vmaGetAllocationInfo(c->vma, selected->alloc, &allocation);
    if (allocation.offset != provider->physical_memory_offset) {
        destroy_rec(c, provider);
        return VK_MEM_IMAGE_ALIAS_STALE;
    }
    VkDeviceSize start = selected->alloc_offset;
    if (!allocation.deviceMemory || start > allocation.size || size > allocation.size - start ||
        start > UINT64_MAX - allocation.offset || size > UINT64_MAX - (allocation.offset + start) ||
        (allocation.offset + start) % requirements->alignment || allocation.memoryType >= VK_MAX_MEMORY_TYPES ||
        !(requirements->memoryTypeBits & (UINT32_C(1) << allocation.memoryType))) {
        destroy_rec(c, provider);
        return VK_MEM_IMAGE_ALIAS_FAILED;
    }
    start += allocation.offset;

    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkMemoryImageHold *previous, *next;
    if (!image_placement_available(allocation.deviceMemory, start, start + size, &previous, &next)) {
        // No bind or transition occurred. The retained owner still supplies its earlier bytes.
        plat_rmutex_unlock(&cmd_lock);
        destroy_rec(c, provider);
        return VK_MEM_IMAGE_ALIAS_OCCUPIED;
    }
    VkMemoryImageHold *hold = image_hold_find_or_add(image);
    if (hold->pool || hold->readers) rt_fatal("vk: image pool alias already published");
    hold->pool = provider;
    hold->pool_memory = allocation.deviceMemory;
    hold->pool_begin = start;
    hold->pool_end = start + size;
    hold->binding_pending = 1;
    image_placement_insert(hold, previous, next);
    plat_rmutex_unlock(&cmd_lock);

    // Recheck the selected topology and raw bytes before entering the driver. A successful reservation pins
    // the original provider and defers a raw borrow over these bytes until the native outcome is known.
    PhysicalOriginBinding raw = {0};
    int native_bound = 0;
    result = physical_origin_begin_image(address, size, selected, mapping, mapping_count, provider, &raw);
    if (result == VK_MEM_IMAGE_ALIAS_DIRECT) {
        const VkResult bound = vmaBindImageMemory2(c->vma, selected->alloc, selected->alloc_offset, image, NULL);
        native_bound = bound == VK_SUCCESS;
        physical_origin_end_image(provider, &raw, native_bound);
        if (native_failure) *native_failure = bound;
        plat_mutex_lock_shared(&pool_lock);
        const int current =
            image_alias_placement_provider_locked(address, size, selected, mapping, mapping_count) == provider;
        plat_mutex_unlock_shared(&pool_lock);
        result = !current ? VK_MEM_IMAGE_ALIAS_STALE : native_bound ? VK_MEM_IMAGE_ALIAS_DIRECT : VK_MEM_IMAGE_ALIAS_FAILED;
    }
    plat_rmutex_lock(&cmd_lock);
    // A successful driver bind owns the old pool through destruction even when logical publication is stale.
    if (native_bound) hold->binding_pending = 0;
    else {
        image_placement_remove(hold);
        if (nvn_hm_del(&image_holds, &image) != hold) rt_fatal("vk: invalid provisional image placement");
        free(hold);
    }
    plat_rmutex_unlock(&cmd_lock);
    if (!native_bound) destroy_rec(c, provider);
    return result;
}

int vk_mem_image_alias_overlaps(uint64_t address, uint64_t size, const VkMemSpan *span, VkImage *images, uint32_t capacity) {
    if (!span->alloc || !size || size > span->avail) return 0;
    // The non-mutating span lookup does not own a pool reference. Acquire its still-current provider
    // before querying VMA, and release it outside the native image placement lock.
    PoolRec *provider = pool_image_alias_acquire(address, size, span);
    if (!provider) return -1;
    VkCtx *ctx = vk_ctx();
    VmaAllocationInfo allocation;
    vmaGetAllocationInfo(ctx->vma, span->alloc, &allocation);
    VkDeviceSize start = span->alloc_offset;
    if (!allocation.deviceMemory || start > allocation.size || size > allocation.size - start ||
        start > UINT64_MAX - allocation.offset || size > UINT64_MAX - (allocation.offset + start)) {
        destroy_rec(ctx, provider);
        return -1;
    }
    start += allocation.offset;
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    uint32_t count = 0;
    int exact = 1;
    const VkMemoryImageHold *hold = nvn_hm_get(&image_placements, &allocation.deviceMemory);
    for (; hold && hold->pool_begin < start + size; hold = hold->placement_next) {
        if (hold->pool_end <= start || hold->detached) continue;
        if (hold->binding_pending || count == capacity) { exact = 0; break; }
        images[count++] = hold->image;
    }
    plat_rmutex_unlock(&cmd_lock);
    destroy_rec(ctx, provider);
    return exact ? (int)count : -1;
}

int vk_mem_image_alias_current(VkImage image, uint64_t addr, uint64_t size, VmaAllocation allocation,
                               const RcInterval *intervals, uint32_t count) {
    if (!image || !size || !allocation || !intervals || !count) return 0;
    // Never nest pool_lock and cmd_lock. The pool reference from the existing image keeps that
    // allocation alive, and this query holds a transient ref on the currently mapped candidate.
    VkMemSpan span = {0};
    PoolRec *provider = NULL;
    plat_mutex_lock_shared(&pool_lock);
    const pidx_ent *entry = pidx_find(&pool_index, addr, size);
    PoolRec *window = entry ? entry->buf : NULL;
    if (window && (window->kind == POOL_PHYS || window->kind == POOL_VIRT) &&
        pool_device_span(window, addr, size, &span, &provider) && span.alloc == allocation &&
        rc_backing_map_intervals_current(pool_backings, window->map, intervals, count, addr, size))
        pool_native_acquire(provider);
    else
        provider = NULL;
    plat_mutex_unlock_shared(&pool_lock);
    if (!provider) return 0;

    plat_rmutex_lock(&cmd_lock);
    const VkMemoryImageHold *hold = nvn_hm_get(&image_holds, &image);
    int current = hold && !hold->detached && !hold->binding_pending && hold->pool == provider;
    plat_rmutex_unlock(&cmd_lock);
    destroy_rec(vk_ctx(), provider);
    return current;
}

VkMemoryImageHold *vk_mem_image_hold(VkImage image) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkMemoryImageHold *hold = image_hold_find_or_add(image);
    if (hold->binding_pending || hold->readers == UINT32_MAX) rt_fatal("vk: invalid native image reader acquire");
    if (!hold->readers) __atomic_fetch_add(&image_readers, 1, __ATOMIC_RELEASE);
    hold->readers++;
    plat_rmutex_unlock(&cmd_lock);
    return hold;
}

VkMemoryImageHold *vk_mem_image_hold_retain(VkMemoryImageHold *hold) {
    plat_rmutex_lock(&cmd_lock);
    if (!hold->readers || hold->readers == UINT32_MAX) rt_fatal("vk: invalid native image reader retain");
    hold->readers++;
    plat_rmutex_unlock(&cmd_lock);
    return hold;
}

VkMemoryImageHold *vk_mem_image_hold_retain_exact(VkMemoryImageHold *hold, VkImage image) {
    if (!hold || !image) return NULL;
    plat_rmutex_lock(&cmd_lock);
    const int exact = hold->image == image && hold->readers && !hold->binding_pending;
    if (exact) {
        if (hold->readers == UINT32_MAX) rt_fatal("vk: native image reader retain exhausted");
        hold->readers++;
    }
    plat_rmutex_unlock(&cmd_lock);
    return exact ? hold : NULL;
}

int vk_mem_image_hold_placement(const VkMemoryImageHold *hold, VkImage image, VkImageMemoryPlacement *placement) {
    if (!hold || !image || !placement) return 0;
    plat_rmutex_lock(&cmd_lock);
    const int current = hold->image == image && hold->readers && !hold->detached && !hold->binding_pending &&
        hold->pool_memory && hold->pool_begin < hold->pool_end;
    if (current)
        *placement = (VkImageMemoryPlacement){hold->pool_memory, hold->pool_begin, hold->pool_end - hold->pool_begin};
    plat_rmutex_unlock(&cmd_lock);
    return current;
}

int vk_mem_image_hold_retired(const VkMemoryImageHold *hold) {
    plat_rmutex_lock(&cmd_lock);
    int retired = hold->detached;
    plat_rmutex_unlock(&cmd_lock);
    return retired;
}

int vk_mem_image_held(VkImage image) {
    if (!__atomic_load_n(&image_readers, __ATOMIC_ACQUIRE)) return 0;
    plat_rmutex_lock(&cmd_lock);
    const VkMemoryImageHold *hold = nvn_hm_get(&image_holds, &image);
    int held = hold && hold->readers;
    plat_rmutex_unlock(&cmd_lock);
    return held;
}

void vk_mem_image_release(VkMemoryImageHold *hold) {
    if (!hold) return;
    plat_rmutex_lock(&cmd_lock);
    if (!hold->readers) rt_fatal("vk: native image reader already released");
    int reclaim = 0;
    if (!--hold->readers) {
        __atomic_fetch_sub(&image_readers, 1, __ATOMIC_RELEASE);
        reclaim = hold->detached;
        if (!hold->detached && !hold->pool && !hold->views) {
            nvn_hm_del(&image_holds, &hold->image);
            free(hold);
        }
    }
    plat_rmutex_unlock(&cmd_lock);
    if (reclaim) resource_core_budget_progress();
}

static void destroy_dead(VkCtx *c, const Dead *d) {
    if (d->image_hold && d->image_hold->readers)
        rt_fatal("vk: deferred image still has a present reader");
    if (d->image_hold) while (d->image_hold->views) {
        OwnedImageView *view = d->image_hold->views;
        d->image_hold->views = view->next;
        vkDestroyImageView(c->device, view->view, NULL);
        free(view);
    }
    switch (d->kind) {
    case DEAD_IMAGE: vkDestroyImage(c->device, (VkImage)(uintptr_t)d->handle, NULL); break;
    case DEAD_IMAGE_VMA: resource_core_vma_destroy_image(c->vma, (VkImage)(uintptr_t)d->handle, d->alloc); break;
    case DEAD_VIEW: vkDestroyImageView(c->device, (VkImageView)(uintptr_t)d->handle, NULL); break;
    case DEAD_BUFFER_VMA: resource_core_vma_destroy_buffer(c->vma, (VkBuffer)(uintptr_t)d->handle, d->alloc); break;
    }
    if (d->image_hold) {
        if (d->image_hold->pool) {
            // Leave the placement excluded until the actual Vulkan image has been destroyed.
            plat_rmutex_lock(&cmd_lock);
            image_placement_remove(d->image_hold);
            plat_rmutex_unlock(&cmd_lock);
            destroy_rec(c, d->image_hold->pool);
        }
        free(d->image_hold);
    }
    if (d->reclaim_bytes) __atomic_fetch_sub(&reclaim_pending_bytes, d->reclaim_bytes, __ATOMIC_RELEASE);
    if (d->destroyed) d->destroyed(d->context);
}

// The objects the GPU is done with: destroyed here (out != NULL: moved to *out, for the caller to destroy without
// cmd_lock; *nout of them).
// ---- The finalize of a used pool (pool_fini)
// The finalize returned only once the queue was idle and every upload batch was through (vkQueueWaitIdle), since the
// guest gets the storage back when it returns. During an area load the queue also holds the load's texture uploads,
// which name no pool's storage: entering Kokiri Forest, one finalize waited 418 ms for them, the frame took 462 ms.
// The game finalizes a pool once its own work with it is done, as on the console, so the finalize now waits for the
// submissions issued so far (the game's work, the mirror copies and write-backs in it) and returns. The pool's buffer
// and memory, which images cut from it (vk_texture_range_gone) and the upload batches open or in flight may still
// name, go later: once those batches and the submission being recorded are through (pool_wait_collect, from the
// upload collector). NX2NAVI_POOL_FINI_IDLE=1 waits for the idle queue as before.
typedef struct { PoolRec *r; uint64_t seq, tl; } PoolWait;
static PoolWait *pwait;   // retire_lock
static int npwait, cpwait;
static uint64_t pf_n, pf_ns, pf_max_ns, pf_logged;   // narrowed finalizes, their waits (any thread; statistics only)

static int pool_fini_idle(void) {
    static int on = -1;
    if (on < 0) {
        on = rt_knob("POOL_FINI_IDLE", "0") == 1;
        rt_log("vk: a pool's finalize waits for %s (NX2NAVI_POOL_FINI_IDLE=%d)",
               on ? "the idle queue, the upload batches included" : "the submissions issued so far", on);
    }
    return on;
}

static void pool_fini_wait(void) {
    uint64_t t0 = kern_now_ns(), v = vk_sync_issued();
    vk_sync_fence_wait(v, UINT64_MAX);
    uint64_t d = kern_now_ns() - t0;
    uint64_t n = __atomic_add_fetch(&pf_n, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&pf_ns, d, __ATOMIC_RELAXED);
    if (d > __atomic_load_n(&pf_max_ns, __ATOMIC_RELAXED)) __atomic_store_n(&pf_max_ns, d, __ATOMIC_RELAXED);
    if (d > 20000000ull && __atomic_add_fetch(&pf_logged, 1, __ATOMIC_RELAXED) <= 20)
        rt_log("vk: a pool's finalize waited %.1f ms for the submissions up to %llu (finalizes so far %llu, %.1f ms in all)",
               d / 1e6, (unsigned long long)v, (unsigned long long)n, __atomic_load_n(&pf_ns, __ATOMIC_RELAXED) / 1e6);
    else if (n % 256 == 0)
        rt_log("vk: pool finalizes that waited for the GPU: %llu, %.1f ms in all, the longest %.1f ms",
               (unsigned long long)n, __atomic_load_n(&pf_ns, __ATOMIC_RELAXED) / 1e6,
               __atomic_load_n(&pf_max_ns, __ATOMIC_RELAXED) / 1e6);
}

static void pool_wait_add(PoolRec *r) {
    // every upload batch handed out so far, and the submission another thread may be recording (vk_sync_submit sets
    // issued after it), as defer stamps a deferred object
    PoolWait w = {r, __atomic_load_n(&next_seq, __ATOMIC_ACQUIRE), vk_sync_issued() + 1};
    plat_mutex_lock(&retire_lock);
    if (npwait == cpwait) grow((void **)&pwait, &cpwait, sizeof *pwait);
    pwait[npwait++] = w;
    plat_mutex_unlock(&retire_lock);
}

// cmd_lock held (collect_locked_out): the waiting pools whose batches and submission are through go now.
static void pool_wait_collect(uint64_t tl) {
    PoolRec *go[64];
    int n = 0;
    plat_mutex_lock(&retire_lock);
    for (int i = 0; i < npwait && n < 64;)
        if (pwait[i].seq <= done_seq && pwait[i].tl <= tl) go[n++] = pwait[i].r, pwait[i] = pwait[--npwait];
        else i++;
    plat_mutex_unlock(&retire_lock);
    for (int i = 0; i < n; i++) pool_retire(go[i]);   // off the retiring list, then destroyed
}

static void collect_locked_out(VkCtx *c, Dead **out, int *nout, int *cout) {
    retire(c, 0);
    uint64_t tl = 0;
    if (c->timeline) vkGetSemaphoreCounterValue(c->device, c->timeline, &tl);
    if (npwait) pool_wait_collect(tl);
    int k = 0;
    for (int i = 0; i < ndead; i++) {
        Dead *d = &dead[i];
        // Upload callback ownership may also release a pool lease. Run it outside cmd_lock.
        if (d->seq > done_seq || d->tl > tl || (d->image_hold && d->image_hold->readers) ||
            (d->kind == DEAD_UPLOAD_CALLBACK && !out)) {
            dead[k++] = *d;
            continue;
        }
        if (d->kind == DEAD_HOST) {   // not a Vulkan object: nothing for the recorder's handle check
            free((void *)(uintptr_t)d->handle);
            continue;
        }
        if (d->kind != DEAD_UPLOAD_CALLBACK)
            rec_check_handle(d->kind == DEAD_VIEW ? REC_VIEW : d->kind == DEAD_BUFFER_VMA ? REC_BUFFER : REC_IMAGE,
                             d->handle, 1, d->tl, tl);
        if (!out) { destroy_dead(c, d); continue; }
        if (*nout == *cout) grow((void **)out, cout, sizeof **out);
        (*out)[(*nout)++] = *d;
    }
    ndead = k;
}

static void collect_locked(VkCtx *c) { collect_locked_out(c, NULL, NULL, NULL); }

// Waiting on uploads already handed to the queue does not require the scene currently being recorded to submit.
// Keep the active upload command buffer untouched; all its recorded commands still belong to its caller.
static void upload_wait_oldest_locked(VkCtx *c) {
    if (!ninflight) return;
    upload_pressure_waits++;
    VkResult result = vkWaitForFences(c->device, 1, &inflight[0].fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        vk_device_lost_check(result, "the staging-pressure wait for an upload batch");
        rt_fatal("vk: staging-pressure upload wait failed (%d)", (int)result);
    }
    collect_locked(c);
}

static void upload_checkpoint_locked(VkCtx *c) {
    if (upload_depth || !open_active || open_b.staging_bytes < upload_batch_window) return;
    RcBudgetSnapshot budget;
    resource_core_budget_snapshot(&budget);
    uint64_t target = budget.sample.total.amount[RC_BUDGET_STAGING];
    uint64_t batch = target / 4, outstanding = target;
    if (batch < UPLOAD_BATCH_MIN) batch = UPLOAD_BATCH_MIN;
    if (batch > UPLOAD_BATCH_MAX) batch = UPLOAD_BATCH_MAX;
    if (outstanding < UPLOAD_INFLIGHT_MIN) outstanding = UPLOAD_INFLIGHT_MIN;
    if (outstanding > UPLOAD_INFLIGHT_MAX) outstanding = UPLOAD_INFLIGHT_MAX;
    // These are scheduling windows derived from the shared staging target, not new allocation allowances.
    // Every actual allocation still passes the RAM/commit/heap admission checks.
    upload_batch_window = batch;
    upload_inflight_window = outstanding;
    if (open_b.staging_bytes < upload_batch_window) return;
    upload_checkpoints++;
    vk_upload_flush(0);
    while (inflight_staging_bytes > upload_inflight_window && ninflight)
        upload_wait_oldest_locked(c);
}

static void upload_drain_submitted(void) {
    // Caller-owned readback staging can request memory before vk_upload_begin, without an ambient upload lock.
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    while (c && ninflight) upload_wait_oldest_locked(c);
    plat_rmutex_unlock(&cmd_lock);
}

VkCommandBuffer vk_upload_begin(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    if (!c) rt_fatal("vk: upload before vk_init");
    upload_depth++;
    if (open_active) return open_b.cb;
    if (!cmd_pool) {
        VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                      .queueFamilyIndex = c->gfx_family};
        if (vkCreateCommandPool(c->device, &pi, NULL, &cmd_pool) != VK_SUCCESS) rt_fatal("vk: upload command pool");
    }
    retire(c, 0);
    memset(&open_b, 0, sizeof open_b);
    if (nspare) {
        Batch s = spare[--nspare];
        open_b.cb = s.cb, open_b.fence = s.fence, open_b.stg = s.stg, open_b.cap = s.cap;
        open_b.chunk = s.chunk, open_b.cap_chunk = s.cap_chunk;
        vkResetCommandBuffer(open_b.cb, 0);
        vkResetFences(c->device, 1, &open_b.fence);
    } else {
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkAllocateCommandBuffers(c->device, &ai, &open_b.cb) != VK_SUCCESS ||
            vkCreateFence(c->device, &fi, NULL, &open_b.fence) != VK_SUCCESS)
            rt_fatal("vk: upload command buffer");
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(open_b.cb, &bi);
    open_b.seq = ++next_seq;
    vk_crumb(open_b.cb, VK_CRUMB_UPLOAD, open_b.seq);
    open_active = 1;
    return open_b.cb;
}

void vk_upload_end(void) {
    if (!upload_depth) rt_fatal("vk: upload end without an active caller");
    upload_depth--;
    if (!upload_depth) upload_checkpoint_locked(vk_ctx());
    plat_rmutex_unlock(&cmd_lock);
}

// These are separate completion domains. A recorded upload is a dependency, not completed contents. Scene
// submissions flush preceding uploads into the same graphics queue before submitting their command buffers.
uint64_t vk_upload_dependency(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    uint64_t seq = next_seq;
    plat_rmutex_unlock(&cmd_lock);
    return seq;
}
uint64_t vk_upload_recording_sequence(VkCommandBuffer cb) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    uint64_t sequence = upload_depth && open_active && open_b.cb == cb ? open_b.seq : 0;
    plat_rmutex_unlock(&cmd_lock);
    return sequence;
}
void vk_upload_progress(uint64_t *submitted, uint64_t *completed) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    if (submitted) *submitted = submitted_seq;
    if (completed) *completed = done_seq;
    plat_rmutex_unlock(&cmd_lock);
}

// Staging memory comes from a pool of its own. Optional shared capacity preallocates STG_MIN_BLOCKS at boot; when
// that allowance is unavailable an empty pool grows through admitted buffer allocations instead. In the
// default pools VMA creates a block under the lock of its memory type (VmaBlockVector::Allocate), so a block made by
// the headroom thread (vk_headroom.c) or a burst of uploads held every staging allocation of the render thread: one
// took 47-50 ms, for a few bytes, in the first frame after a save load. Larger buffers than a block go to the default
// pool.
#define STG_BLOCK (64ull << 20)
#define STG_MIN_BLOCKS 2
static VmaPool stg_pool;
static int stg_pool_tried;
static uint32_t stg_pool_memory_type;

static VkResult staging_create(VkCtx *c, const VkBufferCreateInfo *buffer, const VmaAllocationCreateInfo *allocation,
                                VkBuffer *out, VmaAllocation *memory, VmaAllocationInfo *info, RcAdmissionStatus *admission) {
    RcVmaRequest request = {.producer = RC_BUDGET_STAGE, .required = 1,
        .pool_memory_type = stg_pool_memory_type, .pool_block_size = allocation->pool ? STG_BLOCK : 0};
    return resource_core_vma_create_buffer(&request, c->vma, buffer, allocation, out, memory, info, admission);
}

static void staging_refusal_log(VkDeviceSize size, VkResult result, RcAdmissionStatus admission, const char *phase) {
    RcBudgetSnapshot budget;
    resource_core_budget_snapshot(&budget);
    const RcBudgetProducerStats *stage = &budget.producer[RC_BUDGET_STAGE];
    const RcBudgetRefusal *why = &stage->last_refusal;
    const double M = 1048576.0;
    plat_rmutex_lock(&cmd_lock);
    uint64_t open = open_active ? open_b.staging_bytes : 0, submitted = inflight_staging_bytes;
    uint64_t cached = stg_cache_bytes + (uint64_t)narena_free * ARENA_CHUNK;
    uint64_t owned = stage->retained.amount[RC_BUDGET_STAGING] + stage->pending.amount[RC_BUDGET_STAGING];
    uint64_t outside = owned > open + submitted + cached ? owned - open - submitted - cached : 0;
    plat_rmutex_unlock(&cmd_lock);
    rt_log("vk: staging owners: open %.1f submitted %.1f reusable %.1f outside batches %.1f MiB "
           "(outside includes caller-owned readback/check spans and allocation padding)",
           open / M, submitted / M, cached / M, outside / M);
    rt_log("vk: staging %s: %llu bytes, result %d admission %d; spans reserved %.1f retained %.1f pending %.1f MiB, "
           "optional target %.1f MiB; last shared refusal #%llu status %d required %d capacity/unknown/stale masks %#x/%#x/%#x",
           phase, (unsigned long long)size, (int)result, (int)admission,
           budget.reserved.amount[RC_BUDGET_STAGING] / M, budget.retained.amount[RC_BUDGET_STAGING] / M,
           budget.pending.amount[RC_BUDGET_STAGING] / M, budget.sample.total.amount[RC_BUDGET_STAGING] / M,
           (unsigned long long)stage->refused, (int)why->status, why->required,
           why->capacity_mask, why->unknown_mask, why->stale_mask);
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        if (!why->request.amount[d]) continue;
        const char *name = d == RC_BUDGET_RAM ? "RAM" : d == RC_BUDGET_COMMIT ? "commit" :
            d == RC_BUDGET_STAGING ? "staging spans" : "heap";
        rt_log("vk: staging last shared refusal domain %u (%s): asked %.3f available then %.3f MiB; "
               "now free sample %.1f reserved %.1f retained %.1f pending %.1f unobserved %.1f MiB, refusals %llu",
               d, name, why->request.amount[d] / M, why->available.amount[d] / M,
               budget.sample.free.amount[d] / M, budget.reserved.amount[d] / M, budget.retained.amount[d] / M,
               budget.pending.amount[d] / M, budget.unobserved.amount[d] / M,
               (unsigned long long)stage->refused_domain[d]);
    }
}

// Select only capacity charged to a deficient domain. This counts victims, not available budget;
// actual destruction and the shared sample decide whether the pending request may retry.
static int staging_reclaim_matches(VmaAllocation allocation, RcAdmissionVector *remaining) {
    if (!remaining) return 1;
    RcAdmissionVector cost;
    if (resource_core_vma_allocation_cost(allocation, &cost) != RC_ADMISSION_OK) return 0;
    int matched = 0;
    for (uint32_t d = 0; d < RC_ADMISSION_MAX_DOMAINS; d++) {
        uint64_t bytes = cost.amount[d] < remaining->amount[d] ? cost.amount[d] : remaining->amount[d];
        remaining->amount[d] -= bytes;
        matched |= bytes != 0;
    }
    return matched;
}

// cmd_lock held. Only the completed, reusable lists are eligible.
static int staging_reclaim_take(RcAdmissionVector *remaining, Staging *victims) {
    int count = 0;
    for (int i = 0; i < stg_ncache;) {
        if (!staging_reclaim_matches(stg_cache[i].alloc, remaining)) {
            i++;
            continue;
        }
        stg_cache_bytes -= stg_cache[i].size;
        victims[count++] = stg_cache[i];
        stg_cache[i] = stg_cache[--stg_ncache];
    }
    for (int i = 0; i < narena_free;) {
        if (!staging_reclaim_matches(arena_free[i].alloc, remaining)) {
            i++;
            continue;
        }
        const StgChunk chunk = arena_free[i];
        victims[count++] = (Staging){chunk.buf, chunk.alloc, ARENA_CHUNK};
        arena_free[i] = arena_free[--narena_free];
    }
    return count;
}

static uint64_t staging_reclaim_destroy(VkCtx *c, const Staging *victims, int count) {
    uint64_t bytes = 0;
    for (int i = 0; i < count; i++) {
        bytes += victims[i].size;
        resource_core_vma_destroy_buffer(c->vma, victims[i].buf, victims[i].alloc);
    }
    return bytes;
}

static unsigned staging_budget_reclaim(void *unused, const RcAdmissionVector *deficit) {
    (void)unused;
    VkCtx *c = vk_ctx();
    if (!c || vk_sync_stopping() || vk_device_is_lost()) return RC_BUDGET_RECLAIM_NONE;
    if (!plat_rmutex_trylock(&cmd_lock)) return RC_BUDGET_RECLAIM_BUSY;
    if (upload_depth) {
        plat_rmutex_unlock(&cmd_lock);
        return RC_BUDGET_RECLAIM_BUSY;
    }
    RcAdmissionVector remaining = *deficit;
    Staging victims[STG_CACHE + ARENA_KEEP];
    const int count = staging_reclaim_take(&remaining, victims);
    plat_rmutex_unlock(&cmd_lock);
    staging_reclaim_destroy(c, victims, count);
    return count ? RC_BUDGET_RECLAIM_DESTROYED : RC_BUDGET_RECLAIM_NONE;
}

// A staging buffer the driver refused (vk_staging): the buffers kept for reuse go back to the driver before one more
// try, with the headroom's spare blocks (vk_headroom_release): the staging cache, the arena's free chunks and the
// write-back snapshot pool, up to some hundred MiB. On an integrated GPU the device-local heap is the machine's RAM and
// fills before the commit checks see anything; staging buffers of 11 KiB to 16 MiB refused there ended the game in
// Hyrule Field and on the first scene of play (#465, #567). The replay's transient buffers and descriptor pools do the
// same (vk_exec_draw.c scratch, alloc_set). Returns the bytes given back.
uint64_t vk_mem_give_back(void) {
    VkCtx *c = vk_ctx();
    Staging victims[STG_CACHE + ARENA_KEEP];
    plat_rmutex_lock(&cmd_lock);
    const int count = staging_reclaim_take(NULL, victims);
    plat_rmutex_unlock(&cmd_lock);
    uint64_t freed = staging_reclaim_destroy(c, victims, count);
    plat_mutex_lock(&snap_pool_lock);
    for (int i = 0; i < nsnap_pool; i++) resource_core_vma_destroy_buffer(c->vma, snap_pool[i].buf, snap_pool[i].alloc);
    freed += snap_pool_bytes;
    nsnap_pool = 0, snap_pool_bytes = 0;
    plat_mutex_unlock(&snap_pool_lock);
    return freed + vk_headroom_release();
}

// NX2NAVI_STAGING_FAIL_AT=n (test): the n-th staging buffer the driver makes is taken as refused for memory once, so
// the give-back runs (vk_mem_give_back).
static int staging_fail_now(void) {
    static int64_t at = -1, n;
    if (at < 0) at = cfg_int("NX2NAVI_STAGING_FAIL_AT");
    return at > 0 && plat_inc64(&n) == at;
}

static VkBufferCreateInfo staging_info(VkDeviceSize size) {
    return (VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size ? size : 16,
                                .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
}

static VmaAllocationCreateInfo staging_placement(void) {
    return (VmaAllocationCreateInfo){.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                             VMA_ALLOCATION_CREATE_MAPPED_BIT};
}

static int staging_prepare_take_locked(VkCtx *c, VkDeviceSize size, VkPreparedStaging *stage) {
    Staging cached;
    void *map;
    if (!staging_cache_take(c, size, &cached, &map)) return 0;
    *stage = (VkPreparedStaging){.size = cached.size, .buffer = cached.buf, .allocation = cached.alloc,
                                 .map = map, .from_cache = 1};
    return 1;
}

int vk_staging_prepare_reuse(VkDeviceSize size, VkPreparedStaging *stage) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    int reused = !upload_depth && staging_prepare_take_locked(vk_ctx(), size, stage);
    plat_rmutex_unlock(&cmd_lock);
    return reused;
}

VkResult vk_staging_prepare_plan(VkDeviceSize size, VkPreparedStaging *stage, RcAdmissionVector *peak) {
    VkCtx *c = vk_ctx();
    *stage = (VkPreparedStaging){.size = size};
    *peak = (RcAdmissionVector){{0}};
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    if (upload_depth) {
        plat_rmutex_unlock(&cmd_lock);
        return VK_NOT_READY;
    }
    if (staging_prepare_take_locked(c, size, stage)) {
        plat_rmutex_unlock(&cmd_lock);
        return VK_SUCCESS;
    }
    VmaAllocationCreateInfo placement = staging_placement();
    if (stg_pool && size <= STG_BLOCK / 2) {
        placement.pool = stg_pool;
        stage->pool_memory_type = stg_pool_memory_type;
    }
    plat_rmutex_unlock(&cmd_lock);
    VkBufferCreateInfo info = staging_info(size);
    RcVmaRequest request = {.producer = RC_BUDGET_STAGE, .pool_memory_type = stage->pool_memory_type,
                            .pool_block_size = placement.pool ? STG_BLOCK : 0};
    return resource_core_vma_buffer_peak(&request, &info, &placement, &stage->placement, peak);
}

VkResult vk_staging_prepare_allocate(VkPreparedStaging *stage, RcBudgetTicket *peak, int required) {
    if (stage->buffer) return VK_SUCCESS;
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo info = staging_info(stage->size);
    RcVmaRequest request = {.producer = RC_BUDGET_STAGE, .required = required, .pool_memory_type = stage->pool_memory_type,
                            .pool_block_size = stage->placement.pool ? STG_BLOCK : 0, .peak_ticket = peak};
    VmaAllocationInfo mapped;
    VkResult result = resource_core_vma_create_buffer(&request, c->vma, &info, &stage->placement,
        &stage->buffer, &stage->allocation, &mapped, NULL);
    if (result == VK_SUCCESS) {
        stage->map = mapped.pMappedData;
        if (!stage->map || staging_fail_now()) {
            vk_staging_prepare_cancel(stage);
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        vk_texstat_add(VK_TS_STG_NEW, 1, stage->size);
    }
    return result;
}

void vk_staging_prepare_cancel(VkPreparedStaging *stage) {
    if (stage->buffer) {
        VkCtx *c = vk_ctx();
        if (stage->from_cache) {
            Staging cached = {stage->buffer, stage->allocation, stage->size};
            Batch unused = {.stg = &cached, .nstg = 1};
            plat_rmutex_lock(&cmd_lock);
            free_staging(c, &unused);
            plat_rmutex_unlock(&cmd_lock);
        } else resource_core_vma_destroy_buffer(c->vma, stage->buffer, stage->allocation);
    }
    *stage = (VkPreparedStaging){0};
}

VkBuffer vk_staging_prepare_record(VkCommandBuffer cb, VkPreparedStaging *stage) {
    if (!stage->buffer || !upload_depth || !open_active || open_b.cb != cb)
        rt_fatal("vk: prepared staging must transfer to its actual upload batch");
    if (open_b.nstg == open_b.cap) grow((void **)&open_b.stg, &open_b.cap, sizeof *open_b.stg);
    VkBuffer buffer = stage->buffer;
    open_b.stg[open_b.nstg++] = (Staging){buffer, stage->allocation, stage->size};
    open_b.staging_bytes += stage->size;
    *stage = (VkPreparedStaging){0};
    return buffer;
}

int vk_staging_prepare_available(VkDeviceSize size) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    int ready = 0;
    for (int i = 0; i < stg_ncache && !ready; ++i) ready = staging_size_fits(stg_cache[i].size, size);
    plat_rmutex_unlock(&cmd_lock);
    return ready;
}

uint64_t vk_staging_prepare_supplier(VkDeviceSize size, uint64_t after) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    uint64_t supplier = 0;
    int cached = stg_ncache;
    VkDeviceSize cached_bytes = stg_cache_bytes;
    for (int i = 0; i < ninflight && !supplier; ++i) {
        const Batch *batch = &inflight[i];
        if (batch->seq > submitted_seq) break;
        // Completion retires in queue order. Count the earlier staging allocations that would
        // fill the same cache before choosing a supplier for this request.
        for (int j = 0; j < batch->nstg; ++j) {
            VkDeviceSize bytes = batch->stg[j].size;
            if (!staging_cache_room(cached, cached_bytes, bytes)) continue;
            if (batch->seq > after && staging_size_fits(bytes, size)) {
                supplier = batch->seq;
                break;
            }
            cached++;
            cached_bytes += bytes;
        }
    }
    plat_rmutex_unlock(&cmd_lock);
    return supplier;
}

int vk_staging_prepare_wait(uint64_t handed_upload) {
    VkCtx *c = vk_ctx();
    if (!handed_upload || !c) return 0;
    while (!vk_sync_stopping() && !vk_device_is_lost()) {
        // A fence belongs to its exact sequence only while cmd_lock prevents recycling. This batch was already
        // submitted to gfx_queue without a dependency on the current recorder or its guest final fence.
        plat_rmutex_lock(&cmd_lock);
        if (handed_upload > submitted_seq) {
            plat_rmutex_unlock(&cmd_lock);
            return 0;
        }
        if (handed_upload <= done_seq) {
            plat_rmutex_unlock(&cmd_lock);
            return 1;
        }
        VkFence fence = VK_NULL_HANDLE;
        for (int i = 0; i < ninflight; ++i)
            if (inflight[i].seq == handed_upload) { fence = inflight[i].fence; break; }
        if (!fence) {
            plat_rmutex_unlock(&cmd_lock);
            return 0;
        }
        VkResult status = vkWaitForFences(c->device, 1, &fence, VK_TRUE, UINT64_C(10000000));
        if (status == VK_SUCCESS) retire(c, 0);
        plat_rmutex_unlock(&cmd_lock);
        if (status == VK_SUCCESS) return 1;
        if (status == VK_ERROR_DEVICE_LOST) {
            vk_device_lost_check(status, "waiting for compatible upload staging");
            return 0;
        }
        if (status != VK_TIMEOUT) rt_fatal("vk: compatible upload staging wait failed (%d)", (int)status);
    }
    return 0;
}

void *vk_staging(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc) {
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = staging_info(size);
    VmaAllocationCreateInfo aci = staging_placement();
    if (!__atomic_load_n(&stg_pool_tried, __ATOMIC_ACQUIRE)) {
        plat_once(&cmd_once, cmd_init, NULL);
        plat_rmutex_lock(&cmd_lock);
        if (!stg_pool_tried) {
            uint32_t type;
            VmaPoolCreateInfo pi = {.blockSize = STG_BLOCK, .minBlockCount = STG_MIN_BLOCKS};
            if (vmaFindMemoryTypeIndexForBufferInfo(c->vma, &bi, &aci, &type) == VK_SUCCESS) {
                pi.memoryTypeIndex = type;
                stg_pool_memory_type = type;
                RcVmaRequest request = {.producer = RC_BUDGET_STAGE, .required = 0};
                VkResult made = resource_core_vma_create_pool(&request, c->vma, &pi, &stg_pool, NULL);
                if (made != VK_SUCCESS) {
                    pi.minBlockCount = 0;
                    if (resource_core_vma_create_pool(&request, c->vma, &pi, &stg_pool, NULL) != VK_SUCCESS) stg_pool = NULL;
                }
            }
            rt_log("vk: staging pool %s (%llu blocks of %llu MiB kept)", stg_pool ? "created" : "unavailable: default pool",
                   (unsigned long long)pi.minBlockCount, (unsigned long long)(STG_BLOCK >> 20));
            __atomic_store_n(&stg_pool_tried, 1, __ATOMIC_RELEASE);
        }
        plat_rmutex_unlock(&cmd_lock);
    }
    if (stg_pool && bi.size <= STG_BLOCK / 2) aci.pool = stg_pool;
    if (!alloc) {   // the batch owns it, so a cached one of a fitting size serves
        Staging cs;
        void *cmap;
        if (staging_cache_take(c, bi.size, &cs, &cmap)) {
            *buf = cs.buf;
            if (open_b.nstg == open_b.cap) grow((void **)&open_b.stg, &open_b.cap, sizeof *open_b.stg);
            open_b.stg[open_b.nstg++] = cs;
            open_b.staging_bytes += cs.size;
            return cmap;
        }
    }
    VmaAllocationInfo info;
    VkBuffer b;
    VmaAllocation a;
    RcAdmissionStatus admission;
    VkResult r = staging_create(c, &bi, &aci, &b, &a, &info, &admission);
    if ((r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_NOT_READY) && aci.pool) {
        // The staging pool can grow only in 64 MiB blocks of its one memory type. A smaller allocation may
        // still fit in a default block or another host-visible type. Keep the same mapping requirements.
        aci.pool = NULL;
        r = staging_create(c, &bi, &aci, &b, &a, &info, &admission);
    }
    if (r == VK_SUCCESS && staging_fail_now()) {
        resource_core_vma_destroy_buffer(c->vma, b, a);
        r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY || r == VK_NOT_READY) {
        VkResult first = r;
        static volatile int32_t told;
        int tell = plat_inc32(&told) <= 16;
        if (tell) staging_refusal_log(size, r, admission, "before give-back");
        // Completed uploads can release real capacity even while this operation is recording. The active batch
        // cannot be submitted here because its caller still owns the command buffer and unfilled staging bytes.
        upload_drain_submitted();
        uint64_t freed = vk_mem_give_back();
        resource_core_budget_refresh();
        aci.pool = NULL;
        r = staging_create(c, &bi, &aci, &b, &a, &info, &admission);
        if (tell)
            rt_log("vk: staging buffer of %llu bytes refused (%d): %.1f MiB of kept buffers and spare blocks given back, "
                   "%s", (unsigned long long)size, (int)first, freed / 1048576.0,
                   r == VK_SUCCESS ? "made at the second try" : "refused again");
    }
    if (r != VK_SUCCESS) {
        staging_refusal_log(size, r, admission, "after bounded give-back/retry");
        rt_fatal("vk: staging buffer of %llu bytes failed (%d, admission %d)", (unsigned long long)size, (int)r, (int)admission);
    }
    vk_texstat_add(VK_TS_STG_NEW, 1, bi.size);
    *buf = b;
    if (alloc) *alloc = a;   // the caller uses resource_core_vma_destroy_buffer after vk_upload_flush(1)
    else {
        if (open_b.nstg == open_b.cap) grow((void **)&open_b.stg, &open_b.cap, sizeof *open_b.stg);
        open_b.stg[open_b.nstg++] = (Staging){b, a, bi.size};
        open_b.staging_bytes += bi.size;
    }
    return info.pMappedData;
}

// A range of the staging arena for an upload of the open batch (between vk_upload_begin and vk_upload_end): *buf and
// *off name it, and the batch owns it until its fence signals. Larger requests, and every request with
// NX2NAVI_STG_ARENA=0, get a buffer of their own at offset 0 (vk_staging).
void *vk_staging_at(VkDeviceSize size, VkBuffer *buf, VkDeviceSize *off) {
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("NX2NAVI_STG_ARENA");
        if (!on) rt_log("vk: staging arena off (NX2NAVI_STG_ARENA=0): every upload stages in a buffer of its own");
    }
    *off = 0;
    if (!on || !open_active || size > ARENA_MAX) return vk_staging(size, buf, NULL);
    VkCtx *c = vk_ctx();
    VkDeviceSize at = (open_b.chunk_used + ARENA_ALIGN - 1) & ~(ARENA_ALIGN - 1);
    if (!open_b.nchunk || at + size > ARENA_CHUNK) {
        StgChunk k;
        if (narena_free) {
            k = arena_free[--narena_free];
        } else {
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = ARENA_CHUNK,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo aci = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                           .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                                    VMA_ALLOCATION_CREATE_MAPPED_BIT};
            if (stg_pool) aci.pool = stg_pool;   // half of a staging block (STG_BLOCK)
            VmaAllocationInfo info;
            if (staging_create(c, &bi, &aci, &k.buf, &k.alloc, &info, NULL) != VK_SUCCESS)
                return vk_staging(size, buf, NULL);
            k.map = info.pMappedData;
            vk_texstat_add(VK_TS_STG_NEW, 1, ARENA_CHUNK);
            if (arena_made++ < 16) rt_log("vk: staging arena chunk %llu made (%llu MiB)", (unsigned long long)arena_made,
                                          (unsigned long long)(ARENA_CHUNK >> 20));
        }
        if (open_b.nchunk == open_b.cap_chunk) grow((void **)&open_b.chunk, &open_b.cap_chunk, sizeof *open_b.chunk);
        open_b.chunk[open_b.nchunk++] = k;
        open_b.staging_bytes += ARENA_CHUNK;
        at = 0;
    }
    StgChunk *k = &open_b.chunk[open_b.nchunk - 1];
    open_b.chunk_used = at + (size ? size : 16);
    arena_ranges++;
    *buf = k->buf, *off = at;
    return k->map + at;
}

uint64_t vk_upload_flush(int wait) {
    if (vk_device_is_lost()) return 0;   // the run is ending (vk_device.c vk_device_lost_check): no submit, no wait
    uint64_t f0 = kern_now_ns();
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkCtx *c = vk_ctx();
    uint64_t seq = next_seq;
    if (c && open_active) {
        for (int i = 0; i < open_b.nstg; i++) vmaFlushAllocation(c->vma, open_b.stg[i].alloc, 0, VK_WHOLE_SIZE);
        for (int i = 0; i < open_b.nchunk; i++)
            vmaFlushAllocation(c->vma, open_b.chunk[i].alloc, 0, i + 1 < open_b.nchunk ? VK_WHOLE_SIZE : open_b.chunk_used);
        vk_crumb(open_b.cb, VK_CRUMB_UPLOAD_END, open_b.seq);
        vkEndCommandBuffer(open_b.cb);
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &open_b.cb};
        vk_queue_lock();
        VkResult r = vkQueueSubmit(c->gfx_queue, 1, &si, open_b.fence);
        vk_queue_unlock();
        if (r != VK_SUCCESS) {
            vk_device_lost_check(r, "the submit of an upload batch");
            rt_fatal("vk: upload submit failed (%d)", (int)r);
        }
        if (ninflight == cap_inflight) grow((void **)&inflight, &cap_inflight, sizeof *inflight);
        inflight[ninflight++] = open_b;
        inflight_staging_bytes += open_b.staging_bytes;
        submitted_seq = open_b.seq;
        open_active = 0;
    }
    if (c && wait) {
        while (ninflight && done_seq < seq) {
            // a lost device ends the wait at once and leaves the fence unsignalled: retire would take nothing and this
            // loop would spin for good, holding the upload lock, instead of the run ending with the device's report
            VkResult w = vkWaitForFences(c->device, 1, &inflight[0].fence, VK_TRUE, UINT64_MAX);
            if (w != VK_SUCCESS) {
                vk_device_lost_check(w, "the wait for an upload batch");
                rt_fatal("vk: the wait for an upload failed (%d)", (int)w);
            }
            retire(c, 0);
        }
    }
    if (c) collect_locked(c);
    plat_rmutex_unlock(&cmd_lock);
    vk_texstat_time(VK_TS_FLUSH, f0);
    return seq;
}

static void defer_notify(int kind, uint64_t handle, VmaAllocation alloc, uint64_t reclaim_bytes,
                         void (*destroyed)(void *), void *context) {
    if (alloc && (kind == DEAD_IMAGE_VMA || kind == DEAD_BUFFER_VMA) &&
        resource_core_vma_retire(alloc) != RC_ADMISSION_OK)
        rt_fatal("vk: deferred allocation lost its shared capacity owner");
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    if (ndead == cap_dead) grow((void **)&dead, &cap_dead, sizeof *dead);
    // The open batch may record into the object: it counts as the newest user. The GPU must also be past every
    // submission that names it. tl_noted covers the submitted ones; an object the replay lets go of while it records
    // a submission (vk_exec_recording_fence) may be named by that one too, not yet submitted: vk_texture.c
    // slot_resolve hands a draw a released entry, and entry_of destroys released entries at that address later in
    // the same submission. Stamped with the last submitted value, the object went as soon as the GPU reached it,
    // before the submission that names it ran, and with the recorder thread (vk_rec.c) behind, before the recorder
    // made the call that names it. So every deferral is stamped with the submission being
    // recorded. NX2NAVI_DEFER_STAMP=0 brings back the old stamp.
    static int stamp = -1;
    if (stamp < 0) stamp = rt_knob("DEFER_STAMP", "1") != 0;
    uint64_t tl = tl_noted, rf = stamp ? vk_exec_recording_fence() : 0;
    // vk_exec_recording_fence knows only the recording thread's own submission. A texture a guest thread releases (the
    // synchronous replay: vk_texture_release, entry_of, sweep_released on the game's loader threads) can be named by
    // the submission another thread records meanwhile: that one is issued + 1 (vk_sync_submit sets issued after it).
    if (stamp && !rf) rf = vk_sync_issued() + 1;
    if (rf > tl) tl = rf;
    // Publish the outstanding charge before the record can reach either collector. Only actual owned image
    // allocations count; destroying an image bound to somebody else's memory releases no allocation bytes.
    if (kind != DEAD_IMAGE_VMA || !alloc) reclaim_bytes = 0;
    if (reclaim_bytes) __atomic_fetch_add(&reclaim_pending_bytes, reclaim_bytes, __ATOMIC_RELEASE);
    VkMemoryImageHold *hold = NULL;
    if (kind == DEAD_IMAGE || kind == DEAD_IMAGE_VMA) {
        VkImage image = (VkImage)(uintptr_t)handle;
        hold = nvn_hm_del(&image_holds, &image);
        if (hold) hold->detached = 1;
    }
    dead[ndead++] = (Dead){kind, handle, alloc, open_active ? open_b.seq : next_seq, tl, reclaim_bytes,
                          destroyed, context, hold};
    if (kind != DEAD_HOST)
        rec_check_handle(kind == DEAD_VIEW ? REC_VIEW : kind == DEAD_BUFFER_VMA ? REC_BUFFER : REC_IMAGE, handle, 0, tl,
                         vk_sync_fence_done());
    plat_rmutex_unlock(&cmd_lock);
}

static void defer(int kind, uint64_t handle, VmaAllocation alloc, uint64_t reclaim_bytes) {
    defer_notify(kind, handle, alloc, reclaim_bytes, NULL, NULL);
}

static int dead_pending(void) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    int n = ndead;
    plat_rmutex_unlock(&cmd_lock);
    return n;
}

void vk_defer_image(VkImage img, VmaAllocation alloc) {
    if (img) defer(alloc ? DEAD_IMAGE_VMA : DEAD_IMAGE, (uint64_t)(uintptr_t)img, alloc, 0);
}
void vk_defer_image_reclaim(VkImage img, VmaAllocation alloc, uint64_t bytes) {
    if (img) defer(alloc ? DEAD_IMAGE_VMA : DEAD_IMAGE, (uint64_t)(uintptr_t)img, alloc, bytes);
}
void vk_defer_image_notify(VkImage img, VmaAllocation alloc, uint64_t bytes,
                           void (*destroyed)(void *), void *context) {
    if (!img) rt_fatal("vk: destruction notification requires an actual image");
    defer_notify(alloc ? DEAD_IMAGE_VMA : DEAD_IMAGE, (uint64_t)(uintptr_t)img, alloc, bytes, destroyed, context);
}
int vk_mem_image_destroy_completed(VkImage image, VmaAllocation allocation, uint64_t scene, uint64_t upload,
                                    void (*destroyed)(void *), void *context) {
    VkCtx *c = vk_ctx();
    if (!c || !image || !allocation || (!scene && !upload)) return 0;
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    const uint64_t completed = vk_sync_fence_done();
    VkMemoryImageHold *hold = nvn_hm_get(&image_holds, &image);
    if (scene > completed || upload > done_seq || (hold && (hold->readers || hold->pool || hold->binding_pending))) {
        plat_rmutex_unlock(&cmd_lock);
        return 0;
    }
    if (hold) {
        if (nvn_hm_del(&image_holds, &image) != hold) rt_fatal("vk: completed image placement identity changed");
        hold->detached = 1;
    }
    Dead prior = {.kind = DEAD_IMAGE_VMA, .handle = (uint64_t)(uintptr_t)image, .alloc = allocation,
                  .seq = upload, .tl = scene, .destroyed = destroyed, .context = context, .image_hold = hold};
    rec_check_handle(REC_IMAGE, prior.handle, 1, scene, completed);
    plat_rmutex_unlock(&cmd_lock);
    if (resource_core_vma_retire(allocation) != RC_ADMISSION_OK)
        rt_fatal("vk: completed image retirement lost its shared capacity owner");
    destroy_dead(c, &prior);
    return 1;
}

void vk_defer_view(VkImageView view) { if (view) defer(DEAD_VIEW, (uint64_t)(uintptr_t)view, NULL, 0); }

void vk_defer_image_owned_view(VkImage image, VkImageView view) {
    if (!view) return;
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    VkMemoryImageHold *hold = nvn_hm_get(&image_holds, &image);
    if (hold && (hold->readers || hold->views)) {
        OwnedImageView *retired = malloc(sizeof *retired);
        if (!retired) rt_fatal("vk: allocating retired native image view ownership");
        *retired = (OwnedImageView){hold->views, view};
        hold->views = retired;
    } else vk_defer_view(view);
    plat_rmutex_unlock(&cmd_lock);
}
void vk_defer_free(void *p) { if (p) defer(DEAD_HOST, (uint64_t)(uintptr_t)p, NULL, 0); }
void vk_defer_buffer(VkBuffer buf, VmaAllocation alloc) {
    if (buf) defer(DEAD_BUFFER_VMA, (uint64_t)(uintptr_t)buf, alloc, 0);
}

void vk_upload_defer_buffer(VkBuffer buf, VmaAllocation alloc, uint64_t last_upload_seq) {
    if (!buf) return;
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    if (!alloc || !last_upload_seq || last_upload_seq > next_seq)
        rt_fatal("vk: upload-only buffer retirement requires its recorded upload sequence");
    if (resource_core_vma_retire(alloc) != RC_ADMISSION_OK)
        rt_fatal("vk: upload-only buffer lost its shared capacity owner");
    if (ndead == cap_dead) grow((void **)&dead, &cap_dead, sizeof *dead);
    dead[ndead++] = (Dead){.kind = DEAD_BUFFER_VMA, .handle = (uint64_t)(uintptr_t)buf,
                           .alloc = alloc, .seq = last_upload_seq};
    plat_rmutex_unlock(&cmd_lock);
}

void vk_upload_defer_release(uint64_t sequence, void (*release)(void *), void *owner) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    if (!sequence || sequence > next_seq || !release || !owner)
        rt_fatal("vk: upload ownership requires its original recording sequence");
    if (ndead == cap_dead) grow((void **)&dead, &cap_dead, sizeof *dead);
    dead[ndead++] = (Dead){.kind = DEAD_UPLOAD_CALLBACK, .seq = sequence, .destroyed = release, .context = owner};
    plat_rmutex_unlock(&cmd_lock);
}

void vk_mem_note_submit(uint64_t timeline_value) {
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    if (timeline_value > tl_noted) tl_noted = timeline_value;
    plat_rmutex_unlock(&cmd_lock);
}

// The objects go once cmd_lock is released: destroying an image and its memory takes up to milliseconds (the
// collect's 0.3 slow ones a frame took a median 0.26-0.43 ms, up to 1.9 ms), and the replay's uploads
// (vk_upload_begin) wait for cmd_lock.
void vk_mem_collect(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    plat_once(&cmd_once, cmd_init, NULL);
    Dead *out = NULL;
    int nout = 0, cout = 0;
    plat_rmutex_lock(&cmd_lock);
    collect_locked_out(c, &out, &nout, &cout);
    plat_rmutex_unlock(&cmd_lock);
    for (int i = 0; i < nout; i++) destroy_dead(c, &out[i]);
    free(out);
}

// A newly detached cache victim can be stamped with this recording's final fence. Only a compatible allocation
// already retired to an earlier handed scene can supply progress while the current operation still owns its tail.
uint64_t vk_mem_progress_prior(const RcVmaDependency *need, uint64_t scene) {
    if (!need->selected || !scene) return 0;
    uint64_t handed = vk_sync_handed_fence(), supplier = 0;
    plat_once(&cmd_once, cmd_init, NULL);
    plat_rmutex_lock(&cmd_lock);
    for (int i = 0; i < ndead; i++) {
        const Dead *d = &dead[i];
        if (!d->alloc || d->tl <= supplier || d->tl >= scene || d->tl > handed || d->seq > done_seq ||
            (d->image_hold && d->image_hold->readers)) continue;
        if (resource_core_vma_dependency_retirement(need, d->alloc)) supplier = d->tl;
    }
    plat_rmutex_unlock(&cmd_lock);
    return supplier;
}

void vk_mem_stats(VkMemStats *s) {
    memset(s, 0, sizeof *s);
    plat_mutex_lock_shared(&pool_lock);
    for (int i = 0; i < pool_index.n; i++) {
        const PoolRec *r = pool_index.e[i].buf;
        if (r->kind == POOL_IMPORT) s->imported_pools++;
        else if (r->kind == POOL_PHYS) s->physical_pools++;
        else s->virtual_pools++;
    }
    plat_mutex_unlock_shared(&pool_lock);
    s->import_failures = (uint64_t)n_import_failures;
    s->upload_batches = next_seq;
}

void vk_mem_shutdown(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    vk_upload_flush(1);
    vk_queue_lock();
    vk_device_wait_idle();
    vk_queue_unlock();
    vk_mem_collect();
    native_rmw_pipeline_destroy(c);
    native_rmw_bitmap_pipeline_destroy(c);
    native_rmw_bitmap_free_shapes();
    RB.scene = NULL;
    RB.count = 0;
    RW.scene = NULL;
    RW.workspace = NULL;
    RW.word_count = 0;
    vk_gpa_report();
    vk_gpa_shutdown();
    vk_exec_texture_publications_clear();
    vk_texture_shutdown();
    plat_rmutex_lock(&cmd_lock);
    if (image_holds.n || __atomic_load_n(&image_readers, __ATOMIC_ACQUIRE))
        rt_fatal("vk: native image owners remain at memory shutdown");
    retire(c, 1);
    uint64_t keep = tl_noted;
    tl_noted = 0;
    for (int i = 0; i < ndead; i++) dead[i].seq = 0, dead[i].tl = 0;   // the device is idle
    // Deferred bound images still exclude their physical spans until VkImage destruction.
    collect_locked(c);
    tl_noted = keep;
    if (image_placements.n) rt_fatal("vk: native image placements remain after memory shutdown destruction");
    free(image_holds.keys);
    free(image_holds.vals);
    nvn_hm_init(&image_holds, sizeof(VkImage));
    free(image_placements.keys);
    free(image_placements.vals);
    nvn_hm_init(&image_placements, sizeof(VkDeviceMemory));
    for (int i = 0; i < nspare; i++) {
        vkDestroyFence(c->device, spare[i].fence, NULL);
        free(spare[i].stg), free(spare[i].chunk);
    }
    nspare = 0;
    // The staging cache and the arena's free chunks live in stg_pool: before it goes
    for (int i = 0; i < stg_ncache; i++) resource_core_vma_destroy_buffer(c->vma, stg_cache[i].buf, stg_cache[i].alloc);
    stg_ncache = 0, stg_cache_bytes = 0;
    for (int i = 0; i < narena_free; i++) resource_core_vma_destroy_buffer(c->vma, arena_free[i].buf, arena_free[i].alloc);
    narena_free = 0;
    if (cmd_pool) vkDestroyCommandPool(c->device, cmd_pool, NULL);
    cmd_pool = VK_NULL_HANDLE;
    if (stg_pool) vmaDestroyPool(c->vma, stg_pool);   // after retire: no staging buffer is left in it
    stg_pool = NULL;
    stg_pool_tried = 0;
    plat_rmutex_unlock(&cmd_lock);
    vk_texture_retire_drain();
    plat_mutex_lock(&pool_lock);
    mirror_pressure_first = NULL;
    mirror_pressure_last = NULL;
    for (int i = 0; i < pool_index.n; i++) {
        PoolRec *r = pool_index.e[i].buf;
        if (r->mbuf) mem_lazy_release(r->import_lo, r->import_hi - r->import_lo);
        pool_topology_close(r);
        destroy_rec(c, r);
    }
    pool_index.n = 0;
    for (int i = 0; i < nswl; i++) destroy_rec(c, swl[i].provider);
    nm = npend = nwbl = nsnl = snl_copied = nswl = 0;   // mirrors went with their pools
    swl_marks = -1;
    snap_chunks_shutdown(c);
    for (int i = 0; i < nretiring; i++) destroy_rec(c, retiring[i]);   // finalized ones whose callback never ran
    nretiring = 0;
    plat_mutex_lock(&shadow_lock);
    nstouch = nstab = ntpools = 0;
    s_live_n = s_live_bytes = 0;
    plat_mutex_unlock(&shadow_lock);
    if (pool_backings && rc_backing_store_destroy(pool_backings) != RC_BACKING_OK)
        rt_fatal("vk: pool backing store still has native owners at shutdown");
    pool_backings = NULL;
    npwait = 0;   // those waiting for pool_wait_collect were on the retiring list too
    __atomic_add_fetch(&pool_gen, 1, __ATOMIC_RELEASE);
    nphys = 0;
    plat_mutex_unlock(&pool_lock);
    plat_mutex_lock(&gen_lock);
    for (uint64_t i = 0; i < GEN_CHUNKS; i++) free(gen_chunk[i]), gen_chunk[i] = NULL, free(cgen_chunk[i]), cgen_chunk[i] = NULL;
    plat_mutex_unlock(&gen_lock);
}

// NX2NAVI_TRACE=perf "mem: sizes" (perf.c): pool records, VRAM mirrors, the deferred destruction list, upload batches,
// the staging cache and arena, the write-back merge snapshot pool and the shadow storage's base copies. Counters only:
// cmd_lock is taken by every upload and release, so nothing is walked under it.
void vk_mem_sizes(PerfSizes *s) {
    plat_mutex_lock_shared(&pool_lock);
    uint64_t pools = (uint64_t)pool_index.n, phys = (uint64_t)nphys;
    plat_mutex_unlock_shared(&pool_lock);
    plat_rmutex_lock(&cmd_lock);
    uint64_t dead_n = (uint64_t)ndead, infl = (uint64_t)ninflight, spare_b = (uint64_t)nspare, stg_n = (uint64_t)stg_ncache,
             stg_b = (uint64_t)stg_cache_bytes, arena_n = (uint64_t)narena_free,
             open_staging = open_active ? open_b.staging_bytes : 0, inflight_staging = inflight_staging_bytes,
             checkpoints = upload_checkpoints, pressure_waits = upload_pressure_waits, open_count = open_active != 0;
    plat_rmutex_unlock(&cmd_lock);
    plat_mutex_lock(&snap_pool_lock);
    uint64_t snap_n = (uint64_t)nsnap_pool, snap_b = snap_pool_bytes;
    plat_mutex_unlock(&snap_pool_lock);
    plat_mutex_lock(&shadow_lock);
    uint64_t sb = s_sb_pages;
    plat_mutex_unlock(&shadow_lock);
    perf_size(s, "pool records", pools, 0, PERF_SIZE_NONE);
    perf_size(s, "of them physical", phys, 0, PERF_SIZE_NONE);
    perf_size(s, "VRAM mirrors", (uint64_t)__atomic_load_n(&m_live_n, __ATOMIC_RELAXED), 0,
              (uint64_t)__atomic_load_n(&m_live_bytes, __ATOMIC_RELAXED));
    perf_size(s, "deferred destructions", dead_n, 0, PERF_SIZE_NONE);
    perf_size(s, "upload batches in flight", infl, 0, PERF_SIZE_NONE);
    perf_size(s, "open upload staging", open_count, 0, open_staging);
    perf_size(s, "in-flight upload staging", infl, 0, inflight_staging);
    perf_size(s, "upload checkpoints", checkpoints, 0, PERF_SIZE_NONE);
    perf_size(s, "upload pressure waits", pressure_waits, 0, PERF_SIZE_NONE);
    perf_size(s, "spare upload batches", spare_b, 0, PERF_SIZE_NONE);
    perf_size(s, "staging cache", stg_n, STG_CACHE, stg_b);
    perf_size(s, "free staging arena chunks", arena_n, ARENA_KEEP, arena_n * ARENA_CHUNK);
    perf_size(s, "write-back snapshot pool", snap_n, SNAP_POOL_N, snap_b);
    perf_size(s, "shadow base copies", sb, 0, sb * SB_PAGE);
    // the page generations handed out so far (gen_now, 32 bits, no wrap handling): how fast a session uses them up
    perf_size(s, "page generations handed out", (uint64_t)__atomic_load_n(&gen_now, __ATOMIC_RELAXED), 0, PERF_SIZE_NONE);
}

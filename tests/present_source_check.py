"""Exercise exact present sources and native destruction using the production ownership adapters."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from bind_recipe_check import function as extract_function


def read(name):
    return (ROOT / 'runtime' / name).read_text()


def section(source, start, end):
    offset = source.index(start)
    return source[offset:source.index(end, offset)]


def function(source, name):
    match = re.search(r'^[A-Za-z_][^;\n]*\b' + name + r'\([^;{}]*\)\s*\{', source, re.M)
    if match is None:
        raise RuntimeError('Missing production definition of ' + name)
    return extract_function(source[match.start():], name)


def functions(source, *names):
    return '\n'.join(function(source, name) for name in names)


def fixture():
    tex, uses, content = (read(name) for name in ('vk_texture.c', 'vk_texture_uses.h', 'vk_texture_content.h'))
    mem, hdr, nvn = (read(name) for name in ('vk_memory.c', 'vk_hdr.c', 'nvn_api_cmd.c'))
    shutdown_images = section(function(mem, 'vk_mem_shutdown'), '    plat_rmutex_lock(&cmd_lock);',
                              '    for (int i = 0; i < nspare; i++)')
    return r'''
#include <assert.h>
#include <stdio.h>
#include "resource_core_backing.h"
#include "resource_core_vma.h"
#include "nvn_api.h"
#include "gfx.h"
#include "nvn_hmap.h"
#include "nvn_pool_index.h"
#include "vk_rec.h"

#define HANDLE(type, value) ((type)(uintptr_t)(value))
typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
static int tex_lock, cmd_lock, cmd_once, pool_lock, shutdown_draining;
static void plat_mutex_lock(int *m) { assert(!*m); *m = 1; }
static void plat_mutex_unlock(int *m) { assert(*m); *m = 0; }
#define plat_mutex_lock_shared plat_mutex_lock
#define plat_mutex_unlock_shared plat_mutex_unlock
static void plat_rmutex_lock(int *m) { assert(m == &cmd_lock); ++*m; }
static void plat_rmutex_unlock(int *m) { assert(m == &cmd_lock && *m); --*m; }
static void lock(void) { plat_mutex_lock(&tex_lock); }
static void unlock(void) { plat_mutex_unlock(&tex_lock); }

''' + section(tex, 'typedef struct { uint32_t flags, type, format', '\n') + r'''

#include "vk_texture_material.h"
typedef struct Tex {
    VkTexImage pub;
    tex_t t;
    struct Tex *next_at;
    VmaAllocation alloc;
    MemBackingCapture *image_host_backing;
    int pin_owned, pin_detached, bound_to, source_canonical, source_blocked;
} Tex;
static nvn_hmap by_addr = {.klen = sizeof(uint64_t)};
static Tex *current;
static int tex_deferred;
static unsigned lookup_calls, fresh_calls, pin_dropped, driver_binds, placement_queries, budget_progress;
static unsigned image_destroys, raw_destroys, memory_frees, buffer_destroys, retire_requests;
static uintptr_t last_destroyed;
static unsigned view_destroys, completed_notifications;
static int hdr_wanted, hdr_active, vk_hdr_track;
static VkCtx ctx;
VkCtx *vk_ctx(void) { return &ctx; }
int vk_hdr_wanted(void) { return hdr_wanted; }
int vk_hdr_active(void) { return hdr_active; }
int vk_hdr_native(void) { return 0; }
void rt_log(const char *fmt, ...) { (void)fmt; }
void rt_fatal(const char *fmt, ...) { fputs(fmt, stderr); abort(); }
int mem_backing_current(const MemBackingCapture *backing) { (void)backing; return 1; }
static void destroy_tex(Tex *e) { (void)e; abort(); }
static void touch(Tex *e) { assert(tex_lock && e); }
static Tex *entry_of(const tex_t *t) { assert(tex_lock && t->mem); lookup_calls++; return current; }
static void make_fresh(Tex *e) { assert(tex_lock && e); fresh_calls++; }
static int source_permission_current(Tex *e) { (void)e; abort(); }
static int preserve_pool_contents_current(const Tex *e) { assert(!e->pub.preserve_pool_bytes); return 1; }
static int handed_current;
static RcRegistryPosition picture_cutoff = {31, 812};
RcRegistry *rc_registry_runtime(void) { return (RcRegistry *)(uintptr_t)9; }
RcRegistryPosition rc_registry_handed_cutoff(RcRegistry *registry) {
    assert(tex_lock && registry == rc_registry_runtime());
    return picture_cutoff;
}
static int preserve_pool_contents_handed(const Tex *e) {
    assert(tex_lock && e->pub.image && e->pub.preserve_pool_bytes);
    return handed_current;
}
struct VkTextureAliasRead {
    VkTexImage image;
    VmaAllocation allocation;
    VkMemoryImageHold *target_hold;
    uint32_t owners;
    uint8_t uploaded, upload_canceled;
};
VkTextureAliasRead *vk_texture_alias_read_retain(VkTextureAliasRead *seed) {
    assert(seed && seed->owners);
    seed->owners++;
    return seed;
}
void vk_texture_alias_read_cancel(VkTextureAliasRead *seed) {
    assert(seed && seed->owners);
    seed->owners--;
}
const VkTexImage *vk_texture_alias_target_image(const VkTextureAliasRead *seed) {
    return seed && seed->uploaded ? &seed->image : NULL;
}
void resource_core_budget_progress(void) { assert(!tex_lock && !cmd_lock); budget_progress++; }
''' + section(uses, 'enum { TEXTURE_USE_BUCKETS', 'static TextureUseNode *texture_use_find') + functions(
        uses, 'texture_use_retain_instance', 'texture_use_unlink'
    ) + section(nvn, 'typedef struct {\n    tex_t t;', 'static const tex_t *tex_pin_access(') + functions(
        nvn, 'pin_ref', 'nvn_texture_pin_retain_attachment', 'nvn_texture_pin_release'
    ) + section(hdr, 'static PlatMutex rec_lock', 'void vk_hdr_set_active') + functions(
        hdr, 'vk_hdr_present_source_locked', 'vk_hdr_image_gone'
    ) + r'''
int vk_exec_image_gone(VkImage image) { assert(tex_lock); vk_hdr_image_gone(image); return 0; }

// Supply one selectable native pool placement. Its refcount and destruction stay production-owned.
typedef struct VkMemImageAliasBaseline VkMemImageAliasBaseline;
enum { PHYSICAL_ORIGIN_POISON_LIMIT = 32 };
typedef struct { uint64_t lo, hi; } FlushRange;
typedef struct PhysicalOriginBinding { struct PhysicalOriginBinding *next; uint64_t lo, hi; } PhysicalOriginBinding;
typedef struct PoolRec {
    unsigned native_refs;
    int kind, unpublished;
    uint64_t publication, size, physical_origin_upload, physical_origin_bytes;
    VkDeviceSize physical_memory_offset;
    RcBackingMapId map;
    uint32_t physical_origin_poison_count, physical_raw_use_count;
    int physical_origin_unknown, physical_raw_use_unknown;
    FlushRange physical_origin_poison[PHYSICAL_ORIGIN_POISON_LIMIT], *physical_raw_uses;
    VkMemImageAliasBaseline *physical_origin_borrows;
    PhysicalOriginBinding *physical_origin_pending;
    VkBuffer buf, mbuf;
    VmaAllocation alloc, malloc_, salloc;
    VkDeviceMemory mem;
    RcBackingId backing;
    MemBackingCapture *host_backing;
} PoolRec;
enum {POOL_PHYS = 1, POOL_VIRT};
static RcBackingStore *pool_backings;
static pidx pool_index;
static VkMemSpan placement;
RcRegistryPosition vk_exec_resource_position(void) { return (RcRegistryPosition){0}; }
int rc_backing_map_intervals_current(RcBackingStore *store, RcBackingMapId map,
    const RcInterval *intervals, uint32_t count, uint64_t address, uint64_t bytes) {
    assert(!store && pool_lock && map.id == 1);
    return count == 1 && address == 0x6000 && bytes == 256 && intervals[0].mapping == 1 &&
        intervals[0].address == address && intervals[0].range.offset == 64 && intervals[0].range.size == bytes;
}
static int pool_device_span(PoolRec *r, uint64_t addr, uint64_t size, VkMemSpan *out, PoolRec **provider) {
    assert(pool_lock);
    if (addr < 0x6000 || addr - 0x6000 >= placement.avail || size > placement.avail - (addr - 0x6000)) return 0;
    *provider = r;
    *out = placement;
    out->alloc_offset += addr - 0x6000;
    out->avail -= addr - 0x6000;
    return 1;
}
RcBackingResult rc_backing_reclaim(RcBackingStore *s, RcBackingId b, uintptr_t *p) {
    (void)s; (void)b; (void)p; abort();
}
static void shadow_metadata_destroy(PoolRec *p) { assert(!p->native_refs); }
static void pool_authority_end(PoolRec *p) { assert(!p->native_refs); }
void mem_backing_release(MemBackingCapture **p) { assert(!*p); }
VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice d, VkBuffer b, const VkAllocationCallbacks *a) {
    assert((!cmd_lock || shutdown_draining) && d == ctx.device && b == placement.buffer && !a);
    assert(last_destroyed == 303 && !buffer_destroys++);
}
VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *a) {
    (void)d; (void)m; (void)a; abort();
}
void vmaGetAllocationInfo(VmaAllocator v, VmaAllocation a, VmaAllocationInfo *out) {
    assert(!cmd_lock && !pool_lock && v == ctx.vma && a == placement.alloc);
    PoolRec *provider = pool_index.e[0].buf;
    assert(provider->native_refs >= 2);
    placement_queries++;
    *out = (VmaAllocationInfo){.deviceMemory=HANDLE(VkDeviceMemory, 890), .offset=8192, .size=4096};
}
VkResult vmaBindImageMemory2(VmaAllocator v, VmaAllocation a, VkDeviceSize offset, VkImage image, const void *next) {
    assert(!cmd_lock && v == ctx.vma && a == placement.alloc && offset == placement.alloc_offset && !next);
    assert(image == HANDLE(VkImage, 303));
    driver_binds++;
    return VK_SUCCESS;
}
void resource_core_vma_destroy_buffer(VmaAllocator a, VkBuffer b, VmaAllocation alloc) {
    (void)a; (void)b; (void)alloc; abort();
}
void resource_core_vma_free_memory(VmaAllocator a, VmaAllocation alloc) {
    assert((!cmd_lock || shutdown_draining) && a == ctx.vma && alloc == placement.alloc && buffer_destroys == 1);
    assert(last_destroyed == 303 && !memory_frees++);
}
VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice d, VkImage image, const VkAllocationCallbacks *a) {
    assert((!cmd_lock || shutdown_draining) && d == ctx.device && image == HANDLE(VkImage, 303) && !a);
    assert(!memory_frees && !buffer_destroys && !raw_destroys++);
    last_destroyed = (uintptr_t)image;
}
VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice d, VkImageView image, const VkAllocationCallbacks *a) {
    assert(!cmd_lock && d == ctx.device && image == HANDLE(VkImageView, 1201) && !a);
    assert(!view_destroys++);
}
static unsigned allocation_states[8];
static unsigned allocation_slot(VmaAllocation allocation) {
    uintptr_t index = (uintptr_t)allocation - 200;
    assert(index < sizeof allocation_states / sizeof *allocation_states);
    return (unsigned)index;
}
RcAdmissionStatus resource_core_vma_retire(VmaAllocation a) {
    assert(!cmd_lock && !allocation_states[allocation_slot(a)]++);
    retire_requests++;
    return RC_ADMISSION_OK;
}
void resource_core_vma_destroy_image(VmaAllocator a, VkImage image, VmaAllocation allocation) {
    assert(!cmd_lock && a == ctx.vma && image);
    assert(allocation_states[allocation_slot(allocation)]++ == 1);
    if (image == HANDLE(VkImage, 310)) assert(view_destroys == 1);
    image_destroys++;
    last_destroyed = (uintptr_t)image;
}
static uint64_t done_seq, next_seq = 17, tl_noted, scene_done;
static int ndead, cap_dead, npwait, open_active;
static struct {uint64_t seq;} open_b;
static void cmd_init(void *p) { (void)p; }
static void plat_once(int *once, void (*f)(void *), void *p) { if (!*once) { *once = 1; f(p); } }
int rt_knob(const char *name, const char *value) {
    if (!strcmp(name, "IMAGE_BASELINE_TRACE")) { assert(!strcmp(value, "0")); return 0; }
    return 1;
}
uint64_t vk_exec_recording_fence(void) { return 0; }
uint64_t vk_sync_issued(void) { return 39; }
uint64_t vk_sync_fence_done(void) { return scene_done; }
void rec_check_handle(int k, uint64_t h, int destroy, uint64_t tl, uint64_t done) {
    assert(cmd_lock && k == REC_IMAGE && h && tl == (shutdown_draining ? 0 : 40));
    if (destroy) assert(done >= tl);
}
VKAPI_ATTR VkResult VKAPI_CALL vkGetSemaphoreCounterValue(VkDevice d, VkSemaphore sem, uint64_t *out) {
    assert(cmd_lock && d == ctx.device && sem == ctx.timeline);
    *out = scene_done;
    return VK_SUCCESS;
}
static void retire(VkCtx *c, int wait) { assert(cmd_lock && c == &ctx && (!wait || shutdown_draining)); }
static void pool_wait_collect(uint64_t timeline) { (void)timeline; abort(); }
''' + section(mem, 'enum { DEAD_IMAGE =', 'static PlatRMutex cmd_lock;') + r'''
static Dead *dead;
static nvn_hmap image_holds = {.klen = sizeof(VkImage)};
static nvn_hmap image_placements = {.klen = sizeof(VkDeviceMemory)};
static uint32_t image_readers;
''' + functions(
        mem, 'grow', 'pool_native_acquire', 'destroy_rec', 'pool_image_alias_provider_locked', 'pool_image_alias_acquire',
        'image_hold_find_or_add', 'image_placement_available', 'image_placement_insert', 'image_placement_remove',
    ) + section(mem, 'typedef enum {\n    IMAGE_ALIAS_BASELINE_OPEN,', '// pool_lock held. Track exact raw ranges') + functions(
        mem, 'physical_origin_image_allowed', 'physical_range_add', 'physical_origin_poison', 'physical_use_overlaps',
        'physical_origin_fill_overlaps', 'image_alias_placement_inputs_valid', 'image_alias_placement_provider_locked',
        'physical_image_alias_range', 'image_alias_preplacement_locked', 'physical_origin_begin_image', 'physical_origin_end_image',
        'vk_mem_bind_image_alias', 'vk_mem_image_alias_overlaps', 'vk_mem_image_hold', 'vk_mem_image_held',
        'vk_mem_image_hold_retain_exact', 'vk_mem_image_release', 'destroy_dead', 'collect_locked_out', 'collect_locked', 'defer_notify', 'defer',
        'vk_defer_image_reclaim', 'vk_defer_image_notify', 'vk_mem_image_destroy_completed',
        'vk_defer_view', 'vk_defer_image_owned_view', 'vk_mem_collect'
    ) + functions(content, 'content_defer_instance_mode', 'content_defer_instance') + section(
        tex, 'static Tex *texture_get_locked(', 'static int texture_history_sample_locked('
    ) + section(
        tex, 'static int present_source_retain(', 'void vk_texture_present_release('
    ) + functions(tex, 'vk_texture_present_release', 'vk_texture_present_retain_alias_image') + '\n' + r'''
static void drain_native_images_at_shutdown(VkCtx *c) {
    shutdown_draining = 1;
''' + shutdown_images + r'''
    plat_rmutex_unlock(&cmd_lock);
    shutdown_draining = 0;
}


static tex_t guest = {.mem = HANDLE(uint8_t *, 0x1000), .serial = 7};
static Tex image(uintptr_t handle, uintptr_t allocation, uint64_t address) {
    return (Tex){.pub = {.image = HANDLE(VkImage, handle), .addr = address, .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                        .format = VK_FORMAT_R8G8B8A8_UNORM, .width = 640, .height = 360},
                 .alloc = HANDLE(VmaAllocation, allocation)};
}
static VkTexturePresentSource *capture(const tex_t *texture, VkImage expected) {
    VkTexImage selected = {0};
    VkTexturePresentSource *source = vk_texture_present_capture(texture, &selected);
    assert(source && selected.image == expected && selected.width == 640 && selected.height == 360);
    assert(!tex_lock && !cmd_lock && !rec_lock && !pin_lock);
    return source;
}
static void deferred(Tex *owner, uint64_t bytes) {
    lock();
    vk_exec_image_gone(owner->pub.image);
    vk_defer_image_reclaim(owner->pub.image, owner->alloc, bytes);
    unlock();
}

static void capture_guest_incarnations(void) {
    Tex old = image(401, 201, 0x1000);
    Tex changed = image(402, 202, 0x1000);
    Tex reverted = image(403, 203, 0x1000);
    changed.pub.width = 800;
    changed.pub.height = 450;
    tex_t b = guest, a2 = guest;
    b.serial = 8;
    a2.serial = 9;
    old.t = guest;
    changed.t = b;
    reverted.t = a2;

    current = &old;
    VkTexImage selected;
    VkTexturePresentSource *a = vk_texture_present_capture(&guest, &selected);
    assert(a && selected.image == old.pub.image && selected.width == 640 && selected.height == 360);
    current = &changed;
    VkTexturePresentSource *new_size = vk_texture_present_capture(&b, &selected);
    assert(new_size && selected.image == changed.pub.image && selected.width == 800 && selected.height == 450);
    current = &reverted;
    VkTexturePresentSource *original_size = vk_texture_present_capture(&a2, &selected);
    assert(original_size && selected.image == reverted.pub.image && selected.width == 640 && selected.height == 360);

    const GfxPresentPictureSource first = vk_texture_present_picture(a);
    assert(first.window_texture_serial == 7 && first.source_texture_serial == 7 &&
           first.image == (uintptr_t)old.pub.image && first.window_image == first.image &&
           first.handed_scene == 31 && first.handed_ordinal == 812 &&
           first.width == 640 && first.height == 360 && first.format == VK_FORMAT_R8G8B8A8_UNORM);
    assert(vk_texture_present_picture(new_size).window_texture_serial == 8 && new_size->raw->image == changed.pub.image);
    assert(vk_texture_present_picture(original_size).window_texture_serial == 9 && original_size->raw->image == reverted.pub.image);
    assert(!vk_texture_present_picture(NULL).image);
    VkTexturePresentSource *raw_fork = vk_texture_present_retain_held_image(a->raw, old.pub.image);
    assert(raw_fork && !vk_texture_present_picture(raw_fork).image);
    vk_texture_present_release(raw_fork);

    // A same-address mismatched provider stays untagged even though the legacy
    // selection can hold it. It cannot certify a new guest extent.
    VkTexturePresentSource *unattested = vk_texture_present_capture(&guest, &selected);
    assert(unattested && selected.image == reverted.pub.image && !vk_texture_present_picture(unattested).source_texture_serial);
    vk_texture_present_release(unattested);
    vk_texture_present_release(original_size);
    vk_texture_present_release(new_size);
    vk_texture_present_release(a);
    assert(!image_holds.n);
}

static void capture_raw(void) {
    Tex owner = image(301, 201, 0x1000);
    owner.t = guest;
    owner.pub.preserve_pool_bytes = 1;
    current = &owner;
    VkTexImage refused = {0};
    assert(!vk_texture_present_capture(&guest, &refused) && !refused.image && !image_holds.n);
    handed_current = 1;
    tex_t obsolete = guest;
    obsolete.serial--;
    assert(!vk_texture_present_capture(&obsolete, &refused) && !refused.image && !image_holds.n);
    VkTexturePresentSource *a = capture(&guest, owner.pub.image);
    VkTexturePresentSource *b = capture(&guest, owner.pub.image);
    handed_current = 0; // a later alias winner changes this proof, not the earlier captured native users
    assert(!vk_texture_present_capture(&guest, &refused) && !refused.image);
    assert(a->raw == b->raw && !a->pin && a->raw->readers == 2 && vk_mem_image_held(owner.pub.image));
    unsigned previous_lookups = lookup_calls;
    assert(!vk_texture_present_retain_held_image(a->raw, HANDLE(VkImage, 199)));
    assert(!vk_texture_present_retain_held_image(a->raw, VK_NULL_HANDLE));
    assert(!vk_texture_present_retain_held_image(NULL, owner.pub.image));
    assert(lookup_calls == previous_lookups && a->raw->readers == 2);
    deferred(&owner, 101);
    assert(ndead == 1 && dead->image_hold == a->raw && a->raw->detached && !image_holds.n);
    VkTexturePresentSource *sdk = vk_texture_present_retain_held_image(a->raw, owner.pub.image);
    assert(sdk && sdk->raw == a->raw && sdk->raw->readers == 3 && lookup_calls == previous_lookups);
    assert(vk_mem_reclaim_pending() == 101 && retire_requests == 1);
    done_seq = next_seq;
    scene_done = 40;
    vk_mem_collect();
    assert(ndead == 1 && !image_destroys && vk_mem_reclaim_pending() == 101);
    vk_texture_present_release(b);
    assert(!budget_progress && a->raw->readers == 2);
    vk_mem_collect();
    assert(ndead == 1 && !image_destroys);
    vk_texture_present_release(a);
    assert(!budget_progress && sdk->raw->readers == 1);
    vk_mem_collect();
    assert(ndead == 1 && !image_destroys);
    vk_texture_present_release(sdk);
    assert(budget_progress == 1 && !image_readers);
    vk_mem_collect();
    vk_mem_collect();
    assert(!ndead && !vk_mem_reclaim_pending() && image_destroys == 1 && last_destroyed == 301);
}

static void capture_seed(void) {
    VkTextureAliasRead seed = {.image.image=HANDLE(VkImage, 504),.owners=1};
    assert(!vk_texture_present_retain_alias_image(&seed, seed.image.image) && seed.owners==1);
    seed.uploaded=1;
    assert(!vk_texture_present_retain_alias_image(&seed, seed.image.image) && seed.owners==1);
    seed.allocation=HANDLE(VmaAllocation, 604);
    assert(!vk_texture_present_retain_alias_image(&seed, HANDLE(VkImage, 505)) && seed.owners==1);
    VkTexturePresentSource *held=vk_texture_present_retain_alias_image(&seed, seed.image.image);
    assert(held && held->seed==&seed && !held->raw && seed.owners==2);
    vk_texture_alias_read_cancel(&seed); // caller cancels its own prepared draw reference
    seed.upload_canceled=1; // actual native upload fate distinguishes an unsent contents producer
    assert(seed.owners==1 && !vk_texture_present_retain_alias_image(&seed, seed.image.image));
    vk_texture_present_release(held); // this independent native consumer releases scene ownership
    assert(!seed.owners);
}

static void capture_alias(void) {
    PoolRec *pool = calloc(1, sizeof *pool);
    assert(pool);
    placement = (VkMemSpan){.buffer = HANDLE(VkBuffer, 800), .alloc = HANDLE(VmaAllocation, 900), .alloc_offset = 64, .avail = 1024};
    ctx.props.limits.bufferImageGranularity = 1;
    *pool = (PoolRec){.native_refs = 1, .kind = POOL_PHYS, .buf = placement.buffer, .alloc = placement.alloc,
        .publication = 3, .map = {.id = 1}, .size = 1024, .physical_memory_offset = 8192};
    const VkMemoryRequirements requirements = {.size = 256, .alignment = 16, .memoryTypeBits = 1};
    const RcInterval mapping = {0x6000, 1, {2, 64, 256}};
    assert(pidx_insert(&pool_index, &(pidx_ent){.lo = 0x6000, .hi = 0x7000, .buf = pool}));
    Tex owner = image(303, 0, 0x6000);
    owner.bound_to = 1;
    current = &owner;
    VkMemSpan stale = placement;
    stale.alloc_offset++;
    assert(vk_mem_bind_image_alias(owner.pub.image, 0x6000, &requirements, &stale, &mapping, 1, NULL) ==
        VK_MEM_IMAGE_ALIAS_STALE);
    assert(!driver_binds && pool->native_refs == 1);
    assert(vk_mem_bind_image_alias(owner.pub.image, 0x6000, &requirements, &placement, &mapping, 1, NULL) ==
        VK_MEM_IMAGE_ALIAS_DIRECT);
    assert(driver_binds == 1 && pool->native_refs == 2 && image_placements.n == 1);
    assert(vk_mem_bind_image_alias(HANDLE(VkImage, 313), 0x6000, &requirements, &placement, &mapping, 1, NULL) ==
        VK_MEM_IMAGE_ALIAS_FAILED);
    assert(driver_binds == 1 && pool->native_refs == 2 && image_placements.n == 1);
    VkTexturePresentSource *source = capture(&guest, owner.pub.image);
    assert(source->raw->pool == pool && source->raw->readers == 1);
    assert(!vk_mem_image_destroy_completed(owner.pub.image, HANDLE(VmaAllocation, 206), 40, next_seq, NULL, NULL));
    VkImage selected[1] = {0};
    unsigned queries = placement_queries;
    assert(vk_mem_image_alias_overlaps(0x6000, 256, &placement, selected, 1) == 1 && selected[0] == owner.pub.image);
    assert(vk_mem_image_alias_overlaps(0x6000, 256, &placement, selected, 0) == -1);
    VkMemSpan adjacent = placement;
    adjacent.alloc_offset += 256;
    assert(vk_mem_image_alias_overlaps(0x6100, 256, &adjacent, selected, 1) == 0);
    assert(vk_mem_image_alias_overlaps(0x6000, 256, &stale, selected, 1) == -1);
    assert(vk_mem_image_alias_overlaps(0x6000, 1025, &placement, selected, 1) == 0);
    assert(pool->native_refs == 2 && placement_queries == queries + 3);
    unsigned retires = retire_requests;
    deferred(&owner, 1000);
    vk_mem_collect();
    assert(ndead == 1 && !raw_destroys && !buffer_destroys && !memory_frees && !vk_mem_reclaim_pending());
    assert(retire_requests == retires);
    assert(source->raw->detached && source->raw->image == owner.pub.image);
    assert(vk_mem_image_alias_overlaps(0x6000, 256, &placement, selected, 1) == 0 && !raw_destroys);
    pool_index.n = 0;
    destroy_rec(&ctx, pool);
    assert(pool->native_refs == 1 && !memory_frees && !buffer_destroys);
    assert(vk_mem_image_alias_overlaps(0x6000, 256, &placement, selected, 1) == -1);
    assert(placement_queries == queries + 4);
    vk_texture_present_release(source);
    assert(!image_holds.n && !image_readers && image_placements.n == 1 && ndead == 1);
    drain_native_images_at_shutdown(&ctx);
    assert(raw_destroys == 1 && buffer_destroys == 1 && memory_frees == 1 && !ndead && !image_readers && !image_placements.n);
    assert(vk_mem_image_alias_overlaps(0x6000, 256, &placement, selected, 1) == -1 && placement_queries == queries + 4);
    assert(retire_requests == retires);
}

static void capture_material(void) {
    Tex owner = image(302, 0, 0x1000);
    TextureMaterialLedger ledger = {0};
    TextureMaterial material;
    current = &owner;
    texture_material_init(&material, &ledger, owner.pub.image, HANDLE(VmaAllocation, 202), &(ImgKey){0}, 121, 1);
    material.ready = material.reclaim = 1;
    TextureMaterialInstance *instance = material.resident.instance;
    content_use_image_add(instance);
    VkTexturePresentSource *a = capture(&guest, owner.pub.image);
    assert(a->material.use.instance == instance && !a->raw && texture_uses.nodes == 1);
    assert(rc_lifetime_recovery_set(&material.lifetime, material.lifetime.id, RC_LIFETIME_RECOVERY_IMMUTABLE) ==
           RC_LIFETIME_OK);
    assert(texture_material_take_residency(&material).instance == instance && instance->queued);
    assert(content_use_image_find(owner.pub.image) == instance && !material.resident.instance);
    VkTexturePresentSource *b = capture(&guest, owner.pub.image);
    assert(b->material.use.instance == instance && instance->lifetime.captures == 2);
    assert(instance->use_head == &b->material && b->material.instance_next == &a->material);
    unsigned retires = retire_requests, progress = budget_progress;
    vk_texture_present_release(a);
    assert(instance->use_head == &b->material && !b->material.instance_next && !b->material.instance_prev);
    assert(instance->lifetime.captures == 1 && !ndead && budget_progress == progress);
    vk_texture_present_release(b);
    assert(!texture_uses.nodes && !instance->use_head && !content_use_image_find(owner.pub.image));
    assert(retire_requests == retires + 1 && budget_progress == progress + 1 && ndead == 1);
    assert(vk_mem_reclaim_pending() == 121 && ledger.backend_accounted_bytes == 121 && !instance->destroyed);
    vk_mem_collect();
    vk_mem_collect();
    assert(!ndead && !vk_mem_reclaim_pending() && instance->destroyed && last_destroyed == 302);
    texture_material_instance_reap(instance);
    assert(!ledger.backend_accounted_bytes && !ledger.pending_instances);
}

static void drop_pin_owner(void *arg) {
    Tex *owner = arg;
    pin_dropped++;
    deferred(owner, 0);
}
static void capture_pinned(void) {
    Tex old = image(304, 204, 0x1000), replacement = image(305, 205, 0x1000);
    current = &replacement;
    old.t = guest;
    old.pin_owned = old.pin_detached = 1;
    pin_t pin = {.t = guest};
    nvn_hm_put(&pins, &guest.serial, &pin);
    assert(nvn_texture_pin_attach(&pin.t, &old, drop_pin_owner));
    assert(!nvn_texture_pin_retain_attachment(guest.serial, &replacement));
    unsigned looked_up = lookup_calls, refreshed = fresh_calls;
    VkTexturePresentSource *source = capture(&pin.t, old.pub.image);
    assert(source->pin == &pin.t && source->raw->image == old.pub.image && pin.refs == 1);
    assert(lookup_calls == looked_up && fresh_calls == refreshed && !vk_mem_image_held(replacement.pub.image));
    vk_texture_present_release(source);
    assert(!pin.refs && !pin_dropped && !image_holds.n && !ndead);
    nvn_texture_pin_attachments_clear();
    assert(pin_dropped == 1 && ndead == 1);
    vk_mem_collect();
    assert(last_destroyed == 304 && !ndead);
    nvn_hm_del(&pins, &guest.serial);
    // The exact owner is mandatory, even when an unrelated current image has the same backing address.
    current = &old;
    VkTexImage untouched = {.image = replacement.pub.image};
    assert(!vk_texture_present_capture(&guest, &untouched) && untouched.image == replacement.pub.image);
}

static void repeated_hdr_source(void) {
    Tex window = image(306, 206, 0x1000), picture = image(307, 207, 0x2000);
    picture.pub.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    current = &window;
    window.t.serial = guest.serial;
    hdr_wanted = hdr_active = 1;
    // The old receipt points to the same guest address and a different native incarnation.
    nvn_hm_put(&by_addr, &picture.pub.addr, &picture);
    recs[0] = (typeof(recs[0])){.win = window.pub.image, .src = HANDLE(VkImage, 501),
                              .fmt = picture.pub.format, .addr = picture.pub.addr};
    nrecs = 1;
    VkTexturePresentSource *fallback = capture(&guest, window.pub.image);
    assert(fallback->raw->image == window.pub.image && !vk_mem_image_held(picture.pub.image));
    vk_texture_present_release(fallback);
    recs[0].src = picture.pub.image;
    VkTexImage selected = {0};
    VkTexturePresentSource *float_source = vk_texture_present_capture(&guest, &selected);
    assert(float_source && selected.image == picture.pub.image && selected.addr == picture.pub.addr);
    assert(selected.format == picture.pub.format && float_source->raw->image == picture.pub.image);
    const GfxPresentPictureSource actual = vk_texture_present_picture(float_source);
    assert(actual.window_texture_serial == guest.serial && !actual.source_texture_serial &&
           actual.window_image == (uintptr_t)window.pub.image && actual.image == (uintptr_t)picture.pub.image &&
           actual.format == picture.pub.format && actual.handed_scene == 31 && actual.handed_ordinal == 812);
    assert(!vk_mem_image_held(window.pub.image) && recs[0].src == picture.pub.image);
    // Another present of the same picture forks the existing physical ownership.
    VkTexturePresentSource *repeat = capture(&guest, picture.pub.image);
    assert(repeat->raw == float_source->raw && repeat->raw->readers == 2);
    deferred(&picture, 0);
    assert(!recs[0].src && ndead == 1);
    fallback = capture(&guest, window.pub.image);
    vk_texture_present_release(fallback);
    vk_texture_present_release(float_source);
    vk_mem_collect();
    assert(ndead == 1);
    vk_texture_present_release(repeat);
    vk_mem_collect();
    assert(!ndead && last_destroyed == 307 && !image_readers);
}

static void capture_exact_depth(void) {
    Tex owner = image(308, 200, 0x3000);
    owner.pub.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    owner.pub.format = VK_FORMAT_D32_SFLOAT;
    const VkImage original = owner.pub.image, replacement = HANDLE(VkImage, 309);
    assert(!vk_texture_present_retain_image(NULL, original));
    assert(!vk_texture_present_retain_image(&owner.pub, VK_NULL_HANDLE));
    assert(!vk_texture_present_retain_image(&owner.pub, replacement) && !image_holds.n);
    VkTexturePresentSource *source = vk_texture_present_retain_image(&owner.pub, original);
    assert(source && source->raw && source->raw->image == original && source->raw->readers == 1);
    unsigned destroys = image_destroys;
    deferred(&owner, 2048);
    owner.pub.image = replacement;
    assert(!vk_texture_present_retain_image(&owner.pub, original) && !vk_mem_image_held(replacement));
    vk_mem_collect();
    assert(ndead == 1 && dead->image_hold == source->raw && source->raw->detached);
    assert(image_destroys == destroys && vk_mem_reclaim_pending() == 2048);
    vk_texture_present_release(source);
    vk_mem_collect();
    vk_mem_collect();
    assert(!ndead && !vk_mem_reclaim_pending() && !image_readers);
    assert(image_destroys == destroys + 1 && last_destroyed == (uintptr_t)original);
}

static void completed_notification(void *context) {
    assert(!cmd_lock && context == &completed_notifications && last_destroyed == 310);
    completed_notifications++;
}

static void reclaim_completed(void) {
    VkImage image = HANDLE(VkImage, 310);
    VmaAllocation alloc = HANDLE(VmaAllocation, 206);
    VkMemoryImageHold *hold = vk_mem_image_hold(image);
    vk_defer_image_owned_view(image, HANDLE(VkImageView, 1201));
    const unsigned retires = retire_requests, destroys = image_destroys;
    scene_done = 39;
    assert(!vk_mem_image_destroy_completed(image, alloc, 40, 17, completed_notification, &completed_notifications));
    scene_done = 40;
    done_seq = 16;
    assert(!vk_mem_image_destroy_completed(image, alloc, 40, 17, completed_notification, &completed_notifications));
    done_seq = 17;
    assert(!vk_mem_image_destroy_completed(image, alloc, 40, 17, completed_notification, &completed_notifications));
    assert(retire_requests == retires && image_destroys == destroys && !view_destroys && !completed_notifications);
    vk_mem_image_release(hold);
    assert(vk_mem_image_destroy_completed(image, alloc, 40, 17, completed_notification, &completed_notifications));
    assert(!ndead && !vk_mem_reclaim_pending() && !image_readers && !image_holds.n);
    assert(completed_notifications == 1 && retire_requests == retires + 1 && image_destroys == destroys + 1);
}

int main(void) {
    (void)pins_live;
    (void)pins_freed;
    ctx.timeline = HANDLE(VkSemaphore, 9);
    capture_guest_incarnations();
    capture_raw();
    capture_seed();
    capture_alias();
    capture_material();
    capture_pinned();
    repeated_hdr_source();
    capture_exact_depth();
    reclaim_completed();
    assert(!image_holds.n && !cmd_lock && !pool_lock && !pin_lock && !tex_lock && !rec_lock);
    free(pool_index.e); free(pool_index.maxhi);
    free(dead);
    free(image_holds.keys); free(image_holds.vals);
    free(image_placements.keys); free(image_placements.vals);
    free(by_addr.keys); free(by_addr.vals);
    free(pins.keys); free(pins.vals); free(pin_q);
    puts("Present source: exact material, pin, raw alias, depth, HDR fallback and physical retirement passed");
    return 0;
}
'''


def main():
    cc = os.environ.get('CC') or shutil.which('clang')
    if not cc:
        raise SystemExit('Clang is required (set CC)')
    toolchain = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get('VMA_INCLUDE', toolchain.parent.parent / 'include' / 'vma'))
    env = dict(os.environ, PATH=str(toolchain.parent) + os.pathsep + os.environ.get('PATH', ''))
    with tempfile.TemporaryDirectory(prefix='present-source-') as directory:
        c = Path(directory) / 'check.c'
        exe = Path(directory) / ('check.exe' if os.name == 'nt' else 'check')
        c.write_text(fixture())
        subprocess.run([str(toolchain), '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=undefined',
                        '-I' + str(ROOT / 'runtime'), '-I' + str(vma), str(c), '-o', str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == '__main__':
    main()

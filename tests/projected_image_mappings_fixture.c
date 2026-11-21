#include "resource_core_backing.h"
#include "nvn_tex.h"
#include "texture_source.h"
#define VK_USE_64_BIT_PTR_DEFINES 0
#include <vulkan/vulkan.h>

const int rt_nvn_api_version[2] = {200, 0};
const char *cfg_get(const char *key) { (void)key; return NULL; }
int nvn_texels_box(const nvn_texdesc *, unsigned, unsigned, uint32_t, uint32_t, uint32_t,
                   uint32_t, uint32_t, void *, void *, size_t, int);
int nvn_texels_box_ranges(const nvn_texdesc *, unsigned, unsigned, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t, void (*)(void *, uint64_t, uint64_t), void *);
int nvn_texels_slice_ranges(const nvn_texdesc *, unsigned, unsigned, uint32_t,
                           void (*)(void *, uint64_t, uint64_t), void *);

#define START UINT64_C(0xf000000000010000)
#define BASE_IMAGE 601
#define PATCH_IMAGE 602
#define REPLACEMENT_IMAGE 603
enum Fate { UNISSUED, COMPLETE, UNSUBMITTED, MAY_HAVE_EXECUTED, RECORDED_PENDING };
typedef uint64_t TexKey;
typedef struct { int readers, detached; } VkMemoryImageHold;
typedef struct {
    VkImage image;
    uint64_t addr;
    uint32_t width, height, depth, levels, layers, type, aspect, usage, gone;
    VkFormat format;
    uint32_t scale, preserve_pool_bytes;
    uint32_t view_type, samples, skip, block_views, bound, mirrored, fb_learned, gpu_owned;
} VkTexImage;
/* Alias plane layout */
typedef struct {
    VkTexImage pub;
    tex_t t;
    struct { VkFormat vk; } f;
    TexKey key;
    uint64_t content_revision;
    void *bound_to, *alloc;
    struct { int possible; } owner, *native_output;
    RcInterval *image_private_mapping;
    RcRegistryWrite *image_mapped_holds;
    uint32_t image_private_mapping_count, image_mapped_hold_count;
    uint64_t image_device_size;
    RcInterval *image_device_intervals, *alias_full_demand, *alias_plane_demand;
    uint32_t image_device_count, alias_full_count, alias_plane_count;
    int bound_alias_whole, pin_owned;
    RcRegistryPosition native_write;
    VkImageSubresourceRange alias_plane_range;
    RcInterval *image_host_backing;
    RcRegistryWrite image_authority;
    int gpu_dirty, material, pending_write, released, retiring, superseded, overwritten;
#ifdef TEST_HOST_COLOR
    struct MirroredColorWrite *mirrored_color_write;
    uint64_t wr_fence,gpu_seq,gpu_fence,run_frame,gw_frame,upload_mapping;
    uint32_t gpu_gen;
    uint8_t run,chash_ok,snap_color_texels;
    uint8_t *snap;
#endif
} Tex;
typedef struct VkTextureProjectedTransfer VkTextureProjectedTransfer;
/* Private target layout */

enum { IMAGE_ALIAS_MAX_WRITE_FRAGMENTS = 8192 };
static RcRegistryPosition transfer_read;
static Tex *entry;
static VkMemoryImageHold image_hold;
static int mapping_current=1, scene_abandoned, host_destroyed;
static RcBackingStore *clear_backings;
static RcBackingMapId clear_mapping;
static const int by_key;
static Tex *nvn_hm_get(const int *map,const TexKey *key) {
    assert(map==&by_key);
    return entry && entry->t.serial==*key ? entry : NULL;
}
static void make_key(const tex_t *t, TexKey *key) { *key=t->serial; }
static VkMemoryImageHold *vk_mem_image_hold(VkImage image) {
    assert(entry->pub.image==image);
    image_hold.readers++;
    return &image_hold;
}
static void vk_mem_image_release(VkMemoryImageHold *hold) {
    if (hold) { assert(hold==&image_hold && hold->readers>0);hold->readers--; }
}
static int vk_mem_image_hold_retired(const VkMemoryImageHold *hold) { return hold->detached; }
static int image_alias_guest_mapping_current(uint64_t address, uint64_t size, const RcInterval *intervals,uint32_t count) {
    assert(address==entry->pub.addr && size==entry->t.L.size && intervals && count);
    if (clear_backings)
        return rc_backing_map_intervals_current(clear_backings, clear_mapping, intervals, count, address, size);
    return mapping_current;
}
static int image_alias_exact_layout(const Tex *e, VkTextureAliasLayout *layout) {
    return texture_alias_layout(&e->t.d, &e->pub, layout);
}
static int image_alias_material_layout(const Tex *e,VkTextureAliasLayout *layout) {
    (void)layout;
    return !e->bound_to;
}
static int native_texture_output_reserve(Tex *e) { return e==entry; }
static uint32_t mip(uint32_t size,uint32_t level) {return size>>level ? size>>level : 1;}
static Tex *actual_projected_transfer_current_locked(const VkTextureProjectedTransfer *target);
static int locked, projections;
static void lock(void) { assert(!locked); locked = 1; }
static void unlock(void) { assert(locked); locked = 0; }
static RcRegistryPosition vk_exec_resource_position(void) { return transfer_read; }
static Tex *projected_transfer_current_locked(const VkTextureProjectedTransfer *target) {
    assert(locked && entry && entry->pub.image == target->image);
    return target->copies || (entry->pub.type==VK_IMAGE_TYPE_3D && target->hold)
#ifdef TEST_HOST_COLOR
           || target->host_color
#endif
           ? actual_projected_transfer_current_locked(target) : entry;
}
static int vk_texture_projected_transfer_current(const VkTextureProjectedTransfer *target) {
    lock();
    int current = projected_transfer_current_locked(target) != NULL;
    unlock();
    return current;
}
static const RcInterval *image_backing_intervals(const Tex *e, uint32_t *count) {
    *count = e->image_host_backing ? 1 : e->image_private_mapping_count;
    return e->image_host_backing ? e->image_host_backing : e->image_private_mapping;
}
static void image_alias_possible_written(Tex *e);
static void gpu_written_prepared(Tex *e);
typedef struct { uint64_t id; } TextureUseBatch;
static TextureUseBatch host_part;
static unsigned submit_epoch=1, host_generation=7;
static TextureUseBatch *texture_use_active(uint64_t scene,int create) {
    assert(scene==2 && create);
    return &host_part;
}
static void *texture_use_part_retain(TextureUseBatch *part) { assert(part==&host_part); return part; }
static void texture_use_part_release(void *part) { assert(part==&host_part); }
static int texture_use_part_known_unhanded(const void *part) {
    assert(part==&host_part);
    return scene_abandoned;
}
typedef RcInterval MemBackingCapture;
static RcBackingResult mem_backing_capture(uint64_t address,uint64_t bytes,MemBackingCapture **out) {
    assert(entry && address==entry->pub.addr && bytes==entry->t.L.size);
    *out=entry->image_host_backing;
    return *out ? RC_BACKING_OK : RC_BACKING_STALE;
}
static const RcInterval *mem_backing_intervals(const MemBackingCapture *capture,uint32_t *count) {
    *count=capture ? 1 : 0;
    return capture;
}
static void mem_backing_release(MemBackingCapture **capture) { *capture=NULL; }
static int mem_backing_current(const RcInterval *backing) { return backing && mapping_current; }
#ifdef TEST_HOST_COLOR
static int mem_backing_read_hold(const RcInterval *backing) { return mem_backing_current(backing); }
static void mem_read_release(void) { assert(mapping_current); }
static int fills_no_page(const Tex *e) { return e->t.L.size==128; }
enum { SNAP_MAX = 4096 };
static int wb_exact = 1;
static int cfg_bool(const char *key) { assert(!strcmp(key,"NX2NAVI_WB_EXACT")); return 1; }
enum { TEX_PRIVATE_MIRRORED_COLOR_TRANSFER = 3 };
static void mirrored_color_write_recorded(Tex *, VkTextureProjectedTransfer *);
#endif
static int mem_mapping_range_unchanged(uint64_t addr,uint64_t bytes,uint64_t stamp) {
    assert(addr==entry->pub.addr && bytes==entry->t.L.size && stamp==1);
    return mapping_current;
}
static int mem_protection_range_unchanged(uint64_t addr,uint64_t bytes,uint64_t stamp) {
    assert(addr==entry->pub.addr && bytes==entry->t.L.size && stamp==1);
    return 1;
}
static void vk_mem_observe_subresources(uint64_t addr,uint64_t bytes,const TextureSourceStamp *prior,
    uint32_t epoch,const TextureSourceSpan *spans,uint32_t count,TextureSourceObservation *out) {
    assert(addr==entry->pub.addr && bytes==entry->t.L.size && !prior && epoch==submit_epoch &&
           count==1 && !spans[0].offset && spans[0].size==bytes);
    out->stamp=(TextureSourceStamp){.mapping=1,.protection=1,.lineage=1,.generation=host_generation};
}
static int vk_mem_changed(uint64_t addr,uint64_t bytes,uint32_t since,uint32_t *observed) {
    assert(addr==entry->pub.addr && bytes==entry->t.L.size);
    *observed=host_generation;
    return since!=host_generation;
}
static RcRegistryResult vk_mem_canonical_source_qualify(const RcInterval *backing,uint64_t addr,uint64_t bytes) {
    return backing && backing->address==addr && backing->range.size==bytes ? RC_REGISTRY_OK : RC_REGISTRY_STALE;
}
static void destroy_tex(Tex *e) {
    assert(locked && e==entry);
    host_destroyed++;
#ifdef TEST_HOST_COLOR
    free(e->snap);
    e->snap=NULL;
#endif
    entry=NULL;
}
static void gpu_written_prepared_at(Tex *e,const uint32_t *generation) {
    assert(*generation==host_generation);
    gpu_written_prepared(e);
    e->pub.gpu_owned=1;
#ifdef TEST_HOST_COLOR
    if (e->pub.type==VK_IMAGE_TYPE_2D) {
        if (!e->image_authority.id) {
            assert(rc_registry_hold_begin(rc_registry_runtime(),(RcSourceRange){RC_GUEST_COMPATIBILITY_BACKING,
                e->pub.addr,e->t.L.size},RC_REGISTRY_IMAGE,&e->image_authority)==RC_REGISTRY_OK);
            rc_registry_image_hold_bind(rc_registry_runtime(),e->image_authority,e->pub.image);
        }
        e->native_write=transfer_read;
        e->gpu_gen=*generation;
        e->gpu_dirty=1;
        e->gpu_seq++;
        e->gpu_fence=e->wr_fence=transfer_read.scene;
    }
#endif
}
static void first_mirrored_write_recorded(Tex *,RcRegistryPosition,uint32_t);
static void image_alias_note_private_projection(Tex *e, const RcInterval *covered, uint32_t count, int whole) {
    assert(locked && e == entry && covered && count && (whole == 0 || whole == 1));
    ++projections;
}

typedef struct {
    const tex_t *tex;
    const VkTexImage *img;
    VkImage image, bc_backing;
    VkFormat format;
    uint32_t width, height, level, base_layer, layers, blocks, is3d, aspect, samples;
    VkResult bc_status;
} VkExecTarget;
typedef struct { const tex_t *pin; VkImage image; } VkPinnedTextureWrite;
typedef struct VkTextureAliasRead VkTextureAliasRead;
/* Target binding */
typedef struct {
    VkCommandBuffer cb;
    VkExecTarget color[8], depth;
    uint64_t fence;
    int64_t copy_stride;
    uint32_t recorded, pass_w, pass_h, pass_layers, target_binding_mask, bc_written_mask, bc_target_mask;
    VkExecTargetBinding target_bindings[9];
    struct { uint32_t width; } tonemap_extent;
    int pass_ncolor, has_render_cond, render_disabled, has_scissor, in_pass;
    p_scissor scissor;
} VkExec;
enum { VK_EXEC_GRAPHICS_UNAVAILABLE=99 };
typedef struct { int status; } VkExecPreparationFailure;
static void bc_target_refused(VkExecPreparationFailure *f,unsigned index,VkResult status) {
    (void)f; (void)index; (void)status; abort();
}
static void graphics_preparation_unavailable(VkExec *ex,uint32_t op,int status,
                                              VkExecPreparationFailure *failure,const char *message) {
    (void)ex; (void)op; (void)status; (void)failure; (void)message; abort();
}
static struct { uint32_t clears; } vk_exec_frame;
static unsigned vk_cond_clears_skip, vk_cond_clears_run, native_color_clears;
static unsigned draw_blocked, native_draws;
static VkImageSubresourceRange issued_clear;
enum { W_CLEAR_TARGET, W_CLEAR_MASK, VK_GAP_CLEAR_TARGET };

static Tex *nvn_texture_pin_attachment(const tex_t *t) { (void)t; return NULL; }
static int vk_mem_image_alias_current(VkImage image, uint64_t address, uint64_t size, void *bound,
    const RcInterval *intervals, uint32_t count) {
    (void)image; (void)address; (void)size; (void)bound; (void)intervals; (void)count;
    assert(!"unexpected bound allocation in independent-color check");
    return 0;
}
static void bound_alias_forget(Tex *e) { e->bound_alias_whole = 0; }
static void bound_alias_whole_written(Tex *e, RcRegistryPosition output) { (void)e; (void)output; abort(); }
static void clear_cond_log(const VkExec *ex, const char *kind, int slot) { (void)ex; (void)kind; (void)slot; }
static int clear_direct_attachment(VkExec *ex, unsigned slot, VkExecTarget *selected) {
    (void)ex; (void)slot; (void)selected;
    return 0; // A selected nonbase color view goes through resolve_rt.
}
static void clear_independent_target(VkExec *ex, unsigned slot) { (void)ex; (void)slot; }
static int vk_texture_deferred(void) { return 0; }
static void resolve_rt(VkExec *ex) { assert(ex->color[0].img == &entry->pub); }
static void warn_once(int code, const char *message, ...) { (void)code; (void)message; }
static void vk_dlaa_hud_write(VkImage image) { assert(image == entry->pub.image); }
static void vk_exec_barrier(VkExec *ex) { assert(!ex->in_pass); }
static void vk_gpu_gap_note(int kind, uint64_t area) { assert(kind == VK_GAP_CLEAR_TARGET && area); }
static VkPinnedTextureWrite clear_target_prepare(const VkExecTarget *target) {
    assert(target->image == entry->pub.image);
    resource_image_prepare_mapped(target->image, entry->pub.addr, entry->t.L.size,
                                  entry->image_private_mapping, entry->image_private_mapping_count);
    return (VkPinnedTextureWrite){0};
}
static void clear_target_written(const VkExecTarget *target, VkPinnedTextureWrite *capture) {
    assert(!capture->pin && target->image == entry->pub.image && native_color_clears == 1);
    lock();
    gpu_written_prepared(entry);
    unlock();
}
static void clear_attachment(VkExec *ex, const VkClearAttachment *attachment) {
    (void)ex; (void)attachment; abort();
}
VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(VkCommandBuffer cb, VkImage image, VkImageLayout layout,
    const VkClearColorValue *color, uint32_t count, const VkImageSubresourceRange *ranges) {
    assert(!cb && image == entry->pub.image && layout == VK_IMAGE_LAYOUT_GENERAL && count == 1);
    assert(color->float32[0] == 0.625f);
    issued_clear = ranges[0];
    ++native_color_clears;
}
static void vk_texture_alias_transfer_written(const tex_t *t, VkImage image, const view_t *view, const NVNcopyRegion *box) {
    (void)t; (void)image; (void)view; (void)box; abort(); // Legacy base-rectangle branch is out of scope.
}
static void vk_texture_alias_depth_clear_written(const tex_t *t, VkImage image, const NVNcopyRegion *box) {
    (void)t; (void)image; (void)box; abort();
}
static int image_alias_load_current(Tex *,const VkImageSubresourceRange *,RcRegistryPosition);
static int image_alias_placement_current(const Tex *);
static int image_alias_planes_written_locked(Tex *,const VkImageSubresourceRange *);
static int image_alias_source_current(const Tex *e) { return image_alias_placement_current(e); }
static int image_alias_whole_load_current(const Tex *e, RcRegistryPosition read) {
    return e->alias_full_count && rc_registry_native_image_covers(registry,e->pub.image,e->alias_full_demand,e->alias_full_count,read);
}
static void vk_texture_pinned_write_prepare(VkPinnedTextureWrite *capture) { (void)capture;abort(); }
static void vk_texture_pinned_write_recorded(VkPinnedTextureWrite *capture) { (void)capture;abort(); }
static const VkTexImage *vk_texture_alias_target_replacement(const VkTextureAliasRead *seed) { assert(!seed);return NULL; }
#ifdef OLD_ATTACHMENT_LOAD
static int vk_texture_alias_target_current(VkTextureAliasRead *seed,VkImage image) { assert(!seed);(void)image;return 0; }
static int vk_texture_alias_target_prepare_replacement(VkTextureAliasRead *seed) { (void)seed;abort(); }
static void vk_texture_alias_target_record_replacement(VkTextureAliasRead *seed) { (void)seed;abort(); }
#else
static int vk_texture_alias_target_load_current(VkTextureAliasRead *seed,VkImage image,const VkImageSubresourceRange *planes) {
    assert(!seed);(void)image;(void)planes;return 0;
}
static int vk_texture_alias_target_prepare_replacement(VkTextureAliasRead *seed,const VkImageSubresourceRange *planes) {
    (void)seed;(void)planes;abort();
}
static void vk_texture_alias_target_record_replacement(VkTextureAliasRead *seed,const VkImageSubresourceRange *planes) {
    (void)seed;(void)planes;abort();
}
#endif
static void vk_exec_required_graphics_unavailable(VkExec *ex,unsigned opcode,const char *reason) {
    assert(ex->in_pass && !opcode && reason);draw_blocked++;
}
static int vk_texture_render_target_load_current(VkImage image,uint64_t addr,const VkImageSubresourceRange *planes) {
    assert(image==entry->pub.image && addr==entry->pub.addr);
    return image_alias_load_current(entry,planes,transfer_read);
}
static void vk_texture_image_write_prepare(VkImage image,uint64_t addr,uint64_t size) {
    assert(image==entry->pub.image && addr==entry->pub.addr);
    resource_image_prepare_mapped(image,addr,size,entry->image_private_mapping,entry->image_private_mapping_count);
}
static int vk_texture_render_target_written(VkImage image,uint64_t addr) {
    assert(image==entry->pub.image && addr==entry->pub.addr && native_draws==1);
    lock();gpu_written_prepared(entry);unlock();return 1;
}
static int vk_texture_render_target_loaded_written(VkImage image,uint64_t addr,VkTextureAliasRead *seed,
    const VkImageSubresourceRange *planes) {
    assert(!seed && image==entry->pub.image && addr==entry->pub.addr && native_draws==1);
    lock();
    gpu_written_prepared(entry);
    const int written=image_alias_planes_written_locked(entry,planes);
    unlock();
    return written?2:1;
}

/* Production functions */

static void image_alias_possible_written(Tex *e) {
    assert(locked && e == entry && !e->owner.possible);
    if (e->image_host_backing) return;
    const RcInterval *maps = e->image_private_mapping;
    const uint32_t count = e->image_private_mapping_count;
    e->owner.possible = image_alias_project_write(e, maps, count, maps, count, (RcRegistryPosition){0}, 0);
    assert(e->owner.possible);
}
static void gpu_written_prepared(Tex *e) {
    assert(locked && e == entry);
    e->content_revision++;
    resource_image_written(e->pub.image, e->pub.addr);
    image_hold_mapped(e);
}

static void issue_at(Tex *e, RcRegistryPosition input, RcInterval *covered, uint32_t count, int whole, enum Fate fate) {
    transfer_read = input;
    const RcRegistryPosition output = {input.scene, input.ordinal + 1};
    entry = e;
    e->native_output = &e->owner;
    assert(rc_registry_declare_through(registry, transfer_read) == RC_REGISTRY_OK);
    ResourceCommandEffects *declared = resource_effects_create(output);
    VkTextureProjectedTransfer receipt = {.image=e->pub.image, .read=transfer_read,
        .mappings=e->image_private_mapping_count, .mapping=e->image_private_mapping,
        .pieces=count, .covered=covered, .whole=whole,
        .bytes=e->t.L.size ? e->t.L.size : e->image_private_mapping_count * 64};
    assert(vk_texture_projected_transfer_declare(&receipt, declared, output) == RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry, output) == RC_REGISTRY_OK);
    ResourceCommandEffects *previous = resource_effects_enter(declared);
    {
        RESOURCE_IMAGE_SCOPE(scope, input.scene);
        if (fate != UNISSUED) {
            assert(vk_texture_projected_transfer_begin(&receipt));
            vk_texture_projected_transfer_written(&receipt);
        }
    }
    resource_effects_leave(previous);
    resource_effects_release(declared);
    if (fate == UNISSUED) { assert(nq == 0); return; }
    assert(nq);
    if (fate == RECORDED_PENDING) return;
    if (fate != UNSUBMITTED)
        assert(rc_registry_handoff_through(registry, output) == RC_REGISTRY_OK);
    for (unsigned i=0; i<nq; i++) {
        if (fate == COMPLETE) q[i].done(q[i].arg);
        else q[i].cancel(q[i].arg, fate == MAY_HAVE_EXECUTED);
    }
    nq = 0;
}

static void issue(Tex *e, uint64_t scene, RcInterval *covered, uint32_t count, int whole, enum Fate fate) {
    issue_at(e, (RcRegistryPosition){scene, 2}, covered, count, whole, fate);
}

static void same_native_clear_and_copy(void) {
    registry = rc_registry_create((RcRegistryConfig){16, 24, 4});
    assert(registry);
    RcInterval storage = {START,7,{6,0,64}};
    RcInterval texels = {START,7,{6,0,48}};
    RcInterval box = {START+16,7,{6,16,16}};
    RcInterval padding = {START+48,7,{6,48,16}};
    Tex target = {.pub={.image=BASE_IMAGE,.addr=START},.image_private_mapping_count=1,.image_private_mapping=&storage};
    issue_at(&target, (RcRegistryPosition){1,2}, &texels, 1, 1, COMPLETE);
    issue_at(&target, (RcRegistryPosition){1,4}, &box, 1, 0, RECORDED_PENDING);
    RcRegistryPosition load = {1,6};
    assert(rc_registry_declare_through(registry, load) == RC_REGISTRY_OK);
    RcRegistryNativeImageFragment slices[4];
    uint32_t count = 0;
    assert(rc_registry_native_image_fragments(registry, &texels, 1, load, slices, 4, &count) == RC_REGISTRY_OK);
    assert(count == 3);
    for (uint32_t i=0; i<count; ++i) {
        assert(slices[i].binding.image == BASE_IMAGE && slices[i].interval.range.offset == i*16 &&
               slices[i].interval.range.size == 16 && slices[i].output.scene == 1 &&
               slices[i].output.ordinal == (i == 1 ? 5 : 3));
    }
    assert(rc_registry_native_image_covers(registry, BASE_IMAGE, &texels, 1, load));
    assert(!rc_registry_native_image_covers(registry, BASE_IMAGE, &padding, 1, load));
    assert(rc_registry_handoff_through(registry, (RcRegistryPosition){1,5}) == RC_REGISTRY_OK);
    for (unsigned i=0; i<nq; ++i) q[i].done(q[i].arg);
    nq = 0;
    free(target.image_mapped_holds);
    rc_registry_destroy(registry);
    puts("PASS same-image LOAD spans earlier clear and later partial transfer without claiming padding");
}

static int run(uint32_t mappings, int two_written, enum Fate fate) {
    registry = rc_registry_create((RcRegistryConfig){16, 24, 4});
    assert(registry);
    RcInterval maps[] = {{START,7,{6,0,64}},{START+64,8,{6,64,64}}};
    RcInterval written[] = {{START+16,7,{6,16,8}},{START+80,8,{6,80,8}}};
    Tex prior = {.pub={.image=BASE_IMAGE,.addr=START},.image_private_mapping_count=mappings, .image_private_mapping=maps};
    Tex patch = {.pub={.image=PATCH_IMAGE,.addr=START},.image_private_mapping_count=mappings, .image_private_mapping=maps};
    projections = 0;
    issue(&prior, 1, maps, mappings, 1, COMPLETE);
    issue(&patch, 2, written, two_written ? 2 : 1, 0, fate);
    RcRegistryPosition read = {3,2};
    assert(rc_registry_declare_through(registry, read) == RC_REGISTRY_OK);
    uint64_t expected_patch = fate == COMPLETE ? PATCH_IMAGE : BASE_IMAGE;
    if (fate != MAY_HAVE_EXECUTED)
        assert(rc_registry_native_image_covers(registry, expected_patch, written, two_written ? 2 : 1, read));
    else
        assert(!rc_registry_native_image_covers(registry, BASE_IMAGE, written, 1, read) &&
               !rc_registry_native_image_covers(registry, PATCH_IMAGE, written, 1, read));
    RcInterval first_untouched = {START, 7, {6, 0, 16}};
    RcInterval second_untouched = {START+64, 8, {6, 64, 16}};
    int within_touched_mapping = rc_registry_native_image_covers(registry, BASE_IMAGE, &first_untouched, 1, read);
    int otherwise_untouched_mapping = mappings == 1 ||
        rc_registry_native_image_covers(registry, BASE_IMAGE, &second_untouched, 1, read);
    assert(projections == 1 + (fate != UNISSUED));
    if (fate != MAY_HAVE_EXECUTED) assert(within_touched_mapping);
    const char *fates[] = {"unissued", "completed", "canceled unhanded", "possibly executed"};
    printf("%s %u mappings, %d affected, %s: touched fate correct, untouched mapping retained %d\n",
           otherwise_untouched_mapping ? "PASS" : "FAIL", mappings, two_written ? 2 : 1, fates[fate],
           otherwise_untouched_mapping);
    if (mappings == 2 && !two_written && otherwise_untouched_mapping) {
        RcRegistryNativeImageFragment fragments[5];
        uint32_t count = 0;
        assert(rc_registry_native_image_fragments(registry, &maps[1], 1, read, fragments, 5, &count) == RC_REGISTRY_OK);
        assert(count == 1 && fragments[0].binding.image == BASE_IMAGE && fragments[0].interval.range.size == 64);
        if (fate == COMPLETE) {
            // The sparse image still owns its written slice. A later actual producer can supersede that
            // slice and retire the whole sparse image without making untouched mappings foreign.
            assert(rc_registry_native_image_discard_superseded(registry, PATCH_IMAGE,
                patch.image_mapped_holds, 2) == RC_REGISTRY_BLOCKED);
            Tex replacement = {.pub={.image=REPLACEMENT_IMAGE,.addr=START},.image_private_mapping_count=2,.image_private_mapping=maps};
            issue(&replacement, 4, written, 1, 0, COMPLETE);
            assert(rc_registry_native_image_discard_superseded(registry, PATCH_IMAGE,
                patch.image_mapped_holds, 2) == RC_REGISTRY_OK);
            read = (RcRegistryPosition){5,2};
            assert(rc_registry_declare_through(registry, read) == RC_REGISTRY_OK);
            assert(rc_registry_native_image_covers(registry, REPLACEMENT_IMAGE, written, 1, read));
            assert(rc_registry_native_image_covers(registry, BASE_IMAGE, &maps[1], 1, read));
            puts("PASS sparse output retires after its only written slice is superseded");
            free(replacement.image_mapped_holds);
        }
    }
    free(prior.image_mapped_holds);
    free(patch.image_mapped_holds);
    rc_registry_destroy(registry);
    return otherwise_untouched_mapping;
}

static void contiguous_provider_distinct_mappings(void) {
    RcBackingStore *store = rc_backing_store_create((RcBackingConfig){4,4,8,8,16});
    assert(store);
    RcBackingId backing;
    RcBackingMapId map;
    assert(rc_backing_create(store, 128, 901, &backing) == RC_BACKING_OK);
    assert(rc_backing_map_create(store, START, 128, &map) == RC_BACKING_OK);
    assert(rc_backing_map_replace(store, map, START, 64, backing, 0) == RC_BACKING_OK);
    assert(rc_backing_map_replace(store, map, START+64, 64, backing, 64) == RC_BACKING_OK);
    RcBackingRun run;
    uint32_t count;
    assert(rc_backing_map_runs(store, map, START, &run, 1, &count) == RC_BACKING_OK);
    assert(count == 1 && run.provider == 901 && run.range.offset == 0 && run.range.size == 128);
    RcBackingCapture *capture = NULL;
    assert(rc_backing_capture(store, map, START, 128, RC_BACKING_REQUIRE_MAPPED, &capture) == RC_BACKING_OK);
    const RcBackingSegment *segments = rc_backing_capture_segments(capture, &count);
    assert(count == 2 && segments[0].mapping != segments[1].mapping && segments[1].provider == 901);
    rc_backing_capture_release(&capture);
    rc_backing_store_destroy(store);
    puts("PASS one affine native run contains two distinct normalized mappings");
}
static int byte_in(const RcInterval *intervals,uint32_t count,uint64_t offset) {
    for (uint32_t i=0;i<count;i++) {
        const RcSourceRange r=intervals[i].range;
        if (offset>=r.offset && offset-r.offset<r.size) return 1;
    }
    return 0;
}

uint64_t nvn_bl_offset(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,unsigned,unsigned);
int nvn_deswizzle_geom(const nvn_texdesc *,unsigned,unsigned,uint32_t *,uint32_t *,uint32_t *,
                       uint64_t *,unsigned *,unsigned *,uint32_t *);
static int volume_warnings, volume_refusals, volume_issues;
static enum { B2T_OP = 912 } const volume_opcode = B2T_OP;
static uint32_t max1(uint32_t n) { return n ? n : 1; }
static void report_volume_warning(int kind,const char *format,...) { assert(kind >= 0 && format); volume_warnings++; }
typedef struct { VkFormat vk, ideal; int astc_bc, astc_bw, expand_rgb, depth, stencil; } TexVkFormat;
static int tex_vk_format(int format,TexVkFormat *f) {
    if (format==NVN_FORMAT_RGBA8SN) {
        *f=(TexVkFormat){.vk=VK_FORMAT_R8G8B8A8_SNORM,.ideal=VK_FORMAT_R8G8B8A8_SNORM};
        return 1;
    }
    if (format==NVN_FORMAT_RGBA32F) {
        *f=(TexVkFormat){.vk=VK_FORMAT_R32G32B32A32_SFLOAT,.ideal=VK_FORMAT_R32G32B32A32_SFLOAT};
        return 1;
    }
    *f=(TexVkFormat){.vk=VK_FORMAT_R8_UNORM,.ideal=VK_FORMAT_R8_UNORM};
    return format==NVN_FORMAT_R8;
}
typedef struct { void *alloc; } VkMemSpan;
static int vk_mem_lookup(uint64_t address,uint64_t size,VkMemSpan *span) {
    assert(entry && address==entry->pub.addr && size==entry->t.L.size);
    span->alloc=entry->alloc;
    return 1;
}
static int vk_texture_deferred_take(void) { return 0; }
static int tex_mirrored(const tex_t *t) { return t && entry && t->serial==entry->t.serial && entry->pub.mirrored; }
static Tex *entry_of_mode(const tex_t *t,int private_output, uint32_t *generation) {
    assert(private_output && !generation && locked && t->mem);
    return entry;
}
static const VkTexImage *get_texture_write_mode(const tex_t *t,int prepare,int private_output) {
    assert(locked && t->serial==entry->t.serial && !prepare && private_output);
    return &entry->pub;
}
static int vk_texture_transfer_write_geometry(const tex_t *t,VkTexImage *out) {
    assert(t->serial==entry->t.serial);
    *out=entry->pub; out->image=0;
    return 1;
}
#define OP_COPY_B2T volume_opcode
enum { W_FORMAT=100,W_REGION,VK_GAP_COPY_B2T };
typedef struct {
    VkBufferImageCopy region;
    void *source_hold;
    RcRegistryPosition read;
    uint64_t size;
} VkExecBufferToTexture;
typedef struct {
    VkTextureProjectedTransfer *target;
    VkExecBufferToTexture buffer;
} VkExecProjectedTransfer;
static int buffer_to_texture_capture_source(VkExecBufferToTexture *,uint64_t,const BufferImagePitch *,const Sub *);
static int buffer_to_texture_ready(VkExec *ex,VkExecBufferToTexture *copy) {
    return copy->source_hold && ex->fence==copy->read.scene &&
           !rc_registry_position_compare(copy->read,transfer_read);
}
static void buffer_to_texture_issue(VkExec *ex,VkExecBufferToTexture *copy,VkImage native) {
    const VkBufferImageCopy *region=&copy->region;
    assert(locked && ex->fence==2 && native==REPLACEMENT_IMAGE && copy->source_hold);
    if (entry->image_host_backing) {
        assert(region->imageSubresource.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT && !region->imageSubresource.mipLevel &&
            !region->imageSubresource.baseArrayLayer && region->imageSubresource.layerCount==1);
        assert(!region->imageOffset.x && !region->imageOffset.y && !region->imageOffset.z);
        assert(region->imageExtent.width==32 && region->imageExtent.height==32 && region->imageExtent.depth==32);
        assert(region->bufferRowLength==32 && region->bufferImageHeight==32 && copy->size==0x20000);
        copy->source_hold=NULL;
        ++volume_issues;
        return;
    }
    assert(region->imageSubresource.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT && region->imageSubresource.mipLevel==1 &&
        !region->imageSubresource.baseArrayLayer && region->imageSubresource.layerCount==1);
    assert(region->imageOffset.x==2 && region->imageOffset.y==1 && region->imageOffset.z==1);
    assert(region->imageExtent.width==4 && region->imageExtent.height==3 && region->imageExtent.depth==2);
    assert(region->bufferRowLength==9 && region->bufferImageHeight==3 && copy->size==49);
    copy->source_hold=NULL;
    ++volume_issues;
}
static int b2t_probe_on,b2t_probe_copies;
static void report_transfer_gap(int kind,uint64_t area) { assert(kind==VK_GAP_COPY_B2T && (area==12 || area==1024)); }
static void report_transfer_failure(VkExec *ex,uint32_t opcode,const char *reason) {
    assert(ex->fence==2 && opcode==OP_COPY_B2T && reason);
    ++volume_refusals;
}

/* VOLUME PRODUCTION */
/* HOST PART PRODUCTION */

static int buffer_to_texture_capture_source(VkExecBufferToTexture *copy,uint64_t src,const BufferImagePitch *pitch,
                                             const Sub *dst) {
    if (entry->image_host_backing) {
        assert(src==0xfab0 && pitch->stride==128 && pitch->row_bytes==128 && pitch->rows==1024 &&
               dst->img->type==VK_IMAGE_TYPE_3D && !dst->level && !dst->z && dst->nz==32);
        copy->source_hold=copy;
        copy->read=transfer_read;
        copy->size=pitch->size;
        return 1;
    }
    assert(src==0xffee && pitch->stride==9 && pitch->row_bytes==4 && pitch->rows==6 &&
           dst->img->type==VK_IMAGE_TYPE_3D && dst->level==1 && dst->z==1 && dst->nz==2);
    copy->source_hold=copy;
    copy->size=pitch->size;
    copy->read=transfer_read;
    return 1;
}

static void volume_b2t(enum Fate fate,int remap) {
    Tex old={.pub={.image=BASE_IMAGE,.addr=START,.format=VK_FORMAT_R8_UNORM,
        .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.type=VK_IMAGE_TYPE_3D,.view_type=VK_IMAGE_VIEW_TYPE_3D,
        .usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,.width=19,.height=11,.depth=9,
        .levels=3,.layers=1,.samples=1,.scale=1},.f={.vk=VK_FORMAT_R8_UNORM},
        .t={.serial=36,.mem=(void *)(uintptr_t)START,.d={.target=NVN_TEXTURE_TARGET_3D,.format=NVN_FORMAT_R8,
             .width=19,.height=11,.depth=9,.levels=3}}};
    nvn_tex_layout(&old.t.d,&old.t.L);
    old.alloc=&old;
    const uint64_t bytes=old.t.L.size;
    uint32_t rb,rows,zs,bpb;
    uint64_t base;
    unsigned th,td;
    assert(!nvn_deswizzle_geom(&old.t.d,1,0,&rb,&rows,&zs,&base,&th,&td,&bpb));
    assert(td>0 && zs==4 && rb==9 && rows==5 && bpb==1);
    const uint64_t first=base+nvn_bl_offset(2,1,1,rb,rows,th,td);
    const uint64_t boundary=first+2;
    RcInterval normalized[]={{START,17,{6,0,boundary}},{START+boundary,18,{6,boundary,bytes-boundary}}};
    old.image_private_mapping=normalized;
    old.image_private_mapping_count=2;
    Tex writer=old;
    writer.pub.image=REPLACEMENT_IMAGE;
    writer.alloc=&writer;
    assert(nq==0 && !image_hold.readers);
    registry=rc_registry_create((RcRegistryConfig){16,48,4});
    assert(registry);
    VkTextureAliasLayout layout;
    RcInterval full[IMAGE_ALIAS_MAX_WRITE_FRAGMENTS];
    uint32_t entire=0;
    assert(texture_alias_layout(&old.t.d,&old.pub,&layout) &&
           texture_alias_demand(&old.t.d,&layout,normalized,2,START,bytes,full,IMAGE_ALIAS_MAX_WRITE_FRAGMENTS,&entire));
    issue_at(&old,(RcRegistryPosition){1,2},full,entire,1,COMPLETE);
    entry=&writer;
    writer.native_output=&writer.owner;
    const p_copy_b2t command={.src=0xffee,.tex=&writer.t,.r={2,1,1,4,3,2},.has_view=1,
        .view={.set_target=1,.target=NVN_TEXTURE_TARGET_3D,.set_levels=1,.base_level=1,.nlevels=1,
            .set_format=1,.format=NVN_FORMAT_R8}};
    p_copy_b2t invalid=command;
    invalid.view.set_layers=1;
    VkExec ex={.copy_stride=9,.fence=2};
    VkExecProjectedTransfer plan={0};
    transfer_read=(RcRegistryPosition){2,2};
    const RcRegistryPosition output={2,3}, consume={2,4};
    assert(rc_registry_declare_through(registry,transfer_read)==RC_REGISTRY_OK);
    const int before=volume_issues, refused=volume_refusals, warned=volume_warnings;
    assert(!projected_b2t_capture(&ex,&invalid,&plan) && !plan.target);
    invalid=command;
    invalid.r.zoffset=3; // Mip 1 has Z=[0,4); two slices at 3 are not legal.
    assert(projected_b2t_capture(&ex,&invalid,&plan)==1 && !plan.target && !plan.buffer.source_hold);
    assert(volume_warnings==warned+1);
    assert(projected_b2t_capture(&ex,&command,&plan)==1 && plan.target && image_hold.readers==1);
    VkTextureProjectedTransfer *issued_owner=plan.target;
    assert(!issued_owner->whole && issued_owner->pieces>=6 && issued_owner->mappings==2);
    uint64_t written=0;
    uint8_t *expected=calloc(1,bytes);
    assert(expected);
    for (uint32_t z=1;z<=2;z++)
        for (uint32_t y=1;y<=3;y++)
            for (uint32_t x=2;x<=5;x++) expected[base+nvn_bl_offset(x,y,z,rb,rows,th,td)]=1;
    for (uint32_t i=0;i<issued_owner->pieces;i++) {
        const RcSourceRange span=issued_owner->covered[i].range;
        written+=span.size;
        for (uint64_t b=span.offset;b<span.offset+span.size;b++) assert(expected[b]);
    }
    assert(written==24);
    ResourceCommandEffects *effects=resource_effects_create(output);
    assert(vk_texture_projected_transfer_declare(issued_owner,effects,output)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,output)==RC_REGISTRY_OK);
    ResourceCommandEffects *previous=resource_effects_enter(effects);
    {
        RESOURCE_IMAGE_SCOPE(scope,2);
        if (remap) mapping_current=0;
        if (fate!=UNISSUED || remap)
            projected_b2t_record(&ex,&plan,REPLACEMENT_IMAGE);
        mapping_current=1;
    }
    resource_effects_leave(previous);
    resource_effects_release(effects);
    if (plan.target) {
        assert(fate==UNISSUED && nq==0);
        vk_texture_projected_transfer_cancel(&plan.target);
        assert(volume_issues==before && volume_refusals==refused+remap);
    } else {
        assert(nq>=1 && q[0].arg==issued_owner && image_hold.readers==1);
        assert(volume_issues==before+1 && volume_refusals==refused);
        if (fate==COMPLETE || fate==MAY_HAVE_EXECUTED)
            assert(rc_registry_handoff_through(registry,output)==RC_REGISTRY_OK);
        for (uint32_t i=0;i<nq;i++) {
            if (fate==COMPLETE) q[i].done(q[i].arg);
            else q[i].cancel(q[i].arg,fate==MAY_HAVE_EXECUTED);
        }
        nq=0;
    }
    assert(!plan.target && !image_hold.readers);
    assert(rc_registry_declare_through(registry,consume)==RC_REGISTRY_OK);
    transfer_read=consume;
    const VkImageSubresourceLayers plane={VK_IMAGE_ASPECT_COLOR_BIT,1,0,1};
    assert(vk_texture_transfer_read_prove(&writer.t,&writer.pub,plane,&command.r)==(fate==COMPLETE ? 1 : -1));
    NVNcopyRegion untouched=command.r;
    untouched.zoffset=0; untouched.depth=1;
    assert(vk_texture_transfer_read_prove(&writer.t,&writer.pub,plane,&untouched)<0);
    const RcInterval tested={START+first,17,{6,first,1}};
    if (fate==MAY_HAVE_EXECUTED) {
        assert(!rc_registry_native_image_covers(registry,REPLACEMENT_IMAGE,&tested,1,consume));
    } else {
        for (uint64_t i=0;i<bytes;i++) {
            const RcInterval alias={START+0x20000+i,27,{6,i,1}};
            RcRegistryNativeImageFragment got;
            uint32_t count=0;
            const RcRegistryResult result=rc_registry_native_image_fragments(registry,&alias,1,consume,&got,1,&count);
            if (byte_in(full,entire,i)) {
                const int later=fate==COMPLETE && expected[i];
                assert(result==RC_REGISTRY_OK && count==1 && got.binding.image==(later?REPLACEMENT_IMAGE:BASE_IMAGE));
                assert(got.output.scene==(uint64_t)(later?2:1));
            } else assert(result==RC_REGISTRY_BLOCKED && !count);
        }
    }
    printf("PASS B2T original mip 1, Z [1,3), pitched 24 meaningful bytes, fate %d, remap %d\n",fate,remap);
    free(expected); free(writer.image_mapped_holds); free(old.image_mapped_holds);
    rc_registry_destroy(registry);
}


static void volume_storage_projection(enum Fate fate, int stale) {
    enum { W = 19, H = 11, D = 9, LEVELS = 3, GUARD = 32, MARK = 0xd7 };
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    Tex old = {.pub = {.image = BASE_IMAGE, .addr = START, .format = VK_FORMAT_R8_UNORM,
        .type = VK_IMAGE_TYPE_3D, .view_type = VK_IMAGE_VIEW_TYPE_3D, .usage = usage,
        .aspect = VK_IMAGE_ASPECT_COLOR_BIT, .width = W, .height = H, .depth = D, .levels = LEVELS,
        .layers = 1, .samples = 1, .scale = 1}, .f = {.vk = VK_FORMAT_R8_UNORM},
        .t = {.serial = 35, .mem = (void *)(uintptr_t)START, .d = {.target = NVN_TEXTURE_TARGET_3D,
              .format = NVN_FORMAT_R8, .width = W, .height = H, .depth = D, .levels = LEVELS}}};
    nvn_tex_layout(&old.t.d, &old.t.L);
    assert(old.t.L.tileD > 0 && storage_volume_shape(&old));
    const uint64_t bytes = old.t.L.size;
    VkTextureAliasLayout full, selected;
    assert(texture_alias_layout(&old.t.d, &old.pub, &full));
    selected = full;
    const view_t view = {.set_target = 1, .target = NVN_TEXTURE_TARGET_3D, .set_levels = 1, .base_level = 1, .nlevels = 1};
    VkImageSubresourceRange mip_range;
    assert(storage_target_volume_view(&old, &view, 1, &mip_range));
    assert(mip_range.baseMipLevel == 1 && mip_range.levelCount == 1 && !mip_range.baseArrayLayer && mip_range.layerCount == 1);
    view_t slice = view;
    slice.set_layers = 1;
    slice.min_layer = 2;
    slice.nlayers = 1;
    assert(!storage_target_volume_view(&old, &slice, 1, &mip_range));
    Tex flat = old;
    flat.pub.type = VK_IMAGE_TYPE_2D;
    assert(!storage_volume_shape(&flat));
    flat = old;
    flat.pub.depth--;
    assert(!storage_volume_shape(&flat));
    assert(texture_alias_clear_planes(mip_range, &old.pub, &selected) && selected.count == 1);
    const VkTextureAliasPlane *plane = &selected.planes[0];
    assert(plane->copy.imageExtent.width == W/2 && plane->copy.imageExtent.height == H/2 &&
           plane->copy.imageExtent.depth == D/2 && plane->slices == D/2 &&
           plane->copy.imageSubresource.baseArrayLayer == 0 && plane->copy.imageSubresource.layerCount == 1);
    const uint64_t begin = plane->copy.bufferOffset;

    const uint64_t boundary = nvn_tex_level_offset(&old.t.d, 1) + 4;
    assert(boundary < bytes);
    RcInterval guest_mapping[] = {{START, 17, {6, GUARD, boundary}},
        {START + boundary, 18, {6, GUARD + boundary, bytes - boundary}}};
    RcInterval original[IMAGE_ALIAS_MAX_WRITE_FRAGMENTS], demand[IMAGE_ALIAS_MAX_WRITE_FRAGMENTS];
    uint32_t original_count, demand_count;
    assert(texture_alias_demand(&old.t.d, &full, guest_mapping, 2, START, bytes, original,
                                IMAGE_ALIAS_MAX_WRITE_FRAGMENTS, &original_count));
    assert(texture_alias_demand(&old.t.d, &selected, guest_mapping, 2, START, bytes, demand,
                                IMAGE_ALIAS_MAX_WRITE_FRAGMENTS, &demand_count));
    assert(demand_count > 2 && demand[0].mapping == 17 && demand[demand_count - 1].mapping == 18);
    uint64_t meaningful = 0;
    for (uint32_t i = 0; i < demand_count; ++i) meaningful += demand[i].range.size;
    assert(meaningful == (W/2)*(H/2)*(D/2));

    uint8_t *linear = malloc(full.bytes), *donor = malloc(bytes + 2*GUARD), *seed = malloc(bytes + 2*GUARD);
    uint8_t *upload = malloc(selected.bytes + GUARD);
    VkTextureAliasSlice *slices = calloc(demand_count, sizeof *slices);
    assert(linear && donor && seed && upload && slices);
    memset(linear, MARK, full.bytes);
    memset(donor, MARK, bytes + 2*GUARD);
    memset(seed, MARK, bytes + 2*GUARD);
    memset(upload, MARK, selected.bytes + GUARD);
    for (uint32_t z = 0; z < plane->slices; ++z)
        for (uint32_t y = 0; y < plane->rows; ++y)
            for (uint32_t x = 0; x < plane->row_bytes; ++x)
                linear[begin + ((uint64_t)z * plane->rows + y) * plane->row_bytes + x] = (uint8_t)(z*49 + y*9 + x);
    assert(texture_alias_project(&old.t.d, &full, donor + GUARD, linear, 0));
    for (uint32_t i = 0; i < demand_count; ++i) slices[i] = (VkTextureAliasSlice){.demand = i, .source = 0,
        .covered = demand[i].range, .writer_address = demand[i].address};
    const VkTextureAliasBytes source = {donor + GUARD, START, bytes};
    assert(texture_alias_compose(&source, 1, seed + GUARD, START, bytes, demand, demand_count, slices, demand_count));
    assert(texture_alias_project(&old.t.d, &selected, seed + GUARD, upload, 1));
    assert(!memcmp(upload + begin, linear + begin, meaningful));
    for (uint64_t i = 0; i < begin; ++i) assert(upload[i] == MARK);
    for (uint32_t i = 0; i < GUARD; ++i)
        assert(seed[i] == MARK && seed[GUARD + bytes + i] == MARK && upload[selected.bytes + i] == MARK);
    for (uint64_t i = 0; i < bytes; ++i)
        if (!byte_in(demand, demand_count, GUARD + i)) assert(seed[GUARD + i] == MARK);

    registry = rc_registry_create((RcRegistryConfig){16, 48, 4});
    assert(registry);
    old.alloc = &old;
    old.image_private_mapping = guest_mapping;
    old.image_private_mapping_count = 2;
    Tex writer = old;
    writer.pub.image = REPLACEMENT_IMAGE;
    writer.alloc = &writer;
    issue_at(&old, (RcRegistryPosition){1, 2}, original, original_count, 1, COMPLETE);
    transfer_read = (RcRegistryPosition){2, 2};
    const RcRegistryPosition output = {2, 3}, next = {2, 4};
    assert(rc_registry_declare_through(registry, transfer_read) == RC_REGISTRY_OK);
    assert(rc_registry_native_image_covers(registry, BASE_IMAGE, demand, demand_count, transfer_read));
    ResourceCommandEffects *effects = resource_effects_create(output);
    assert(resource_effects_add_pinned_image(effects, writer.pub.image, guest_mapping, 2) == RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry, output) == RC_REGISTRY_OK);
    ResourceCommandEffects *previous = resource_effects_enter(effects);
    entry = &writer;
    writer.native_output = &writer.owner;
    const int old_projections = projections;
    {
        RESOURCE_IMAGE_SCOPE(scope, 2);
        resource_image_prepare_pinned(writer.pub.image, START, bytes, guest_mapping, 2, 0);
        lock();
        assert(!image_alias_planes_written_locked(&writer, &mip_range)); // Graphics clears keep their original 2D adapter.
        if (stale) {
            mapping_current = 0;
            assert(!image_alias_private_planes_written_locked(&writer, &mip_range, output));
            mapping_current = 1;
        } else {
            gpu_written_prepared(&writer);
            assert(image_alias_private_planes_written_locked(&writer, &mip_range, output));
        }
        unlock();
    }
    resource_effects_leave(previous);
    resource_effects_release(effects);
    assert(projections - old_projections == !stale);
    if (fate == COMPLETE && !stale)
        assert(rc_registry_handoff_through(registry, output) == RC_REGISTRY_OK);
    for (uint32_t i = 0; i < nq; ++i) {
        if (fate == COMPLETE && !stale) q[i].done(q[i].arg);
        else q[i].cancel(q[i].arg, 0);
    }
    nq = 0;
    assert(rc_registry_declare_through(registry, next) == RC_REGISTRY_OK);
    for (uint64_t i = 0; i < bytes; ++i) {
        const uint32_t map = i >= boundary;
        const RcInterval byte = {START + i, guest_mapping[map].mapping, {6, GUARD + i, 1}};
        RcRegistryNativeImageFragment fragment;
        uint32_t count;
        const RcRegistryResult result = rc_registry_native_image_fragments(registry, &byte, 1, next, &fragment, 1, &count);
        if (byte_in(original, original_count, GUARD + i)) {
            const int new_mip = fate == COMPLETE && !stale && byte_in(demand, demand_count, GUARD + i);
            assert(result == RC_REGISTRY_OK && count == 1 && fragment.binding.image == (new_mip ? REPLACEMENT_IMAGE : BASE_IMAGE));
            assert(fragment.output.scene == (uint64_t)(new_mip ? 2 : 1));
        } else assert(result == RC_REGISTRY_BLOCKED && !count);
    }
    free(writer.alias_plane_demand);
    free(writer.image_mapped_holds);
    free(old.image_mapped_holds);
    free(slices);
    free(upload);
    free(seed);
    free(donor);
    free(linear);
    rc_registry_destroy(registry);
    printf("PASS selected 3D mip, %u distinct Z slices, %llu normalized texels, canceled=%d stale=%d\n",
           plane->slices, (unsigned long long)meaningful, fate == UNSUBMITTED, stale);
}

static void color_after_transfer(enum Fate fate, int remap, int other_view, int prior_clear, int draw) {
    enum { W = 16, H = 8, LEVELS = 2, LAYERS = 3 };
    registry = rc_registry_create((RcRegistryConfig){16, 48, 4});
    assert(registry);
    Tex target = {.pub = {.image = BASE_IMAGE, .addr = START, .width = W, .height = H, .depth = 1,
        .levels = LEVELS, .layers = LAYERS, .type = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
        .aspect = VK_IMAGE_ASPECT_COLOR_BIT, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT},
        .t = {.serial = 4, .mem = (void *)(uintptr_t)START, .d = {.target = NVN_TEXTURE_TARGET_2D_ARRAY,
              .format = NVN_FORMAT_RGBA8, .width = W, .height = H, .depth = LAYERS, .levels = LEVELS}}};
    target.alloc = &target;
    target.pub.preserve_pool_bytes = draw;
    nvn_tex_layout(&target.t.d, &target.t.L);
    const uint64_t bytes = target.t.L.size;
    RcBackingId before, after;
    clear_backings = rc_backing_store_create((RcBackingConfig){4, 4, 8, 8, 16});
    assert(clear_backings);
    assert(rc_backing_create(clear_backings, bytes, 901, &before) == RC_BACKING_OK);
    assert(rc_backing_create(clear_backings, bytes, 902, &after) == RC_BACKING_OK);
    assert(rc_backing_map_create(clear_backings, START, bytes, &clear_mapping) == RC_BACKING_OK);
    assert(rc_backing_map_replace(clear_backings, clear_mapping, START, bytes, before, 0) == RC_BACKING_OK);
    RcInterval mapping;
    uint32_t count;
    assert(rc_backing_map_intervals(clear_backings, clear_mapping, START, bytes, &mapping, 1, &count) == RC_BACKING_OK);
    assert(count == 1);
    target.image_private_mapping = &mapping;
    target.image_private_mapping_count = 1;
    VkTextureAliasLayout layout;
    assert(texture_alias_layout(&target.t.d, &target.pub, &layout));
    RcInterval original[IMAGE_ALIAS_MAX_WRITE_FRAGMENTS], selected[IMAGE_ALIAS_MAX_WRITE_FRAGMENTS];
    uint32_t original_count, selected_count;
    assert(texture_alias_demand(&target.t.d, &layout, &mapping, 1, START, bytes, original,
                              IMAGE_ALIAS_MAX_WRITE_FRAGMENTS, &original_count));
    const VkImageSubresourceRange planes = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 1, draw ? 1u : 2u};
    const VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, LEVELS, 0, LAYERS};
    assert(texture_alias_clear_planes(planes, &target.pub, &layout));
    assert(texture_alias_demand(&target.t.d, &layout, &mapping, 1, START, bytes, selected,
                              IMAGE_ALIAS_MAX_WRITE_FRAGMENTS, &selected_count));
    if (prior_clear) {
        int inversion = 0;
        for (uint32_t i = 1; i < original_count; i++)
            inversion |= original[i - 1].range.offset > original[i].range.offset;
        assert(inversion); // Native plane order differs from the guest byte order on this shape.
        entry = &target;
        target.native_output = &target.owner;
        transfer_read = (RcRegistryPosition){1, 2};
        assert(rc_registry_declare_through(registry, transfer_read) == RC_REGISTRY_OK);
        ResourceCommandEffects *transfer = resource_effects_create((RcRegistryPosition){1, 3});
        assert(resource_effects_add_intervals(transfer, &mapping, 1, &mapping, 1, RC_REGISTRY_IMAGE) == RC_REGISTRY_OK);
        assert(rc_registry_declare_through(registry, (RcRegistryPosition){1, 3}) == RC_REGISTRY_OK);
        ResourceCommandEffects *outer = resource_effects_enter(transfer);
        {
            RESOURCE_IMAGE_SCOPE(scope, 1);
            resource_image_prepare_mapped(BASE_IMAGE, START, bytes, &mapping, 1);
            VkClearColorValue color = {.float32 = {0.625f}};
            vkCmdClearColorImage(NULL, BASE_IMAGE, VK_IMAGE_LAYOUT_GENERAL, &color, 1, &all);
            lock();
            gpu_written_prepared(&target);
            unlock();
            vk_texture_alias_clear_written(&target.t, BASE_IMAGE, &all);
        }
        resource_effects_leave(outer);
        resource_effects_release(transfer);
        assert(rc_registry_handoff_through(registry, (RcRegistryPosition){1, 3}) == RC_REGISTRY_OK);
        for (unsigned i = 0; i < nq; i++) q[i].done(q[i].arg);
        nq = 0;
    } else {
        projected_transfer_union(original, &original_count);
        issue_at(&target, (RcRegistryPosition){1, 2}, original, original_count, 1, COMPLETE);
    }
    const int prior_projections = projections;
    transfer_read = (RcRegistryPosition){1, 4};
    assert(rc_registry_declare_through(registry, transfer_read) == RC_REGISTRY_OK);
    if (!rc_registry_native_image_covers(registry, BASE_IMAGE, original, original_count, transfer_read)) {
        fputs("FAIL complete prior image planes lost their original coverage\n", stderr);
        abort();
    }

    const RcRegistryPosition output = {1, 5}, read = {1, 6};
    ResourceCommandEffects *effect = resource_effects_create(output);
    assert(resource_effects_add_intervals(effect, &mapping, 1, &mapping, 1, RC_REGISTRY_IMAGE) == RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry, output) == RC_REGISTRY_OK);
    if (remap) assert(rc_backing_map_replace(clear_backings, clear_mapping, START, bytes, after, 0) == RC_BACKING_OK);
    if (remap == 2) assert(rc_backing_map_replace(clear_backings, clear_mapping, START, bytes, before, 0) == RC_BACKING_OK);
    ResourceCommandEffects *previous = resource_effects_enter(effect);
    VkExec ex = {.fence = 1, .pass_w = W / 2, .pass_h = H / 2, .pass_ncolor = 1, .pass_layers = 1,
                 .render_disabled = fate == UNISSUED, .has_render_cond = 1,
                 .color = {{.tex = &target.t, .img = &target.pub, .image = BASE_IMAGE,
                    .width = W / 2, .height = H / 2, .level = 1, .base_layer = 1, .layers = 2,
                    .aspect = VK_IMAGE_ASPECT_COLOR_BIT, .samples = 1,
                    .format = other_view ? VK_FORMAT_R8G8B8A8_UINT : target.pub.format}}};
    native_color_clears = native_draws = draw_blocked = 0;
    RcInterval drawn[16];
    uint32_t drawn_count = 0;
    uint8_t *physical = NULL;
    if (draw) {
        physical = malloc(bytes);
        assert(physical);
        for (uint64_t i = 0; i < bytes; i++) physical[i] = byte_in(original,original_count,i) ? (uint8_t)(i % 151 + 1) : 0xd6;
        const NVNcopyRegion clipped = {1,1,0,3,2,1};
        assert(texture_alias_demand_box_plane(&target.t.d,&clipped,4,1,1,&mapping,1,START,bytes,drawn,16,&drawn_count));
        if (fate != UNISSUED) {
            ex.in_pass = ex.target_binding_mask = 1;
            ex.target_bindings[0] = (VkExecTargetBinding){.binding = {.image = BASE_IMAGE},.range = {1,START,bytes}};
            RESOURCE_IMAGE_SCOPE(scope, 1);
            vk_exec_targets_prepare(&ex);
            if (!draw_blocked) {
                native_draws++;
                // Model a conditional raster write to a small part of the selected initialized plane.
                if (fate == COMPLETE)
                    for (uint32_t i = 0; i < drawn_count; i++) memset(physical + drawn[i].range.offset,0xef,drawn[i].range.size);
                resource_effects_emitted();
                vk_exec_targets_written(&ex);
            }
        }
    } else clear_color(&ex, &(p_clear_color){.mask = 0xf, .rgba = {0.625f}});
    resource_effects_leave(previous);
    resource_effects_release(effect);
    if (draw) assert(draw_blocked == (unsigned)(remap != 0) && native_draws == (unsigned)(fate != UNISSUED && !remap));
    else assert(native_color_clears == (fate != UNISSUED) && ex.recorded == native_color_clears);
    if (native_color_clears) {
        assert(issued_clear.aspectMask == planes.aspectMask && issued_clear.baseMipLevel == planes.baseMipLevel &&
               issued_clear.levelCount == planes.levelCount && issued_clear.baseArrayLayer == planes.baseArrayLayer &&
               issued_clear.layerCount == planes.layerCount);
    }
    const int qualifying = fate != UNISSUED && !remap && !other_view;
    if (projections - prior_projections != qualifying) {
        fputs(draw ? "FAIL conditional draw lost selected prior color plane\n" :
            "FAIL issued mip1/layers1+2 not handed to exact image coverage\n", stderr);
        abort();
    }
    if (fate == COMPLETE) assert(rc_registry_handoff_through(registry, output) == RC_REGISTRY_OK);
    for (unsigned i = 0; i < nq; i++) {
        if (fate == COMPLETE) q[i].done(q[i].arg);
        else { assert(fate == UNSUBMITTED); q[i].cancel(q[i].arg, 0); }
    }
    nq = 0;
    assert(rc_registry_declare_through(registry, read) == RC_REGISTRY_OK);
    if (remap) {
        RcInterval new_mapping;
        assert(rc_backing_map_intervals(clear_backings, clear_mapping, START, bytes, &new_mapping, 1, &count) == RC_BACKING_OK);
        assert(count == 1 && new_mapping.mapping != mapping.mapping);
        assert((new_mapping.range.backing == mapping.range.backing) == (remap == 2));
        new_mapping.address = selected[0].address;
        new_mapping.range.offset = selected[0].range.offset;
        new_mapping.range.size = 1;
        if (draw && remap == 2) {
            RcRegistryNativeImageFragment fragment;
            assert(rc_registry_native_image_fragments(registry,&new_mapping,1,read,&fragment,1,&count)==RC_REGISTRY_OK);
            assert(count==1 && fragment.binding.image==BASE_IMAGE && fragment.output.ordinal==3);
        } else assert(!rc_registry_native_image_covers(registry, BASE_IMAGE, &new_mapping, 1, read));
    }
    if (!remap && !other_view) {
        uint64_t meaningful = 0, newer = 0, padding = 0;
        for (uint64_t i = 0; i < bytes; i++) {
            const RcInterval byte = {START + i, mapping.mapping, {mapping.range.backing, i, 1}};
            const int had_prior = byte_in(original, original_count, i);
            const int changed = fate == COMPLETE && byte_in(selected, selected_count, i);
            RcRegistryNativeImageFragment fragment;
            const RcRegistryResult result = rc_registry_native_image_fragments(registry, &byte, 1, read,
                                                                             &fragment, 1, &count);
            if (had_prior) {
                assert(result == RC_REGISTRY_OK && count == 1 && fragment.binding.image == BASE_IMAGE);
                assert(fragment.output.scene == 1 && fragment.output.ordinal == (changed ? 5 : 3));
                if (draw) assert(physical[i] == (fate==COMPLETE && byte_in(drawn,drawn_count,i) ? 0xef : i % 151 + 1));
                meaningful++;
                newer += changed;
            } else {
                assert(!changed && result == RC_REGISTRY_BLOCKED && !count);
                if (draw) assert(physical[i]==0xd6);
                padding++;
            }
        }
        assert(meaningful == LAYERS * 4 * (W * H + W * H / 4) && padding);
        assert(newer == (fate == COMPLETE ? (draw ? 1 : 2) * 4 * W * H / 4 : 0));
    }
    free(physical);
    free(target.alias_plane_demand);
    free(target.image_mapped_holds);
    rc_registry_destroy(registry);
    rc_backing_store_destroy(clear_backings);
    clear_backings = NULL;
    printf("PASS layered color %s after %s, disposition %d, remap %d, different view %d\n",
        draw ? "conditional draw" : "clear", prior_clear ? "literal multi-mip clear" : "projected transfer", fate, remap, other_view);
}

static void self_copy_edges(int format,VkImageAspectFlags aspect,enum Fate fate) {
    enum { W=21,H=13,PIXEL=4 };
    registry=rc_registry_create((RcRegistryConfig){16,48,4});
    assert(registry);
    Tex target={.pub={.image=BASE_IMAGE,.addr=START,.width=W,.height=H,.depth=1,.levels=1,.layers=1,
                      .type=VK_IMAGE_TYPE_2D,.aspect=aspect,.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT},
                .t={.serial=4,.d={.target=NVN_TEXTURE_TARGET_2D,.format=format,.width=W,.height=H,.depth=1,.levels=1}}};
    target.t.mem=(void *)(uintptr_t)START;
    nvn_tex_layout(&target.t.d,&target.t.L);
    const uint64_t bytes=target.t.L.size, boundary=bytes/2;
    assert(nvn_fmt_bpb(format,NULL,NULL)==PIXEL && bytes>W*H*PIXEL);
    RcInterval maps[]={{START,7,{6,0,boundary}},{START+boundary,8,{6,boundary,bytes-boundary}}};
    target.image_private_mapping=maps;
    target.image_private_mapping_count=2;
    RcInterval interior[IMAGE_ALIAS_MAX_WRITE_FRAGMENTS];
    uint32_t count=0;
    const NVNcopyRegion inside={0,0,0,W-1,H,1};
    assert(texture_alias_demand_box_plane(&target.t.d,&inside,PIXEL,0,0,maps,2,START,bytes,
        interior,IMAGE_ALIAS_MAX_WRITE_FRAGMENTS,&count));
    const uint32_t interior_count=count;
    issue_at(&target,(RcRegistryPosition){1,2},interior,interior_count,0,COMPLETE);

    const VkImageSubresourceLayers plane={aspect,0,0,1};
    VkImageCopy copies[]={{plane,{W-2,0,0},plane,{W-1,0,0},{1,H,1}},
                          {plane,{0,H-2,0},plane,{0,H-1,0},{W,1,1}}};
    VkImageCopy bad_order[]={copies[1],copies[0]};
    transfer_read=(RcRegistryPosition){1,4};
    const RcRegistryPosition output={1,5};
    assert(rc_registry_declare_through(registry,transfer_read)==RC_REGISTRY_OK);
    VkTextureProjectedTransfer *receipt=NULL;
    assert(vk_texture_projected_self_copy_capture(&target.t,&target.pub,bad_order,2,&receipt)==-1);
    assert(!receipt && !image_hold.readers); // Without its preceding column the row has an unproved last source pixel.
    VkImageCopy overlapping=copies[0];
    overlapping.dstOffset=overlapping.srcOffset;
    assert(vk_texture_projected_self_copy_capture(&target.t,&target.pub,&overlapping,1,&receipt)==-1);
    assert(!receipt && !image_hold.readers);
    const RcInterval before=maps[1];
    maps[1].range.offset=maps[0].range.offset;
    assert(vk_texture_projected_self_copy_capture(&target.t,&target.pub,copies,2,&receipt)==-1);
    assert(!receipt && !image_hold.readers);
    maps[1]=before;
    assert(vk_texture_projected_self_copy_capture(&target.t,&target.pub,copies,2,&receipt)==1);
    assert(receipt && image_hold.readers==1);
    const VkImageCopy *selected=vk_texture_projected_self_copy_regions(receipt,&count);
    assert(count==2 && selected!=copies && !memcmp(selected,copies,sizeof copies));

    // Compare the projected exact guest byte mask against per-pixel ranges from the real NVN swizzle.
    uint8_t *written=calloc(bytes,1), *original=calloc(bytes,1);
    assert(written && original);
    for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
        RcInterval pixel[4];
        const NVNcopyRegion cell={x,y,0,1,1,1};
        assert(texture_alias_demand_box_plane(&target.t.d,&cell,PIXEL,0,0,maps,2,START,bytes,pixel,4,&count));
        for (uint32_t j=0;j<count;j++) for (uint64_t k=0;k<pixel[j].range.size;k++) {
            const uint64_t at=pixel[j].range.offset+k;
            written[at]=(x==W-1 || y==H-1);
            original[at]=(x==W-2 || (y==H-2 && x<W-1));
        }
    }
    for (uint64_t i=0;i<bytes;i++) {
        assert(byte_in(receipt->covered,receipt->pieces,i)==written[i]);
        assert(byte_in(receipt->prior,receipt->prior_count,i)==original[i]);
    }

    ResourceCommandEffects *effect=resource_effects_create(output);
    assert(vk_texture_projected_transfer_declare(receipt,effect,output)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,output)==RC_REGISTRY_OK);
    ResourceCommandEffects *previous=resource_effects_enter(effect);
    {
        RESOURCE_IMAGE_SCOPE(scope,output.scene);
        mapping_current=0;
        assert(!vk_texture_projected_transfer_begin(receipt) && !locked);
        mapping_current=1;
        image_hold.detached=1;
        assert(!vk_texture_projected_transfer_begin(receipt) && !locked);
        image_hold.detached=0;
        target.content_revision++;
        assert(!vk_texture_projected_transfer_begin(receipt) && !locked);
        target.content_revision--;
        if (fate!=UNISSUED) {
            assert(vk_texture_projected_transfer_begin(receipt) && locked);
            unsigned native_copies=0;
            for (uint32_t i=0;i<receipt->copy_count;i++) {
                assert(selected[i].srcSubresource.aspectMask==aspect);
                native_copies++; // Native command sink. This is one command scope with two ordered copies.
            }
            assert(native_copies==2);
            vk_texture_projected_transfer_written(receipt);
            assert(!locked);
        }
    }
    resource_effects_leave(previous);
    resource_effects_release(effect);
    vk_texture_projected_transfer_cancel(&receipt);
    vk_texture_projected_transfer_cancel(&receipt);
    assert(!receipt && !image_hold.readers);
    if (fate==COMPLETE) assert(rc_registry_handoff_through(registry,output)==RC_REGISTRY_OK);
    for (unsigned i=0;i<nq;i++) {
        if (fate==COMPLETE) q[i].done(q[i].arg);
        else { assert(fate==UNSUBMITTED);q[i].cancel(q[i].arg,0); }
    }
    nq=0;

    const RcRegistryPosition read={1,6};
    assert(rc_registry_declare_through(registry,read)==RC_REGISTRY_OK);
    for (uint64_t i=0;i<bytes;i++) {
        const uint32_t map=i<boundary?0:1;
        const RcInterval byte={START+i,maps[map].mapping,{6,i,1}};
        RcRegistryNativeImageFragment slice;
        const RcRegistryResult result=rc_registry_native_image_fragments(registry,&byte,1,read,&slice,1,&count);
        const int expect_later=fate==COMPLETE && written[i];
        if (expect_later || byte_in(interior,interior_count,i)) {
            assert(result==RC_REGISTRY_OK && count==1 && slice.binding.image==target.pub.image &&
                   slice.output.ordinal==(expect_later?5:3));
        } else assert(result==RC_REGISTRY_BLOCKED && !count);
    }
    free(written);free(original);free(target.image_mapped_holds);
    rc_registry_destroy(registry);
    printf("PASS same-image ordered edge union, aspect %u, disposition %d\n",aspect,fate);
}

static void host_volume_b2t(enum Fate fate,int stale) {
    const uint64_t address=0x41000000;
    Tex dst={.pub={.image=REPLACEMENT_IMAGE,.addr=address,.format=VK_FORMAT_R8G8B8A8_SNORM,
        .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.type=VK_IMAGE_TYPE_3D,.view_type=VK_IMAGE_VIEW_TYPE_3D,
        .usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,.width=32,.height=32,.depth=32,
        .levels=1,.layers=1,.samples=1,.scale=1,.mirrored=1},.f={.vk=VK_FORMAT_R8G8B8A8_SNORM},
        .t={.serial=37,.mem=(void *)(uintptr_t)address,.d={.target=NVN_TEXTURE_TARGET_3D,.format=NVN_FORMAT_RGBA8SN,
             .width=32,.height=32,.depth=32,.levels=1}}};
    nvn_tex_layout(&dst.t.d,&dst.t.L);
    const RcInterval normalized={address,9,{6,0,dst.t.L.size}};
    dst.image_host_backing=(RcInterval *)&normalized;
    dst.alloc=&dst; dst.key=dst.t.serial;
    dst.native_output=&dst.owner;
    entry=&dst;
    assert(!nq && !image_hold.readers);
    registry=rc_registry_create((RcRegistryConfig){16,48,4});
    assert(registry);
    assert(rc_registry_cpu_changed(registry, normalized.range)==RC_REGISTRY_OK);
    const p_copy_b2t command={.src=0xfab0,.tex=&dst.t,.r={0,0,0,32,32,32},.has_view=1,
        .view={.set_target=1,.target=NVN_TEXTURE_TARGET_3D,.set_levels=1,.base_level=0,.nlevels=1}};
    VkExec ex={.fence=2};
    VkExecProjectedTransfer plan={0};
    transfer_read=(RcRegistryPosition){2,2};
    const RcRegistryPosition output={2,3};
    assert(rc_registry_declare_through(registry,transfer_read)==RC_REGISTRY_OK);
    assert(projected_b2t_capture(&ex,&command,&plan)==1 && plan.target && image_hold.readers==1);
    VkTextureProjectedTransfer *owned=plan.target;
    const int before=volume_issues, refused=volume_refusals, destroyed=host_destroyed;
    assert(!dst.pub.gpu_owned && !dst.image_mapped_holds);
    ResourceCommandEffects *effects=resource_effects_create(output);
    assert(vk_texture_projected_transfer_declare(owned,effects,output)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,output)==RC_REGISTRY_OK);
    ResourceCommandEffects *previous=resource_effects_enter(effects);
    {
        RESOURCE_IMAGE_SCOPE(scope,2);
        if (stale==1) mapping_current=0;
        if (stale==2) host_generation++;
        if (fate!=UNISSUED || stale) projected_b2t_record(&ex,&plan,REPLACEMENT_IMAGE);
        mapping_current=1;
    }
    resource_effects_leave(previous);
    resource_effects_release(effects);
    if (fate==UNISSUED) {
        assert(plan.target==owned && !nq && !dst.pub.gpu_owned);
        assert(volume_issues==before && volume_refusals==refused+(stale!=0));
        vk_texture_projected_transfer_cancel(&plan.target);
    } else {
        assert(!plan.target && nq>=1 && volume_issues==before+1 && dst.pub.gpu_owned);
        assert(image_hold.readers==1 && q[0].arg==owned);
        if (fate!=UNSUBMITTED) assert(rc_registry_handoff_through(registry,output)==RC_REGISTRY_OK);
        scene_abandoned=fate==UNSUBMITTED;
        for (uint32_t i=0;i<nq;i++) {
            if (fate==COMPLETE) q[i].done(q[i].arg);
            else q[i].cancel(q[i].arg,fate==MAY_HAVE_EXECUTED);
        }
        nq=0;
        scene_abandoned=0;
        assert(host_destroyed==destroyed+(fate==UNSUBMITTED));
    }
    assert(!image_hold.readers);
    if (dst.image_mapped_holds) {
        rc_registry_image_hold_release_extents(registry,dst.image_mapped_holds,dst.image_mapped_hold_count,0);
        free(dst.image_mapped_holds);
    }
    rc_registry_destroy(registry);
    printf("PASS cold host RGBA8SN 32^3 B2T target custody, disposition %d, stale %d\n", fate, stale);
}

#ifdef TEST_HOST_COLOR
static void resolve_color_part(enum Fate fate) {
    scene_abandoned=fate==UNSUBMITTED;
    for (uint32_t i=0;i<nq;++i) {
        if (fate==COMPLETE) q[i].done(q[i].arg);
        else q[i].cancel(q[i].arg,fate==MAY_HAVE_EXECUTED);
    }
    nq=0;
    scene_abandoned=0;
}

static void record_mirrored_color(Tex *destination, RcRegistryPosition read, int remap) {
    VkTexImage shape=destination->pub;
    const NVNcopyRegion box={0,0,0,1,1,1};
    const view_t identity={.set_format=1,.format=NVN_FORMAT_RGBA32F};
    const RcRegistryPosition output={read.scene,read.ordinal+1};
    VkTextureProjectedTransfer *recipe=NULL;
    transfer_read=read;
    assert(rc_registry_declare_through(registry,read)==RC_REGISTRY_OK);
    assert(vk_texture_projected_transfer_capture(&destination->t,&identity,&box,&shape,&recipe)==1);
    assert(recipe && shape.image==destination->pub.image && recipe->pieces==1 &&
        recipe->covered[0].range.size==16 && recipe->covered[0].range.offset==0 && recipe->bytes==128 &&
        image_hold.readers==1);
    ResourceCommandEffects *effects=resource_effects_create(output);
    assert(vk_texture_projected_transfer_declare(recipe,effects,output)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,output)==RC_REGISTRY_OK);
    ResourceCommandEffects *previous=resource_effects_enter(effects);
    {
        RESOURCE_IMAGE_SCOPE(scope,2);
        if (remap) mapping_current=0;
        if (remap==2) {
            mapping_current=1;
            host_generation++;
        }
        if (remap) assert(!vk_texture_projected_transfer_begin(recipe));
        else {
            assert(vk_texture_projected_transfer_begin(recipe) && locked && recipe->issuing==destination);
            vk_texture_projected_transfer_written(recipe);
        }
    }
    resource_effects_leave(previous);
    resource_effects_release(effects);
    vk_texture_projected_transfer_cancel(&recipe);
    mapping_current=1;
    assert(!image_hold.readers && !locked && !recipe);
}

static void mirrored_color_transfer(enum Fate fate, int remap, int reassign) {
    uint8_t raw[160], second[160];
    memset(raw,0xA7,sizeof raw);
    memset(second,0xC3,sizeof second);
    const uint64_t address=(uintptr_t)&raw[16];
    Tex destination={.pub={.image=REPLACEMENT_IMAGE,.addr=address,.format=VK_FORMAT_R32G32B32A32_SFLOAT,
        .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.type=VK_IMAGE_TYPE_2D,.view_type=VK_IMAGE_VIEW_TYPE_2D,
        .usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,.width=1,.height=1,.depth=1,
        .levels=1,.layers=1,.samples=1,.scale=1,.mirrored=1},
        .t={.serial=39,.mem=(void *)(uintptr_t)address,.d={.target=NVN_TEXTURE_TARGET_2D,
             .format=NVN_FORMAT_RGBA32F,.width=1,.height=1,.depth=1,.levels=1,
             .flags=NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET}}};
    nvn_tex_layout(&destination.t.d,&destination.t.L);
    assert(destination.t.L.size==128);
    RcInterval normalized={address,9,{6,0,128}};
    destination.image_host_backing=&normalized;
    destination.alloc=&destination;
    destination.key=destination.t.serial;
    entry=&destination;
    registry=rc_registry_create((RcRegistryConfig){16,48,4});
    assert(registry && rc_registry_cpu_changed(registry,normalized.range)==RC_REGISTRY_OK);
    host_part.id++;
    record_mirrored_color(&destination,(RcRegistryPosition){2,2},0);
    assert(destination.pub.gpu_owned && destination.gpu_dirty && destination.mirrored_color_write);
    assert(rc_registry_handoff_through(registry,(RcRegistryPosition){2,3})==RC_REGISTRY_OK);
    resolve_color_part(COMPLETE);
    assert(!destination.mirrored_color_write);
    assert(!mirrored_color_storage_changed(&destination));
    raw[16+16]^=17;
    assert(!mirrored_color_storage_changed(&destination));
    raw[16+15]^=1;
    assert(mirrored_color_storage_changed(&destination));
    raw[16+15]^=1;

    const uint64_t prior_sequence=destination.gpu_seq;
    const RcRegistryWrite prior_hold=destination.image_mapped_holds[0];
    host_part.id++;
    record_mirrored_color(&destination,(RcRegistryPosition){2,4},remap);
    MirroredColorWrite *owner=destination.mirrored_color_write;
    if (remap) {
        assert(!owner && destination.gpu_seq==prior_sequence && destination.native_write.ordinal==2);
    } else {
        assert(owner && destination.gpu_seq==prior_sequence+1 && destination.native_write.ordinal==4);
        record_mirrored_color(&destination,(RcRegistryPosition){2,6},0);
        assert(destination.mirrored_color_write==owner && destination.gpu_seq==prior_sequence+2);
    }
    Tex sibling={0};
    if (reassign) {
        sibling.key=destination.key;
        sibling.t.serial=destination.t.serial;
        sibling.pub.image=PATCH_IMAGE;
        sibling.pub.addr=(uintptr_t)&second[16];
        sibling.content_revision=destination.content_revision;
        sibling.native_write=(RcRegistryPosition){2,8};
        sibling.gpu_seq=43;
        entry=&sibling;
    }
    if (!remap && fate!=UNSUBMITTED)
        assert(rc_registry_handoff_through(registry,(RcRegistryPosition){2,7})==RC_REGISTRY_OK);
    resolve_color_part(fate);
    if (reassign) {
        assert(entry==&sibling && sibling.pub.image==PATCH_IMAGE && sibling.gpu_seq==43 && sibling.native_write.ordinal==8);
    } else {
        assert(!destination.mirrored_color_write && destination.pub.gpu_owned && destination.gpu_dirty);
        assert(destination.gpu_seq==prior_sequence+((remap || fate==UNSUBMITTED)?0:2));
        assert(destination.native_write.ordinal==((remap || fate==UNSUBMITTED)?2:6));
    }
    assert(destination.image_mapped_holds[0].id==prior_hold.id && !image_hold.readers);
    for (unsigned i=0;i<sizeof second;++i) assert(second[i]==0xC3);
    assert(raw[0]==0xA7 && raw[16+16]==(0xA7^17) && raw[159]==0xA7);
    free(destination.snap);
    free(destination.image_mapped_holds);
    rc_registry_destroy(registry);
    entry=NULL;
    printf("PASS 16-byte mirrored RGBA32F T2T custody, padding 112, fate %d, stale %d, reassigned %d\n",fate,remap,reassign);
}
#endif

int main(int argc,char **argv) {
    const int only_host_color=argc==2 && !strcmp(argv[1],"--host-color");
#ifdef TEST_HOST_COLOR
    if (only_host_color || argc==1) {
        mirrored_color_transfer(COMPLETE,0,0);
        mirrored_color_transfer(UNSUBMITTED,0,0);
        mirrored_color_transfer(MAY_HAVE_EXECUTED,0,0);
        mirrored_color_transfer(UNISSUED,1,0);
        mirrored_color_transfer(UNISSUED,2,0);
        mirrored_color_transfer(UNSUBMITTED,0,1);
        if (only_host_color) return 0;
    }
#else
    if (only_host_color) {
        fputs("FAIL ordinary host color T2T has no selected custody branch\n",stderr);
        return 1;
    }
#endif
    const int only_host_volume = argc==2 && !strcmp(argv[1],"--host-volume");
#if defined(TEST_HOST_VOLUME)
    const int run_host_volume = 1;
#else
    const int run_host_volume = only_host_volume;
#endif
    if (run_host_volume) {
        host_volume_b2t(COMPLETE,0);
        host_volume_b2t(UNISSUED,1);
        host_volume_b2t(UNISSUED,2);
        host_volume_b2t(UNSUBMITTED,0);
        host_volume_b2t(MAY_HAVE_EXECUTED,0);
        if (only_host_volume) return 0;
    }
    volume_b2t(COMPLETE,0);
    volume_b2t(UNSUBMITTED,0);
    volume_b2t(MAY_HAVE_EXECUTED,0);
    volume_b2t(UNISSUED,1);
    volume_storage_projection(COMPLETE, 0);
    volume_storage_projection(UNSUBMITTED, 0);
    volume_storage_projection(UNISSUED, 1);
    color_after_transfer(COMPLETE, 0, 0, 0, 1);
    color_after_transfer(UNISSUED, 0, 0, 0, 1);
    color_after_transfer(UNSUBMITTED, 0, 0, 0, 1);
    color_after_transfer(COMPLETE, 1, 0, 0, 1);
    color_after_transfer(COMPLETE, 2, 0, 0, 1);
    color_after_transfer(COMPLETE, 0, 0, 0, 0);
    color_after_transfer(COMPLETE, 0, 0, 1, 0);
    color_after_transfer(UNISSUED, 0, 0, 0, 0);
    color_after_transfer(UNSUBMITTED, 0, 0, 0, 0);
    color_after_transfer(COMPLETE, 1, 0, 0, 0);
    color_after_transfer(COMPLETE, 2, 0, 0, 0);
    color_after_transfer(COMPLETE, 0, 1, 0, 0);
    contiguous_provider_distinct_mappings();
    self_copy_edges(NVN_FORMAT_RGBA8, VK_IMAGE_ASPECT_COLOR_BIT, COMPLETE);
    self_copy_edges(NVN_FORMAT_DEPTH32F, VK_IMAGE_ASPECT_DEPTH_BIT, COMPLETE);
    self_copy_edges(NVN_FORMAT_RGBA8, VK_IMAGE_ASPECT_COLOR_BIT, UNISSUED);
    self_copy_edges(NVN_FORMAT_DEPTH32F, VK_IMAGE_ASPECT_DEPTH_BIT, UNSUBMITTED);
    same_native_clear_and_copy();
    assert(run(1, 0, COMPLETE));
    assert(run(2, 1, COMPLETE));
    assert(run(2, 0, UNISSUED));
    int okay = run(2, 0, COMPLETE);
    okay &= run(2, 0, UNSUBMITTED);
    okay &= run(2, 0, MAY_HAVE_EXECUTED);
    return !okay;
}

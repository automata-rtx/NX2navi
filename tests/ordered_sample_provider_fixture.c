#include <assert.h>
#include <stdio.h>

#define RC_REGISTRY_SINGLE_THREADED 1
#define rc_registry_runtime unused_registry_runtime
#include "resource_core_registry.c"
#undef rc_registry_runtime

#define VK_NULL_HANDLE 0
#define VK_IMAGE_USAGE_SAMPLED_BIT 1
#define VK_IMAGE_ASPECT_DEPTH_BIT 2
#define VK_TEXTURE_ALIAS_MAX_BYTES 65536
#define IMAGE_ALIAS_MAX_WRITE_FRAGMENTS 32
enum {VK_TEXTURE_ALIAS_SAMPLE_FORCE = 1, VK_TEXTURE_ALIAS_SAMPLE_LINEAR_RGBA8_VIEW = 2, VK_TEXTURE_ALIAS_SAMPLE_ONLY = 4};
enum {VK_TEXTURE_READ_FORCE_COMPAT = 2, VK_TEXTURE_READ_LINEAR_RGBA8_VIEW = 4};
enum {VK_TEXTURE_REGISTRATION_CURRENT, VK_TEXTURE_REGISTRATION_EXACT};
typedef enum {VK_TEXTURE_ALIAS_LAYOUT, VK_TEXTURE_ALIAS_ORDER, VK_TEXTURE_ALIAS_OWNER,
              VK_TEXTURE_ALIAS_COVERAGE, VK_TEXTURE_ALIAS_CAPACITY, VK_TEXTURE_ALIAS_MAPPING} VkTextureAliasPhase;
typedef enum {VK_TEXTURE_SAMPLE_READY, VK_TEXTURE_SAMPLE_PREPARE, VK_TEXTURE_SAMPLE_FALLBACK,
              VK_TEXTURE_SAMPLE_ABSENT, VK_TEXTURE_SAMPLE_UNAVAILABLE} VkTextureSampleStatus;
typedef uintptr_t VkImage;
typedef unsigned VkImageSubresourceRange;
typedef unsigned ImgKey, view_t, VkMemSpan;
typedef struct {uint64_t addr;} TexKey;
typedef struct {void *mem; uint64_t serial; int d; struct {uint64_t size;} L;} tex_t;
typedef struct {
    VkImage image;
    uint64_t addr;
    unsigned usage, bound, gpu_owned, preserve_pool_bytes, gone, aspect;
} VkTexImage;
typedef struct {
    tex_t t;
    TexKey key;
    VkTexImage pub;
    RcInterval *image_private_mapping, *alias_full_demand;
    uint32_t image_private_mapping_count, alias_full_count;
    unsigned alias_fragment_owner, released, retiring, superseded, overwritten, content_revision;
    RcRegistryPosition native_write;
} Tex;
typedef struct {RcRegistryHandle handle; tex_t descriptor; view_t view; int has_view;} TextureRegistration;
typedef struct {TextureRegistration *registration; int finalized;} TexSlot;
typedef struct {
    const VkTexImage *image;
    view_t view;
    int has_view, native_slot, historical;
    uint64_t registration;
    int canonical, projected;
} VkTextureSample;
typedef struct {
    VkTextureAliasPhase phase;
    RcRegistryResult status, raw_status;
    RcInterval receiver;
    RcRegistryNativeImageFailure coverage;
    RcRegistryBufferLocation raw_origin;
    uint64_t registration, material_serial;
    VkImage image;
    unsigned raw_attempted, raw_spans;
    const char *layout_gate;
} VkTextureAliasFailure;
typedef struct {RcRegistryBufferSpan span;} VkMemImageAliasBaseline;
typedef struct {uint64_t bytes;} VkTextureAliasLayout;
typedef struct {
    const tex_t *texture;
    TexKey key;
    ImgKey placement;
    VkTexImage image;
    const RcInterval *mapping;
    uint32_t mapping_count;
    const view_t *view;
    const VkImageSubresourceRange *load_planes;
    VkMemImageAliasBaseline **prepared_baseline;
    const RcInterval *transfer_demand;
    uint32_t transfer_demand_count;
    int allow_rgba8_linear_view, allow_submitted_raw;
} ImageAliasReceiver;
typedef struct VkTextureAliasRead {
    uint32_t owners, mapping_count, demand_count, fragment_count;
    RcRegistryPosition read;
    TexKey key;
    const tex_t *receiver;
    VkTexImage result;
    unsigned registration_only, predecessor_dormant, predecessor_revision;
    uint8_t transfer_source;
    VkImage predecessor, target_hold, predecessor_hold;
    ImgKey placement;
    VkImageSubresourceRange load_planes;
    uint64_t predecessor_serial, guest_bytes, workspace_bytes, upload_offset, stage_bytes;
    RcInterval *receiver_mapping, *demands;
    RcRegistryNativeImageSnapshotFragment *fragments, *current;
    uint32_t *slices;
    VkMemImageAliasBaseline *baseline;
    uint64_t raw_upload;
    VkTextureAliasLayout receiver_layout;
    uint64_t receiver_registration, receiver_pool;
    int receiver_logical, receiver_native;
} VkTextureAliasRead;

static RcRegistry *registry;
static const RcRegistryPosition read = {1, 4};
static const RcInterval mapping = {.address=0x5000, .mapping=18, .range={9,0,64}};
static TextureRegistration registration = {.handle={19,1}, .descriptor={.mem=(void *)0x5000,.serial=7,.L={64}},
    .view=3,.has_view=1};
static Tex attached = {.t={.mem=(void *)0x5000,.serial=7,.L={64}}, .key={0x5000},
    .pub={.image=44,.addr=0x5000,.usage=1,.gpu_owned=1}};
static TexSlot tslots[3] = {[2]={&registration,0}};
static int by_key, raw_available, pinned, linear_view, exact, cold;
static uint64_t recorded_fill, submitted_fill;
static unsigned receiver_aspect = 1;
static unsigned peeks, resolutions, raw_attempts, provider_captures;

static int single_color_view;
static int image_alias_single_color_view(const tex_t *t, const view_t *view) {
    assert(t == &registration.descriptor && (!view || *view == registration.view));
    return single_color_view;
}

static RcRegistry *rc_registry_runtime(void) { return registry; }
static RcRegistryPosition vk_exec_resource_position(void) { return read; }
static uint64_t vk_exec_texture_pool(void) { return 5; }
static int texture_registration_resolve(int logical,const TextureRegistration **history,int *native) {
    assert(logical==6);
    *history=NULL;
    *native=2;
    resolutions++;
    return VK_TEXTURE_REGISTRATION_CURRENT;
}
static VkTextureSampleStatus vk_texture_select_sample_locked(int logical,unsigned flags,VkTextureSample *sample) {
    assert(logical==6 && !(flags & ~(VK_TEXTURE_READ_LINEAR_RGBA8_VIEW | VK_TEXTURE_READ_FORCE_COMPAT)));
    peeks++;
    *sample=(VkTextureSample){.native_slot=2};
    if (exact && !(flags & VK_TEXTURE_READ_FORCE_COMPAT)) {
        sample->canonical=1;
        sample->registration=19;
        sample->image=&attached.pub;
        return VK_TEXTURE_SAMPLE_READY;
    }
    return VK_TEXTURE_SAMPLE_FALLBACK;
}
static const Tex *image_alias_sample_entry_locked(const TexSlot *slot,int native) {
    assert(slot==&tslots[2] && native==2);
    return cold ? NULL : &attached;
}
static int nvn_gpu_only(uint64_t address) { return address==mapping.address; }
static int vk_mem_lookup(uint64_t address,uint64_t bytes,VkMemSpan *span) {
    assert(address==mapping.address && bytes==mapping.range.size);
    *span=1;
    return 1;
}
static uint32_t vk_mem_device_intervals(const VkMemSpan *span,uint64_t address,uint64_t bytes,RcInterval **out) {
    assert(*span==1 && address==mapping.address && bytes==mapping.range.size && (*out=malloc(sizeof **out)));
    **out=mapping;
    return 1;
}
static const char *image_alias_sample_recipe(const TextureRegistration *record, ImageAliasReceiver *receiver) {
    assert(record==&registration && receiver->allow_rgba8_linear_view==linear_view);
    receiver->image=(VkTexImage){.addr=mapping.address,.usage=1,.aspect=receiver_aspect};
    receiver->key=(TexKey){mapping.address};
    receiver->texture=&record->descriptor;
    receiver->view=&record->view;
    return NULL;
}

static const tex_t *nvn_texture_pin_acquire(const tex_t *t) {
    assert(t==&registration.descriptor && !pinned);
    pinned=1;
    return t;
}
static VkImage vk_mem_image_hold(VkImage image) { return image; }
static void image_alias_release(VkTextureAliasRead *alias) {
    if (alias->receiver) pinned=0;
    free(alias->receiver_mapping);
    free(alias->demands);
    free(alias->fragments);
    free(alias->current);
    free(alias->slices);
    free(alias->baseline);
    free(alias);
}
static int texture_alias_layout(const int *d,const VkTexImage *image,VkTextureAliasLayout *layout) {
    assert(d==&registration.descriptor.d && image->addr==mapping.address);
    *layout=(VkTextureAliasLayout){16};
    return 1;
}
static int image_alias_material_layout(const Tex *entry,VkTextureAliasLayout *layout) {
    (void)entry; (void)layout; abort();
}
static int texture_alias_clear_planes(VkImageSubresourceRange range,const VkTexImage *image,VkTextureAliasLayout *layout) {
    (void)range; (void)image; (void)layout; abort();
}
static int texture_alias_selected_planes(const view_t *view,const VkTexImage *image,VkTextureAliasLayout *layout) {
    assert(view==&registration.view && image->addr==mapping.address && layout->bytes==16);
    return 1;
}
static int texture_alias_demand(const int *d,const VkTextureAliasLayout *layout,const RcInterval *captured,
    uint32_t count,uint64_t address,uint64_t guest_bytes,RcInterval *out,uint32_t capacity,uint32_t *used) {
    assert(d==&registration.descriptor.d && count==1 && captured->mapping==mapping.mapping &&
           address==mapping.address && guest_bytes==mapping.range.size && capacity);
    *out=*captured;
    out->range.size=layout->bytes;
    *used=1;
    return 1;
}
static RcRegistryResult vk_mem_image_alias_baseline_capture(uint64_t address,uint64_t bytes,const RcInterval *capture,
    uint32_t count,RcRegistryPosition stamp,VkMemImageAliasBaseline **out) {
    assert(address==mapping.address && bytes==mapping.range.size && count==1 && capture->mapping==mapping.mapping &&
           !rc_registry_position_compare(stamp,read) && !*out && !pinned);
    raw_attempts++;
    if (raw_available) {
        *out=malloc(sizeof **out);
        assert(*out);
        **out=(VkMemImageAliasBaseline){.span={mapping,{20,1,512}}};
    }
    return RC_REGISTRY_OK;
}
static const RcRegistryBufferSpan *vk_mem_image_alias_baseline_spans(const VkMemImageAliasBaseline *baseline,uint32_t *count) {
    *count=1;
    return &baseline->span;
}
static RcRegistryResult vk_mem_image_alias_baseline_capture_submitted(uint64_t address,uint64_t bytes,
    const RcInterval *capture,uint32_t count,RcRegistryPosition stamp,VkMemImageAliasBaseline **out,uint64_t *fill) {
    assert(!raw_available && address==mapping.address && bytes==mapping.range.size && count==1 &&
           capture->mapping==mapping.mapping && !rc_registry_position_compare(stamp,read) && !*out && !pinned);
    if (recorded_fill && recorded_fill<=submitted_fill) {
        *out=malloc(sizeof **out);
        assert(*out);
        **out=(VkMemImageAliasBaseline){.span={mapping,{20,1,512}}};
        *fill=recorded_fill;
    }
    return RC_REGISTRY_OK;
}
static const RcRegistryBufferSpan *vk_mem_image_alias_baseline_submitted_spans(const VkMemImageAliasBaseline *baseline,
    uint64_t fill,uint32_t *count) {
    assert(fill==recorded_fill && fill<=submitted_fill);
    return vk_mem_image_alias_baseline_spans(baseline,count);
}
static RcRegistryResult image_alias_capture_sources(VkTextureAliasRead *alias,int retain,VkTextureAliasFailure *failure) {
    assert(retain && failure && alias->fragment_count && pinned);
    provider_captures++;
    return RC_REGISTRY_OK;
}
static void make_key(const tex_t *t,TexKey *key) { key->addr=(uintptr_t)t->mem; }
static int same_but_swizzle(const TexKey *a,const TexKey *b) { return a->addr==b->addr; }
static int rt_knob(const char *key,const char *fallback) { (void)key; (void)fallback; return 0; }
static void rt_log(const char *fmt,...) { (void)fmt; abort(); }
static const Tex *nvn_hm_get(const int *table,const TexKey *key) {
    assert(table==&by_key && key->addr==mapping.address);
    return &attached;
}
static int image_alias_source_current(const Tex *entry) { (void)entry; abort(); }
static int image_alias_read_capture_locked(const tex_t *t,const Tex *entry,VkTextureAliasRead **out,VkTexImage *shape,
    int force,int target,const void *planes,void *baseline,VkTextureAliasFailure *failure) {
    (void)t; (void)entry; (void)out; (void)shape; (void)force; (void)target; (void)planes; (void)baseline; (void)failure;
    abort();
}
static void slot_result_view(const TexSlot *slot,const Tex *entry,view_t *view,int *has) {
    (void)slot; (void)entry; (void)view; (void)has; abort();
}

/* Extract provider */

#define OK(call) assert((call)==RC_REGISTRY_OK)
static void begin(uint64_t image_start,uint64_t image_bytes,int physically_bound) {
    registry=rc_registry_create((RcRegistryConfig){12,16,4});
    assert(registry);
    if (!image_bytes) {
        OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,8}));
        return;
    }
    RcInterval donor={.address=0x1000,.mapping=17,.range={9,0,64}};
    RcRegistryImageWriteExtent owner={.interval=donor};
    RcRegistryWrite draw;
    const RcRegistryPosition output={1,3};
    const RcSourceRange region={9,image_start,image_bytes};
    OK(rc_registry_image_hold_begin_extents(registry,100,&donor,1,&owner.hold));
    OK(rc_registry_write_begin_at(registry,donor.range,RC_REGISTRY_IMAGE,output,&draw));
    OK(rc_registry_image_producer_add(registry,draw,output));
    OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,8}));
    const RcRegistryNativeImageWrite write={owner,&region,1};
    const RcRegistryImageBinding binding={100,owner.hold};
    if (physically_bound) {
        assert(rc_registry_native_image_written(registry,100,&donor,&owner.hold,1,output));
        OK(rc_registry_native_image_bound_written(registry,100,&write,1,output));
    } else {
        OK(rc_registry_native_image_write_fragments(registry,100,&write,1,output));
    }
    OK(rc_registry_image_producer_resolve(registry,draw,output,&binding,0));
}

int main(void) {
    VkTextureAliasRead *alias=NULL;
    VkTextureSample sample;
    VkTexImage shape;
    VkTextureAliasFailure failure;
    assert(attached.pub.image && !attached.alias_fragment_owner && !attached.pub.preserve_pool_bytes);
    begin(0,0,0);
    raw_available=1;
    assert(vk_texture_ordered_sample_select_locked(6,(RcRegistryPosition){2,4},0,0,&sample,&alias,&shape,&failure)
           ==VK_TEXTURE_SAMPLE_UNAVAILABLE);
    assert(failure.phase==VK_TEXTURE_ALIAS_ORDER && failure.status==RC_REGISTRY_STALE && !peeks && !raw_attempts);
    assert(vk_texture_ordered_sample_select_locked(6,(RcRegistryPosition){0},0,0,&sample,&alias,&shape,&failure)
           ==VK_TEXTURE_SAMPLE_FALLBACK && peeks==1 && !raw_attempts);
    assert(vk_texture_ordered_sample_select_locked(6,(RcRegistryPosition){0},0,VK_TEXTURE_ALIAS_SAMPLE_ONLY,
           &sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_FALLBACK && peeks==1 && !resolutions);
    exact=1;
    assert(vk_texture_ordered_sample_select_locked(6,read,0,0,&sample,&alias,&shape,&failure)
           ==VK_TEXTURE_SAMPLE_READY && !alias && sample.image==&attached.pub && sample.canonical && !raw_attempts);
    exact=0;
    assert(vk_texture_ordered_sample_select_locked(6,read,0,VK_TEXTURE_ALIAS_SAMPLE_ONLY,
           &sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_FALLBACK && resolutions==1 && !alias && !raw_attempts);
    assert(vk_texture_ordered_sample_select_locked(6,read,0,0,&sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_READY);
    assert(alias && !shape.image && sample.registration==19 && sample.view==3 && sample.has_view && raw_attempts==1 &&
           alias->receiver_registration==19 && alias->receiver_logical==6 && alias->receiver_native==2 &&
           alias->receiver_pool==5 && alias->registration_only && alias->fragment_count==1 && provider_captures==1 &&
           alias->fragments[0].baseline==0 && alias->fragments[0].native_buffer.offset==512 &&
           alias->fragments[0].fragment.interval.range.size==16 && resolutions==1);
    image_alias_release(alias);
    alias=NULL;
    rc_registry_destroy(registry);

    begin(4,4,0);
    linear_view=1;
    cold=1;
    assert(vk_texture_ordered_sample_select_locked(6,read,VK_TEXTURE_READ_LINEAR_RGBA8_VIEW,VK_TEXTURE_ALIAS_SAMPLE_ONLY,
           &sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_READY);
    assert(alias && alias->fragment_count==3 && alias->fragments[0].baseline==0 &&
           alias->fragments[1].baseline==UINT32_MAX && alias->fragments[1].fragment.binding.image==100 &&
           alias->fragments[1].fragment.interval.range.size==4 && alias->fragments[2].baseline==0 &&
           alias->fragments[2].native_buffer.offset==520 && resolutions==2 && raw_attempts==2);
    image_alias_release(alias);
    alias=NULL;
    raw_available=0;
    cold=0;
    assert(vk_texture_ordered_sample_select_locked(6,read,VK_TEXTURE_READ_LINEAR_RGBA8_VIEW,0,
           &sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_UNAVAILABLE);
    assert(failure.phase==VK_TEXTURE_ALIAS_COVERAGE && failure.status==RC_REGISTRY_BLOCKED && !alias && !pinned);
    rc_registry_destroy(registry);

    begin(32,8,0);
    linear_view=0;
    assert(rc_registry_native_image_has_prior_projection(registry,&mapping,1,read));
    assert(vk_texture_ordered_sample_select_locked(6,read,0,0,&sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_FALLBACK &&
           !alias && !pinned && failure.raw_attempted && !failure.raw_spans);
    raw_available=1;
    assert(vk_texture_ordered_sample_select_locked(6,read,0,0,&sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_READY &&
           alias && alias->fragment_count==1 && alias->fragments[0].baseline==0 &&
           alias->fragments[0].fragment.interval.range.size==16);
    image_alias_release(alias);
    rc_registry_destroy(registry);
    alias=NULL;

    // The real registry must adjudicate the submitted RAW span against each image winner.
    begin(4,4,0);
    cold=1;
    recorded_fill=17;
    submitted_fill=16;
    raw_available=0;
    assert(vk_texture_ordered_sample_select_locked(6,read,0,0,&sample,&alias,&shape,&failure)
           ==VK_TEXTURE_SAMPLE_UNAVAILABLE && !alias && !pinned && failure.status==RC_REGISTRY_BLOCKED);
    submitted_fill=17;
    assert(vk_texture_ordered_sample_select_locked(6,read,0,VK_TEXTURE_ALIAS_SAMPLE_ONLY,&sample,&alias,&shape,&failure)
           ==VK_TEXTURE_SAMPLE_UNAVAILABLE && !alias && !pinned && failure.status==RC_REGISTRY_BLOCKED);
    assert(vk_texture_ordered_sample_select_locked(6,read,0,0,&sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_READY);
    assert(alias && alias->raw_upload==17 && !shape.image && !alias->stage_bytes && alias->fragment_count==3 &&
           alias->fragments[0].native_buffer.offset==512 && alias->fragments[2].native_buffer.offset==520 &&
           alias->fragments[1].fragment.binding.image==100);
    image_alias_release(alias);
    alias=NULL;
    cold=0;
    rc_registry_destroy(registry);

    begin(0,0,0);
    assert(vk_texture_ordered_sample_select_locked(6,read,0,0,&sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_READY);
    assert(alias && alias->raw_upload==17 && !shape.image && !alias->stage_bytes && alias->fragment_count==1 &&
           alias->fragments[0].native_buffer.offset==512 && alias->fragments[0].fragment.interval.range.size==16);
    image_alias_release(alias);
    alias=NULL;
    recorded_fill=submitted_fill=0;
    rc_registry_destroy(registry);

    begin(0,16,1);
    raw_available=0;
    assert(vk_texture_ordered_sample_select_locked(6,read,0,0,&sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_FALLBACK &&
           !alias && !pinned);
    single_color_view=1;
    assert(vk_texture_ordered_sample_select_locked(6,read,0,0,&sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_READY &&
           alias && !alias->baseline && alias->fragment_count==1 && alias->fragments[0].fragment.binding.image==100);
    image_alias_release(alias);
    alias=NULL;
    single_color_view=0;
    receiver_aspect=VK_IMAGE_ASPECT_DEPTH_BIT;
    assert(vk_texture_ordered_sample_select_locked(6,read,0,0,&sample,&alias,&shape,&failure)==VK_TEXTURE_SAMPLE_READY &&
           alias && !alias->baseline && alias->fragment_count==1 && alias->fragments[0].fragment.binding.image==100);
    image_alias_release(alias);
    rc_registry_destroy(registry);
    puts("Ordered sample provider preserves preflight and exact residents and captures warm RAW and mixed texels at the selected read");
}

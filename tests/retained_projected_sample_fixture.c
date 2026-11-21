/* Host adapters and cases. The Python runner inserts the production function definitions. */
#define VK_NO_PROTOTYPES
#include <assert.h>
#include <stdio.h>
#include "nvn_api.h"
#include "vk_exec.h"
#include "nvn_hmap.h"
#define rc_registry_runtime registry_unused_runtime
#define main maintained_fixture_main
#include "resource_core_native_image_check.c"
#undef main
#undef rc_registry_runtime
#include "vk_texture_alias_bytes.h"
#define NVN_STAGE_VS 0
#define NVN_STAGE_TASK 6
#define NVN_STAGE_MESH 7
const int rt_nvn_api_version[2]={200,0};
const char *cfg_get(const char *config){(void)config;return NULL;}

typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    uint32_t aspect, level0, levels, layer0, layers, swz;
} ViewKey;
typedef struct { VkImage image; uint32_t l0, l1, a0, a1; } SubRange;
typedef struct {
    const GfxShader *shader;
    const GfxShaderRes *resource;
    uint64_t handle, token;
    VkTextureSample sample;
#if TEST_QUERY_GEOMETRY
    VkTextureSampleGeometry geometry;
#endif
    VkTexImage image;
    VkTexturePresentSource *hold;
    VkTextureAliasRead *alias;
    int stage, member, seen;
} GraphicsSample;
struct VkExecGraphicsDraw {
    GraphicsSample *samples;
    VkTextureRestore *sample_restore;
    const GfxShader *restore_shader;
    const GfxShaderRes *restore_resource;
    uint64_t restore_handle;
    int restore_stage, restore_member;

    uint32_t sample_count, opcode, sampler_generation;
    uint64_t registration, scene, program_serial, tex_serial, texture_pool;
    const prog_t *program;
    const dpool_t *sampler_pool;
    const void *payload;
    int pool_index_retry;
    RcRegistryPosition read_position;
    VkTextureAliasRead *attachment_seeds[9];
    VkExecPreparationFailure failure;
    int mesh, compatibility, feedback_selected;
#if TEST_QUERY_GEOMETRY
    VkGraphicsPipelineUse use;
    int geometry_present, geometry_recording_attested;
#endif
};
typedef struct VkExecGpaSampleScope VkExecGpaSampleScope;
struct GfxShader { int number; };
struct VkTexturePresentSource { VkImage image; RcRegistryPosition read; };
struct VkTextureAliasRead { VkImage image; int live; uint64_t receiver_registration; };
typedef struct { uint64_t address; } TexKey;
typedef struct { struct { uint64_t id; } handle; tex_t descriptor; view_t view; int has_view; } TextureRegistration;
typedef struct {
    VkTexImage pub;
    tex_t t;


    TexVkFormat f;
    RcInterval *alias_plane_demand;
    VkImageSubresourceRange alias_plane_range;
    uint32_t alias_plane_count;
    RcInterval *alias_full_demand, image_private_mapping[1];
    uint32_t alias_full_count, image_private_mapping_count, image_mapped_hold_count;
    RcRegistryWrite *image_mapped_holds;
    void *bound_to;
    RcRegistryPosition native_write;
    TexKey key;
    int source_canonical, source_blocked, uploaded, up_retry, released, alias_fragment_owner;
    unsigned retiring, superseded, overwritten;
    uint32_t checked;
    uint64_t checked_authority;
} Tex;
typedef struct {
    Tex *ce;
    TextureRegistration *registration;
    view_t cview, v;
    uint32_t cgen, cepoch;
    uint8_t used, sample_owned, chas_view, cheld, has_view, finalized;
} TexSlot;

enum { LOGICAL_SLOT = 31, NATIVE_SLOT = 1, SUBMISSION = 42, HORIZON = 12 };
static TexSlot tslots[2];
static uint32_t agen[1], sinfo[2], submit_epoch=22;
static int ntslots=2, pool_ctx, resolve_held, in_lookup, tex_deferred, round_checks;
static struct { unsigned canonical_ready; } resource_bridge;
static VkExecGraphicsDraw *graphics_current;
static RcRegistryPosition resource_read = {SUBMISSION, HORIZON};
static Tex resident;
static nvn_hmap by_key={.klen=sizeof(TexKey)};
static TextureRegistration registration;
static VkTexturePresentSource native_hold;
static VkTextureAliasRead load_seed;
static const VkImage native_image = (VkImage)(uintptr_t)0x558;
static const VkImage attachment_image = (VkImage)(uintptr_t)0x338;
static const prog_t program={.serial=1001};
static const GfxShader shader = { .number = 23 };
static GfxShaderRes resource = { .slot=0, .count=1, .type=2 };
static int is_absent, exact_source=1, current_mapping=1, live_refs, positions_rejected, legacy_uses, selected_queries;
static int ordered_resolve, unsupported_reason, compatibility_releases, native_handoffs, moved_results, stage_result;
static int same_read(RcRegistryPosition a, RcRegistryPosition b){return a.scene==b.scene&&a.ordinal==b.ordinal;}

RcRegistryPosition vk_exec_resource_position(void) { return resource_read; }
static RcRegistry *registry_fixture;
RcRegistry *rc_registry_runtime(void) { return registry_fixture; }
static Tex *texture_registration_source(const TexSlot *s) { return s->used && s->registration == &registration ? &resident : NULL; }
static int texture_registration_source_attach(TexSlot *s, Tex *e) {
    assert(s==&tslots[NATIVE_SLOT]&&e==&resident);
    return 0; /* This exact resident material does not belong to the canonical CPU source lane. */
}
static int texture_history_sample_locked(int logical,VkTextureSample *out,int *native,int resident_only) {
    assert(logical == LOGICAL_SLOT); (void)out;
    if (resident_only) ++ordered_resolve;
    *native = is_absent ? -1 : NATIVE_SLOT;
    return is_absent ? VK_TEXTURE_REGISTRATION_ABSENT : VK_TEXTURE_REGISTRATION_CURRENT;
}
VkTextureSampleStatus vk_texture_sample_query_locked(int native,VkTextureSample *out) {
    assert(native==NATIVE_SLOT); (void)out; ++selected_queries; return VK_TEXTURE_SAMPLE_FALLBACK;
}
static VkTextureSampleStatus texture_sample_prepare_locked(int native,VkTextureSample *out,int budgeted,VkTextureSampleRefusal *refused) {
    (void)native; (void)out; (void)budgeted; (void)refused; abort();
}
void vk_texture_sample_use_locked(const VkTextureSample *out) { (void)out; abort(); }
static Tex *image_alias_sample_entry_locked(const TexSlot *s,int native){
    assert(native==NATIVE_SLOT); return texture_registration_source(s);
}
static int image_alias_sample_entry_stale(const Tex *e,const TextureRegistration *r){
    return e->released||e->retiring||e->superseded||e->overwritten||e->pub.gone||e->t.serial!=r->descriptor.serial;
}
static int image_alias_source_current(const Tex *e){
    return e==&resident&&current_mapping&&e->image_private_mapping_count==1&&
           e->image_private_mapping[0].address==e->pub.addr&&e->image_private_mapping[0].mapping==1349;
}
static void make_key(const tex_t *t,TexKey *out){*out=(TexKey){(uintptr_t)t->mem};}
static int same_but_swizzle(const TexKey *a,const TexKey *b){return a->address==b->address;}
static void slot_result_view(const TexSlot *,const Tex *,view_t *,int *);
enum { IMAGE_ALIAS_MAX_WRITE_FRAGMENTS = 8192 };
static int image_alias_whole_load_current(const Tex *, RcRegistryPosition);
/* Extract projected */
static int image_alias_whole_load_current(const Tex *entry, RcRegistryPosition read) {
    // The retained authority cases below use private native image allocations. The existing attachment
    // and native-registry fixtures exercise the bound publication/receipt path separately.
    assert(!entry->bound_to);
    return image_alias_source_current(entry) && entry->alias_full_count &&
        rc_registry_native_image_covers(registry_fixture,(uintptr_t)entry->pub.image,
                                         entry->alias_full_demand,entry->alias_full_count,read);
}
static int slot_absence_valid(TexSlot *s,uint32_t g) { (void)s; (void)g; abort(); }
static Tex *slot_resolve(TexSlot *s,view_t *v,int *hv){(void)s;(void)v;(void)hv;abort();}
static int preserve_pool_contents_current(Tex *e) {
    assert(e==&resident);
    return !e->pub.preserve_pool_bytes ||
        (exact_source && rc_registry_native_image_covers(registry_fixture,(uintptr_t)e->pub.image,
                                                        e->alias_full_demand,e->alias_full_count,resource_read));
}
static void sample_refusal_record(VkTextureSampleRefusal *refused,const TexSlot *s,const Tex *e,const char *reason) {
    assert(s==&tslots[NATIVE_SLOT]&&(!e||e==&resident)&&reason);
    if (refused) {
        refused->registration=registration.handle.id;
        refused->stage=reason;
    }
    unsupported_reason++;
}
static void content_use(Tex *e){assert(e==&resident && exact_source);++legacy_uses;}
static void touch(Tex *e){assert(e==&resident);}
static int source_mapping_due(const Tex *e){assert(e==&resident);return !exact_source;}
static int source_binding_ready(const Tex *e){assert(e==&resident);return exact_source;}
static int pool_deps_certificate(Tex *e){assert(e==&resident);return exact_source;}
static void make_fresh_snap(Tex *e){assert(e==&resident&&!in_lookup&&round_checks>0);abort();}
static int source_permission_current(const Tex *e){assert(e==&resident);return exact_source;}
static void source_note_refusal(Tex *e,int failure,RcRegistryResult status){(void)e;(void)failure;(void)status;abort();}
enum { SOURCE_REFUSAL_PERMISSION, SOURCE_REFUSAL_UPLOAD };
static void slot_result_view(const TexSlot *s,const Tex *e,view_t *v,int *hv){assert(s==&tslots[NATIVE_SLOT]&&e==&resident); *v=s->cview; *hv=s->chas_view;}
static const TexSlot *texture_read_key_slot(VkTextureReadKey key) {
    if (is_absent||key.native_slot!=NATIVE_SLOT||key.registration!=registration.handle.id) return NULL;
    return &tslots[NATIVE_SLOT];
}
static Tex *texture_binding_entry(const TexSlot *s,int native) {
    assert(native==NATIVE_SLOT);
    return texture_registration_source(s);
}
static Tex *texture_read_key_image(const TexSlot *s,VkImage expected){(void)s;(void)expected;abort();}
static struct { unsigned calls; } binding_path;
static int source_binding_receipt(Tex *e,RcRegistryPosition *next){(void)e;(void)next;abort();}
int vk_texture_binding_receipt_locked(int native,VkImage expected,RcRegistryPosition *next) {
    (void)next; return native==NATIVE_SLOT && exact_source && expected==resident.pub.image;
}
int vk_texture_registration_expected_locked(int logical,VkImage *expected) {
    assert(logical==LOGICAL_SLOT); if (expected) *expected=NULL;
    return is_absent ? VK_TEXTURE_REGISTRATION_ABSENT : VK_TEXTURE_REGISTRATION_CURRENT;
}
int vk_texture_history_sample_current_locked(int logical,uint64_t reg,VkImage image){(void)logical;(void)reg;(void)image;abort();}
int vk_texture_sample_unbacked_current_locked(VkTextureReadKey key){(void)key;abort();}
int vk_texture_alias_sample_current_locked(int logical,VkTextureAliasRead *alias){(void)logical;(void)alias;abort();}
static uint64_t token_of(uint64_t handle){assert(NVN_HANDLE_TEX(handle)==LOGICAL_SLOT);return 24;}
static int texture_registration_resolve(int logical,const TextureRegistration **record,int *native){
    assert(logical==LOGICAL_SLOT);
    *record=is_absent ? NULL : &registration;
    *native=is_absent ? -1 : NATIVE_SLOT;
    return is_absent ? VK_TEXTURE_REGISTRATION_ABSENT : VK_TEXTURE_REGISTRATION_CURRENT;
}
static int image_alias_sample_prospective_locked(const TextureRegistration *r,int logical,int native,
    RcRegistryPosition read,VkTextureAliasRead **alias,VkTextureSample *sample,VkTexImage *shape,unsigned options,
    VkTextureAliasFailure *failure){(void)r;(void)logical;(void)native;(void)read;(void)alias;(void)sample;
    (void)shape;(void)options;(void)failure;abort();}
static int image_alias_refuse(VkTextureAliasFailure *failure,VkTextureAliasPhase phase,RcRegistryResult status,RcInterval mapped){
    if(failure) {*failure=(VkTextureAliasFailure){.phase=phase,.status=status,.receiver=mapped};} return -1;
}
static int image_alias_refuse_layout(VkTextureAliasFailure *failure,RcInterval mapped,const char *reason){
    assert(reason);return image_alias_refuse(failure,VK_TEXTURE_ALIAS_LAYOUT,RC_REGISTRY_UNSUPPORTED,mapped);
}
static int image_alias_read_capture_locked(const tex_t *t,const Tex *e,VkTextureAliasRead **alias,VkTexImage *shape,
    int force,int target,const VkImageSubresourceRange *planes,VkMemImageAliasBaseline **base,VkTextureAliasFailure *failure){

    (void)t;(void)alias;(void)shape;(void)force;(void)target;(void)planes;(void)base;
    assert(e==&resident);
    RcRegistryNativeImageSnapshotFragment fragments[100]; uint32_t count;
    RcRegistryNativeImageFailure refused={0};
    assert(rc_registry_native_image_snapshot_mixed(registry_fixture,e->alias_full_demand,e->alias_full_count,
        NULL,0,resource_read,fragments,100,&count,&refused)==RC_REGISTRY_BLOCKED);
    assert(!count&&refused.gate==RC_NATIVE_IMAGE_READ_BLOCKED&&refused.cause==RC_NATIVE_IMAGE_BLOCK_UNKNOWN_OUTPUT);
    int result=image_alias_refuse(failure,VK_TEXTURE_ALIAS_COVERAGE,RC_REGISTRY_BLOCKED,e->image_private_mapping[0]);
    failure->coverage=refused;
    return result;
}
static void rt_log_stub(const char *unused,...);
#define rt_log rt_log_stub
/* Extract alias */
static void rt_fatal_stub(const char *unused,...){(void)unused;abort();}
#define rt_fatal rt_fatal_stub
static void rt_log_stub(const char *format,...){assert(strstr(format,"graphics preparation input moved"));++moved_results;}
#define rt_log rt_log_stub
int rt_knob(const char *name,const char *value){(void)name;(void)value;return 0;}
static VkExecGraphicsResult graphics_restore_sample(VkExec *ex,VkExecGraphicsDraw *draw,VkTextureSample *sample,int nvstage){
    (void)ex;(void)draw;(void)sample;(void)nvstage;abort();
}
static int graphics_sample_wait(VkExec *ex,int logical,RcRegistryPosition attempted,VkExecGpaSampleScope **pending){
    (void)ex;(void)logical;(void)attempted;(void)pending;return 0;
}
static int graphics_sample_prefix_prepared(VkExecGraphicsDraw *draw,int selecting){(void)draw;(void)selecting;abort();}
VkTextureRestore *vk_texture_sample_restore_capture_locked(int logical){(void)logical;abort();}
void vk_texture_restore_cancel(VkTextureRestore *restore){(void)restore;abort();}
void vk_texture_alias_read_cancel(VkTextureAliasRead *alias){(void)alias;abort();}
void vk_texture_lock(void){abort();}
void vk_texture_unlock(void){abort();}
VkTextureSampleStatus vk_texture_sample_prepare_locked(int native,VkTextureSample *out){(void)native;(void)out;abort();}
VkTexturePresentSource *vk_texture_history_sample_retain_locked(int logical,uint64_t reg,VkImage image){
    (void)logical;(void)reg;(void)image;abort();
}
VkTexturePresentSource *vk_texture_present_retain_image(const VkTexImage *selected,VkImage image) {
    assert(selected==&resident.pub && exact_source && image==native_image && !live_refs);
    native_hold=(VkTexturePresentSource){image,resource_read};
    ++live_refs;
    return &native_hold;
}
static VkImageViewType res_view_type(uint16_t type){assert(type==2);return VK_IMAGE_VIEW_TYPE_2D;}
int gfx_shader_resources(const GfxShader *s,int kind,const GfxShaderRes **res){
    assert(s==&shader&&kind==GFX_RES_TEX);
    *res=&resource;return 1;
}
static VkExecPreparationReason image_alias_preparation_reason(VkTextureAliasPhase phase){assert(phase==VK_TEXTURE_ALIAS_COVERAGE);return VK_EXEC_PREPARATION_ORDER;}
VkCtx *vk_ctx(void){abort();}
static int texture_view_key(const VkTexImage *image,const view_t *view,int has_view,uint16_t type,int storage,ViewKey *key){
    (void)image;(void)view;(void)has_view;(void)type;(void)storage;(void)key;abort();
}
static uint32_t feedback_overlap(const VkExecTarget *colors,const VkExecTarget *depth,int count,const SubRange *range){
    (void)colors;(void)depth;(void)count;(void)range;abort();
}
static void graphics_release_gpa_inputs(VkExecGraphicsDraw *draw,int keep_arguments){
    assert(draw && keep_arguments); ++compatibility_releases;
}
void vk_exec_required_graphics_unavailable(VkExec *ex,uint32_t opcode,const char *reason){
    assert(ex->fence==SUBMISSION&&opcode==26&&reason);++positions_rejected;
}
typedef struct { int native; } TextureSlotSelection;
static TextureSlotSelection texture_slot_select_locked(int native,int private_output) {
    assert(!private_output && native<ntslots && !tex_deferred && !pool_ctx && !resolve_held);
    (void)agen;
    (void)sinfo;
    (void)resource_bridge;
    return (TextureSlotSelection){native};
}
static const VkTexImage *texture_slot_refresh_locked(TextureSlotSelection selected, view_t *view, int *has_view,
    VkTextureSampleStatus *status, VkTextureSampleRefusal *refused) {
    assert(selected.native==NATIVE_SLOT && status);
    if (view && has_view) slot_result_view(&tslots[NATIVE_SLOT],&resident,view,has_view);
    if (refused && !exact_source) sample_refusal_record(refused,&tslots[NATIVE_SLOT],&resident,"source-owned");
    return exact_source ? &resident.pub : NULL;
}
/* Extract resolve */
#if TEST_QUERY_GEOMETRY
// This fixture exercises the projected-content lane. A query-only grant is outside its shader proof.
static VkExecSampledFailure graphics_direct_sample_receipt(const VkExecGraphicsDraw *draw,
    const GfxShader *shader, const GfxShaderRes *resource, uint64_t handle, int member, int status,
    const VkTextureSample *sample, const VkTextureAliasFailure *alias, const VkTextureSampleRefusal *legacy,
    const char *gate, const char *geometry) {
    (void)draw; (void)shader; (void)resource; (void)handle; (void)member; (void)status;
    (void)sample; (void)alias; (void)legacy; (void)gate; (void)geometry;
    return (VkExecSampledFailure){.present=1};
}
static int sample_redirect_query_carrier_locked(GraphicsSample *sample, const VkTextureSample *source,
                                              const char **reason) {
    (void)sample; (void)source; (void)reason; abort();
}
static int graphics_try_geometry_sample(VkExec *ex, VkExecGraphicsDraw *draw, const GfxShader *shader,
    const GfxShaderRes *resource, int stage, uint64_t handle, int ordered, GraphicsSample *sample, const char **reason) {
    (void)ex; (void)draw; (void)shader; (void)resource; (void)stage; (void)handle; (void)ordered; (void)sample;
    assert(reason && !*reason);
    return 0;
}
static void graphics_log_geometry_refusal(const VkExecGraphicsDraw *draw, const GfxShaderRes *resource,
    int logical, int ordered, const char *reason) {
    (void)draw; (void)resource; (void)logical; (void)ordered;
    assert(!reason);
}
int vk_texture_query_image_current_locked(const VkTextureQueryImage *image) { (void)image; abort(); }
int vk_texture_sample_geometry_current_locked(int logical, const VkTextureSample *sample, const VkTexImage *exact,
    const VkTextureSampleGeometry *geometry, const RcInterval **ranges, uint32_t *count) {
    (void)logical; (void)sample; (void)exact; (void)geometry; (void)ranges; (void)count;
    abort();
}
static int graphics_geometry_final_binding(const VkGraphicsPipelineUse *use, const GraphicsSample *sample) {
    (void)use; (void)sample;
    abort();
}
static int graphics_geometry_peers_current(const VkExec *ex, const VkExecGraphicsDraw *draw,
    const GraphicsSample *sample, int active) {
    (void)ex; (void)draw; (void)sample; (void)active;
    abort();
}
static void graphics_sample_release(GraphicsSample *sample) {
    assert(sample->hold==&native_hold && live_refs==1 && !sample->alias && !sample->geometry.registration);
    sample->hold=NULL;
    live_refs--;
}
#endif

static int retained_compat_sample_current(const VkExec *ex,const GraphicsSample *sample) {
    (void)ex; (void)sample;
    return 0; // This fixture exclusively exercises native projected/alias providers.
}
static void retain_compat_sample(GraphicsSample *sample) {(void)sample;}
static void retained_sample_refusal(VkExec *ex,const GraphicsSample *sample,RcRegistryPosition read,
    uint32_t opcode,int stage,const char *reason) {
    (void)sample; (void)read; (void)stage;
    vk_exec_required_graphics_unavailable(ex,opcode,reason);
}
/* Extract sample */
static struct DrawCtx *dc;
static int n_skip_prog;
static const VkExecGraphicsPassShape *prepare_shape;
int vk_sync_stopping(void){return 0;}
volatile int32_t vk_lost;
int vk_exec_recording_failed(void){return 0;}
uint64_t vk_texture_registration_publication(void){return 21;}
uint32_t vk_sampler_gen(void){return 3;}
static int skip_prog_wanted(uint64_t serial){(void)serial;abort();}
static int graphics_inputs_match(const VkExec *ex,uint32_t opcode,const void *payload,const VkExecGraphicsDraw *draw){
    return ex->fence==SUBMISSION && ex->prog==&program && opcode==26 && payload==NULL && draw->attachment_seeds[0]==&load_seed;
}
static void graphics_log_identity(const VkExec *ex,uint32_t opcode,const void *payload,const VkExecGraphicsDraw *draw){
    (void)ex;(void)opcode;(void)payload;(void)draw;abort();
}
static VkExecGraphicsResult graphics_draw_select(VkExec *ex,VkExecGraphicsDraw *draw,int ordered,
    RcRegistryPosition attempted,VkExecGpaSampleScope **pending){
    uint32_t overlap=0;
    stage_result=graphics_stage_samples(ex,draw,prepare_shape,&shader,0,NVN_STAGE_VS,ordered,attempted,pending,&overlap);
    assert(!overlap && draw->attachment_seeds[0]==&load_seed && load_seed.live);
    if(stage_result==VK_EXEC_GRAPHICS_READY) draw->feedback_selected=1;
    return stage_result;
}
static VkExecGraphicsResult graphics_compat_pool_prepare(VkExec *ex,VkExecGraphicsDraw *draw,
    RcRegistryPosition attempted,VkExecGpaSampleScope **pending){
    (void)ex;(void)draw;(void)attempted;(void)pending;abort();
}
/* Extract prepare */
static VkExecGraphicsResult prepare(VkExec *ex,VkExecGraphicsDraw *draw,const VkExecGraphicsPassShape *shape,int ordered){
    VkExecGraphicsDraw *same=draw;
    VkExecGpaSampleScope *pending=NULL;
    prepare_shape=shape;
    VkExecGraphicsResult result=vk_exec_graphics_draw_prepare(ex,26,NULL,ordered,(RcRegistryPosition){0},&pending,&same);
    assert(same==draw&&!pending&&draw->attachment_seeds[0]==&load_seed&&load_seed.live);
    return result;
}
static void reset_draw(VkExecGraphicsDraw *draw){
#if TEST_QUERY_GEOMETRY
    for (uint32_t i=0;i<draw->sample_count;i++)
        if (draw->samples[i].hold) graphics_sample_release(&draw->samples[i]);
#endif
    free(draw->samples);
    *draw=(VkExecGraphicsDraw){.read_position=resource_read,.opcode=26,.attachment_seeds={&load_seed}};
}
static VkExecGraphicsResult stage(VkExec *ex,VkExecGraphicsDraw *draw,const VkExecGraphicsPassShape *shape,int ordered){
    uint32_t overlap=0;
    VkExecGpaSampleScope *pending=NULL;
    VkExecGraphicsResult result=graphics_stage_samples(ex,draw,shape,&shader,0,NVN_STAGE_VS,ordered,
        (RcRegistryPosition){0},&pending,&overlap);
    assert(!overlap&&!pending&&draw->attachment_seeds[0]==&load_seed&&load_seed.live);
    return result;
}
static int issue_retained(VkExec *ex,VkExecGraphicsDraw *draw){
    graphics_current=draw;
    VkTextureSample out={0};
    VkTextureAliasRead *alias=NULL;
    const VkTexturePresentSource *owner=NULL;
    const uint64_t handle=ex->tex[NVN_STAGE_VS][0];
    const int result=graphics_sample(ex,NVN_STAGE_VS,0,&resource,handle,&out,&alias,&owner);
    if (result!=1) return result;
    assert(!alias&&out.image&&out.image->image==native_image&&out.image->image!=load_seed.image);
    assert(out.registration==registration.handle.id&&out.native_slot==NATIVE_SLOT&&
           out.has_view&&out.view.set_levels&&out.view.base_level==2&&out.view.nlevels==1&&
           out.view.set_layers&&out.view.min_layer==1&&out.view.nlayers==1);
    assert(owner==&native_hold&&owner->image==native_image&&same_read(owner->read,draw->read_position)&&live_refs==1);
    ++native_handoffs;
    return 1;
}
static RcInterval whole_demand[100], selected_demand[100];
static RcInterval attachment_demand;
static uint32_t selected_count;
static void install_projection(void){
    nvn_tex_layout(&registration.descriptor.d,&registration.descriptor.L);
    resident.t=registration.descriptor;
    resident.key=(TexKey){resident.pub.addr};
    resident.image_private_mapping[0]=interval(1346,0x4000,resident.t.L.size,resident.pub.addr,1349);
    resident.image_private_mapping_count=1;
    resident.alias_fragment_owner=1;
    resident.f=(TexVkFormat){.vk=VK_FORMAT_R8_UNORM,.ideal=VK_FORMAT_R8_UNORM};
    resident.alias_full_demand=whole_demand;
    VkTextureAliasLayout layout;
    assert(texture_alias_layout(&resident.t.d,&resident.pub,&layout));
    assert(texture_alias_demand(&resident.t.d,&layout,resident.image_private_mapping,1,
        resident.pub.addr,resident.t.L.size,whole_demand,100,&resident.alias_full_count));
    assert(texture_alias_clear_planes((VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,2,1,1,1},&resident.pub,&layout));
    assert(texture_alias_demand(&resident.t.d,&layout,resident.image_private_mapping,1,
        resident.pub.addr,resident.t.L.size,selected_demand,100,&selected_count));
    attachment_demand=interval(1346,0x20000,64,0x8000020000,1559);
    uint64_t selected_bytes=0;
    for(uint32_t i=0;i<selected_count;i++){
        assert(!rc_range_overlap(selected_demand[i].range,attachment_demand.range));
        assert(selected_demand[i].mapping==1349);
        selected_bytes+=selected_demand[i].range.size;
    }
    assert(selected_bytes==64);
    registry_fixture=rc_registry_create((RcRegistryConfig){24,24,4});
    assert(registry_fixture);
    RcInterval whole=resident.image_private_mapping[0];
    RcRegistryImageWriteExtent material_owner=hold(registry_fixture,(uintptr_t)native_image,whole);
    RcRegistryImageWriteExtent render_owner=hold(registry_fixture,(uintptr_t)attachment_image,attachment_demand);
    RcRegistryWrite material_writer=declare_image(registry_fixture,whole,(RcRegistryPosition){SUBMISSION,3});
    RcRegistryWrite render_writer=declare_image(registry_fixture,attachment_demand,(RcRegistryPosition){SUBMISSION,5});
    OK(rc_registry_declare_through(registry_fixture,resource_read));
    for(uint32_t i=0;i<resident.alias_full_count;i++)
        assert(!rc_range_overlap(whole_demand[i].range,attachment_demand.range));
    RcSourceRange writes[100];
    for(uint32_t i=0;i<resident.alias_full_count;i++) writes[i]=whole_demand[i].range;
    for(uint32_t i=1;i<resident.alias_full_count;i++){
        for(uint32_t j=i;j>0 && writes[j].offset < writes[j-1].offset;j--){
            RcSourceRange tmp=writes[j]; writes[j]=writes[j-1]; writes[j-1]=tmp;
        }
    }
    publish(registry_fixture,(uintptr_t)native_image,material_owner,writes,resident.alias_full_count,(RcRegistryPosition){SUBMISSION,3});
    record(registry_fixture,material_writer,(uintptr_t)native_image,material_owner,(RcRegistryPosition){SUBMISSION,3});
    publish(registry_fixture,(uintptr_t)attachment_image,render_owner,&attachment_demand.range,1,(RcRegistryPosition){SUBMISSION,5});
    record(registry_fixture,render_writer,(uintptr_t)attachment_image,render_owner,(RcRegistryPosition){SUBMISSION,5});
    assert(rc_registry_native_image_covers(registry_fixture,(uintptr_t)native_image,selected_demand,selected_count,resource_read));
    assert(rc_registry_native_image_covers(registry_fixture,(uintptr_t)native_image,whole_demand,resident.alias_full_count,resource_read));
    assert(rc_registry_native_image_covers(registry_fixture,(uintptr_t)attachment_image,&attachment_demand,1,resource_read));
}

static void verify_unchanged_flat(void) {
    resident=(Tex){.pub={.image=native_image,.type=VK_IMAGE_TYPE_2D,.view_type=VK_IMAGE_VIEW_TYPE_2D,
        .width=8,.height=8,.depth=1,.addr=0x8000004000,.usage=VK_IMAGE_USAGE_SAMPLED_BIT,
        .format=VK_FORMAT_R8_UNORM,.aspect=VK_IMAGE_ASPECT_COLOR_BIT,.samples=1,.levels=1,.layers=1,.gpu_owned=1},
        .f={.vk=VK_FORMAT_R8_UNORM,.ideal=VK_FORMAT_R8_UNORM},.alias_fragment_owner=1,.image_private_mapping_count=1};
    registration.descriptor=(tex_t){.serial=480,.mem=(void *)(uintptr_t)resident.pub.addr,
        .d={.width=8,.height=8,.depth=1,.target=NVN_TEXTURE_TARGET_2D,.format=NVN_FORMAT_R8,.levels=1,.samples=1}};
    nvn_tex_layout(&registration.descriptor.d,&registration.descriptor.L);
    resident.t=registration.descriptor;
    resident.key=(TexKey){resident.pub.addr};
    resident.image_private_mapping[0]=interval(1346,0x4000,resident.t.L.size,resident.pub.addr,1349);
    VkTextureAliasLayout layout;
    assert(texture_alias_layout(&resident.t.d,&resident.pub,&layout));
    assert(texture_alias_demand(&resident.t.d,&layout,resident.image_private_mapping,1,resident.pub.addr,
                               resident.t.L.size,whole_demand,100,&resident.alias_full_count));
    resident.alias_full_demand=whole_demand;
    RcSourceRange writes[100];
    for(uint32_t i=0;i<resident.alias_full_count;i++) writes[i]=whole_demand[i].range;
    registry_fixture=rc_registry_create((RcRegistryConfig){24,24,4});
    assert(registry_fixture);
    const RcRegistryPosition original_read={SUBMISSION,HORIZON};
    resource_read=original_read;
    RcRegistryImageWriteExtent owner=hold(registry_fixture,(uintptr_t)native_image,resident.image_private_mapping[0]);
    RcRegistryWrite writer=declare_image(registry_fixture,resident.image_private_mapping[0],(RcRegistryPosition){SUBMISSION,3});
    OK(rc_registry_declare_through(registry_fixture,resource_read));
    publish(registry_fixture,(uintptr_t)native_image,owner,writes,resident.alias_full_count,(RcRegistryPosition){SUBMISSION,3});
    record(registry_fixture,writer,(uintptr_t)native_image,owner,(RcRegistryPosition){SUBMISSION,3});
    registration.view=(view_t){0};
    registration.has_view=0;
    tslots[NATIVE_SLOT].cview=(view_t){0};
    tslots[NATIVE_SLOT].chas_view=0;
    VkTextureSample sample = {0};
    RcRegistryPosition next = {0};
    assert(texture_projected_sample_locked(&tslots[NATIVE_SLOT], NATIVE_SLOT, native_image, &sample, &next) == &resident);
    assert(sample.projected && sample.image == &resident.pub && !sample.has_view &&
           sample.registration == registration.handle.id && sample.native_slot == NATIVE_SLOT);
    assert(next.scene == resource_read.scene && next.ordinal == resource_read.ordinal + 1);
    resident.alias_full_count=0;
    assert(!texture_projected_sample_locked(&tslots[NATIVE_SLOT],NATIVE_SLOT,native_image,NULL,NULL));
    rc_registry_destroy(registry_fixture);
}
static void verify_depth_plane(void) {
    resource_read=(RcRegistryPosition){SUBMISSION,HORIZON};
    resident=(Tex){.pub={.image=native_image,.type=VK_IMAGE_TYPE_2D,.view_type=VK_IMAGE_VIEW_TYPE_2D,
        .width=64,.height=32,.depth=1,.addr=0x8000004000,
        .usage=VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .format=VK_FORMAT_D32_SFLOAT,.aspect=VK_IMAGE_ASPECT_DEPTH_BIT,.samples=1,.levels=1,.layers=1,.gpu_owned=1},
        .f={.vk=VK_FORMAT_D32_SFLOAT,.ideal=VK_FORMAT_D32_SFLOAT,.depth=1},.alias_fragment_owner=1,
        .uploaded=1,.checked=submit_epoch,.image_private_mapping_count=1};
    registration=(TextureRegistration){.handle.id=17002,.has_view=1,
        .view={.set_dsmode=1,.dsmode=NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH},
        .descriptor={.serial=499,.mem=(void *)(uintptr_t)resident.pub.addr,
            .d={.width=64,.height=32,.depth=1,.target=NVN_TEXTURE_TARGET_2D,.format=NVN_FORMAT_DEPTH32F,.levels=1,.samples=1}}};
    nvn_tex_layout(&registration.descriptor.d,&registration.descriptor.L);
    resident.t=registration.descriptor;
    resident.key=(TexKey){resident.pub.addr};
    tslots[NATIVE_SLOT]=(TexSlot){.used=1,.registration=&registration,.chas_view=1,.cview=registration.view};
    const RcInterval mapping=interval(1346,0x4000,resident.t.L.size,resident.pub.addr,1349);
    resident.image_private_mapping[0]=mapping;
    const VkImageSubresourceRange plane={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1};
    uint32_t count=0;
    const RcInterval *demand=image_alias_plane_demand(&resident,&plane,&count);
    assert(demand&&count>0&&count<100);
    RcSourceRange writes[100];
    uint64_t meaningful=0;
    for (uint32_t i=0;i<count;i++) {
        writes[i]=demand[i].range;
        meaningful+=writes[i].size;
    }
    assert(meaningful==64*32*4);
    registry_fixture=rc_registry_create((RcRegistryConfig){24,24,4});
    assert(registry_fixture);
    RcRegistryImageWriteExtent owner=hold(registry_fixture,(uintptr_t)native_image,mapping);
    const RcRegistryPosition partial_position={SUBMISSION,3}, complete_position={SUBMISSION,5};
    RcRegistryWrite partial=declare_image(registry_fixture,mapping,partial_position);
    RcRegistryWrite complete=declare_image(registry_fixture,mapping,complete_position);
    OK(rc_registry_declare_through(registry_fixture,(RcRegistryPosition){SUBMISSION,4}));
    resource_read.ordinal=4;
    RcSourceRange clipped=writes[0];
    clipped.size/=2;
    publish(registry_fixture,(uintptr_t)native_image,owner,&clipped,1,partial_position);
    record(registry_fixture,partial,(uintptr_t)native_image,owner,partial_position);
    VkTextureSample sample;
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,VK_TEXTURE_READ_EXACT_DETAIL,&sample)==VK_TEXTURE_SAMPLE_FALLBACK);
    const char *gate=NULL;
    assert(!texture_projected_sample_report_locked(&tslots[NATIVE_SLOT],NATIVE_SLOT,native_image,&sample,NULL,&gate));
    assert(!strcmp(gate,"depth-private-plane"));
    VkTextureSampleRefusal refused={0};
    assert(vk_texture_resolve_sample_report_locked(LOGICAL_SLOT,VK_TEXTURE_READ_REQUIRED|VK_TEXTURE_READ_EXACT_DETAIL,
                                                  &sample,&refused)==VK_TEXTURE_SAMPLE_FALLBACK);
    assert(!sample.projected && sample.image==&resident.pub && refused.registration==registration.handle.id &&
           !strcmp(refused.stage,"depth-private-plane") && same_read(refused.position,resource_read));
    resident.pub.gpu_owned=0;
    assert(!texture_projected_sample_report_locked(&tslots[NATIVE_SLOT],NATIVE_SLOT,native_image,&sample,NULL,&gate));
    assert(!strcmp(gate,"depth-material"));
    resident.pub.gpu_owned=1;

    resource_read.ordinal=HORIZON;
    OK(rc_registry_declare_through(registry_fixture,resource_read));
    publish(registry_fixture,(uintptr_t)native_image,owner,writes,count,complete_position);
    record(registry_fixture,complete,(uintptr_t)native_image,owner,complete_position);
    assert(!resident.alias_full_count);
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,VK_TEXTURE_READ_EXACT_DETAIL,&sample)==VK_TEXTURE_SAMPLE_READY);
    assert(sample.projected && sample.registration==registration.handle.id && sample.image==&resident.pub);
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,VK_TEXTURE_READ_FORCE_COMPAT,&sample)==VK_TEXTURE_SAMPLE_FALLBACK);
    tslots[NATIVE_SLOT].cview.dsmode=NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL;
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,0,&sample)==VK_TEXTURE_SAMPLE_FALLBACK);
    tslots[NATIVE_SLOT].cview.dsmode=NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH;
    resident.pub.aspect|=VK_IMAGE_ASPECT_STENCIL_BIT;
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,0,&sample)==VK_TEXTURE_SAMPLE_FALLBACK);
    resident.pub.aspect=VK_IMAGE_ASPECT_DEPTH_BIT;
    resident.pub.scale=2;
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,0,&sample)==VK_TEXTURE_SAMPLE_FALLBACK);
    resident.pub.scale=1;
    resident.pub.height--;
    assert(!texture_projected_sample_report_locked(&tslots[NATIVE_SLOT],NATIVE_SLOT,native_image,&sample,NULL,&gate));
    assert(!strcmp(gate,"depth-native-shape"));
    resident.pub.height++;
    assert(texture_projected_sample_report_locked(&tslots[NATIVE_SLOT],NATIVE_SLOT,native_image,&sample,NULL,&gate));
    assert(!gate);
    registration.descriptor.d.levels++;
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,0,&sample)==VK_TEXTURE_SAMPLE_FALLBACK);
    registration.descriptor.d.levels--;
    current_mapping=0;
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,0,&sample)==VK_TEXTURE_SAMPLE_FALLBACK);
    current_mapping=1;
    const RcInterval separate=interval(1346,0x20000,64,0x8000020000,1559);
    RcRegistryWrite sibling=declare_image(registry_fixture,separate,(RcRegistryPosition){SUBMISSION,13});
    resource_read.ordinal=14;
    OK(rc_registry_declare_through(registry_fixture,resource_read));
    OK(rc_registry_image_producer_resolve(registry_fixture,sibling,(RcRegistryPosition){SUBMISSION,13},NULL,0));
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,0,&sample)==VK_TEXTURE_SAMPLE_READY);
    RcRegistryWrite overlap=declare_image(registry_fixture,demand[0],(RcRegistryPosition){SUBMISSION,15});
    resource_read.ordinal=16;
    OK(rc_registry_declare_through(registry_fixture,resource_read));
    OK(rc_registry_image_producer_resolve(registry_fixture,overlap,(RcRegistryPosition){SUBMISSION,15},NULL,0));
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,0,&sample)==VK_TEXTURE_SAMPLE_FALLBACK);
    OK(rc_registry_write_cancel(registry_fixture,overlap));
    assert(vk_texture_select_sample_locked(LOGICAL_SLOT,0,&sample)==VK_TEXTURE_SAMPLE_READY);
    free(resident.alias_plane_demand);
    rc_registry_destroy(registry_fixture);
}

int main(int argc, char **argv) {

    registration=(TextureRegistration){.handle.id=17001,.descriptor={.serial=412,.mem=(void *)(uintptr_t)0x8000004000,
        .d={.target=NVN_TEXTURE_TARGET_2D_ARRAY,.format=NVN_FORMAT_R8,.width=32,.height=32,.depth=2,.levels=3,.samples=1}}};
    resident=(Tex){
        .pub={.image=native_image,.type=VK_IMAGE_TYPE_2D,.view_type=VK_IMAGE_VIEW_TYPE_2D_ARRAY,
              .width=32,.height=32,.depth=1,.addr=(uintptr_t)registration.descriptor.mem,
              .usage=VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,.format=VK_FORMAT_R8_UNORM,
              .aspect=VK_IMAGE_ASPECT_COLOR_BIT,.samples=1,.levels=3,.layers=2,.gpu_owned=1,.preserve_pool_bytes=1},
                   .t.serial=registration.descriptor.serial,.checked=submit_epoch,.uploaded=1};
    tslots[NATIVE_SLOT]=(TexSlot){.used=1,.registration=&registration,.chas_view=1,
       .cview={.set_levels=1,.base_level=2,.nlevels=1,.set_layers=1,.min_layer=1,.nlayers=1}};
    install_projection();
    registration.view=tslots[NATIVE_SLOT].cview;
    registration.has_view=1;
    load_seed=(VkTextureAliasRead){.image=attachment_image,.live=1};
    VkExec ex={.fence=SUBMISSION,.prog=&program,.tex={[NVN_STAGE_VS]={LOGICAL_SLOT|(UINT64_C(3)<<20)}}};
    VkExecGraphicsPassShape shape={.pass_ncolor=1,.color={{.image=attachment_image,.layers=1}}};
    VkExecGraphicsDraw draw={0};

    reset_draw(&draw);
    is_absent=1;
    assert(stage(&ex,&draw,&shape,1)==VK_EXEC_GRAPHICS_READY);
    assert(draw.sample_count==1&&draw.samples[0].sample.historical==VK_TEXTURE_REGISTRATION_ABSENT&&
           !draw.samples[0].hold&&!draw.samples[0].image.image&&!legacy_uses);
    assert(!graphics_compatibility_begin(&draw)&&!compatibility_releases);
    is_absent=0;
    reset_draw(&draw);
    resource_read.ordinal=14;
    stage_result=-1;
    assert(prepare(&ex,&draw,&shape,1)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(stage_result==-1&&moved_results==1&&!draw.sample_count&&!live_refs);
    reset_draw(&draw);
    const RcRegistryPosition unknown_output={SUBMISSION,13};
    RcRegistryWrite unknown=declare_image(registry_fixture,selected_demand[0],unknown_output);
    OK(rc_registry_declare_through(registry_fixture,resource_read));
    OK(rc_registry_image_producer_resolve(registry_fixture,unknown,unknown_output,NULL,0));
    assert(rc_registry_native_image_covers(registry_fixture,(uintptr_t)attachment_image,&attachment_demand,1,resource_read));
    assert(!rc_registry_native_image_covers(registry_fixture,(uintptr_t)native_image,selected_demand,selected_count,resource_read));
    assert(prepare(&ex,&draw,&shape,0)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!draw.sample_count&&!live_refs);
    assert(prepare(&ex,&draw,&shape,1)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(draw.failure.coverage.cause==RC_NATIVE_IMAGE_BLOCK_UNKNOWN_OUTPUT&&!draw.sample_count&&!live_refs);
    OK(rc_registry_write_cancel(registry_fixture,unknown));
    assert(rc_registry_native_image_covers(registry_fixture,(uintptr_t)native_image,selected_demand,selected_count,resource_read));
    resource_read.ordinal=HORIZON;
    reset_draw(&draw);
    const int expect_blocked = argc == 2 && !strcmp(argv[1], "--expect-blocked");
    resource.exact=1; // A real retained projection satisfies the shader's positional full-detail read.
    const VkExecGraphicsResult outcome = prepare(&ex, &draw, &shape, 1);
    if (outcome != VK_EXEC_GRAPHICS_READY) {
        assert(outcome == VK_EXEC_GRAPHICS_UNAVAILABLE && stage_result == VK_EXEC_GRAPHICS_COMPAT);
        assert(draw.failure.owner == VK_EXEC_PREPARATION_IMAGE && draw.failure.role == NVN_STAGE_VS &&
               draw.failure.slot == LOGICAL_SLOT);
        assert(!draw.compatibility && !draw.sample_count && !live_refs && !compatibility_releases && load_seed.live);
        assert(rc_registry_native_image_covers(registry_fixture, (uintptr_t)native_image,
                                               selected_demand, selected_count, resource_read));
        reset_draw(&draw);
        rc_registry_destroy(registry_fixture);
        verify_unchanged_flat();
        puts("Baseline returned FALLBACK and could not retain the sample alongside the LOAD seed.");
        return expect_blocked ? 0 : 42;
    }
    assert(!expect_blocked);
    VkTextureSample lower = {0};
    RcRegistryPosition next = {0};
    assert(texture_projected_sample_locked(&tslots[NATIVE_SLOT],NATIVE_SLOT,native_image,&lower,&next)==&resident);
    assert(lower.projected&&lower.registration==registration.handle.id&&lower.image==&resident.pub&&
           lower.view.base_level==2&&lower.view.min_layer==1&&next.scene==SUBMISSION&&next.ordinal==HORIZON+1);
    uint32_t count=0;
    const RcInterval *actual_demand=image_alias_plane_demand(&resident,
        &(VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,2,1,1,1},&count);
    assert(actual_demand&&count==selected_count&&!memcmp(actual_demand,selected_demand,count*sizeof *actual_demand));
    assert(draw.sample_count==1&&!draw.compatibility&&!compatibility_releases&&issue_retained(&ex,&draw)==1);
    /* The prepared path validates its prefix before final receipt lookup. A stale input must not reach issue. */
    current_mapping=0;
    assert(!graphics_sample_shape_current(&draw.samples[0]));
    current_mapping=1;
    registration.view.set_format=1;
    registration.view.format=NVN_FORMAT_RGBA8;
    assert(!graphics_sample_shape_current(&draw.samples[0]));
    registration.view.set_format=0;
    tslots[NATIVE_SLOT].cview.min_layer=0;
    assert(!graphics_sample_shape_current(&draw.samples[0]));
    tslots[NATIVE_SLOT].cview.min_layer=1;
    assert(!texture_projected_sample_locked(&tslots[NATIVE_SLOT],NATIVE_SLOT,attachment_image,NULL,NULL));
    resident.t.serial++;
    assert(!graphics_sample_shape_current(&draw.samples[0]));
    resident.t.serial--;
    assert(graphics_sample_shape_current(&draw.samples[0]));
    assert(native_handoffs==1&&positions_rejected==0&&live_refs==1&&load_seed.live);
    resident.pub.skip=1;
    assert(!graphics_sample_shape_current(&draw.samples[0]) && load_seed.live);
    resident.pub.skip=0;
    const RcInterval sibling=whole_demand[0];
    for(uint32_t i=0;i<selected_count;i++) assert(!rc_range_overlap(sibling.range,selected_demand[i].range));
    const RcRegistryPosition output={SUBMISSION,15};
    RcRegistryWrite sibling_unknown=declare_image(registry_fixture,sibling,output);
    resource_read.ordinal=16;
    OK(rc_registry_declare_through(registry_fixture,resource_read));
    OK(rc_registry_image_producer_resolve(registry_fixture,sibling_unknown,output,NULL,0));
    assert(!rc_registry_native_image_covers(registry_fixture,(uintptr_t)native_image,whole_demand,
                                            resident.alias_full_count,resource_read));
    assert(rc_registry_native_image_covers(registry_fixture,(uintptr_t)native_image,selected_demand,selected_count,resource_read));
    assert(selected_queries>0 && ordered_resolve==0 && unsupported_reason==0);
    resident.alias_full_count=0;
    VkTextureSample still_selected={0};
    assert(texture_projected_sample_locked(&tslots[NATIVE_SLOT],NATIVE_SLOT,native_image,&still_selected,NULL));
    assert(still_selected.projected&&still_selected.view.base_level==2&&still_selected.view.min_layer==1);
    reset_draw(&draw);
    free(resident.alias_plane_demand);
    rc_registry_destroy(registry_fixture);
    verify_unchanged_flat();
    verify_depth_plane();
    puts("Projected full-detail samples and depth planes retain their original producers alongside attachment LOAD.");
    return 0;
}

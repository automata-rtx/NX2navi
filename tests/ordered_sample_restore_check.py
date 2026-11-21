"""Exercise the required draw's real sampled-member continuation and exact sibling checks."""
import os
import runpy
from pathlib import Path
import subprocess
import tempfile

function = runpy.run_path(str(Path(__file__).with_name("ordered_graphics_feedback_check.py")))["function"]


ROOT = Path(__file__).resolve().parents[1]
DRAW = Path(os.environ.get("GRAPHICS_RESTORE_SOURCE", ROOT / "runtime/vk_exec_draw.c")).read_text()

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NVN_STAGE_VS = 0, NVN_STAGE_FS = 1, NVN_STAGE_TASK = 6, NVN_STAGE_MESH = 7,
       VK_FEEDBACK_OFF = 0, VK_FEEDBACK_RESTART, VK_FEEDBACK_LAYOUT,
       GFX_RES_TEX, VK_IMAGE_TYPE_2D = 0, VK_IMAGE_ASPECT_DEPTH_BIT = 2,
       VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_3D, VK_TEXTURE_REGISTRATION_CURRENT = 0,
       VK_TEXTURE_REGISTRATION_EXACT = 1, VK_TEXTURE_REGISTRATION_ABSENT = -1 };
#define NVN_HANDLE_TEX(h) ((int)(h))
#define VK_NULL_HANDLE 0
#define VK_SUCCESS 0
#define VK_ERROR_OUT_OF_HOST_MEMORY -1
#define VK_ERROR_OUT_OF_DEVICE_MEMORY -2
typedef uint64_t VkImage;
typedef int VkResult;
typedef struct { VkImage image; int skip, type; uint32_t width, height, format, aspect; } VkTexImage;
typedef struct { uint32_t width; } VkExtent2D;
typedef struct { int id; } GfxShader;
typedef struct { int pool, slot, count, exact, type; unsigned binding; } GfxShaderRes;
typedef struct { GfxShader *vs,*ts,*fs; uint8_t feedback; } VkPipeState;
typedef struct { VkImage image; } VkExecTarget;
typedef struct {
    VkExecTarget color[8], depth;
    uint32_t pass_ncolor, feedback_dirty, fbl_mask;
    int in_pass;
    VkExtent2D tonemap_extent;
} VkExecGraphicsPassShape;
typedef struct { uint32_t sampled, layouts, dirty; int restart; } GraphicsFeedback;
typedef struct { uint64_t tex[8][3]; } VkExec;
typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct { uint64_t address, mapping; struct { uint64_t backing,offset,size; } range; } RcInterval;
typedef int VkTextureAliasPhase;
typedef struct { unsigned gate; } RcRegistryNativeImageFailure;
typedef struct { VkTextureAliasPhase phase; int status; RcInterval receiver; uint64_t registration,material_serial;
    VkImage image; RcRegistryNativeImageFailure coverage; const char *layout_gate; } VkTextureAliasFailure;
enum { VK_TEXTURE_ALIAS_ORDER, VK_TEXTURE_ALIAS_MAPPING, VK_TEXTURE_ALIAS_LAYOUT, VK_TEXTURE_ALIAS_COVERAGE };
enum { RC_REGISTRY_OK, RC_REGISTRY_STALE, RC_REGISTRY_UNSUPPORTED, RC_REGISTRY_CAPACITY, RC_REGISTRY_BLOCKED };
typedef struct { unsigned id; } VkExecGpaSampleScope;
typedef struct { int live; } VkExecGraphicsPass, VkExecGraphicsScratch, VkGraphicsPipelineWait;
typedef uintptr_t VkImageView;
typedef struct { int live; } VkTextureAliasRead;
enum { VK_EXEC_ALIAS_READ_LIMIT = 8 };
#if REAL_DRAW_POOL
typedef struct { int held, current; } VkExecPoolGrant;
#endif
typedef struct { int revision; } VkGraphicsPipelineUse;
typedef struct { int input; } VkMemSpan, VkMemSpanCapture;
typedef struct { unsigned image; } VkTexturePresentSource;
typedef struct { unsigned logical; } VkTextureRestore;
typedef struct { uint64_t registration; int native_slot, canonical, unbacked, projected; } VkTextureReadKey;
typedef struct {
    const VkTexImage *image;
    int view, has_view, native_slot, historical, canonical, projected;
    uint64_t registration;
    void *query_image;
} VkTextureSample;
static int vk_texture_query_image_current_locked(const void *image) { (void)image; assert(0); return 0; }
static void vk_texture_query_image_release(void **image) { assert(image && !*image); }
typedef struct { uint64_t registration; } VkTextureSampleGeometry;
typedef struct { uint64_t registration; } VkTextureSampleRefusal;
typedef struct { VkImage image; uint32_t level0, levels, layer0, layers; } ViewKey;
typedef struct { VkImage image; uint32_t l0, l1, a0, a1; } SubRange;
typedef enum { VK_TEXTURE_SAMPLE_READY, VK_TEXTURE_SAMPLE_PREPARE, VK_TEXTURE_SAMPLE_FALLBACK,
    VK_TEXTURE_SAMPLE_DEFERRED, VK_TEXTURE_SAMPLE_UNAVAILABLE, VK_TEXTURE_SAMPLE_ABSENT, VK_TEXTURE_SAMPLE_GONE,
    VK_TEXTURE_SAMPLE_CANONICAL_REFUSED, VK_TEXTURE_SAMPLE_UNBACKED } VkTextureSampleStatus;
typedef enum { VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_COMPAT, VK_EXEC_GRAPHICS_NEEDS_ORDERED,
    VK_EXEC_GRAPHICS_UNAVAILABLE, VK_EXEC_GRAPHICS_STOPPED, VK_EXEC_GRAPHICS_INVALIDATED } VkExecGraphicsResult;
enum { VK_EXEC_PREPARATION_NONE, VK_EXEC_PREPARATION_IMAGE, VK_EXEC_PREPARATION_ORDER, VK_EXEC_PREPARATION_MAPPING,
    VK_EXEC_PREPARATION_ATTACHMENT, VK_EXEC_PREPARATION_PLACEMENT, VK_EXEC_PREPARATION_PRESSURE,
    VK_EXEC_PREPARATION_FEEDBACK };
typedef int VkExecPreparationReason;
typedef struct {
    RcRegistryPosition read;
    const GfxShader *shader;
    const GfxShaderRes *resource;
    const char *gate;
    uint64_t program, handle;
    VkImage image;
    int logical, member, status, present;
    uint8_t attachment_seed;
} VkExecSampledFailure;
typedef struct { int owner,reason; VkExecGraphicsResult result; int detail; uint32_t role,slot; RcInterval range;
    uint64_t registration,material_serial; VkImage image; RcRegistryNativeImageFailure coverage; const char *stage;
    VkTextureSampleRefusal sampled_source;
    VkExecSampledFailure sampled;
    struct { RcRegistryPosition position; int member; VkTextureSampleStatus status; int present; } sample;
} VkExecPreparationFailure;
typedef enum { VK_EXEC_RESTORE_READY, VK_EXEC_RESTORE_STOPPED, VK_EXEC_RESTORE_STALE,
    VK_EXEC_RESTORE_FAILED } VkExecRestoreResult;
enum { OP_DRAW_MESH_IND = 8, VK_TEXTURE_READ_REQUIRED = 1, VK_TEXTURE_READ_FORCE_COMPAT = 2,
       VK_TEXTURE_READ_EXACT_DETAIL = 8,
       VK_TEXTURE_READ_LINEAR_RGBA8_VIEW = 4 };
typedef struct {
    const GfxShader *shader;
    const GfxShaderRes *resource;
    uint64_t handle, token;
    VkTextureSample sample;
    VkTexImage image;
    VkTexturePresentSource *hold;
    VkTextureAliasRead *alias;
    VkTextureSampleGeometry geometry;
    int stage, member, seen;
} GraphicsSample;
typedef struct {
    struct { VkExecGraphicsScratch *scratch; unsigned descriptors; } task, mesh;
    VkExecGraphicsScratch *scratch_owner;
    struct { int native; } task_buffer;
} MfbDraw;
static void mfb_task_pointer_release(MfbDraw *mfb) { assert(mfb && !mfb->task_buffer.native); }
typedef struct { VkMemSpanCapture *native; void *mapped; VkMemSpan span; } EffectStorageCapture;
typedef struct { int prepared; EffectStorageCapture output; } DeviceStoragePointer;
typedef struct {
    GraphicsSample *samples;
    uint32_t sample_count;
    uint64_t scene, program_serial;
    const void *payload;
    RcRegistryPosition read_position;
    VkTextureRestore *sample_restore;
    const GfxShader *restore_shader;
#if REAL_REPLAN
    const GfxShaderRes *restore_resource;
#else
    int restore_resource;
    uint32_t selected_overlap;
#endif
    uint64_t restore_handle;
    int restore_stage, restore_member, mesh;
    int activated, compatibility, feedback_selected, hud_incomplete, geometry_present;
    VkExecGraphicsPass *pass;
    MfbDraw *fallback;
    void *storage, *selected_writers;
    DeviceStoragePointer *pointer_sources;
    int pointer_outputs_ready;
    unsigned descriptors;
    uint32_t pointer_count;
#if REAL_DRAW_POOL
    VkExecPoolGrant *pool_grant;
#endif
    VkExecGraphicsScratch *scratch;
    VkGraphicsPipelineWait *wait;
    VkGraphicsPipelineUse use;
    VkMemSpan index, indirect;
    VkMemSpanCapture *index_hold, *indirect_hold;
#if REAL_VERTEX_INPUT
    int exact_index, vertices;
#endif
    VkPipeState state;
    GraphicsFeedback feedback;
    VkExecPreparationFailure failure;
    VkTextureAliasRead *attachment_seeds[9];
    uint32_t opcode;
} VkExecGraphicsDraw;

static const GfxShaderRes resources[] = {{.slot=0,.count=3}};
static VkTexImage images[] = {{.image=101}, {.image=202}, {.image=303}};
static VkTexturePresentSource holds[3];
static VkTextureRestore retained;
static int locked, sibling_current, change_sibling, remove_restore_slot;
static int override_selection, refuse_alias, refuse_native_hold;
static int select_alias, remap_alias, select_geometry;
static unsigned alias_captures, alias_cancels, alias_uploads;
static RcRegistryPosition alias_read;
static VkTextureAliasRead captured_alias;
static const VkTexImage alias_image={.image=515};
static VkTextureSampleStatus selection_status, resolve_status;
static const RcInterval selected_range={.address=24,.mapping=7,.range={.backing=5,.offset=32,.size=64}};
static unsigned queries[3], keeps[3], drops[3], captured, prepared, installed, canceled, sibling_checks;
static RcRegistryPosition position = {19, 22};
#if REAL_REPLAN
static int scratch_finished, pipeline_released, pass_finished, canonical_refresh;
#if REAL_DRAW_POOL
static int pool_released;
static int vk_exec_pool_grant_current(const VkExec *ex,const VkExecPoolGrant *grant) {
    assert(ex); return grant && grant->held && grant->current;
}
static void vk_exec_pool_grant_release(VkExecPoolGrant **out) {
    if (*out) { assert((*out)->held); (*out)->held=0; *out=NULL; pool_released++; }
}
#endif
static RcRegistryPosition vk_exec_resource_position(void) { return position; }
static int rc_registry_position_valid(RcRegistryPosition now) { return now.scene && now.ordinal; }
static int rc_registry_position_compare(RcRegistryPosition a, RcRegistryPosition b) {
    return a.scene != b.scene || a.ordinal != b.ordinal;
}
static int graphics_inputs_match(const VkExec *ex, uint32_t opcode, const void *payload, const VkExecGraphicsDraw *draw) {
    return ex && draw && opcode == draw->opcode && payload == draw->payload;
}
static int vk_mem_span_capture_current(const VkMemSpanCapture *hold, const VkMemSpan *span) {
    return hold->input == span->input;
}
#if REAL_VERTEX_INPUT
#if REAL_INDEXED_SNAPSHOT
static void indexed_draw_snapshot_release(VkExecGraphicsDraw *draw) { assert(draw); }
#endif
static void vk_mem_span_capture_release(VkMemSpanCapture **hold) { assert(*hold);*hold=NULL; }
static int graphics_vertices_current(const VkExec *ex,const VkExecGraphicsDraw *draw) { assert(ex&&draw);return 1; }
static void vk_exec_graphics_vertices_release(int *vertices) { assert(vertices&&!*vertices); }
#endif
static int vk_exec_graphics_pass_current(const VkExec *ex, const VkExecGraphicsPass *pass) {
    assert(ex); return pass->live;
}
static void vk_exec_graphics_pass_cancel(VkExecGraphicsPass **pass) { assert(*pass); *pass = NULL; pass_finished++; }
static void vk_exec_draw_graphics_scratch_finish(VkExecGraphicsScratch **scratch) {
    if (*scratch) { scratch_finished++; *scratch = NULL; }
}
static void vk_exec_storage_phase_cancel(void **storage) { assert(!*storage); }
static void selected_storage_writers_release(void **writers) { assert(writers && !*writers); }
static void draw_descriptor_grant_release(unsigned *grant) { assert(grant && !*grant); }
static int graphics_ubo_merges_current(const VkExec *ex,const VkExecGraphicsScratch *scratch) {
    assert(ex);(void)scratch;return 1;
}
static int device_shader_pointers_current(void *sources, uint32_t count) { assert(!sources && !count); return 1; }
static int device_storage_pointer_current(const DeviceStoragePointer *pointer) {
    assert(pointer); return 1;
}
static int device_storage_pointer_value_current(const VkExec *ex, const DeviceStoragePointer *pointer) {
    assert(ex && pointer); return pointer->prepared;
}
static void graphics_device_pointers_release(VkExecGraphicsDraw *draw) { (void)draw; assert(0); }
static void vk_pipeline_graphics_cancel(VkGraphicsPipelineWait **wait) { assert(!*wait); }
static void vk_pipeline_graphics_release(VkGraphicsPipelineUse *use) {
    if (use->revision) { pipeline_released++; use->revision = 0; }
}
#endif
static VkTextureRestore *active;
static VkExecRestoreResult progress_result;
static void *vk_dlaa_swap_img;
static struct { int feedback; } context;
static const typeof(context) *vk_ctx(void) { return &context; }
static void vk_texture_lock(void) { assert(!locked); locked=1; }
static void vk_texture_unlock(void) { assert(locked); locked=0; }
static uint64_t token_of(uint64_t handle) { assert(locked); return handle+20; }
static int vk_texture_sample_unbacked_current_locked(VkTextureReadKey key) {
    (void)key; assert(0); return 0;
}
static int vk_texture_registration_expected_locked(int logical, VkImage *exact) {
    assert(locked && logical>=1 && logical<=3 && !exact);
    if ((logical==2 && remove_restore_slot) ||
        (logical==1 && override_selection && selection_status==VK_TEXTURE_SAMPLE_ABSENT))
        return VK_TEXTURE_REGISTRATION_ABSENT;
    return VK_TEXTURE_REGISTRATION_CURRENT;
}
static int vk_texture_resolved_receipt_locked(VkTextureReadKey key, VkImage expected, void *next) {
    assert(locked && !next && key.canonical && key.registration == (uint64_t)(25+key.native_slot));
    if (key.native_slot==0) {
        sibling_checks++;
        return sibling_current && expected==images[0].image;
    }
    return expected==images[key.native_slot].image;
}
static int vk_texture_binding_receipt_locked(int slot, VkImage image, void *next) {
    (void)slot; (void)image; (void)next; assert(0); return 0;
}
static VkTextureSample ready_sample(int native) {
    return (VkTextureSample){.image=&images[native],.native_slot=native,.canonical=1,.registration=25+native};
}
static VkTextureSampleStatus vk_texture_ordered_sample_select_locked(int logical,RcRegistryPosition read,
    unsigned flags,unsigned options,VkTextureSample *sample,VkTextureAliasRead **alias,VkTexImage *shape,
    VkTextureAliasFailure *failure) {
    assert(locked && !flags && !options && logical>=1 && logical<=3 && !*alias && failure);
    assert(read.scene==position.scene && read.ordinal==position.ordinal);
    int native=logical-1;
    *sample=(VkTextureSample){.native_slot=native};
    queries[native]++;
    if (refuse_alias) {
        *failure=(VkTextureAliasFailure){.phase=VK_TEXTURE_ALIAS_COVERAGE,.status=RC_REGISTRY_BLOCKED,
            .receiver=selected_range,.registration=825,.material_serial=903,.image=444,.coverage={10}};
        return VK_TEXTURE_SAMPLE_UNAVAILABLE;
    }
    if (select_alias) {
        assert(logical==1 && !captured_alias.live);
        captured_alias.live=1;
        alias_read=read;
        alias_captures++;
        sample->registration=42;
        *alias=&captured_alias;
        *shape=(VkTexImage){0};
        return VK_TEXTURE_SAMPLE_READY;
    }
    if (override_selection && selection_status!=VK_TEXTURE_SAMPLE_READY) return selection_status;
    if (native==1) return VK_TEXTURE_SAMPLE_PREPARE;
    *sample=ready_sample(native);
    return VK_TEXTURE_SAMPLE_READY;
}
#if REAL_REPLAN
static VkTextureSampleStatus vk_texture_sample_prepare_locked(int native, VkTextureSample *sample) {
    assert(locked && native==0);
    canonical_refresh++;
    sibling_current=1;
    images[0].image=404;
    *sample=ready_sample(native);
    return VK_TEXTURE_SAMPLE_READY;
}
#endif
static VkTextureSampleStatus vk_texture_resolve_sample_report_locked(int logical, unsigned flags, VkTextureSample *sample,
    VkTextureSampleRefusal *refused) {
    assert(locked && logical == 1 && override_selection && flags == VK_TEXTURE_READ_REQUIRED && sample && refused);
    if (resolve_status==VK_TEXTURE_SAMPLE_FALLBACK) *sample=ready_sample(0);
    return resolve_status;
}
static VkTextureRestore *vk_texture_sample_restore_capture_locked(int logical) {
    assert(locked && !active && logical==2);
    captured++; retained.logical=logical; active=&retained; return active;
}
static int vk_texture_sample_restore_current(void *ptr) {
    assert(!locked && ptr==active);
    return !remove_restore_slot;
}
static VkExecRestoreResult vk_exec_prepare_texture_restore(VkExec *ex,VkTextureRestore *exact,
    int (*current)(void *),void *arg,const VkTexImage **selected,VkResult *failure) {
    assert(!locked && ex && exact==active && arg==active && exact->logical==2);
    prepared++;
    if (!current(arg)) return VK_EXEC_RESTORE_STALE;
    if (change_sibling) sibling_current=0;
    *selected=&images[1]; *failure=-17;
    return progress_result;
}
static VkTextureSampleStatus vk_texture_sample_restore_install_locked(VkTextureRestore *exact, VkTextureSample *sample) {
    assert(locked && exact==active);
    installed++; *sample=ready_sample(1); return VK_TEXTURE_SAMPLE_READY;
}
static void vk_texture_restore_cancel(VkTextureRestore *exact) {
    if (!exact) return;
#if REAL_REPLAN
    assert(!locked);
#endif
    assert(exact==active); active=NULL; canceled++;
}
static int vk_texture_history_sample_current_locked(int logical,uint64_t registration,VkImage image) {
    (void)logical;(void)registration;(void)image;assert(0);return 0;
}
static VkTexturePresentSource *vk_texture_history_sample_retain_locked(int logical,uint64_t registration,VkImage image) {
    (void)logical;(void)registration;(void)image;assert(0);return NULL;
}
static VkTexturePresentSource *vk_texture_present_retain_image(const VkTexImage *img,VkImage expected) {
    assert(locked && img>=images && img<&images[3] && expected==img->image);
    if (refuse_native_hold) return NULL;
    int i=(int)(img-images); keeps[i]++; return &holds[i];
}
static void vk_texture_present_release(VkTexturePresentSource *hold) {
    if (!hold) return;
    assert(!locked && hold>=holds && hold<&holds[3]);
    drops[hold-holds]++;
}
static void vk_exec_graphics_failure_set(VkExecGraphicsDraw *draw,const VkExecPreparationFailure *failure) {
    if (!draw->failure.owner) draw->failure=*failure;
}
static int vk_texture_alias_sample_current_locked(int logical, VkTextureAliasRead *alias);
static int vk_texture_alias_sample_selection_current_locked(int logical, VkTextureAliasRead *alias) {
    return vk_texture_alias_sample_current_locked(logical, alias);
}
static int vk_texture_alias_sample_current_locked(int logical, VkTextureAliasRead *alias) {
    assert(locked && logical==1 && alias==&captured_alias && alias->live);
    return !remap_alias && alias_read.scene==position.scene && alias_read.ordinal==position.ordinal;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *alias) {
    if (!alias) return;
    assert(alias==&captured_alias && alias->live);
    alias->live=0;
    alias_cancels++;
}
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *alias,uint32_t *count) {
    assert(alias==&captured_alias && alias->live); *count=1; return &selected_range;
}
static VkExecRestoreResult vk_exec_prepare_alias_texture_read(VkExec *ex, VkTextureAliasRead *const *aliases,
    uint32_t count, int (*current)(void *), void *scope, const VkTexImage **out) {
    assert(ex && !locked && count==1 && aliases[0]==&captured_alias && captured_alias.live);
    if (!current(scope)) return VK_EXEC_RESTORE_STALE;
    alias_uploads++;
    out[0]=&alias_image;
    return VK_EXEC_RESTORE_READY;
}
static VkImageView alias_sample_view_locked(VkTextureAliasRead *alias, const VkTexImage *image, const int *view,
    int has_view, uint16_t type) {
    assert(locked && alias==&captured_alias && image->image==alias_image.image && view && !has_view && !type);
    return 1;
}
static int vk_texture_sample_geometry_current_locked(int logical,const VkTextureSample *sample,const VkTexImage *image,
    const VkTextureSampleGeometry *geometry,void *unused,void *also_unused) {
    assert(locked && logical==1 && sample->registration==geometry->registration &&
           image->image==images[0].image && !unused && !also_unused);
    return 1;
}
static VkExecSampledFailure graphics_direct_sample_receipt(const VkExecGraphicsDraw *draw,
    const GfxShader *shader, const GfxShaderRes *resource, uint64_t handle, int member, int status,
    const VkTextureSample *sample, const VkTextureAliasFailure *alias, const VkTextureSampleRefusal *legacy,
    const char *gate, const char *geometry) {
    (void)alias; (void)legacy; (void)geometry;
    return (VkExecSampledFailure){.read=draw->read_position, .program=draw->program_serial,
        .shader=shader, .resource=resource, .logical=NVN_HANDLE_TEX(handle), .handle=handle,
        .member=member, .status=status, .image=sample && sample->image ? sample->image->image : 0,
        .gate=gate, .present=1};
}
static void graphics_drop_geometry_samples(VkExecGraphicsDraw *draw) {
    assert(draw && !draw->geometry_present);
}
static void graphics_release_gpa_inputs(VkExecGraphicsDraw *draw, int keep_arguments) {
    assert(draw && keep_arguments && !draw->sample_count && !draw->failure.owner);
}
static int sample_redirect_query_carrier_locked(GraphicsSample *sample, const VkTextureSample *source,
                                              const char **reason) {
    (void)sample; (void)source; (void)reason; abort();
}
static int graphics_try_geometry_sample(VkExec *ex,VkExecGraphicsDraw *draw,const GfxShader *shader,
    const GfxShaderRes *resource,int stage,uint64_t handle,int ordered,GraphicsSample *sample,const char **reason) {
    assert(ex && draw && shader && resource && handle && reason);
    if (!select_geometry) return 0;
    assert(ordered && !stage && handle==1);
    *sample=(GraphicsSample){.shader=shader,.resource=resource,.handle=handle,.sample=ready_sample(0),
        .image=images[0],.geometry={.registration=25},.seen=1};
    return 1;
}
static VkExecGpaSampleScope *vk_exec_gpa_sample_enter(VkExec *ex,RcRegistryPosition position) {
    (void)ex; (void)position; assert(0); return NULL;
}
static int vk_texture_sample_wait_gpa_locked(VkExec *ex,int logical) { (void)ex;(void)logical;assert(0);return 0; }
static int vk_exec_gpa_sample_leave(VkExecGpaSampleScope *scope) { (void)scope;assert(0);return 0; }
static void vk_exec_gpa_sample_cancel(VkExecGpaSampleScope *scope) { (void)scope;assert(0); }
static int gfx_shader_resources(const GfxShader *shader,int kind,const GfxShaderRes **out) {
    assert(shader && kind==GFX_RES_TEX);*out=resources;return 1;
}
static int res_view_type(int type) { return type; }
static int texture_view_key(const VkTexImage *image,const int *view,int has_view,int type,int storage,ViewKey *key) {
    (void)view;(void)has_view;(void)type;
    assert(context.feedback && !storage);
    *key=(ViewKey){.image=image->image,.levels=1,.layers=1};
    return 1;
}
static uint32_t feedback_overlap(const VkExecTarget *color,const VkExecTarget *depth,unsigned n,const SubRange *range) {
    assert(context.feedback && color && depth && n==1 && range->l1==1 && range->a1==1);
    return (range->image==depth->image ? 1u<<8 : 0) | (range->image==color[0].image ? 1u : 0);
}
static GraphicsFeedback feedback_plan_shape(const VkExecTarget *color,const VkExecTarget *depth,unsigned n,
    uint32_t overlap,uint32_t dirty,uint32_t native,int reopen,int tonemap,int mode) {
    assert(color&&depth&&n==1&&!tonemap);
    if (mode!=VK_FEEDBACK_OFF) {
        assert(!reopen);
        const uint32_t pending=overlap&dirty&~(mode==VK_FEEDBACK_LAYOUT ? native : 0u);
        return (GraphicsFeedback){.sampled=overlap,.dirty=overlap&dirty,.restart=!!pending};
    }
    assert(!overlap&&!dirty&&!native&&reopen);
    return (GraphicsFeedback){0};
}
static void rt_fatal(const char *msg) { assert(msg); abort(); }
static int rt_knob(const char *key,const char *fallback) { assert(key && fallback); return 0; }
static void rt_log(const char *fmt,...) { assert(fmt); }
'''

CASES = r'''
static void test(VkExecRestoreResult progress, int stale_sibling, int changed_slot, int stage) {
    static GfxShader shader, older;
    VkExec ex={0};
    VkExecGraphicsPass pass={.live=1};
    VkExecGraphicsScratch scratch={.live=1};
    VkExecGraphicsDraw draw={.read_position=position,.payload=&ex,.pass=&pass};
    int pending_stage;
    if (stage==NVN_STAGE_TASK) {
        draw.mesh=1; draw.state.vs=&older; draw.state.ts=&shader;
        ex.tex[NVN_STAGE_MESH][0]=1;
        if (stale_sibling) ex.tex[NVN_STAGE_MESH][2]=3;
        pending_stage=1;
    } else if (stage==NVN_STAGE_FS) {
        draw.state.vs=&older; draw.state.fs=&shader;
        ex.tex[NVN_STAGE_VS][0]=1;
        pending_stage=2;
    } else {
        draw.state.vs=&shader;
        ex.tex[NVN_STAGE_VS][0]=1;
        pending_stage=0;
    }
    ex.tex[stage][1]=2; ex.tex[stage][2]=3;
    VkExecGraphicsPassShape shape={.pass_ncolor=1};
    images[0].image=101;
    sibling_current=1; change_sibling=stale_sibling; remove_restore_slot=0; progress_result=progress;
    for (int i=0;i<3;i++) queries[i]=keeps[i]=drops[i]=0;
    captured=prepared=installed=canceled=sibling_checks=0;
#if REAL_REPLAN
    scratch_finished=pipeline_released=pass_finished=canonical_refresh=0;
#endif
    int prefix_count=stale_sibling ? 2 : 1;
    assert(graphics_feedback_select(&ex,&draw,&shape,0,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(draw.sample_count==(unsigned)prefix_count && draw.sample_restore==&retained &&
           draw.restore_stage==pending_stage && draw.restore_member==1 && draw.restore_handle==2);
    assert(!locked && captured==1 && keeps[0]==1 && queries[0]==1 && !prepared);
    remove_restore_slot=changed_slot;
    VkExecGraphicsResult got=graphics_feedback_select(&ex,&draw,&shape,1,(RcRegistryPosition){0},NULL);
    assert(!locked && captured==1 && prepared==1 && keeps[0]==1 && queries[0]==1 && sibling_checks);
    if (stale_sibling) {
        // The native work already happened. Finish installing once before pruning the invalid old sibling.
        assert(installed==1 && canceled==1 && !active && keeps[1]==1 && got==VK_EXEC_GRAPHICS_INVALIDATED);
        assert(draw.failure.owner==VK_EXEC_PREPARATION_IMAGE && draw.failure.reason==VK_EXEC_PREPARATION_MAPPING &&
               draw.failure.result==got && draw.failure.role==NVN_STAGE_MESH && draw.failure.slot==1 &&
               draw.failure.detail==RC_REGISTRY_STALE);
        draw.failure=(VkExecPreparationFailure){0};
#if REAL_REPLAN
        draw.scratch=&scratch; draw.use.revision=1; draw.feedback_selected=1;
        position.ordinal++;
        assert(!vk_exec_graphics_draw_replan(&ex,draw.opcode,&ex,&draw));
        position.ordinal--;
        draw.activated=1;
        assert(!vk_exec_graphics_draw_replan(&ex,draw.opcode,&ex,&draw));
        draw.activated=0;
        assert(!drops[0] && !drops[1] && !drops[2] && !scratch_finished && !pipeline_released);
        assert(vk_exec_graphics_draw_replan(&ex,draw.opcode,&ex,&draw));
        assert(drops[0]==1 && !drops[1] && !drops[2] && draw.sample_count==2);
        assert(scratch_finished==1 && pipeline_released==1 && !draw.feedback_selected && !pass_finished);
        assert(!draw.state.vs && !draw.state.ts); // caller recomputes shader answers from the actual source
        draw.state.vs=&older; draw.state.ts=&shader;
        assert(graphics_feedback_select(&ex,&draw,&shape,1,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
        assert(draw.sample_count==4 && captured==1 && prepared==1 && installed==1 && canceled==1);
        assert(queries[0]==2 && canonical_refresh==1 && keeps[0]==2 && queries[1]==1 && keeps[1]==1);
        assert(queries[2]==2 && keeps[2]==2); // same raw handle at two distinct stage/member positions
        draw.scratch=&scratch; draw.use.revision=2;
        assert(vk_exec_graphics_draw_replan(&ex,draw.opcode,&ex,&draw));
        assert(draw.scratch==&scratch && draw.use.revision==2 && scratch_finished==1 && pipeline_released==1);
#if REAL_DRAW_POOL
        VkExecPoolGrant grant = {.held = 1, .current = 1};
        draw.pool_grant = &grant;
        assert(vk_exec_graphics_draw_replan(&ex, draw.opcode, &ex, &draw));
        assert(draw.pool_grant == &grant && draw.sample_count == 4 && draw.use.revision == 2);
        grant.current = 0;
        assert(vk_exec_graphics_draw_replan(&ex, draw.opcode, &ex, &draw));
        assert(!draw.pool_grant && !grant.held && pool_released == 1 &&
               draw.sample_count == 4 && draw.use.revision == 2);
#endif
        VkMemSpanCapture invalid_span={2};
        draw.index_hold=&invalid_span;
        pass.live=0;
        assert(!vk_exec_graphics_draw_replan(&ex,draw.opcode,&ex,&draw));
        assert(draw.pass==&pass && !pass_finished);
        draw.index.input=2;
        assert(vk_exec_graphics_draw_replan(&ex,draw.opcode,&ex,&draw));
        assert(!draw.pass && pass_finished==1 && !draw.state.vs && draw.sample_count==4);
#endif
    } else if (changed_slot || progress!=VK_EXEC_RESTORE_READY) {
        assert(got==(progress==VK_EXEC_RESTORE_STOPPED?VK_EXEC_GRAPHICS_STOPPED:VK_EXEC_GRAPHICS_UNAVAILABLE));
        assert(draw.sample_count==1 && !installed && !canceled && !keeps[1] && !queries[2]);
        if (changed_slot)
            assert(draw.failure.owner==VK_EXEC_PREPARATION_IMAGE && draw.failure.reason==VK_EXEC_PREPARATION_ORDER &&
                   draw.failure.result==got && draw.failure.role==(unsigned)stage && draw.failure.slot==2 &&
                   draw.failure.detail==VK_EXEC_RESTORE_STALE);
        else if (progress==VK_EXEC_RESTORE_FAILED)
            assert(draw.failure.reason==VK_EXEC_PREPARATION_PLACEMENT && draw.failure.detail==-17);
        else assert(!draw.failure.owner);
        vk_texture_restore_cancel(draw.sample_restore);
    } else {
        assert(got==VK_EXEC_GRAPHICS_READY && draw.sample_count==3 && !draw.sample_restore);
        assert(installed==1 && keeps[1]==1 && keeps[2]==1 && queries[2]==1);
        assert(draw.samples[1].handle==2 && draw.samples[1].image.image==202 && draw.samples[1].sample.registration==26);
    }
    assert(canceled==1 && !active);
    for (uint32_t i=0;i<draw.sample_count;i++) vk_texture_present_release(draw.samples[i].hold);
    for (int i=0;i<3;i++) assert(keeps[i]==drops[i]);
    free(draw.samples);
    (void)scratch;
}

#if REAL_REPLAN
static void test_pointer_replan(void) {
    VkExec ex = {0};
    VkMemSpanCapture target = {2};
    DeviceStoragePointer pointer = {.prepared=1,.output={.native=&target,.span={1},.mapped=malloc(8)}};
    assert(pointer.output.mapped);
    VkExecGraphicsDraw draw = {.read_position=position,.payload=&ex,.pointer_sources=&pointer,
        .pointer_count=1,.pointer_outputs_ready=1};
    assert(vk_exec_graphics_draw_replan(&ex,draw.opcode,&ex,&draw));
    assert(draw.pointer_sources==&pointer&&draw.pointer_count==1&&pointer.prepared);
    assert(!pointer.output.native&&!pointer.output.mapped&&!draw.pointer_outputs_ready);
}

static void test_fallback_replan(void) {
    VkExec ex = {0};
    VkExecGraphicsPass pass = {.live = 1};
    VkExecGraphicsScratch final = {1}, task = {1}, mesh = {1}, cohort = {1};
    MfbDraw *fallback = malloc(sizeof *fallback);
    assert(fallback);
    *fallback = (MfbDraw){.task.scratch = &task, .mesh.scratch = &mesh, .scratch_owner = &cohort};
    VkExecGraphicsDraw draw = {.read_position = position, .payload = &ex, .pass = &pass,
        .fallback = fallback, .scratch = &final, .use.revision = 1, .feedback_selected = 1};
    scratch_finished = pipeline_released = pass_finished = 0;
#if REAL_DRAW_POOL
    VkExecPoolGrant pool = {.held = 1, .current = 1};
    draw.pool_grant = &pool;
    pool_released = 0;
#endif
    draw.activated = 1;
    assert(!vk_exec_graphics_draw_replan(&ex, draw.opcode, &ex, &draw));
    assert(draw.fallback == fallback && !scratch_finished);
    draw.activated = 0;
    assert(vk_exec_graphics_draw_replan(&ex, draw.opcode, &ex, &draw));
    assert(!draw.fallback && !draw.scratch && !draw.feedback_selected && draw.pass == &pass);
    assert(scratch_finished == 4 && pipeline_released == 1 && !pass_finished);
#if REAL_DRAW_POOL
    assert(!draw.pool_grant && !pool.held && pool_released == 1);
#endif
}
#endif

static void test_sample_refusal(VkTextureSampleStatus select, VkTextureSampleStatus resolve, int alias, int no_hold,
    VkExecGraphicsResult expected, VkExecPreparationReason reason, int detail) {
    static GfxShader shader;
    VkExec ex={0};
    VkExecGraphicsPassShape shape={.pass_ncolor=1};
    VkExecGraphicsDraw draw={.state.fs=&shader,.read_position=position,.program_serial=917};
    ex.tex[NVN_STAGE_FS][0]=1;
    override_selection=1; selection_status=select; resolve_status=resolve; refuse_alias=alias;
    refuse_native_hold=no_hold; sibling_current=1;
    assert(graphics_feedback_select(&ex,&draw,&shape,1,(RcRegistryPosition){0},NULL)==expected);
    if (select==VK_TEXTURE_SAMPLE_ABSENT) {
        assert(draw.sample_count==1 && draw.samples[0].sample.historical==VK_TEXTURE_REGISTRATION_ABSENT &&
               !draw.samples[0].image.image && !draw.samples[0].hold);
        free(draw.samples);
    } else assert(!draw.sample_count);
    if (expected!=VK_EXEC_GRAPHICS_READY) {
        assert(draw.failure.owner==VK_EXEC_PREPARATION_IMAGE && draw.failure.reason==reason &&
               draw.failure.result==expected && draw.failure.role==NVN_STAGE_FS && draw.failure.slot==1 &&
               draw.failure.detail==detail);
        assert(draw.failure.range.mapping==(alias?selected_range.mapping:0));
        assert(draw.failure.sampled.present && draw.failure.sampled.program==917 && draw.failure.sampled.shader==&shader &&
               draw.failure.sampled.resource==resources && draw.failure.sampled.logical==1 &&
               draw.failure.sampled.member==0 && !rc_registry_position_compare(draw.failure.sampled.read,position));
        if (no_hold) {
            assert(!strcmp(draw.failure.sampled.gate,"sample-retain") &&
                   draw.failure.sampled.status==VK_TEXTURE_SAMPLE_READY && draw.failure.sampled.image==images[0].image);
        }
        if (expected==VK_EXEC_GRAPHICS_COMPAT) {
            assert(!strcmp(draw.failure.sampled.gate,"sample-placement") && draw.failure.sampled.status==(int)resolve);
            if (resolve==VK_TEXTURE_SAMPLE_FALLBACK) assert(draw.failure.sampled.image==images[0].image);
            VkTextureAliasRead seed={.live=1};
            draw.attachment_seeds[4]=&seed;
            RcRegistryPosition original=position;
            position.ordinal+=2; // The observer keeps the selected origin rather than this replay cursor.
            assert(!graphics_compatibility_begin(&draw) && draw.attachment_seeds[4]==&seed && seed.live &&
                   draw.failure.sampled.attachment_seed==5 && !draw.compatibility &&
                   !rc_registry_position_compare(draw.failure.sampled.read,original));
            position=original;
            draw.attachment_seeds[4]=NULL;
            assert(graphics_compatibility_begin(&draw) && draw.compatibility && !draw.failure.owner &&
                   !draw.failure.sampled.present);
        }
        if (alias) assert(draw.failure.registration==825 && draw.failure.material_serial==903 &&
                          draw.failure.image==444 && draw.failure.coverage.gate==10);
    } else assert(!draw.failure.owner);
    override_selection=refuse_alias=refuse_native_hold=0;
}

static void test_ordered_alias_and_geometry(void) {
    static GfxShader shader;
    VkExec ex={0};
    VkExecGraphicsPassShape shape={.pass_ncolor=1};
    VkExecGraphicsDraw draw={.state.fs=&shader,.read_position=position};
    ex.tex[NVN_STAGE_FS][0]=1;
    select_alias=1;
    const RcRegistryPosition attempted={0};
    assert(graphics_feedback_select(&ex,&draw,&shape,0,attempted,NULL)==VK_EXEC_GRAPHICS_NEEDS_ORDERED);
    assert(!draw.sample_count && alias_captures==1 && alias_cancels==1 && !alias_uploads);
    remap_alias=1;
    assert(graphics_feedback_select(&ex,&draw,&shape,1,attempted,NULL)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(draw.sample_count==1 && draw.failure.detail==VK_EXEC_RESTORE_STALE && !alias_uploads);
    graphics_sample_release(&draw.samples[0]);
    free(draw.samples);
    assert(alias_captures==2 && alias_cancels==2 && !captured_alias.live);

    remap_alias=0;
    draw=(VkExecGraphicsDraw){.state.fs=&shader,.read_position=position};
    assert(graphics_feedback_select(&ex,&draw,&shape,1,attempted,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(draw.sample_count==1 && draw.samples[0].sample.registration==42 &&
           draw.samples[0].image.image==alias_image.image && alias_uploads==1);
    graphics_sample_release(&draw.samples[0]);
    free(draw.samples);
    assert(alias_captures==3 && alias_cancels==3 && !captured_alias.live);
    select_alias=0;

    select_geometry=1;
    const unsigned previous_queries=queries[0];
    draw=(VkExecGraphicsDraw){.state.vs=&shader,.read_position=position};
    ex.tex[NVN_STAGE_VS][0]=1;
    assert(graphics_feedback_select(&ex,&draw,&shape,1,attempted,NULL)==VK_EXEC_GRAPHICS_READY);
    assert(draw.geometry_present && draw.sample_count==1 && draw.samples[0].geometry.registration==25);
    assert(queries[0]==previous_queries && alias_uploads==1);
    graphics_sample_release(&draw.samples[0]);
    free(draw.samples);
    select_geometry=0;
}

static void test_active_depth_feedback(void) {
    static GfxShader shader;
    VkExec ex={0};
    ex.tex[NVN_STAGE_FS][0]=1;
    const unsigned depth_slot=1u<<8;
    VkExecGraphicsPassShape shape={.pass_ncolor=1,.depth={images[0].image},
        .in_pass=1,.feedback_dirty=depth_slot};
    VkExecGraphicsDraw draw={.state.fs=&shader,.read_position=position,.program_serial=918};
    VkTextureAliasRead seed={.live=1};
    draw.attachment_seeds[2]=&seed;
    images[0].aspect=VK_IMAGE_ASPECT_DEPTH_BIT;
    context.feedback=VK_FEEDBACK_RESTART;
    sibling_current=1;
    assert(graphics_feedback_select(&ex,&draw,&shape,1,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_COMPAT);
    assert(draw.failure.reason==VK_EXEC_PREPARATION_FEEDBACK && draw.failure.sampled.present &&
           draw.failure.sampled.program==918 && draw.failure.sampled.image==images[0].image &&
           !strcmp(draw.failure.sampled.gate,"active-depth-feedback"));
    assert(!graphics_compatibility_begin(&draw) && seed.live && draw.failure.sampled.attachment_seed==3);
    graphics_sample_release(&draw.samples[0]);
    free(draw.samples);
    draw=(VkExecGraphicsDraw){.state.fs=&shader,.read_position=position};
    shape.depth.image++;
    assert(graphics_feedback_select(&ex,&draw,&shape,1,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_READY);
    assert(!draw.failure.sampled.present && !draw.feedback.restart);
    graphics_sample_release(&draw.samples[0]);
    free(draw.samples);

    // Color still requires a restart, while the simultaneous depth sample has a legal feedback layout.
    draw=(VkExecGraphicsDraw){.state.fs=&shader,.read_position=position};
    ex.tex[NVN_STAGE_FS][1]=3;
    shape.depth.image=images[0].image;
    shape.color[0].image=images[2].image;
    shape.feedback_dirty=depth_slot|1u;
    shape.fbl_mask=depth_slot;
    context.feedback=VK_FEEDBACK_LAYOUT;
    assert(graphics_feedback_select(&ex,&draw,&shape,1,(RcRegistryPosition){0},NULL)==VK_EXEC_GRAPHICS_COMPAT);
    assert(draw.sample_count==2 && draw.feedback.dirty==(depth_slot|1u) && !draw.failure.sampled.present);
    for (uint32_t i=0;i<draw.sample_count;++i) graphics_sample_release(&draw.samples[i]);
    free(draw.samples);
    images[0].aspect=0;
    context.feedback=0;
}

int main(void) {
    test_ordered_alias_and_geometry();
    test_sample_refusal(VK_TEXTURE_SAMPLE_ABSENT,0,0,0,VK_EXEC_GRAPHICS_READY,0,0);
    test_sample_refusal(VK_TEXTURE_SAMPLE_UNAVAILABLE,0,0,0,VK_EXEC_GRAPHICS_UNAVAILABLE,
        VK_EXEC_PREPARATION_PLACEMENT,VK_TEXTURE_SAMPLE_UNAVAILABLE);
    test_sample_refusal(VK_TEXTURE_SAMPLE_FALLBACK,VK_TEXTURE_SAMPLE_GONE,0,0,VK_EXEC_GRAPHICS_UNAVAILABLE,
        VK_EXEC_PREPARATION_MAPPING,VK_TEXTURE_SAMPLE_GONE);
    test_sample_refusal(VK_TEXTURE_SAMPLE_FALLBACK,VK_TEXTURE_SAMPLE_FALLBACK,0,0,VK_EXEC_GRAPHICS_COMPAT,
        VK_EXEC_PREPARATION_PLACEMENT,VK_TEXTURE_SAMPLE_FALLBACK);
    test_sample_refusal(VK_TEXTURE_SAMPLE_FALLBACK,VK_TEXTURE_SAMPLE_DEFERRED,0,0,VK_EXEC_GRAPHICS_COMPAT,
        VK_EXEC_PREPARATION_ORDER,VK_TEXTURE_SAMPLE_DEFERRED);
    test_sample_refusal(VK_TEXTURE_SAMPLE_FALLBACK,0,1,0,VK_EXEC_GRAPHICS_UNAVAILABLE,
        VK_EXEC_PREPARATION_ORDER,RC_REGISTRY_BLOCKED);
    test_sample_refusal(VK_TEXTURE_SAMPLE_READY,0,0,1,VK_EXEC_GRAPHICS_UNAVAILABLE,
        VK_EXEC_PREPARATION_PLACEMENT,RC_REGISTRY_STALE);
    test(VK_EXEC_RESTORE_FAILED,0,0,NVN_STAGE_TASK);
    test(VK_EXEC_RESTORE_READY,1,0,NVN_STAGE_TASK);
    test(VK_EXEC_RESTORE_READY,0,0,NVN_STAGE_FS);
    test(VK_EXEC_RESTORE_READY,0,0,NVN_STAGE_VS);
    test(VK_EXEC_RESTORE_READY,0,0,NVN_STAGE_TASK);
    test(VK_EXEC_RESTORE_READY,0,1,NVN_STAGE_FS);
    test(VK_EXEC_RESTORE_STOPPED,0,0,NVN_STAGE_VS);
#if REAL_REPLAN
    test_fallback_replan();
    test_pointer_replan();
#endif
    test_active_depth_feedback();
    puts("ordered sample restore: same-position replan, exact stages and members, one installation and stop passed");
}
'''

names = ["graphics_sample_shape_current", "graphics_sample_admit"]
real_replan = "int vk_exec_graphics_draw_replan(" in DRAW
if real_replan:
    names += ["graphics_log_sample"]
names += ["graphics_sample_selection_current", "graphics_stale_sample", "graphics_sample_prefix_current", "graphics_sample_release"]
if real_replan:
    names += ["graphics_prune_samples", "vk_exec_graphics_draw_replan"]
names += ["image_alias_preparation_reason", "graphics_sample_failure"]
if "static VkExecGraphicsResult graphics_sample_selection_failure(" in DRAW:
    names += ["graphics_sample_selection_failure"]
names += ["graphics_sample_prefix_prepared", "graphics_restore_sample", "graphics_log_geometry_refusal",
          "graphics_compatibility_begin"]
if real_replan:
    names += ["graphics_retained_sample", "graphics_sample_wait"]
names += ["graphics_keep_sample", "graphics_alias_samples_current", "graphics_prepare_alias_samples",
          "graphics_stage_samples", "graphics_direct_samples_select", "graphics_feedback_select"]
real_draw_pool = "VkExecPoolGrant *pool_grant;" in DRAW
real_vertex_input = "static int graphics_vertices_record(" in DRAW
source = (f"#define REAL_REPLAN {int(real_replan)}\n#define REAL_DRAW_POOL {int(real_draw_pool)}\n"
          f"#define REAL_VERTEX_INPUT {int(real_vertex_input)}\n"
          f"#define REAL_INDEXED_SNAPSHOT {int('indexed_draw_snapshot_release(draw)' in DRAW)}\n") + PREFIX + "\n".join(function(DRAW, name) for name in names) + CASES
layout = (Path(__file__).resolve().parents[1] / "runtime/vk_texture_alias_bytes.h").read_text()
view_test = r"""
#include <assert.h>
#include <stdint.h>
typedef struct { unsigned levels,layers; } VkTexImage;
typedef struct { unsigned set_levels,set_layers,base_level,nlevels,min_layer,nlayers; } view_t;
typedef struct { uint32_t aspectMask,mipLevel,baseArrayLayer,layerCount; } VkImageSubresourceLayers;
typedef struct { uint32_t aspectMask,baseMipLevel,levelCount,baseArrayLayer,layerCount; } VkImageSubresourceRange;
typedef struct { uint64_t bufferOffset; VkImageSubresourceLayers imageSubresource; } VkBufferImageCopy;
typedef struct { VkBufferImageCopy copy; uint32_t rows,row_bytes,slices; } VkTextureAliasPlane;
typedef struct { VkTextureAliasPlane planes[8]; uint64_t bytes; uint32_t count; } VkTextureAliasLayout;
enum { VK_IMAGE_ASPECT_COLOR_BIT = 1, VK_IMAGE_ASPECT_DEPTH_BIT = 2 };
""" + "\n".join(function(layout, name) for name in ("texture_alias_selected_planes", "texture_alias_view_covered")) + r"""
int main(void) {
    const VkTexImage image = {.levels=3,.layers=2};
    const VkTextureAliasLayout all = {.count=6,.bytes=96,.planes={
        {{0, {1,0,0,1}},1,16,1}, {{16, {1,0,1,1}},1,16,1},
        {{32, {1,1,0,1}},1,16,1}, {{48, {1,1,1,1}},1,16,1},
        {{64, {1,2,0,1}},1,16,1}, {{80, {1,2,1,1}},1,16,1}
    }};
    const view_t view = {.set_levels=1,.base_level=1,.nlevels=2,.set_layers=1,.min_layer=1,.nlayers=1};
    VkTextureAliasLayout selected = all;
    assert(texture_alias_selected_planes(&view,&image,&selected) && selected.count==2 && selected.bytes==96);
    assert(selected.planes[0].copy.bufferOffset==48 && selected.planes[1].copy.bufferOffset==80);
    assert(texture_alias_view_covered((VkImageSubresourceRange){1,1,2,1,1},&selected));
    assert(texture_alias_view_covered((VkImageSubresourceRange){1,2,1,1,1},&selected));
    assert(!texture_alias_view_covered((VkImageSubresourceRange){1,0,2,1,1},&selected));
    assert(!texture_alias_view_covered((VkImageSubresourceRange){1,1,2,0,2},&selected));
    assert(!texture_alias_view_covered((VkImageSubresourceRange){2,1,2,1,1},&selected));
    assert(!texture_alias_view_covered((VkImageSubresourceRange){1,1,UINT32_MAX,1,0},&selected));
    const view_t malformed = {.set_layers=1,.min_layer=2};
    assert(!texture_alias_selected_planes(&malformed,&image,&selected));
}
"""
cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="ordered-sample-restore-") as directory:
    c = Path(directory) / "check.c"
    exe = c.with_suffix(".exe")
    for code in (source, view_test):
        c.write_text(code)
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", str(c), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True, timeout=20)

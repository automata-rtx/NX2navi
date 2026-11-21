#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct {
    uint8_t set_levels, set_layers, set_format, set_target, set_swizzle, set_dsmode;
    uint8_t base_level, nlevels, target, dsmode, swz[4];
    uint16_t min_layer, nlayers, format;
} view_t;
#include "types.inc"

typedef struct { uint8_t key[32]; } GfxShader;
typedef struct { int slot, type, flags; } GfxShaderRes;
typedef struct {
    uint32_t binding, stage, kind, count, set;
    VkDescriptorType type;
    const GfxShaderRes *res;
} VkPipeBinding;
typedef struct { uint32_t n_bindings; VkPipeBinding *bindings; const GfxShader *compute_effect_shader; } VkPipeInfo;
typedef struct { VkImage image; uint64_t addr; uint32_t width,height,depth,format,type,levels,layers,samples,aspect; } VkTexImage;
typedef struct {
    const GfxShaderRes *resource;
    uint64_t handle, size;
    int slot, written, has_view, sampled;
    VkTexImage image;
    view_t view;
    const void *read_hold, *independent, *write_target;
    struct { const void *pin; } pinned;
    VkImageSubresourceRange issued_planes;
} EffectImageCapture;
typedef struct {
    uint64_t handle;
    VkTexImage image;
    struct { uint64_t registration; int historical, has_view; view_t view; void *query_image; } sample;
    struct { uint64_t registration, history_capture; int registration_only, association_only; VkImage image; } geometry;
    void *hold, *alias;
} GraphicsSample;
typedef struct { uint32_t count; EffectImageCapture *images; GraphicsSample *samples; } VkExecDispatchImages;
typedef struct {
    int binding, slot, member;
    const char *reason;
    uint64_t handle, image, peer_image, address, bytes;
#if TRACED
    VkExecDispatchInputTrace trace;
#endif
} VkExecDispatchInputFailure;
typedef struct {
    const GfxShader *shader;
    const VkPipeInfo *info;
    VkExecDispatchImages *selection;
    VkExecDispatchInputFailure input_failure;
    RcRegistryPosition read_position;
    uint64_t program_serial;
    uint32_t opcode;
    int compatibility;
} VkExecDispatch;
typedef struct { uint64_t tex[1][32]; } VkExec;
typedef struct { VkImage image; uint32_t aspect, level0, levels, layer0, layers; } ViewKey;
enum { GFX_RES_TEX=2, GFX_RES_IMG=4, NVN_STAGE_CS=0, VK_TEXTURE_REGISTRATION_ABSENT=-1, VK_TEXTURE_REGISTRATION_EXACT=1 };

static int trace_enabled, view_bad, alias_unproved;
static VkImage view_bad_image;
static int vk_dlaa_swap_img;

#if TRACED
static unsigned shader_keys, logs;
static char last_log[2048];
static int resource_effects_image_trace_enabled(void) { return trace_enabled; }
static const uint8_t *gfx_shader_key(const GfxShader *shader) { shader_keys++; return shader->key; }
typedef int PlatMutex;
#define PLAT_MUTEX_INIT 0
static void plat_mutex_lock(PlatMutex *lock) { assert(!(*lock)++); }
static void plat_mutex_unlock(PlatMutex *lock) { assert((*lock)-- == 1); }
static void rt_log(const char *format, const char *text) {
    assert(!strcmp(format, "%s"));
    const int len = snprintf(last_log, sizeof last_log, "[host] %s\n", text);
    assert(len > 0 && len < sizeof last_log - 1 && last_log[len - 1] == '\n');
    logs++;
}
#endif
static int texture_view_key(const VkTexImage *image, const view_t *v, int has_view, uint16_t type, int write, ViewKey *key) {
    (void)type;
    if ((view_bad == (write ? 2 : 1)) && image->image == view_bad_image) return 0;
    *key = (ViewKey){.image=image->image,.aspect=image->aspect,
        .level0=has_view && v->set_levels ? v->base_level : 0,.levels=1,.layers=1};
    return 1;
}
static int vk_texture_storage_target_host_mirror(const void *target) { (void)target; return 0; }
static int dispatch_storage_mirror_buffers_disjoint(const VkExec *ex, VkExecDispatch *d, const EffectImageCapture *output) {
    (void)ex; (void)d; (void)output; assert(0); return 0;
}
static int vk_texture_storage_targets_disjoint(const void *first, const void *second) {
    return first && second && first != second && !alias_unproved;
}
static uint32_t compute_sample_members(const VkPipeBinding *b, int selecting) {
    assert(selecting);
    return b->res && b->res->slot >= 0 ? (b->count ? b->count : 1) : 0;
}
static const EffectImageCapture *effect_image_lookup(EffectImageCapture *images, uint32_t count,
    const GfxShaderRes *res, uint64_t handle) {
    for (uint32_t i = 0; i < count; i++) if (images[i].resource == res && images[i].handle == handle) return images+i;
    return NULL;
}
static int vk_dlaa_swap_active(const VkExec *ex) { (void)ex; return 1; }
static VkImage vk_texture_write_target_prepared_image(const void *write) { return *(const VkImage *)write; }
static const GraphicsSample *compute_sample_receipt(const GraphicsSample *samples, uint32_t member,
    const GfxShaderRes *res, uint64_t handle) {
    assert(res && !member);
    return samples && samples->handle == handle ? samples : NULL;
}
static int compute_query_geometry_binding(const VkExec *ex, const VkPipeInfo *info, const VkPipeBinding *binding) {
    (void)ex; (void)info; (void)binding; return 0;
}
static int twin_compare_resource(const GfxShaderRes *res) { (void)res; return 0; }
static int twin_format(const VkTexImage *image) { (void)image; return 0; }
static int pool_comparison_format(const ViewKey *key, uint64_t *feature) { (void)key; *feature=1; return 1; }
static int dispatch_storage_material_sample_disjoint(const EffectImageCapture *out, const GraphicsSample *sample
#if HELD_FOOTPRINT
    , RcRegistryPosition read
#endif
) {
#if HELD_FOOTPRINT
    assert(read.scene==9 && read.ordinal==22);
#endif
    (void)out; (void)sample;
    return !alias_unproved;
}

#include "capture.inc"
#include "logger.inc"

#if TRACED
enum { VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_INVALIDATED, VK_EXEC_GRAPHICS_UNAVAILABLE };
static int optional_readonly_result(VkExecDispatch *dispatch,int optional_bound,int result) {
#include "optional.inc"
}
#endif

int main(void) {
    GfxShader shader = {{0xab}};
    GfxShaderRes write = {.slot=2,.type=3,.flags=4}, read = {.slot=8,.type=0x403,.flags=2}, peer = {.slot=11,.type=3,.flags=4};
    VkPipeBinding bindings[] = {{.binding=1,.kind=GFX_RES_IMG,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.res=&write},
        {.binding=5,.kind=GFX_RES_TEX,.type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.res=&read},
        {.binding=6,.kind=GFX_RES_IMG,.type=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,.res=&peer}};
    VkPipeInfo info={.n_bindings=3,.bindings=bindings,.compute_effect_shader=&shader};
    const VkImage a=(VkImage)(uintptr_t)0x1234, b=(VkImage)(uintptr_t)0x5678;
    EffectImageCapture captures[2]={{.resource=&write,.handle=0x31001025,.slot=2,.written=1,.size=0x700,.independent=&captures[0],
        .image={.image=a,.width=64,.height=32,.depth=1,.aspect=1,.format=VK_FORMAT_R8_UNORM,.addr=0x20000},
        .issued_planes={1,0,1,0,1}},
        {.resource=&peer,.handle=0x3a004384,.slot=11,.written=1,.size=0x200,.independent=&captures[1],
        .image={.image=b,.width=4,.height=2,.depth=1,.aspect=1,.format=VK_FORMAT_R8_UNORM,.addr=0x93000}}};
    const uint64_t original_input=UINT64_C(0xad001280);
    GraphicsSample sample={.handle=original_input,.hold=&sample,.image={.image=a,.width=64,.height=32,.depth=1,.aspect=1,
        .format=VK_FORMAT_R8_UNORM,.addr=0x21000},.sample={.registration=136839,.has_view=1,.view={.set_levels=1,.nlevels=1}}};
    VkExec ex={0};
    ex.tex[0][8]=original_input;
    VkExecDispatchImages selection={.count=1,.images=captures,.samples=&sample};
    VkExecDispatch d={.shader=&shader,.info=&info,.selection=&selection,.read_position={9,22},.opcode=27,.program_serial=978,.compatibility=1};
    assert(!dispatch_storage_output_disjoint(&ex,&d,captures));
    assert(!strcmp(d.input_failure.reason,"sample-view-conflict") && d.input_failure.binding==5 && d.input_failure.slot==2);
    assert(d.input_failure.handle==captures[0].handle && d.input_failure.handle!=original_input);
#if TRACED
    vk_exec_dispatch_input_trace_log(&d,"compat-storage-output",2);
    assert(!shader_keys && !logs); // Disabled keeps the exact old output and allocates no collector.
    trace_enabled=1;
    assert(!dispatch_storage_output_disjoint(&ex,&d,captures));
    assert(d.input_failure.trace.input.handle==original_input && d.input_failure.trace.output.handle==captures[0].handle);
    assert(d.input_failure.trace.input.registration==136839 && !d.input_failure.trace.output.registration);
    assert(!strcmp(d.input_failure.trace.detail,"shared-subresources"));
    assert(d.input_failure.trace.input.view.nlevels==1 && d.input_failure.trace.output.planes.levelCount==1);
    memset(shader.key,0,sizeof shader.key); // The terminal API only receives scalar copies, even after teardown.
    memset(&sample,0,sizeof sample);
    d.info=NULL;d.selection=NULL;d.shader=NULL;
    vk_exec_dispatch_input_trace_log(&d,"compat-storage-output",2);
    assert(logs==1&&strstr(last_log,"gate=sample-view-conflict detail=shared-subresources source-key=ab00"));
    assert(strstr(last_log,"output=storage-write(")&&strstr(last_log,"handle=0x31001025,image=0x1234"));
    assert(strstr(last_log,"input=sampled(")&&strstr(last_log,"handle=0xad001280,image=0x1234")&&strstr(last_log,"registration=136839"));
    vk_exec_dispatch_input_trace_log(&d,"compat-storage-output",2);
    assert(logs==1);
    d.info=&info;d.selection=&selection;d.shader=&shader;
    selection.samples=NULL;
    assert(!dispatch_storage_output_disjoint(&ex,&d,captures));
    assert(!strcmp(d.input_failure.trace.gate,"sample-receipt-missing")&&d.input_failure.trace.input.handle==original_input);
    assert(d.input_failure.trace.input.known==(VK_DISPATCH_TRACE_BINDING|VK_DISPATCH_TRACE_HANDLE));
    vk_exec_dispatch_input_trace_log(&d,"compat-storage-output",2);
    assert(logs==2);
    ex.tex[0][8]=0;
    assert(!dispatch_storage_output_disjoint(&ex,&d,captures));
    assert(!strcmp(d.input_failure.trace.detail,"zero-handle")&&!d.input_failure.trace.input.handle);
    assert(d.input_failure.trace.input.known==(VK_DISPATCH_TRACE_BINDING|VK_DISPATCH_TRACE_HANDLE));
    ex.tex[0][8]=original_input;
    selection.samples=&sample;
    sample.handle=original_input;
    sample.sample.historical=VK_TEXTURE_REGISTRATION_ABSENT;
    assert(!dispatch_storage_output_disjoint(&ex,&d,captures));
    assert(!strcmp(d.input_failure.trace.detail,"selected-no-image")&&
        d.input_failure.trace.input.known&VK_DISPATCH_TRACE_HISTORY);
    assert(d.input_failure.trace.input.history==VK_TEXTURE_REGISTRATION_ABSENT);
    read.type=3;
    assert(dispatch_storage_output_disjoint(&ex,&d,captures)); // Absent ordinary reads legitimately use a stand-in.
    selection.count=2;
    alias_unproved=1;
    assert(!dispatch_storage_output_disjoint(&ex,&d,captures));
    assert(!strcmp(d.input_failure.trace.gate,"storage-peer-mapping")&&!d.input_failure.trace.detail);
    assert(!strcmp(d.input_failure.trace.input.role,"storage-written-peer")&&d.input_failure.trace.input.handle==captures[1].handle);
    assert(d.input_failure.trace.input.binding==6 && d.input_failure.trace.input.image==(uint64_t)(uintptr_t)b);
    dispatch_storage_target_trace(&d,&(EffectImageCapture){.resource=&peer,.slot=11},&bindings[2],"storage-target-unavailable","zero-handle");
    assert(!strcmp(d.input_failure.trace.input.role,"storage-read") && !d.input_failure.trace.output.role); // Two writers, no unique context.
    selection.count=1;
    dispatch_storage_target_trace(&d,&(EffectImageCapture){.resource=&peer,.slot=11},&bindings[2],"storage-target-unavailable","zero-handle");
    assert(!strcmp(d.input_failure.trace.output.role,"retained-storage-write"));
    assert(optional_readonly_result(&d,1,VK_EXEC_GRAPHICS_INVALIDATED)==VK_EXEC_GRAPHICS_INVALIDATED);
    assert(d.input_failure.trace.gate);
    assert(optional_readonly_result(&d,1,VK_EXEC_GRAPHICS_UNAVAILABLE)==VK_EXEC_GRAPHICS_READY);
    vk_exec_dispatch_input_trace_log(&d,"unrelated-terminal-stage",2);
    assert(!d.input_failure.trace.gate && logs==2);
    selection.count=1;
    sample.image=captures[0].image;
    sample.hold=&sample;
    sample.sample.historical=0;
    view_bad_image=a;
    view_bad=1;
    assert(!dispatch_storage_output_disjoint(&ex,&d,captures));
    assert(!strcmp(d.input_failure.trace.detail,"input-view-unavailable"));
    view_bad=2;
    assert(!dispatch_storage_output_disjoint(&ex,&d,captures));
    assert(!strcmp(d.input_failure.trace.detail,"output-view-unavailable"));

    VkExecDispatchTraceOperand full={.role="storage-written-peer",.known=UINT32_MAX,.holds=UINT32_MAX,.handle=UINT64_MAX,
        .registration=UINT64_MAX,.image=UINT64_MAX,.address=UINT64_MAX,.bytes=UINT64_MAX,.width=UINT32_MAX,
        .height=UINT32_MAX,.depth=UINT32_MAX,.format=UINT32_MAX,.type=UINT32_MAX,.aspect=UINT32_MAX,.levels=UINT32_MAX,
        .layers=UINT32_MAX,.samples=UINT32_MAX,.count=UINT32_MAX,.flags=UINT32_MAX,.resource_type=UINT32_MAX,
        .set=INT_MIN,.binding=INT_MIN,.slot=INT_MIN,.stage=INT_MIN,.member=INT_MIN,.kind=INT_MIN,.descriptor_type=INT_MIN,
        .history=INT_MIN,.has_view=INT_MIN,.view={1,1,1,1,1,1,255,255,255,255,{255,255,255,255},65535,65535,65535},
        .planes={UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}};
    d.input_failure.trace.input=full;d.input_failure.trace.output=full;
    d.program_serial=UINT64_MAX;d.read_position=(RcRegistryPosition){UINT64_MAX,UINT64_MAX-1};d.opcode=UINT32_MAX;
    vk_exec_dispatch_input_trace_log(&d,"maximum-scalar-width-control",INT_MIN);
    assert(logs==3 && strstr(last_log,"planes=0xffffffff/4294967295/4294967295/4294967295/4294967295)\n"));
    for (unsigned i=0;i<90;i++) {
        d.read_position.ordinal=i*2;
        vk_exec_dispatch_input_trace_log(&d,"budget-control",2);
    }
    assert(logs==64);
    puts("dispatch input trace: actual input vs output, absence, written peer and views, scalar teardown, formatting and 64-row budget passed");
#else
    (void)trace_enabled;
    puts("legacy failure reproduced: input b5 carries output s2 and output handle");
#endif
}

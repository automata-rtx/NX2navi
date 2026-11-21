"""Check original-read sampled pool admission for selected entries and whole-kind cohorts."""
import os
from pathlib import Path
import subprocess
import tempfile

from bind_recipe_check import function


ROOT = Path(__file__).resolve().parents[1]

MODEL = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define VK_NULL_HANDLE 0
#define VK_POOL_TEX 64
#define VK_IMAGE_ASPECT_COLOR_BIT 1
#define VK_TEXTURE_READ_EXACT_DETAIL 8
#define VK_POOL_KINDS 16
#define VK_EXEC_ALIAS_READ_LIMIT 8
enum {VK_SUCCESS, VK_NOT_READY, VK_ERROR_INITIALIZATION_FAILED=-3};
enum { VK_TEXTURE_ALIAS_SAMPLE_FORCE = 1, VK_TEXTURE_ALIAS_SAMPLE_LINEAR_RGBA8_VIEW = 2,
       VK_TEXTURE_ALIAS_SAMPLE_ONLY = 4 };
typedef enum { VK_TEXTURE_SAMPLE_READY, VK_TEXTURE_SAMPLE_FALLBACK, VK_TEXTURE_SAMPLE_UNAVAILABLE }
    VkTextureSampleStatus;
enum {VK_TEXTURE_ALIAS_CAPACITY, VK_TEXTURE_ALIAS_LAYOUT, VK_TEXTURE_ALIAS_ORDER};
enum {RC_REGISTRY_CAPACITY=3, RC_REGISTRY_UNSUPPORTED, RC_REGISTRY_STALE};
enum {RC_BUDGET_TEXTURE};
typedef int VkResult, RcRegistryResult, VkTextureAliasPhase;
typedef uint64_t VkImage, VkImageView;
typedef struct {int owner;} RcBudgetTicket;
typedef struct {uint64_t mapping;} RcInterval;
typedef struct {uint64_t scene, ordinal;} RcRegistryPosition;
typedef struct {int unused;} VkExecPoolGrant, VkTextureRestore, VkExecGpaSampleScope;
typedef struct {uint32_t count; struct {uint32_t slot,kinds;} entries[4]; int domain;} VkExecPoolIndices;
typedef struct {int unused;} VkExecGraphicsDraw;
typedef struct VkExecGraphicsPass VkExecGraphicsPass;
typedef struct {unsigned record_count;} VkExec;
typedef struct {VkImage image; unsigned aspect;} VkTexImage;
typedef struct {uint32_t slot; int live; uint64_t registration;} VkTextureAliasRead;
typedef struct {int view, has_view; uint64_t registration; const VkTexImage *image; int projected;} VkTextureSample;
typedef struct {int unused;} ViewKey;
typedef struct {int detail;} VkExecPreparationFailure;
typedef struct {RcInterval receiver; VkTextureAliasPhase phase; RcRegistryResult status; const char *layout_gate;}
    VkTextureAliasFailure;
typedef struct {unsigned kinds; int finite;} GfxShader;
typedef struct {RcRegistryPosition attempted; VkExecGpaSampleScope **pending;} GraphicsCompatibilityPool;
typedef enum {VK_EXEC_GRAPHICS_READY, VK_EXEC_GRAPHICS_COMPAT, VK_EXEC_GRAPHICS_UNAVAILABLE,
              VK_EXEC_GRAPHICS_STOPPED} VkExecGraphicsResult;
typedef enum {VK_EXEC_RESTORE_READY, VK_EXEC_RESTORE_STOPPED, VK_EXEC_RESTORE_STALE} VkExecRestoreResult;
static VkTextureAliasRead retained[VK_POOL_TEX];
static VkTexImage outputs[VK_POOL_TEX];
static const RcInterval range={73};
static unsigned charged, refusals, released, failed_allocation, locked, prepared_count, absence, epoch=1;
static unsigned expected_alias_options=VK_TEXTURE_ALIAS_SAMPLE_ONLY;
static unsigned alias_captures, detached, invalidated, layout_reports;
static unsigned expected_first_slot, expected_prepare_count=9, selected_fallbacks;
static uint32_t registered_format;
static int has_registered_format;
static int finite_calls, whole_calls, pointer_calls, finite_result, enable_pool=1;
static int vk_replay_metadata_allocate(size_t bytes, int clear, int producer, RcBudgetTicket *ticket, void **out) {
    assert(bytes && producer==RC_BUDGET_TEXTURE && !ticket->owner);
    if (failed_allocation && !--failed_allocation) return VK_NOT_READY;
    *out=clear?calloc(1,bytes):malloc(bytes);
    assert(*out);
    ticket->owner=1;
    charged++;
    return VK_SUCCESS;
}
static void vk_replay_metadata_free(void *data,RcBudgetTicket *ticket) {
    assert(!!data == ticket->owner);
    if (!data) return;
    assert(charged);
    charged--;
    ticket->owner=0;
    free(data);
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *read) {
    assert(read && read->live==1);
    read->live=0;
    released++;
}
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *read, uint32_t *count) {
    assert(read->live);
    *count=1;
    return &range;
}
static int resident_color_ready;
static const VkTexImage resident_color={.image=101,.aspect=VK_IMAGE_ASPECT_COLOR_BIT};
static RcRegistryPosition vk_exec_resource_position(void) { return (RcRegistryPosition){7,20}; }
static int rc_registry_position_valid(RcRegistryPosition position) { return position.scene && position.ordinal; }
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b) {
    if (a.scene != b.scene) return a.scene < b.scene ? -1 : 1;
    return a.ordinal < b.ordinal ? -1 : a.ordinal > b.ordinal;
}
static int in_fbl(const VkExec *ex,VkImage image) { assert(ex && image==101); return 0; }
static VkTextureSampleStatus vk_texture_select_sample_locked(int slot,unsigned flags,VkTextureSample *sample) {
    assert(slot==3 && flags==VK_TEXTURE_READ_EXACT_DETAIL);
    if (!resident_color_ready) return VK_TEXTURE_SAMPLE_FALLBACK;
    *sample=(VkTextureSample){.registration=703,.image=&resident_color,.projected=1};
    return VK_TEXTURE_SAMPLE_READY;
}
static VkTextureSampleStatus vk_texture_ordered_sample_select_locked(int slot,RcRegistryPosition read,
    unsigned flags,unsigned options,VkTextureSample *sample,VkTextureAliasRead **out,VkTexImage *image,
    VkTextureAliasFailure *failure) {
    assert(slot>=0 && slot<VK_POOL_TEX && !flags && options==expected_alias_options && failure);
    assert(read.scene==7 && read.ordinal==20);
    alias_captures++;
    if ((unsigned)slot==absence) return VK_TEXTURE_SAMPLE_FALLBACK;
    VkTextureAliasRead *r=&retained[slot];
    assert(!r->live);
    *r=(VkTextureAliasRead){.slot=(unsigned)slot,.live=1,.registration=epoch*100+slot};
    sample->registration=r->registration;
    *out=r;
    // Two logical materials can select a receiver with identical shape and no staged VkImage yet.
    image->image=0;
    return VK_TEXTURE_SAMPLE_READY;
}
static uint16_t pool_kind_type(uint32_t kind) {return kind==4?4:3;}
static int vk_texture_registration_view_format_locked(int slot,uint32_t *format) {
    assert(slot>=0 && slot<VK_POOL_TEX);
    if (!has_registered_format) return 0;
    *format=registered_format;
    return 1;
}
static int int_view_format(uint32_t format,VkImage *vk) { assert(!vk); return format==11; }
static void vk_texture_alias_sample_report_layout_locked(int slot,const VkTextureAliasFailure *failure) {
    assert(slot==3 && failure->layout_gate && !strcmp(failure->layout_gate,"comparison-numeric-type"));
    layout_reports++;
}
static int texture_view_key(const VkTexImage *image,const int *view,int has_view,uint16_t kind,int storage,ViewKey *key) {
    assert(image&&view&&!has_view&&kind==3&&!storage&&key);
    return 1;
}
static void vk_texture_lock(void) {assert(!locked++);}
static void vk_texture_unlock(void) {assert(locked--==1);}
static int vk_texture_alias_sample_current_locked(int slot,const VkTextureAliasRead *r) {
    assert(locked && r->slot==(unsigned)slot && r->live);
    return r->registration==epoch*100+(unsigned)slot;
}
static VkExecRestoreResult vk_exec_prepare_alias_texture_read(VkExec *ex,VkTextureAliasRead *const *reads,uint32_t count,
    int (*current)(void *),void *scope,const VkTexImage **images) {
    assert(!ex->record_count && !locked && count==expected_prepare_count);
    if (!current(scope)) return VK_EXEC_RESTORE_STALE;
    prepared_count=count;
    ex->record_count++;
    for (unsigned i=0; i<count; i++) {
        assert(reads[i]->live && reads[i]->slot==expected_first_slot+i);
        outputs[i].image=10+reads[i]->slot;
        images[i]=&outputs[i];
    }
    return VK_EXEC_RESTORE_READY;
}
static uint32_t vk_exec_pool_shader_kinds(const GfxShader *s) {return s?s->kinds:0;}
static int gfx_shader_has_pool_index_plan(const GfxShader *s) {return s->finite;}
static int graphics_pool_has_domain(const GfxShader *const shaders[3]) { (void)shaders; return 0; }
enum { GRAPHICS_POOL_COMPARISON_NONE };
static int graphics_pool_comparison_usage(const GfxShader *const shaders[3]) { (void)shaders; return GRAPHICS_POOL_COMPARISON_NONE; }
static int pool_on(void) {return enable_pool;}
static VkExecGraphicsResult graphics_pool_prepare_selected(VkExec *ex,VkExecGraphicsDraw *draw,
    const GfxShader *const shaders[3],uint32_t kinds,int ordered,int allow_prefix,RcRegistryPosition attempted,
    VkExecGpaSampleScope **pending,int finite_only) {
    assert(ex&&draw&&shaders[0]&&kinds==(1u<<10)&&ordered&&allow_prefix&&attempted.ordinal==20&&!pending&&!finite_only);
    finite_calls++;
    if (finite_result == VK_EXEC_GRAPHICS_COMPAT) {
        whole_calls++; pointer_calls++;
        return VK_EXEC_GRAPHICS_READY;
    }
    return finite_result;
}
static VkExecGraphicsResult graphics_required_device_inputs(VkExec *ex,VkExecGraphicsDraw *draw) {
    assert(ex&&draw);
    pointer_calls++;
    return VK_EXEC_GRAPHICS_READY;
}
static VkExecGraphicsResult graphics_pool_prepare_whole(VkExec *ex,uint32_t kinds,VkExecGraphicsDraw *draw,
    int allow_prefix,RcRegistryPosition attempted,VkExecGpaSampleScope **pending,const void *pass) {
    assert(ex&&draw&&kinds==(1u<<10)&&allow_prefix&&attempted.ordinal==20&&!pending&&!pass);
    whole_calls++;
    return VK_EXEC_GRAPHICS_READY;
}
'''

CALLS = r'''
static int pool_prepare_depth_alias_locked(GraphicsPoolRestore *r,uint32_t slot,uint32_t kinds,const VkTextureSample *s) {
    (void)r;(void)slot;(void)kinds;(void)s;return 0; // The depth companion has its own retained source fixture.
}
static int vk_texture_alias_comparison_range_refused(const VkTextureAliasRead *alias) {(void)alias;return 0;}
static int pool_domain_comparison_supported(GraphicsPoolRestore *request,uint32_t slot,uint32_t kind,const ViewKey *key,uint64_t reg) {
    (void)request;(void)slot;(void)kind;assert(key&&reg);return 1;
}
static void pool_alias_refuse(PoolAliases *aliases,uint32_t id,uint32_t kind,const char *gate,const VkTextureAliasFailure *failure) {
    assert(id<VK_POOL_TEX && kind==10);
    if (!strcmp(gate,"comparison-view")) {
        assert(failure->phase==VK_TEXTURE_ALIAS_LAYOUT && failure->status==RC_REGISTRY_UNSUPPORTED);
    } else {
        assert(!strcmp(gate,"cohort-metadata") && failure->phase==VK_TEXTURE_ALIAS_CAPACITY &&
               failure->status==RC_REGISTRY_CAPACITY && failure->receiver.mapping==73);
    }
    aliases->refused=1;
    refusals++;
}
static void pool_alias_progress_refused(PoolAliases *aliases,const PoolAliasMember *member,const char *gate,
    VkTextureAliasPhase phase,RcRegistryResult result) {
    assert(!member && !strcmp(gate,"cohort-workspace") && phase==VK_TEXTURE_ALIAS_CAPACITY && result==RC_REGISTRY_CAPACITY);
    aliases->refused=1;
    refusals++;
}

typedef struct { uint32_t selected_kinds, stamp, fmask; } PoolSlot;
static struct {
    PoolSlot ps[VK_POOL_TEX];
    VkImage image[VK_POOL_KINDS*VK_POOL_TEX];
    uint64_t tok[VK_POOL_TEX];
    int wanted_reconcile, finite_reconcile;
} PL;
typedef struct {int unused;} TextureImagePublication;
static const TextureImagePublication *pool_member_installed_images(void) { return NULL; }
static uint32_t pool_indexed_selected_kinds(const TextureImagePublication *images,uint32_t slot,uint32_t kind) {
    assert(!images && slot<VK_POOL_TEX && kind==(1u<<10));
    return 0;
}
static VkTextureAliasRead *vk_texture_publication_alias_locked(const TextureImagePublication *images,
                                                               uint32_t index,VkImage image) {
    assert(!images && index>=10*VK_POOL_TEX && !image);
    return NULL;
}
static int vk_texture_alias_published_current_locked(int slot,VkTextureAliasRead *alias,RcRegistryPosition read) {
    (void)slot;(void)alias;(void)read;
    assert(0);
    return 0;
}
static uint64_t pool_slot_token(int slot) { assert(slot==3 || slot==9); return 7; }
static int pool_keep_selected(uint32_t slot,uint32_t kinds) { (void)slot;(void)kinds; return 0; }
static int pool_need_restore(uint32_t slot,uint32_t kinds) { assert((slot==3 || slot==9) && kinds==(1u<<2)); return 0; }
static int pool_prepare_selected_member(VkExec *ex,uint32_t slot,uint32_t kinds,int required) {
    assert(ex && slot==9 && kinds==(1u<<2) && required);
    selected_fallbacks++;
    return 1;
}
static struct { struct { VkImageView view; } img[1]; } Z={.img={{71}}};
static VkImage native_cells[VK_POOL_KINDS][VK_POOL_TEX], sealed_cells[VK_POOL_KINDS][VK_POOL_TEX];
static void pool_association_detach(uint32_t slot) { assert(slot==3); detached++; }
static void pool_member_invalidate(uint32_t slot) { assert(slot==3); invalidated++; }
static int res_view_type(uint16_t type) { return type; }
static int dummy_index(int kind,int ms) { assert(kind==3 && !ms); return 0; }
static VkImageView pool_absent_view(uint32_t kind) {
    return Z.img[dummy_index(res_view_type(pool_kind_type(kind)),0)].view;
}
static void pool_set_image(uint32_t kind,uint32_t slot,VkImage image,VkImageView view) {
    assert(kind==10 && slot==3 && !image && view==71);
    native_cells[kind][slot]=image;
}
'''

TEST = r'''
int main(void) {
    PoolAliases cohort={0};
    GraphicsPoolRestore request={.aliases=&cohort,.kinds=1u<<10,.read={7,20}};
    absence=9;
    const VkTexImage shape={0};
    const VkTextureSample visible={.image=&shape};
    for (uint32_t i=0; i<9; i++) graphics_pool_plan_dynamic_alias((int)i,&visible,&request);
    assert(!cohort.refused && cohort.count==9 && charged==2);
    assert(pool_need_alias_locked(&request,8,10,0)==-1 && cohort.count==9);
    assert(pool_need_alias_locked(&request,absence,10,0)==0 && cohort.count==9 && !pool_alias_member(&cohort,absence));
    PoolAliasMember *ninth=pool_alias_member(&cohort,8);
    assert(ninth && ninth->read==&retained[8]);
    assert(ninth->sample.registration!=pool_alias_member(&cohort,7)->sample.registration);
    VkExec ex={0};
    epoch++;
    assert(graphics_pool_alias_prepare(&ex,&cohort)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!prepared_count && !ex.record_count && !ninth->image.image && charged==2);
    epoch--;
    assert(graphics_pool_alias_prepare(&ex,&cohort)==VK_EXEC_GRAPHICS_READY);
    assert(prepared_count==9 && ex.record_count==1 && ninth->image.image==18 && charged==2);
    pool_aliases_cancel(&cohort);
    assert(!charged && released==9);

    cohort=(PoolAliases){0};
    for (uint32_t i=0; i<8; i++) assert(pool_need_alias_locked(&request,i,10,0)==-1);
    failed_allocation=2;  // Grow storage arrives, but its independent slot index cannot be admitted.
    assert(pool_need_alias_locked(&request,8,10,0)==-1 && cohort.count==8 && cohort.refused);
    assert(refusals==1 && cohort.failure.detail==VK_NOT_READY && !charged && released==10);
    pool_aliases_cancel(&cohort);
    assert(released==18);
    cohort=(PoolAliases){0};

    // Mutable current set replaces only the incompatible cell. The sealed set owns its old native image.
    const uint32_t ordinary=1u<<2, compare=1u<<10;
    has_registered_format=1;
    registered_format=11; // R32UI on the registered view, regardless of the base texture format.
    unsigned captures=alias_captures;
    assert(pool_incompatible_comparison_kinds(3,ordinary|compare)==compare);
    assert(pool_need_alias_locked(&request,3,10,0)==0 && alias_captures==captures && !cohort.refused);
    native_cells[2][3]=61;
    sealed_cells[10][3]=native_cells[10][3]=60;
    PL.ps[3]=(PoolSlot){.selected_kinds=ordinary|compare,.stamp=18,.fmask=ordinary|compare};
    pool_install_incompatible_comparisons(3,compare);
    assert(native_cells[2][3]==61 && !native_cells[10][3] && sealed_cells[10][3]==60);
    assert(PL.ps[3].selected_kinds==ordinary && !PL.ps[3].stamp && !PL.ps[3].fmask && detached==1 && invalidated==1);
    request.kinds=ordinary|compare;
    assert(pool_need_alias_locked(&request,3,10,0)==-1 && alias_captures==captures+1);
    assert(cohort.count==1 && cohort.members[0].kinds==ordinary && !cohort.refused);
    pool_aliases_cancel(&cohort);
    cohort=(PoolAliases){0};
    request.kinds=compare;
    VkExecPoolIndices finite_slot={.count=1,.entries={{3,compare}}};
    request.indices=&finite_slot;
    assert(pool_need_alias_locked(&request,3,10,0)==-1 && alias_captures==captures+1 && cohort.refused);
    assert(layout_reports==1);
    pool_aliases_cancel(&cohort);
    cohort=(PoolAliases){0};
    request.indices=NULL;
    registered_format=10; // R32F is still considered for valid comparison providers.
    assert(!pool_incompatible_comparison_kinds(3,ordinary|compare));
    assert(pool_need_alias_locked(&request,3,10,0)==-1 && alias_captures==captures+2);
    pool_aliases_cancel(&cohort);

    cohort=(PoolAliases){0};
    finite_slot.entries[0].kinds=ordinary;
    request.kinds=ordinary;
    request.indices=&finite_slot;
    finite_slot.count=2;
    finite_slot.entries[1].slot=absence;
    finite_slot.entries[1].kinds=ordinary;
    VkExec selected_ex={0};
    request.ex=&selected_ex;
    expected_alias_options=VK_TEXTURE_ALIAS_SAMPLE_ONLY|VK_TEXTURE_ALIAS_SAMPLE_LINEAR_RGBA8_VIEW;
    pool_sync_selected_entries(&request);
    assert(cohort.count==1 && !cohort.refused && selected_fallbacks==1 && PL.finite_reconcile);
    assert(cohort.members[0].kinds==ordinary && cohort.members[0].read==&retained[3] &&
           cohort.members[0].sample.registration==retained[3].registration);
    expected_first_slot=3;
    expected_prepare_count=1;
    epoch++;
    assert(graphics_pool_alias_prepare(&selected_ex,&cohort)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    assert(!selected_ex.record_count && !cohort.members[0].image.image);
    epoch--;
    assert(graphics_pool_alias_prepare(&selected_ex,&cohort)==VK_EXEC_GRAPHICS_READY);
    assert(prepared_count==1 && selected_ex.record_count==1 && cohort.members[0].image.image==13);
    pool_aliases_cancel(&cohort);

    // Replacing an installed alias preserves the same original read and view option.
    cohort=(PoolAliases){0};
    expected_alias_options|=VK_TEXTURE_ALIAS_SAMPLE_FORCE;
    assert(pool_need_alias_locked(&request,3,2,1)==-1 && cohort.count==1 && !cohort.refused);
    assert(cohort.members[0].read==&retained[3] && cohort.members[0].kinds==ordinary);
    pool_aliases_cancel(&cohort);
    assert(!charged);

    // Same typed native color can be selected at the original cut without a byte reconstruction.
    cohort=(PoolAliases){0};
    request.indices=NULL;
    resident_color_ready=1;
    const unsigned before_native=alias_captures;
    assert(pool_need_alias_locked(&request,3,2,0)==0 && !cohort.count && alias_captures==before_native);

    VkExecGraphicsDraw draw={0};
    const GraphicsCompatibilityPool original={.attempted={7,20}};
    const GfxShader dynamic={1u<<10,0}, finite={1u<<10,1}, flatten={1u<<4,0};
    const GfxShader *stages[]={&dynamic,NULL,NULL};
    assert(graphics_compat_pool_prepare_members(&ex,&draw,stages,&original)==VK_EXEC_GRAPHICS_READY);
    assert(whole_calls==1 && pointer_calls==1 && !finite_calls);
    stages[0]=&finite;
    assert(graphics_compat_pool_prepare_members(&ex,&draw,stages,&original)==VK_EXEC_GRAPHICS_READY);
    assert(finite_calls==1 && whole_calls==1);
    finite_result=VK_EXEC_GRAPHICS_COMPAT;
    assert(graphics_compat_pool_prepare_members(&ex,&draw,stages,&original)==VK_EXEC_GRAPHICS_READY);
    assert(finite_calls==2 && whole_calls==2 && pointer_calls==2);
    finite_result=VK_EXEC_GRAPHICS_UNAVAILABLE;
    assert(graphics_compat_pool_prepare_members(&ex,&draw,stages,&original)==VK_EXEC_GRAPHICS_UNAVAILABLE);
    stages[0]=&flatten;
    assert(graphics_compat_pool_prepare_members(&ex,&draw,stages,&original)==VK_EXEC_GRAPHICS_READY);
    enable_pool=0; stages[0]=&dynamic;
    assert(graphics_compat_pool_prepare_members(&ex,&draw,stages,&original)==VK_EXEC_GRAPHICS_READY);
    assert(finite_calls==3 && whole_calls==2 && pointer_calls==2);
    puts("Pool alias cohort: whole ninth member, finite original-read preparation, absence and aggregate ownership passed");
}
'''


def main():
    source = (ROOT / "runtime/vk_exec_draw.c").read_text(encoding="utf-8")
    start = source.index("typedef struct {\n    VkTextureAliasRead *read;\n    VkTextureSample sample;")
    end = source.index("// Only the ordered graphics preflight", start)
    functions = ("pool_request_kinds", "pool_incompatible_comparison_kinds", "pool_install_incompatible_comparisons",
                 "pool_alias_member", "pool_aliases_append", "pool_aliases_cancel",
                 "pool_resident_color_locked", "pool_need_alias_locked", "pool_sync_selected_entries",
                 "graphics_pool_alias_reads_current",
                 "graphics_pool_alias_prepare", "graphics_pool_plan_dynamic_alias", "graphics_compat_pool_prepare_members")
    depth_kinds = next(line for line in source.splitlines() if line.startswith("#define POOL_DEPTH_TWIN_KINDS "))
    text = MODEL + source[start:end] + CALLS + depth_kinds + "\n" + "\n".join(function(source, name) for name in functions) + TEST
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="pool-alias-cohort-") as temp:
        path = Path(temp) / "check.c"
        path.write_text(text, encoding="utf-8")
        exe = path.with_suffix(".exe")
        subprocess.run([str(cc), "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                        "-fsanitize-trap=all", str(path), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

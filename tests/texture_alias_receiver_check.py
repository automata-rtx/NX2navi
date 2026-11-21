"""Run the ordinary selected alias receiver decisions with held or absent producer receipts."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from present_source_check import ROOT, functions


def main():
    cc = Path(os.environ.get("CC") or shutil.which("clang") or "").resolve()
    if not cc.is_file():
        raise SystemExit("Clang is required (set CC)")
    path = Path(os.getenv("ALIAS_RECEIVER_CHECK_SOURCE", ROOT / "runtime" / "vk_texture.c"))
    production = functions(path.read_text(encoding="utf-8"), "image_alias_refuse_layout", "image_alias_sample_entry_stale",
        "image_alias_sample_select_locked", "image_alias_sample_receiver_locked", "image_alias_pool_member_locked")
    prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define VK_NULL_HANDLE 0
#define VK_IMAGE_USAGE_SAMPLED_BIT 1
typedef uintptr_t VkImage;
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
typedef struct {uint64_t mapping; struct {uint64_t backing,offset,size;} range;} RcInterval;
typedef enum {RC_REGISTRY_OK, RC_REGISTRY_UNSUPPORTED, RC_REGISTRY_STALE, RC_REGISTRY_BLOCKED} RcRegistryResult;
enum { VK_TEXTURE_ALIAS_SAMPLE_FORCE = 1, VK_TEXTURE_ALIAS_SAMPLE_LINEAR_RGBA8_VIEW = 2,
       VK_TEXTURE_ALIAS_SAMPLE_ONLY = 4 };
enum {VK_TEXTURE_ALIAS_LAYOUT, VK_TEXTURE_ALIAS_OWNER, VK_TEXTURE_ALIAS_ORDER, VK_TEXTURE_ALIAS_COVERAGE};
enum {VK_TEXTURE_REGISTRATION_CURRENT, VK_TEXTURE_REGISTRATION_GONE, VK_TEXTURE_REGISTRATION_ABSENT};
typedef struct {uint64_t addr;} TexKey;
typedef struct {uint64_t serial; struct {uint64_t size;} L;} tex_t;
typedef struct {VkImage image; uint64_t addr; unsigned gone,gpu_owned,preserve_pool_bytes,usage;} VkTexImage;
typedef struct {
    tex_t t;
    TexKey key;
    VkTexImage pub;
    const RcInterval *image_private_mapping, *alias_full_demand;
    unsigned image_private_mapping_count,alias_full_count,released,retiring,superseded,overwritten,alias_fragment_owner;
    RcRegistryPosition native_write;
} Tex;
typedef struct {struct {uint64_t id;} handle; tex_t descriptor; int view,has_view;} TextureRegistration;
typedef struct {const TextureRegistration *registration; int finalized;} TexSlot;
typedef struct {
    const tex_t *receiver;
    int registration_only,receiver_logical,receiver_native;
    uint64_t receiver_registration,receiver_pool;
} VkTextureAliasRead;
typedef struct {uint64_t registration; int native_slot,view,has_view; const VkTexImage *image;} VkTextureSample;
typedef struct {uint64_t registration,material_serial; VkImage image; int phase,status; const char *layout_gate;
    RcInterval receiver;} VkTextureAliasFailure;
typedef struct {VkTexImage image;} ImageAliasReceiver;
typedef void (*VkAliasPoolMember)(int,const VkTextureSample *,void *);

static unsigned prospective, normal, visit_count, traces;
static int outcome=1, scope_status, trace, by_key;
static uint64_t expected_pool = 42;
static RcRegistryPosition now = {1,2};
static const RcInterval scope = {11,{7,400,8192}};
static TextureRegistration registration = {{9},{17,{8192}},4,1};
static Tex cached = {.t={17,{8192}},.key={700},.pub={.image=5,.addr=700,.usage=1,.preserve_pool_bytes=1},
    .image_private_mapping=&scope,.image_private_mapping_count=1,.alias_full_count=1,.alias_full_demand=&scope};
static const Tex *entry = &cached;
static TexSlot tslots[4] = {[3]={&registration,0}};
static tex_t retained_receiver;
static VkTextureAliasRead receipt, *chosen;
static VkTextureSample sample;
static VkTexImage shape;
static VkTextureAliasFailure failure;
static int texture_registration_resolve(int logical,const TextureRegistration **history,int *native) {
    assert(logical==7);
    *history=&registration;
    *native=3;
    return scope_status;
}
static const Tex *image_alias_sample_entry_locked(const TexSlot *slot,int native) {
    assert(slot==&tslots[3] && native==3);
    return entry;
}
static void make_key(const tex_t *t,TexKey *key) { assert(t==&registration.descriptor); key->addr=700; }
static int same_but_swizzle(const TexKey *a,const TexKey *b) { return a->addr==b->addr; }
int rt_knob(const char *name, const char *fallback) { assert(!strcmp(name, "GRAPHICS_PREP_TRACE") && !strcmp(fallback, "0")); return trace; }
static const Tex *nvn_hm_get(const int *table,const TexKey *key) {
    assert(table==&by_key && key->addr==700);
    return entry;
}
static void rt_log(const char *fmt,...) { assert(strstr(fmt,"image selected owner stale")); traces++; }
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b) {
    return a.scene!=b.scene || a.ordinal!=b.ordinal;
}
static RcRegistryPosition vk_exec_resource_position(void) { return now; }
static uint64_t vk_exec_texture_pool(void) { return expected_pool; }
static int vk_texture_pool_logical_locked(uint64_t pool,int native) { return pool==expected_pool && native==3 ? 7 : -1; }
static int rc_registry_runtime(void) { return 1; }
static int image_alias_source_current(const Tex *t) { return t==entry && !t->retiring; }
static int rc_registry_native_image_covers(int registry,uint64_t image,const RcInterval *demand,
    unsigned count,RcRegistryPosition position) {
    assert(registry==1 && image==5 && demand==&scope && count==1 && !rc_registry_position_compare(position,now));
    return 1;
}
static int image_alias_refuse(VkTextureAliasFailure *reason,int phase,int status,RcInterval interval) {
    if (reason) reason->phase=phase, reason->status=status, reason->receiver=interval;
    return -1;
}
static int image_alias_sample_prospective_locked(const TextureRegistration *record,int logical,int native,
    RcRegistryPosition read,VkTextureAliasRead **out,VkTextureSample *selected,VkTexImage *image,unsigned options,VkTextureAliasFailure *reason) {
    assert(!options);
    prospective++;
    assert(record==&registration && logical==7 && native==3 && !rc_registry_position_compare(read,now));
    if (outcome==1 && out) {
        retained_receiver=record->descriptor;
        receipt=(VkTextureAliasRead){&retained_receiver,1,logical,native,record->handle.id,expected_pool};
        *out=&receipt;
        *selected=(VkTextureSample){.registration=record->handle.id,.native_slot=native,.view=record->view,.has_view=1};
        *image=(VkTexImage){.addr=700,.usage=1}; // receiver recipe, no old VkImage published
    }
    if (outcome<0) return image_alias_refuse(reason,VK_TEXTURE_ALIAS_COVERAGE,RC_REGISTRY_BLOCKED,scope);
    return outcome;
}
static int image_alias_read_capture_locked(const tex_t *tex,const Tex *input,VkTextureAliasRead **out,
    VkTexImage *image,int force,int target,const void *planes,void *initial,VkTextureAliasFailure *reason) {
    assert(tex==&entry->t && input==entry && !target && !planes && !initial && out && image && force && reason);
    normal++;
    return -1;
}
static void slot_result_view(const TexSlot *slot,const Tex *t,int *view,int *has) {
    assert(slot==&tslots[3] && t==entry);
    *view=9; *has=1;
}
static const char *image_alias_sample_recipe(const TextureRegistration *record,ImageAliasReceiver *receiver) {
    assert(record==&registration);
    receiver->image=(VkTexImage){.addr=700,.usage=1};
    return NULL;
}
static int nvn_gpu_only(uint64_t address) { return address==700; }
'''
    cases = r'''
static void visit(int logical,const VkTextureSample *selected,void *private) {
    assert(logical==7 && selected->registration==registration.handle.id && selected->native_slot==3 &&
        selected->image && selected->image->addr==700 && selected->has_view);
    int stale = *(const int *)private;
    assert(selected->image->image==(stale ? 0 : 5) && selected->view==(stale ? 4 : 9));
    visit_count++;
}
static void recover(void) {
    chosen=NULL;
    unsigned before=prospective;
    int code=image_alias_sample_select_locked(7,-1,now,&chosen,&sample,&shape,0,&failure);
    if (code!=1)
        fprintf(stderr,"recovery refused code %d prospective %u released %d retiring %d superseded %d overwritten %d gone %d, image %llu serial %llu phase %d result %d\n",
                code, prospective, cached.released,cached.retiring,cached.superseded,cached.overwritten,cached.pub.gone,
                (unsigned long long)cached.pub.image,(unsigned long long)cached.t.serial,failure.phase,failure.status);
    assert(code==1);
    assert(prospective==before+1 && !normal && chosen==&receipt && sample.registration==9 && !shape.image);
    assert(image_alias_sample_receiver_locked(7,chosen));
}
int main(void) {
    assert(!image_alias_sample_select_locked(7,-1,now,&chosen,&sample,&shape,0,&failure) && !prospective);
    int binding_is_stale=0;
    image_alias_pool_member_locked(expected_pool,3,visit,&binding_is_stale);
    trace=1;
    cached.released=1; recover(); cached.released=0;
    cached.retiring=1; recover(); cached.retiring=0;
    cached.superseded=1; recover(); cached.superseded=0;
    cached.overwritten=1; recover();
    binding_is_stale=1;
    image_alias_pool_member_locked(expected_pool,3,visit,&binding_is_stale);
    cached.overwritten=0;
    cached.pub.gone=1; recover(); cached.pub.gone=0;
    cached.pub.image=0; recover(); cached.pub.image=5;
    cached.t.serial--; recover();
    image_alias_pool_member_locked(expected_pool,3,visit,&binding_is_stale);
    assert(visit_count==3 && traces==7);
    registration.handle.id++;
    assert(!image_alias_sample_receiver_locked(7,chosen));  // reused slot, different immutable registration
    registration.handle.id--;
    registration.descriptor.serial++;
    assert(!image_alias_sample_receiver_locked(7,chosen));  // same handle, different descriptor version
    registration.descriptor.serial--;
    expected_pool++;
    assert(!image_alias_sample_receiver_locked(7,chosen));
    expected_pool--;
    scope_status=VK_TEXTURE_REGISTRATION_GONE;
    assert(!image_alias_sample_receiver_locked(7,chosen));
    unsigned before=prospective;
    assert(!image_alias_sample_select_locked(7,-1,now,&chosen,&sample,&shape,0,&failure) && prospective==before);
    scope_status=VK_TEXTURE_REGISTRATION_CURRENT;
    chosen=NULL;
    outcome=0;
    assert(image_alias_sample_select_locked(7,-1,now,&chosen,&sample,&shape,0,&failure)<0 && !chosen);
    assert(failure.phase==VK_TEXTURE_ALIAS_OWNER && failure.status==RC_REGISTRY_STALE);
    outcome=-1;
    assert(image_alias_sample_select_locked(7,-1,now,&chosen,&sample,&shape,0,&failure)<0 && !chosen);
    assert(failure.phase==VK_TEXTURE_ALIAS_COVERAGE && failure.status==RC_REGISTRY_BLOCKED &&
        failure.receiver.mapping==scope.mapping);
    before=prospective;
    assert(image_alias_sample_select_locked(7,-1,(RcRegistryPosition){2,4},&chosen,&sample,&shape,0,&failure)<0 &&
        prospective==before && failure.phase==VK_TEXTURE_ALIAS_ORDER);
    assert(!normal);
    puts("Sampled alias receiver: seven stale material cases, exact registered replacement, dynamic shape, changed identity and missing donor refusal");
}
'''
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="alias-receiver-") as temp:
        source = Path(temp) / "check.c"
        exe = Path(temp) / ("check.exe" if os.name == "nt" else "check")
        source.write_text(prefix + production + cases, encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", str(source), "-o", str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

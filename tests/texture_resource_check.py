"""Exercise production texture preparation and active-membership transitions without a game or GPU."""
from pathlib import Path
import subprocess
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "runtime/vk_texture.c").read_text()
OUT = ROOT / "build/test/texture-resource"
CC = Path(os.environ.get("CC", r"C:\msys64\ucrt64\bin\clang.exe"))
if not CC.is_file():
    CC = Path(r"C:\Users\Andrew\AppData\Local\Programs\2NX2Navi\toolchain\msys64\ucrt64\bin\clang.exe")
sys.path.insert(0, str(ROOT / "tools"))
from check_content import GENERATED_MARKER
from bind_recipe_check import function

def section(name):
    return SOURCE.split("// RESOURCE_" + name + "_BEGIN:", 1)[1].split("\n", 1)[1].split("// RESOURCE_" + name + "_END", 1)[0]

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static struct { uint64_t queries,ready,prepared,deferred,unavailable,fallback,used;
uint64_t dependency_detach,live_visits,avoided_visits,material_used,versioned_used; } resource_path;
'''

MEMBERSHIP = PREFIX + r'''
#define POOL_TOUCH_DIV 64
typedef struct { struct { unsigned gpu_owned; } pub; int ph,pidx; unsigned touches; } Tex;
static unsigned next_dep,detach_count;
static void rt_fatal(const char *s) { (void)s; abort(); }
static void touch(Tex *e) { assert(e); e->touches++; }
static void pool_unlist(Tex *e) { if(e->pidx) detach_count++; e->pidx=0; }
static void pool_list(Tex *e) { if(!e->pub.gpu_owned&&!e->pidx)e->pidx=++next_dep; }
''' + section("MEMBERSHIP") + r'''
static void verify(void) {
    unsigned count[2]={0},seen[8192]={0};
    for(unsigned kind=0;kind<2;kind++) {
        unsigned prev=0;
        for(unsigned id=pool_active[kind].head;id;id=pref[id-1].active_next) {
            assert(id<8192&&!seen[id]++);
            PRef *p=&pref[id-1];
            assert(p->active&&p->e&&p->refs&&p->gown==kind&&p->active_prev==prev);
            prev=id; count[kind]++;
        }
        assert(count[kind]==pool_active[kind].count);
        if(pool_active[kind].cursor)assert(seen[pool_active[kind].cursor]);
    }
    for(int i=0;i<npref;i++)assert(!!seen[i+1]==!!(pref[i].e&&pref[i].refs));
}
int main(void) {
    Tex textures[5000]={0}; unsigned ids[5000];
    for(unsigned i=0;i<5000;i++) {
        ids[i]=pool_ref(&textures[i]);
        vk_texture_pool_hold_locked(0,ids[i]);
        vk_texture_pool_hold_locked(ids[i],0);
        assert(!textures[i].pidx);
    }
    verify(); uint64_t before=resource_path.live_visits;
    pool_touch(); assert(resource_path.live_visits==before);
    assert(resource_path.avoided_visits>=5000);
    for(unsigned i=0;i<130;i++)vk_texture_pool_hold_locked(0,ids[i]);
    unsigned saved=next_dep; vk_texture_pool_hold_locked(ids[0],ids[0]);
    assert(next_dep==saved&&textures[0].pidx);
    for(unsigned k=0;k<64;k++)pool_touch();
    for(unsigned i=0;i<130;i++)assert(textures[i].touches);
    textures[5].pub.gpu_owned=1; pool_gpu_owned(&textures[5]);
    assert(!textures[5].pidx&&pool_active[1].count==1); verify();
    unsigned touched=textures[5].touches; pool_touch(); assert(textures[5].touches==touched+1);
    pool_forget(&textures[5]); verify();
    vk_texture_pool_hold_locked(ids[5],0); verify();
    unsigned recycled=pool_ref(&textures[4999]); assert(recycled==ids[4999]);
    for(unsigned i=0;i<130;i++)if(i!=5)vk_texture_pool_hold_locked(ids[i],0);
    verify(); assert(!pool_active[0].count&&!pool_active[1].count);
    assert(detach_count>=5129);
    free(pref); puts("texture live membership: passed"); return 0;
}
'''

SAMPLE_PREFIX = PREFIX + r'''
typedef unsigned VkImage;
typedef uintptr_t VkCommandBuffer;
typedef int VkResult;
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
static RcRegistryPosition current_read = {31, 24};
static RcRegistryPosition vk_exec_resource_position(void) {return current_read;}
static int rc_registry_position_valid(RcRegistryPosition p) {return p.scene && p.ordinal;}
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b) {
    return a.scene != b.scene || a.ordinal != b.ordinal;
}
typedef struct { unsigned value; } view_t;
typedef struct { uint64_t addr; unsigned format; } TexKey;
typedef struct { void *mem; unsigned format, cpu; uint64_t serial; struct { uint64_t size; } L; } tex_t;
typedef struct { unsigned mirrored,gpu_owned,gone,image; } VkTexImage;
typedef struct TextureRegistration TextureRegistration;
typedef struct VkTextureRestore VkTextureRestore;
typedef struct VkTexturePresentSource VkTexturePresentSource;
typedef struct Tex { VkTexImage pub; TexKey key; tex_t t; unsigned sfill,superseded,overwritten;
void *material; unsigned ordered,current;struct {uint64_t id;} material_version;
void *source_mapping;TextureRegistration *source_registrations;unsigned material_writable;
unsigned released,retiring,uploaded,up_retry,checked,touches,source_blocked,source_canonical;
unsigned source_refusal;int source_refusal_result;RcRegistryPosition source_refusal_position; uint64_t checked_authority; struct Tex *next_at; } Tex;
static int source_binding_ordered(const Tex *e){return e->ordered;}
static struct { unsigned canonical_ready; } resource_bridge;
static int source_binding_current(Tex *e){return e->current;}
static int source_mapping_due(const Tex *e){(void)e;return 0;}
static int source_permission_current(Tex *e){return e->current;}
static uint64_t authority_epoch=1;
static int source_binding_compat_current(const Tex *e){return authority_epoch && e->checked_authority==authority_epoch;}
static int content_require_resident(Tex *e){e->pub.image=1;return 1;}
''' + SOURCE[SOURCE.index('enum {\n    SOURCE_REFUSAL_NONE,'):
             SOURCE.index('typedef struct TextureRegistration TextureRegistration;')] + r'''
typedef enum { RC_REGISTRY_UNSUPPORTED, RC_REGISTRY_BLOCKED } RcRegistryResult;
static void source_note_refusal(Tex *e, int stage, int result){
    e->source_refusal=stage;e->source_refusal_result=result;e->source_refusal_position=current_read;
}
static void *rc_registry_runtime(void) { return NULL; }
static uint64_t rc_registry_authority_epoch(void *r) { (void)r; return authority_epoch; }
typedef struct { Tex *ce; uint32_t cgen,cepoch; view_t cview; unsigned chas_view,used,has_view,finalized;
unsigned cheld,cwant,sample_owned; uint64_t absent_mapping,absent_protection; view_t v; tex_t t; TextureRegistration *registration; } TexSlot;
struct TextureRegistration { Tex *source;TextureRegistration *source_next,**source_previous;tex_t descriptor;int native_slot;
    struct { uint64_t id; } handle; };
static TexSlot tslots[2]; static int ntslots=2;
static TextureRegistration registrations[2];
static unsigned agen[2]={1,1},submit_epoch=1; static uint64_t sinfo[2]={0,1};
static Tex image,alias; static int by_key,by_addr;
static unsigned creates,uploads,deferred,fail_create,fail_upload,certificate,mutate_upload,round_checks;
static Tex *existing,*address_head; static int in_lookup;
static unsigned slots_gen,key_lookups,dirty_slots;

typedef struct {unsigned unused;} MemBackingCapturePlan;
typedef struct {unsigned unused;} RcInterval;
typedef struct VkTextureAliasFailure VkTextureAliasFailure;
enum { RC_BACKING_OK, RC_BACKING_ABSENT };
static int create_budget=-1, create_deferred, backing_hole, readable=1;
static uint64_t mapping_version=1;
static int want_take(void) { return 0; }
static int mem_backing_capture_plan(uint64_t address,uint64_t size,MemBackingCapturePlan *plan) {
    assert(address==4096&&size==64&&plan);return backing_hole?RC_BACKING_ABSENT:RC_BACKING_OK;
}
static int mem_mapping_range_unchanged(uint64_t address,uint64_t size,uint64_t version) {
    assert(address==4096&&size==64);return version==mapping_version;
}
static int mem_protection_range_unchanged(uint64_t address,uint64_t size,uint64_t version) {
    assert(address==4096&&size==64);return version==mapping_version;
}
static int slot_absence_valid(const TexSlot *slot,uint32_t generation) {
    return !slot->ce&&slot->cgen==generation&&slot->absent_mapping==mapping_version&&
        slot->absent_protection==mapping_version;
}
static int slot_storage_readable(TexSlot *slot) {
    if(!readable)slot->absent_mapping=slot->absent_protection=mapping_version;
    return readable;
}

static int tex_mirrored(const tex_t *t) { return t->cpu; }
static void make_key(const tex_t *t,TexKey *k) { memset(k,0,sizeof *k); k->addr=(uintptr_t)t->mem; k->format=t->format; }
static void *nvn_hm_get(const int *map,const void *key) {
    if(map==&by_addr)return address_head;
    key_lookups++;
    return existing&&!memcmp(&existing->key,key,sizeof(TexKey))?existing:NULL;
}
static int alias_has_gpu_content(const Tex *e) { return e->pub.gpu_owned; }
static int vk_texture_deferred(void) { return deferred; }
static Tex *entry_of(const tex_t *t) {
    assert(in_lookup); creates++;
    if(fail_create)return NULL;
    if(!existing) { memset(&image,0,sizeof image); existing=&image; make_key(t,&image.key); image.pub.mirrored=1; image.pub.image=1; }
    if(existing->t.serial!=t->serial) existing->checked=0;
    existing->t=*t;existing->released=existing->retiring=0; return existing;
}
static int pool_deps_certificate(Tex *e) { (void)e; return certificate; }
static void make_fresh_snap(Tex *e) { uploads++; if(!fail_upload)e->uploaded=1; if(mutate_upload)agen[0]++; }
static void slot_result_view(const TexSlot *s,const Tex *e,view_t *v,int *have) { (void)e; *v=s->v; *have=s->has_view; }
static void touch(Tex *e) { e->touches++; }
static int source_private_sample(const Tex *e){return e->pub.mirrored && !e->pub.gpu_owned && !e->material_writable && !e->pub.gone;}
static void dirty_push(uint32_t slot){dirty_slots|=1u<<slot;}
'''
HEADER = "\n".join(line for line in (ROOT / "runtime/texture_resource.h").read_text().splitlines()
                   if not line.startswith("#"))
registrations = (ROOT / "runtime/vk_texture_registrations.h").read_text()
REGISTRATION_SOURCE = registrations[registrations.index("static Tex *texture_registration_source("):
                                   registrations.index("static uint64_t texture_registration_key(")]
REGISTRATION_ID = function(SOURCE, "texture_slot_registration")
SAMPLE = SAMPLE_PREFIX + HEADER + r"""
static int texture_history_sample_locked(int logical,VkTextureSample *out,int *native,int recording) {
    assert(out && native && !recording);
    *native=logical;
    return logical==0 ? VK_TEXTURE_REGISTRATION_CURRENT : VK_TEXTURE_REGISTRATION_ABSENT;
}
#define VK_NULL_HANDLE 0
static const Tex *texture_projected_sample_locked(const TexSlot *slot,int native,VkImage expected,
                                                  VkTextureSample *out,RcRegistryPosition *edge) {
    assert(slot==&tslots[native] && !expected && out && !edge);
    return NULL; // CPU preparation here, actual GPU projected selections have their own fixture.
}
static const TexSlot *texture_read_key_slot(VkTextureReadKey key) {
    if(key.native_slot != 0 || !tslots[0].registration ||
       key.registration != tslots[0].registration->handle.id) return NULL;
    return &tslots[0];
}
""" + REGISTRATION_SOURCE + REGISTRATION_ID + section("SAMPLE") + r'''
static void reset(void) {
    memset(tslots,0,sizeof tslots); memset(&image,0,sizeof image); memset(&alias,0,sizeof alias);
    existing=address_head=NULL; creates=uploads=deferred=fail_create=fail_upload=certificate=mutate_upload=0;
    memset(registrations,0,sizeof registrations);key_lookups=dirty_slots=slots_gen=0;
    agen[0]=submit_epoch=1; round_checks=0;
    readable=1; backing_hole=create_deferred=0; create_budget=-1; mapping_version=1;
    tslots[0]=(TexSlot){.used=1,.has_view=1,.v={7},.t={(void *)(uintptr_t)4096,3,1,100,{64}}};
}
static void collect_dirty(int slot,void *arg){*(unsigned *)arg|=1u<<slot;}
int main(void) {
    VkTextureSample out;
    VkTextureSampleRefusal refusal = {0};
    reset();
    registrations[0]=(TextureRegistration){.descriptor=tslots[0].t,.native_slot=0,.handle={71}};
    tslots[0].registration=&registrations[0];
    create_budget=0;
    assert(texture_sample_prepare_locked(0,&out,1,NULL)==VK_TEXTURE_SAMPLE_PREPARE&&create_deferred&&!creates);
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&creates==1);
    reset();
    registrations[0]=(TextureRegistration){.descriptor=tslots[0].t,.native_slot=0,.handle={72}};
    tslots[0].registration=&registrations[0];
    readable=0; backing_hole=1;
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_UNBACKED&&!creates&&!out.image);
    VkTextureReadKey empty={.native_slot=out.native_slot,.registration=out.registration,.unbacked=1};
    assert(vk_texture_sample_unbacked_current_locked(empty));
    registrations[0].handle.id++;
    assert(!vk_texture_sample_unbacked_current_locked(empty)); // No absence transfer to a new registration.
    registrations[0].handle.id--;
    mapping_version++; backing_hole=0; fail_create=1;
    assert(!vk_texture_sample_unbacked_current_locked(empty));
    assert(texture_sample_prepare_locked(0,&out,0,&refusal)==VK_TEXTURE_SAMPLE_UNAVAILABLE&&!out.image&&creates==1);
    assert(refusal.registration==72 && !strcmp(refusal.stage,"no-provider") && !refusal.recorded);
    fail_create=0; readable=1;
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&creates==2&&out.image);
    assert(!vk_texture_sample_unbacked_current_locked(empty));
    reset();
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE&&!creates&&!uploads);
    mutate_upload=1;
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&creates==1&&uploads==1);
    assert(tslots[0].cgen==agen[0]&&out.view.value==7&&round_checks==1);
    assert(!image.touches); vk_texture_sample_use_locked(&out); assert(image.touches==1);
    unsigned n=uploads;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&uploads==n);
    assert(!out.registration && !out.canonical && out.native_slot==0);
    tslots[0].t.serial++;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE);
    certificate=0;
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&image.t.serial==101&&creates==2);
    image.ordered=image.current=1;
    authority_epoch++; assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_READY);
    image.current=0; assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE);
    image.ordered=0;
    authority_epoch++; assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE);
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY);
    n=uploads;
    submit_epoch++; assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE);
    certificate=1; assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&uploads==n);
    tslots[0].finalized=1; assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_FALLBACK);
    tslots[0].finalized=0; image.released=1;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE);
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&!image.released);
    image.pub.gpu_owned=1; assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_FALLBACK);
    reset(); address_head=&alias; alias.pub.gpu_owned=1;
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_FALLBACK&&!creates);
    reset(); fail_create=deferred=1;
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_DEFERRED&&!out.image);
    reset(); fail_upload=1;
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_UNAVAILABLE&&!out.image);
    image.source_canonical=image.source_blocked=1;
    assert(texture_sample_prepare_locked(0,&out,0,&refusal)==VK_TEXTURE_SAMPLE_CANONICAL_REFUSED&&!out.image);
    assert(refusal.registration==0 && refusal.material==100 && refusal.recorded &&
           !strcmp(refusal.stage,"binding-permission") && refusal.result==RC_REGISTRY_BLOCKED &&
           refusal.position.scene==31 && refusal.position.ordinal==24);
    current_read.ordinal++;
    sample_refusal_record(&refusal,&tslots[0],&image,"material-restore");
    assert(!refusal.recorded && !strcmp(refusal.stage,"material-restore"));
    current_read.ordinal--;
    reset(); vk_texture_sample_prepare_locked(0,&out); image.key.format++; tslots[0].sample_owned=0;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE);
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_FALLBACK);
    // Stale cached pointers may be freed; querying must reject the generation before reading the pointer.
    tslots[0].ce=(Tex *)(uintptr_t)1; agen[0]++;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE);
    reset();
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY);
    image.source_canonical=image.ordered=image.current=1;
    image.source_mapping=&image;
    registrations[0]=(TextureRegistration){.descriptor=tslots[0].t,.native_slot=0,.handle={71}};
    tslots[0].registration=&registrations[0];
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&out.canonical);
    assert(out.registration==71&&out.native_slot==0);
    assert(registrations[0].source==&image && !tslots[0].ce && !tslots[0].sample_owned);
    n=key_lookups; agen[0]++;tslots[0].ce=(Tex *)(uintptr_t)1;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_READY && out.image==&image.pub);
    assert(out.view.value==7&&key_lookups==n&&out.registration==71&&out.native_slot==0&&out.canonical);
    image.current=0;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE&&key_lookups==n);
    image.current=1;image.pub.image=0;image.material=&image;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE);
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&out.canonical&&key_lookups==n);
    tslots[1]=tslots[0];registrations[1]=(TextureRegistration){.descriptor=tslots[1].t,.native_slot=1,.handle={93}};
    tslots[1].registration=&registrations[1];
    assert(texture_registration_source_attach(&tslots[1],&image));
    assert(vk_texture_sample_query_locked(1,&out)==VK_TEXTURE_SAMPLE_READY);
    assert(out.registration==93&&out.native_slot==1&&out.canonical&&out.image==&image.pub);
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_READY);
    assert(out.registration==71&&out.native_slot==0&&out.canonical&&out.image==&image.pub&&key_lookups==n);
    unsigned visited=0;
    texture_registration_source_dirty(&image,collect_dirty,&visited);assert(visited==3);
    texture_registration_source_unlink(&registrations[0]);assert(!registrations[0].source);
    visited=0;texture_registration_source_dirty(&image,collect_dirty,&visited);assert(visited==2);
    texture_registration_source_clear(&image);assert(!image.source_registrations && !registrations[1].source);
    assert(dirty_slots==2&&slots_gen==1);
    image.material_writable=1;
    assert(!texture_registration_source_attach(&tslots[0],&image));
    image.material_writable=0;image.t.serial++;
    assert(!texture_registration_source_attach(&tslots[0],&image));
    puts("texture query/preparation/use: passed"); return 0;
}
'''

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.resolve().relative_to(ROOT.resolve())
    with tempfile.TemporaryDirectory(prefix='check-', dir=OUT) as temporary:
        run(Path(temporary))

def run(out):
    for name, source in (("membership", MEMBERSHIP), ("sample", SAMPLE)):
        src, exe = out / (name + ".c"), out / (name + ".exe")
        marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
        src.write_text("// " + marker + "\n" + source)
        subprocess.run([str(CC), "-std=gnu11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", str(src), "-o", str(exe)], check=True, env=dict(os.environ, PATH=str(CC.parent)+os.pathsep+os.environ.get("PATH", "")))
        subprocess.run([str(exe)], check=True)

if __name__ == "__main__":
    main()

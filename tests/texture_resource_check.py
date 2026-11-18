"""Exercise production texture preparation and active-membership transitions without a game or GPU."""
from pathlib import Path
import subprocess
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "runtime/vk_texture.c").read_text()
OUT = ROOT / "build/test/texture-resource"
CC = Path(os.environ.get("CC", r"C:\msys64\ucrt64\bin\clang.exe"))
if not CC.is_file():
    CC = Path(r"C:\Users\Andrew\AppData\Local\Programs\2NX2Navi\toolchain\msys64\ucrt64\bin\clang.exe")
sys.path.insert(0, str(ROOT / "tools"))
from check_content import GENERATED_MARKER

def section(name):
    return SOURCE.split("// RESOURCE_" + name + "_BEGIN:", 1)[1].split("\n", 1)[1].split("// RESOURCE_" + name + "_END", 1)[0]

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static struct { uint64_t queries,ready,prepared,deferred,unavailable,fallback,used;
uint64_t dependency_detach,live_visits,avoided_visits; } resource_path;
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
typedef struct { unsigned value; } view_t;
typedef struct { uint64_t addr; unsigned format; } TexKey;
typedef struct { void *mem; unsigned format, cpu; } tex_t;
typedef struct { unsigned mirrored,gpu_owned,gone,image; } VkTexImage;
typedef struct Tex { VkTexImage pub; TexKey key; unsigned sfill,superseded,overwritten;
unsigned released,retiring,uploaded,up_retry,checked,touches; struct Tex *next_at; } Tex;
typedef struct { Tex *ce; uint32_t cgen,cepoch; view_t cview; unsigned chas_view,used,has_view,finalized;
unsigned cheld,cwant,sample_owned; uint64_t absent_mapping,absent_protection; view_t v; tex_t t; } TexSlot;
static TexSlot tslots[2]; static int ntslots=2;
static unsigned agen[2]={1,1},submit_epoch=1; static uint64_t sinfo[2]={0,1};
static Tex image,alias; static int by_key,by_addr;
static unsigned creates,uploads,deferred,fail_create,fail_upload,certificate,mutate_upload,round_checks;
static Tex *existing,*address_head; static int in_lookup;
static int tex_mirrored(const tex_t *t) { return t->cpu; }
static void make_key(const tex_t *t,TexKey *k) { memset(k,0,sizeof *k); k->addr=(uintptr_t)t->mem; k->format=t->format; }
static void *nvn_hm_get(const int *map,const void *key) {
    if(map==&by_addr)return address_head;
    return existing&&!memcmp(&existing->key,key,sizeof(TexKey))?existing:NULL;
}
static int alias_has_gpu_content(const Tex *e) { return e->pub.gpu_owned; }
static int vk_texture_deferred(void) { return deferred; }
static Tex *entry_of(const tex_t *t) {
    assert(in_lookup); creates++;
    if(fail_create)return NULL;
    if(!existing) { memset(&image,0,sizeof image); existing=&image; make_key(t,&image.key); image.pub.mirrored=1; image.pub.image=1; }
    existing->released=existing->retiring=0; return existing;
}
static int pool_deps_certificate(Tex *e) { (void)e; return certificate; }
static void make_fresh_snap(Tex *e) { uploads++; if(!fail_upload)e->uploaded=1; if(mutate_upload)agen[0]++; }
static void slot_result_view(const TexSlot *s,const Tex *e,view_t *v,int *have) { (void)e; *v=s->v; *have=s->has_view; }
static void touch(Tex *e) { e->touches++; }
'''
HEADER = "\n".join(line for line in (ROOT / "runtime/texture_resource.h").read_text().splitlines()
                   if not line.startswith("#"))
SAMPLE = SAMPLE_PREFIX + HEADER + section("SAMPLE") + r'''
static void reset(void) {
    memset(tslots,0,sizeof tslots); memset(&image,0,sizeof image); memset(&alias,0,sizeof alias);
    existing=address_head=NULL; creates=uploads=deferred=fail_create=fail_upload=certificate=mutate_upload=0;
    agen[0]=submit_epoch=1; round_checks=0;
    tslots[0]=(TexSlot){.used=1,.has_view=1,.v={7},.t={(void *)(uintptr_t)4096,3,1}};
}
int main(void) {
    VkTextureSample out; reset();
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE&&!creates&&!uploads);
    mutate_upload=1;
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&creates==1&&uploads==1);
    assert(tslots[0].cgen==agen[0]&&out.view.value==7&&round_checks==1);
    assert(!image.touches); vk_texture_sample_use_locked(&out); assert(image.touches==1);
    unsigned n=uploads;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_READY&&uploads==n);
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
    reset(); vk_texture_sample_prepare_locked(0,&out); image.key.format++; tslots[0].sample_owned=0;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE);
    assert(vk_texture_sample_prepare_locked(0,&out)==VK_TEXTURE_SAMPLE_FALLBACK);
    // Stale cached pointers may be freed; querying must reject the generation before reading the pointer.
    tslots[0].ce=(Tex *)(uintptr_t)1; agen[0]++;
    assert(vk_texture_sample_query_locked(0,&out)==VK_TEXTURE_SAMPLE_PREPARE);
    puts("texture query/preparation/use: passed"); return 0;
}
'''

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, source in (("membership", MEMBERSHIP), ("sample", SAMPLE)):
        src, exe = OUT / (name + ".c"), OUT / (name + ".exe")
        marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
        src.write_text("// " + marker + "\n" + source)
        subprocess.run([str(CC), "-std=gnu11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=undefined", str(src), "-o", str(exe)], check=True, env=dict(os.environ, PATH=str(CC.parent)+os.pathsep+os.environ.get("PATH", "")))
        subprocess.run([str(exe)], check=True)

if __name__ == "__main__":
    main()

"""Verify the actual by-ID preparation decision across predeclared writers."""
from pathlib import Path
import os, subprocess, tempfile
from bind_recipe_check import function
ROOT=Path(__file__).resolve().parents[1]
CC=Path(os.environ["CC"])
source=(ROOT/"runtime/vk_texture.c").read_text()
start=source.rindex("static TextureSlotSelection texture_slot_select_locked(")
body="\n".join(function(source[start:], name) for name in (
    "texture_slot_select_locked", "texture_slot_refresh_locked", "texture_by_slot_locked",
))
prefix=r'''
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>
typedef int VkTextureSampleStatus;
typedef struct {int recorded;} VkTextureSampleRefusal;
typedef struct { unsigned value; } view_t;
typedef struct { unsigned image,gpu_owned,gone; } VkTexImage;
enum {VK_TEXTURE_SAMPLE_DEFERRED,VK_TEXTURE_SAMPLE_CANONICAL_REFUSED,VK_TEXTURE_SAMPLE_READY,VK_TEXTURE_SAMPLE_UNAVAILABLE};
enum { SOURCE_REFUSAL_UPLOAD, SOURCE_REFUSAL_PERMISSION, RC_REGISTRY_UNSUPPORTED, RC_REGISTRY_BLOCKED };
typedef struct {uint64_t addr;} TexKey;
typedef struct {uint64_t serial;struct {uint64_t size;} L;} tex_t;
typedef struct {int released,uploaded,up_retry,source_canonical,source_blocked;VkTexImage pub;
 uint64_t checked,checked_authority;uint64_t prepared,next;TexKey key;tex_t t;void *image_host_backing;} Tex;
typedef struct {Tex *source; view_t view;int has_view;tex_t descriptor;} TextureRegistration;
typedef struct {Tex *ce;int used,cheld,chas_view,has_view,sample_owned,finalized;uint32_t cgen;uint64_t cepoch;
    view_t cview,v;TextureRegistration *registration;} TexSlot;
typedef struct {TexSlot *slot;Tex *entry;int associated;} TextureSlotSelection;
static void sample_refusal_record(VkTextureSampleRefusal *failure,const TexSlot *s,const Tex *e,const char *stage){
 (void)s;(void)e;(void)stage;assert(!failure);
}
static int preserve_pool_contents_current(const Tex *e){(void)e;return 1;}
static TexSlot tslots[1];static int ntslots=1;
static uint32_t agen[1]={1};static uint64_t sinfo[1]={0};
static int pool_ctx,resolve_held,in_lookup;
static Tex *legacy_source;
static unsigned legacy_resolves;
static Tex *texture_registration_source(const TexSlot *s){return s->registration?s->registration->source:NULL;}
static int texture_registration_source_attach(const TexSlot *s,Tex *e){
 assert(s->registration&&e->source_canonical);s->registration->source=e;return 1;
}
static int slot_absence_valid(const TexSlot *s,uint32_t g){(void)s;(void)g;return 0;}
static Tex *slot_resolve(TexSlot *s,view_t *view,int *has_view,int private_output,uint32_t *installed_generation){
 assert(in_lookup && !private_output && installed_generation && !*installed_generation);
 legacy_resolves++;*view=s->v;*has_view=s->has_view;return legacy_source;
}
static const int by_key;
static int storage_target_identity_current(const Tex *e){(void)e;assert(0);return 0;}
static void make_key(const tex_t *t,TexKey *key){(void)t;(void)key;assert(0);}
static Tex *nvn_hm_get(const int *table,const TexKey *key){(void)table;(void)key;assert(0);return NULL;}
static int nvn_gpu_only(uint64_t addr){(void)addr;assert(0);return 0;}
static int mem_backing_current(const void *backing){(void)backing;assert(0);return 0;}
static void slot_result_view(const TexSlot *s,const Tex *e,view_t *view,int *has_view){
 assert(s->registration->source==e);*view=s->registration->view;*has_view=s->registration->has_view;
}
static void content_use(Tex *e){assert(e);}
static void source_note_refusal(Tex *e,int stage,int result){(void)e;(void)stage;(void)result;}
static uint64_t submit_epoch=1,position=2;
static struct { unsigned canonical_ready; } resource_bridge;
static int tex_deferred,force_deferred,replace_mapping;
static unsigned round_checks,preparations,certificates,touches;
static void *rc_registry_runtime(void){return NULL;}
static uint64_t rc_registry_authority_epoch(void *r){(void)r;return 17;}
static int source_binding_ready(Tex *e){return e->checked==submit_epoch && position>=e->prepared && position<e->next;}
static int source_mapping_due(const Tex *e){(void)e;return 0;}
static int source_permission_current(Tex *e){return source_binding_ready(e);}
static int pool_deps_certificate(Tex *e){(void)e;certificates++;return 1;}
static void touch(Tex *e){(void)e;touches++;}
static void make_fresh_snap(Tex *e){
 preparations++;if(force_deferred){tex_deferred=1;return;}
 if(replace_mapping){
    assert(tslots[0].registration->source==e);
    tslots[0].registration->source=NULL;e->pub.image++;
 }
 e->prepared=position;e->next=position+4;e->uploaded=1;
}
'''
tests=r'''
int main(void){
 Tex e={.pub.image=77,.uploaded=1,.checked=1,.checked_authority=17,.prepared=2,.next=4};
 tslots[0]=(TexSlot){.used=1,.cepoch=1,.cgen=1,.ce=&e};
 legacy_source=&e;
 assert(texture_by_slot_locked(0,NULL,NULL,NULL,NULL)==&e.pub && !preparations && !certificates);
 position=4; // A previously declared producer is reached; the global authority value remains exactly17.
 assert(texture_by_slot_locked(0,NULL,NULL,NULL,NULL)==&e.pub && preparations==1 && !certificates && touches==3);
 assert(e.prepared==4 && e.next==8 && e.checked_authority==17);
 position=6;for(int i=0;i<100;i++)assert(texture_by_slot_locked(0,NULL,NULL,NULL,NULL)==&e.pub);
 assert(preparations==1 && !certificates); // No repeated work while the new actual preparation remains usable.
 position=8;force_deferred=1;
 assert(!texture_by_slot_locked(0,NULL,NULL,NULL,NULL) && preparations==2 && e.prepared==4);
 force_deferred=tex_deferred=0;
 assert(texture_by_slot_locked(0,NULL,NULL,NULL,NULL)==&e.pub && preparations==3 && e.prepared==8);
 e.released=1;position=99;assert(texture_by_slot_locked(0,NULL,NULL,NULL,NULL)==&e.pub && preparations==3);
 assert(round_checks==3);
 // A supported already-selected ordinary provider does not touch the unrelated address cache.
 TextureRegistration registration={.source=&e,.view={21},.has_view=1};
 tslots[0]=(TexSlot){.used=1,.ce=(Tex *)(uintptr_t)1,.registration=&registration,.v={13}};
 e.released=0;e.source_canonical=1;position=8;view_t view;int has_view;
 assert(texture_by_slot_locked(0,&view,&has_view,NULL,NULL)==&e.pub && view.value==21 && has_view);
 assert(!legacy_resolves && preparations==3);
 position=12;replace_mapping=1;
 assert(texture_by_slot_locked(0,&view,&has_view,NULL,NULL)==&e.pub && !legacy_resolves && preparations==4);
 assert(e.prepared==12 && e.next==16 && registration.source==&e && e.pub.image==78);
 replace_mapping=0;
 registration.source=NULL;e.source_canonical=0;
 tslots[0].ce=NULL;
 assert(texture_by_slot_locked(0,&view,&has_view,NULL,NULL)==&e.pub && legacy_resolves==1 && view.value==13);
 puts("texture by-ID boundary: unchanged global epoch forces one actual preparation; certificates cannot skip it; retry and released lifetime passed");
}
'''
env=dict(os.environ,PATH=str(CC.parent)+os.pathsep+os.environ.get("PATH",""))
with tempfile.TemporaryDirectory(prefix="texture-boundary-",dir=ROOT/"build/test") as tmp:
 p=Path(tmp)/"boundary.c";p.write_text(prefix+body+tests);exe=p.with_suffix(".exe")
 subprocess.run([str(CC),"-std=gnu11","-O1","-Wall","-Wextra","-Werror","-fsanitize=undefined","-fsanitize-trap=all",str(p),"-o",str(exe)],env=env,check=True)
 subprocess.run([str(exe)],env=env,check=True)

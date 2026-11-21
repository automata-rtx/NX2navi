"""Compile production exact group keys, live/retained DIRECT admission and selector generations."""
from pathlib import Path
import os, re, subprocess, sys, tempfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bind_recipe_check import function
ROOT=Path(__file__).resolve().parents[1]
CC=Path(os.environ['CC'])
source=(ROOT/'runtime/vk_texture.c').read_text()
header=(ROOT/'runtime/vk.h').read_text()
key=source[source.index('typedef struct {\n    uint64_t addr;'):source.index('} TexKey;')+len('} TexKey;')]
group=re.search(r'typedef struct \{[^\n]+\} VkSlotGroup;',header).group(0)
start=source.index('typedef struct {\n    TexKey texture;')
end=source.index('void vk_texture_groups_unstamp_locked(',start)
fixture=r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "nvn_obj.h"
#include "nvn_hmap.h"
typedef uint64_t VkImage;
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
enum {VK_TEXTURE_BINDING_DIRECT=1,VK_TEXTURE_BINDING_SOURCE=2};
'''+key+r'''
typedef struct Tex {TexKey key;tex_t t;struct Tex *next_at;void *material;uint64_t uid;
 struct {VkImage image;int gone,gpu_owned,mirrored;uint8_t swz[4];} pub;
 int released,retiring,up_retry,sfill,uploaded,superseded,overwritten;} Tex;
typedef struct {Tex *source;tex_t descriptor;view_t view;int has_view;} TextureRegistration;
typedef struct {tex_t t;view_t v;uint64_t khash,absent_mapping,absent_protection,pool_owner;Tex *ce;
 uint32_t cgen;uint8_t used,has_view,finalized,cwant;TextureRegistration *registration;} TexSlot;
#define AGEN_N 4
static TexSlot tslots[8];static int ntslots=8;
static uint32_t agen[AGEN_N]={1,1,1,1};static uint64_t sinfo[8];
static uint32_t agen_top[AGEN_N]={1,1,1,1},bepoch[AGEN_N],bump_count[AGEN_N],tex_gen=1,slots_gen=1;
static uint64_t bsig[AGEN_N],n_bumps,n_undone;
static struct {uint64_t sig;uint32_t b,epoch,gen,valid;} gs[8192];
static int in_lookup;static unsigned dirty_events;
#define DIRTY_BUCKET 0x80000000u
static void dirty_push(uint32_t event){assert(event&DIRTY_BUCKET);dirty_events++;}
static int rt_knob(const char *name,const char *value){(void)name;(void)value;return 1;}
static int32_t bucket_head[AGEN_N],next_array[8],previous_array[8],*slot_next=next_array,*slot_prev=previous_array;
static uint16_t buckets[8],*slot_bucket=buckets;
static uint32_t agen_idx(uint64_t addr){return (uint32_t)(addr%AGEN_N);}
__attribute__((noreturn)) static void rt_fatal(const char *s){fputs(s,stderr);abort();}
static unsigned ref_calls;static uint32_t pool_ref(Tex *e){assert(e);ref_calls++;return 7;}
static int proof=VK_TEXTURE_BINDING_DIRECT;
static int vk_texture_binding_receipt_locked(int id,VkImage expected,RcRegistryPosition *edge){assert(tslots[id].ce->pub.image==expected);*edge=(RcRegistryPosition){0};return proof;}
static nvn_hmap by_key={.klen=sizeof(TexKey)},by_addr={.klen=sizeof(uint64_t)},holds={.klen=sizeof(TexKey)};
static int pool_ctx=1,resolve_held,resolve_stale,create_budget=-1,create_deferred,created,restored;
static uint64_t n_stale_kept,n_stale,n_stale_hm;
static int tex_mirrored(const tex_t *t){return (t->pool_flags&1)!=0;}
static int alias_has_gpu_content(const Tex *e){return e->pub.gpu_owned;}
static int follow_size_ok(const Tex *e,const TexKey *k){return e->key.d.width==k->d.width&&e->key.d.height==k->d.height;}
static int stale_kept(const Tex *e){return e&&!e->superseded&&!e->overwritten&&(e->uploaded||e->pub.gpu_owned);}
static int content_require_resident(Tex *e){(void)e;restored++;return 0;}
static int slot_storage_readable(TexSlot *s){(void)s;return 1;}
static int want_take(void){return 1;}
static Tex *entry_of_mode(const tex_t *t,int private_output,uint32_t *installed_generation){
 (void)t;assert(!private_output && installed_generation && !*installed_generation);created++;return NULL;
}
static void retire_cancel(Tex *e){e->retiring=0;}
static int terrain_heightmap(const tex_t *t){(void)t;return 0;}
'''+group+source[start:end]+function(source[source.index('static void make_key(const tex_t *t, TexKey *k) {'):],'make_key')+''.join(function(source,n) for n in ('gs_at','tex_gen_bump','tex_gen_flip','from_storage','same_but_swizzle','slot_finalized_owner'))+function(source[source.rindex('static void slot_result_view('):],'slot_result_view')+''.join(function(source,n) for n in ('slot_resolve','vk_texture_group_direct_locked'))+r'''
static Tex *resolve_slot(TexSlot *slot,view_t *view,int *has_view){
 uint32_t installed_generation=0;
 Tex *owner=slot_resolve(slot,view,has_view,0,&installed_generation);
 assert(!installed_generation); // This fixture exercises existing owners, not creation or revival.
 return owner;
}
int main(void){
 slot_grp=malloc(sizeof(*slot_grp)*8);assert(slot_grp);
 for(unsigned i=0;i<AGEN_N;i++)bucket_head[i]=bucket_ghead[i]=-1;
 for(unsigned i=0;i<8;i++){slot_prev[i]=-2;slot_next[i]=slot_grp[i]=-1;}
 Tex image={.pub={.image=100}};
 tslots[0]=(TexSlot){.t={.mem=(uint8_t*)(uintptr_t)100,.d={.width=64,.height=64,.target=1,.format=2},.L={.size=4096}},
  .ce=&image,.cgen=1,.used=1,.has_view=1,.khash=123};
 tslots[0].v.set_layers=1;tslots[0].v.nlayers=1;
 tslots[1]=tslots[0];tslots[1].t.serial=99;tslots[1].v.pad1[0]=255;
 slot_place(0,0,123);slot_place(1,0,123);int g=slot_grp[0];uint64_t incarnation=grp[g].pub.incarnation;
 assert(slot_grp[1]==g&&grp[g].pub.n==2&&vk_texture_group_member_locked(g,incarnation,1));
 // Deliberately identical supplied hashes cannot merge differing semantic keys.
 tslots[2]=tslots[0];tslots[2].t.d.width=128;slot_place(2,0,123);assert(slot_grp[2]!=g);
 tslots[3]=tslots[0];tslots[3].v.min_layer=1;slot_place(3,0,123);assert(slot_grp[3]!=g&&slot_grp[3]!=slot_grp[2]);
 tslots[4]=tslots[0];tslots[4].t.L.size++;slot_place(4,0,123);assert(slot_grp[4]!=g);
 tslots[5]=tslots[0];tslots[5].t.pool_flags=1;slot_place(5,0,123);assert(slot_grp[5]!=g);
 uint32_t entry=0;assert(vk_texture_group_direct_locked(g,incarnation,0,100,&entry)&&entry==7&&ref_calls==1);
 image.released=1;assert(!vk_texture_group_direct_locked(g,incarnation,0,100,&entry));image.released=0;
 image.retiring=1;assert(!vk_texture_group_direct_locked(g,incarnation,0,100,&entry));image.retiring=0;
 image.pub.mirrored=1;assert(!vk_texture_group_direct_locked(g,incarnation,0,100,&entry));image.pub.mirrored=0;
 image.sfill=1;assert(!vk_texture_group_direct_locked(g,incarnation,0,100,&entry));image.sfill=0;
 assert(!vk_texture_group_direct_locked(g,incarnation,0,101,&entry));
 tslots[0].cgen=0;assert(!vk_texture_group_direct_locked(g,incarnation,0,100,&entry));tslots[0].cgen=1;
 tslots[0].finalized=1;assert(!vk_texture_group_member_locked(g,incarnation,0));
 assert(!vk_texture_group_direct_locked(g,incarnation,0,100,&entry));tslots[0].finalized=0;
 proof=VK_TEXTURE_BINDING_SOURCE;assert(!vk_texture_group_direct_locked(g,incarnation,0,100,&entry));proof=VK_TEXTURE_BINDING_DIRECT;
 assert(ref_calls==1); // Failed source/identity proofs never create an association owner.
 // Finalized selectors form a distinct exact-key group, retaining the same live DIRECT eligibility rules.
 tslots[0].finalized=1;group_finalize_slot(0);int finished=slot_grp[0];
 assert(finished!=g&&grp[finished].key.finalized&&grp[g].pub.n==1);
 assert(vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,100,&entry));
 tslots[1].finalized=1;group_finalize_slot(1);
 assert(slot_grp[1]==finished&&grp[finished].pub.n==2&&!grp[g].pub.incarnation);
 assert(vk_texture_group_member_locked(finished,grp[finished].pub.incarnation,1));
 image.released=1;assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,100,&entry));image.released=0;
 image.retiring=1;assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,100,&entry));image.retiring=0;
 image.sfill=1;assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,100,&entry));image.sfill=0;
 proof=VK_TEXTURE_BINDING_SOURCE;assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,100,&entry));proof=VK_TEXTURE_BINDING_DIRECT;
 // Re-registration changes selector state and cannot keep the finalized association.
 tslots[0].finalized=tslots[1].finalized=0;slot_place(0,0,123);slot_place(1,0,123);
 g=slot_grp[0];incarnation=grp[g].pub.incarnation;
 assert(slot_grp[1]==g&&!grp[g].key.finalized&&!grp[finished].pub.incarnation);
 tslots[1].t.d.width++;assert(!vk_texture_group_member_locked(g,incarnation,1));tslots[1].t.d.width--;
 slot_place(0,0,0);slot_place(1,0,0);assert(!grp[g].pub.incarnation);
 slot_place(0,0,123);assert(slot_grp[0]==g&&grp[g].pub.incarnation!=incarnation);
 assert(!vk_texture_group_direct_locked(g,incarnation,0,100,&entry));
 assert(vk_texture_group_direct_locked(g,grp[g].pub.incarnation,0,100,&entry));
 assert(ngrp_live==5);
 // Exercise the actual selector: a finalized descriptor follows the shared live by-address selection while
 // its otherwise identical nonfinalized registration keeps its own live image. Swizzle remains the slot's.
 Tex own={.pub={.image=500}},live={.pub={.image=600,.gpu_owned=1}},other={.pub={.image=700,.gpu_owned=1}};
 make_key(&tslots[0].t,&own.key);live.key=other.key=own.key;live.key.d.swizzle_r=2;
 other.key.d.target++;other.next_at=&live;
 nvn_hm_put(&by_key,&own.key,&own);nvn_hm_put(&by_addr,&own.key.addr,&other);
 view_t view0,view1;int has0,has1;
 assert(resolve_slot(&tslots[0],&view0,&has0)==&own);
 slot_place(1,0,123);
 tslots[0].finalized=1;group_finalize_slot(0);finished=slot_grp[0];
 tslots[1].finalized=1;group_finalize_slot(1);assert(slot_grp[1]==finished);
 tslots[0].ce=resolve_slot(&tslots[0],&view0,&has0);tslots[1].ce=resolve_slot(&tslots[1],&view1,&has1);
 assert(tslots[0].ce==&live&&tslots[1].ce==&live&&has0==has1);
 view1.pad1[0]=view0.pad1[0];assert(!memcmp(&view0,&view1,sizeof view0));
 assert(vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,600,&entry));
 // A changed by-address selection invalidates the old generation; the next stable live selection can replace it.
 other.key.d.target=own.key.d.target;other.key.d.swizzle_r=3;tex_gen_bump(own.key.addr);
 assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,600,&entry));
 tslots[0].ce=resolve_slot(&tslots[0],&view0,&has0);tslots[0].cgen=agen[0];
 assert(tslots[0].ce==&other&&vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,700,&entry));
 // Selection alone is insufficient: live CPU/split-fill and retirement stay individual.
 other.sfill=1;other.pub.gpu_owned=0;
 assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,700,&entry));other.sfill=0;other.pub.gpu_owned=1;
 other.retiring=1;assert(!resolve_slot(&tslots[0],&view0,&has0)&&resolve_held);other.retiring=0;
 other.released=live.released=own.released=1;
 tslots[0].ce=resolve_slot(&tslots[0],&view0,&has0);assert(tslots[0].ce==&own);
 assert(vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,500,&entry));
 // The same numeric physical image in another logical instance is not the cached slot's retained own entry.
 Tex replacement=own;nvn_hm_put(&by_key,&own.key,&replacement);
 assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,500,&entry));
 nvn_hm_put(&by_key,&own.key,&own);
 own.retiring=1;assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,500,&entry));own.retiring=0;
 // The retained-own admission does not override a later live alias. Its stale generation refuses before
 // dereferencing/accepting the old selected entry; the real selector then chooses the live image instead.
 other.released=0;tex_gen_bump(own.key.addr);
 assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,500,&entry));
 tslots[0].ce=resolve_slot(&tslots[0],&view0,&has0);tslots[0].cgen=agen[0];
 assert(tslots[0].ce==&other&&vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,700,&entry));
 other.released=1;tex_gen_bump(own.key.addr);
 tslots[0].ce=resolve_slot(&tslots[0],&view0,&has0);tslots[0].cgen=agen[0];assert(tslots[0].ce==&own);
 own.superseded=1;assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,500,&entry));own.superseded=0;
 own.sfill=1;assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,500,&entry));own.sfill=0;
 // A stale cache pointing at another released entry is not the exact descriptor's retained-own case.
 tslots[0].ce=&live;assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,600,&entry));tslots[0].ce=&own;
 // Even if a CPU slot were placed into a group in future, its stale-kept history is not a DIRECT group proof.
 tslots[0].t.pool_flags=1;group_finalize_slot(0);int cpu=slot_grp[0];own.pub.gpu_owned=1;
 assert(!vk_texture_group_direct_locked(cpu,grp[cpu].pub.incarnation,0,500,&entry));own.pub.gpu_owned=0;
 tslots[0].t.pool_flags=0;group_finalize_slot(0);finished=slot_grp[0];
 // Real generation revival preserves the same retained selection, but a lookup-created transition invalidates
 // the old state permanently. No fixture restamps a result after a selector changed the generation.
 own.released=0;tex_gen_bump(own.key.addr);own.released=1;tex_gen_flip(&own,0);
 uint32_t retained_generation=agen[0];tslots[0].ce=&own;tslots[0].cgen=retained_generation;
 assert(vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,500,&entry));
 own.released=0;tex_gen_flip(&own,0);assert(agen[0]!=retained_generation);
 own.released=1;tex_gen_flip(&own,0);assert(agen[0]==retained_generation);
 assert(vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,500,&entry));
 own.released=0;in_lookup=1;tex_gen_flip(&own,0);in_lookup=0;
 own.released=1;tex_gen_flip(&own,0);assert(agen[0]!=retained_generation);
 assert(!vk_texture_group_direct_locked(finished,grp[finished].pub.incarnation,0,500,&entry));
 assert(dirty_events&&n_undone&&n_bumps&&slots_gen>1);
 assert(!created&&!restored&&!resolve_stale&&!create_deferred);
 free(by_key.keys);free(by_key.vals);free(by_addr.keys);free(by_addr.vals);
 free(slot_grp);free(grp);free(grp_by.keys);free(grp_by.vals);
 puts("texture groups: exact keys, live/retained DIRECT selection, replacement identity and generation revival passed");
}
'''
env=dict(os.environ,PATH=str(CC.parent)+os.pathsep+os.environ.get('PATH',''))
with tempfile.TemporaryDirectory(prefix='texture-group-',dir=ROOT/'build/test') as tmp:
    src=Path(tmp)/'check.c';src.write_text(fixture);exe=src.with_suffix('.exe')
    subprocess.run([str(CC),'-std=gnu11','-O1','-Wall','-Wextra','-Werror','-Wno-unused-function','-Wno-unused-variable',
                   '-fsanitize=undefined','-fsanitize-trap=all','-I',str(ROOT/'runtime'),str(src),'-o',str(exe)],env=env,check=True)
    subprocess.run([str(exe)],env=env,check=True)

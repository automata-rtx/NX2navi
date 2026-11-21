"""Exercise production finalization across individual and grouped registrations."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bind_recipe_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ["CC"])
source = (ROOT / "runtime/vk_texture.c").read_text()
body = "\n".join(function(source, name) for name in
                 ("group_finalize_slot", "finalize_slot", "vk_texture_release"))
fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef struct {uint64_t addr,selector;} TexKey;
typedef struct {void *mem;uint64_t serial,selector;struct {uint64_t size;} L;} tex_t;
typedef struct {uint64_t backing,offset,size;} RcSourceRange;
typedef struct {RcSourceRange range;} RcInterval;
typedef struct {uint32_t count;const RcInterval *intervals;} TextureSourceMapping;
typedef struct {int released,source_canonical;tex_t t;TextureSourceMapping *source_mapping;} Tex;
typedef struct {tex_t t;int finalized,sample_owned;void *ce;uint64_t khash;} TexSlot;
enum {RC_GUEST_COMPATIBILITY_BACKING=1};
static TexSlot tslots[6];
static int slot_next[6],slot_group[6],*slot_grp=slot_group,bucket_head[1],bucket_ghead[1];
static struct {struct {int head;} pub;int gnext;} grp[4];
static int sinfo=1,by_key,bumps,finalizations,mapped_finalizations,entry_present;
static uint64_t n_fin_other,n_fin_other_hm,n_fin_own;
static uint64_t expected_serial=7;
static Tex entry;
static uint32_t agen_idx(uint64_t address){(void)address;return 0;}
static void make_key(const tex_t *t,TexKey *key){*key=(TexKey){(uintptr_t)t->mem,t->selector};}
static int terrain_heightmap(const tex_t *t){(void)t;return 0;}
static int retired_registrations;
static void texture_registration_finalize(TexSlot *slot,int id){
 assert(slot==&tslots[id]&&!slot->finalized);retired_registrations++;
}
static void lock(void){}
static void unlock(void){}
static void *rc_registry_runtime(void){return NULL;}
static void rc_registry_finalize_object(void *r,uint64_t serial,RcSourceRange range){
 (void)r;assert(serial==expected_serial);
 if(range.backing==RC_GUEST_COMPATIBILITY_BACKING){assert(range.offset==100&&range.size==64);finalizations++;}
 else {assert(range.backing==5&&range.offset==10&&range.size==64);mapped_finalizations++;}
}
static Tex *nvn_hm_get(void *map,const TexKey *key){(void)map;(void)key;return entry_present?&entry:NULL;}
static void tex_gen_bump(uint64_t address){assert(address==100);bumps++;}
static void mark_released(Tex *e){e->released=1;tex_gen_bump(100);}
static void sweep_released(void){}
static void slot_place(int id,uint32_t bucket,uint64_t hash){
 assert(bucket==0&&hash==3&&slot_group[id]>=0&&slot_group[id]<2);
 int g=slot_group[id],*link=&grp[g].pub.head;
 while(*link!=id){assert(*link>=0);link=&slot_next[*link];}
 *link=slot_next[id];
 if(grp[g].pub.head<0){
  int *group_link=&bucket_ghead[0];
  while(*group_link!=g){assert(*group_link>=0);group_link=&grp[*group_link].gnext;}
  *group_link=grp[g].gnext;grp[g].gnext=-77; // Recycled group's previous link is no longer readable.
 }
 int final=g+2;
 if(grp[final].pub.head<0){grp[final].gnext=bucket_ghead[0];bucket_ghead[0]=final;}
 link=&grp[final].pub.head;while(*link>=0)link=&slot_next[*link];
 *link=id;slot_next[id]=-1;slot_group[id]=final;
}
''' + body + r'''
int main(void){
 tex_t released={.mem=(void *)(uintptr_t)100,.serial=7,.selector=1,.L.size=64};
 for(int g=0;g<4;g++)grp[g].pub.head=grp[g].gnext=-1;
 for(int id=0;id<6;id++){
  tslots[id]=(TexSlot){.t=released,.ce=(void *)(uintptr_t)1,.sample_owned=1,.khash=3};
  slot_next[id]=-1;slot_group[id]=-1;
 }
 // Two view groups can share one texture key; one group is removed completely while the next retains a newer owner.
 grp[0].pub.head=0;grp[0].gnext=1;grp[1].pub.head=2;grp[1].gnext=-1;bucket_ghead[0]=0;
 slot_next[0]=1;slot_group[0]=slot_group[1]=0;
 slot_next[2]=3;slot_group[2]=slot_group[3]=1;tslots[3].t.serial=8;
 bucket_head[0]=4;slot_next[4]=5;tslots[5].t.serial=8;
 vk_texture_release(&released);
 assert(finalizations==1&&bumps==1&&n_fin_own==4&&n_fin_other==2&&!n_fin_other_hm&&retired_registrations==4);
 for(int id=0;id<6;id++){
  int own=id!=3&&id!=5;
  assert(tslots[id].finalized==own);
  if(own)assert(!tslots[id].ce&&!tslots[id].sample_owned);
 }
 assert(slot_group[0]==2&&slot_group[1]==2&&slot_group[2]==3&&slot_group[3]==1);
 assert(slot_group[4]<0&&slot_group[5]<0&&grp[1].pub.head==3&&slot_next[3]==-1);
 assert(grp[2].pub.head==0&&slot_next[0]==1&&slot_next[1]==-1&&grp[3].pub.head==2);
 vk_texture_release(&released);assert(bumps==1&&n_fin_own==4);
 // A later object's release joins an already-existing finalized group while that list is being traversed.
 released.serial=expected_serial=8;vk_texture_release(&released);
 assert(bumps==2&&n_fin_own==6&&n_fin_other==4&&slot_group[3]==3&&tslots[5].finalized);
 assert(grp[3].pub.head==2&&slot_next[2]==3&&slot_next[3]==-1);
 vk_texture_release(&released);assert(bumps==2&&n_fin_own==6);
 entry_present=1;vk_texture_release(&released);assert(entry.released&&bumps==3);
 RcInterval extent={{5,10,64}};TextureSourceMapping mapping={1,&extent};
 entry=(Tex){.source_canonical=1,.t={.serial=9},.source_mapping=&mapping};
 vk_texture_release(&released);assert(!entry.released && !mapped_finalizations && bumps==3);
 released.serial=expected_serial=9;vk_texture_release(&released);
 assert(entry.released && mapped_finalizations==1 && bumps==4);
 puts("texture finalization: serial-specific group rekey, recycled links, grouped and individual traversal passed");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="group-finalize-", dir=ROOT / "build/test") as tmp:
    path = Path(tmp) / "check.c"
    path.write_text(fixture)
    exe = path.with_suffix(".exe")
    subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", str(path), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

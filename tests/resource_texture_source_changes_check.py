"""Exercise the production source-change adapter with real registry history."""
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
                 ("source_dirty_slot", "vk_texture_source_dirty_slots_locked", "vk_texture_schedule_dirty_slots_locked"))
fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#define rc_registry_runtime unused_registry_runtime
#include "resource_core_registry.c"
#undef rc_registry_runtime
static RcRegistry *registry;
static RcRegistry *rc_registry_runtime(void){return registry;}
#define OK(x) assert((x)==RC_REGISTRY_OK)
#define VK_POOL_TEX 8
typedef struct Tex {
 RcRegistrySourceAccess source_access;
 uint64_t source_dirty_visit,schedule_dirty_visit;
 uint64_t source_registration_visit,schedule_registration_visit;
 unsigned source_registrations;
 struct {uint64_t serial;struct {uint64_t size;} L;} t;
 struct {uint64_t addr;} pub;
 struct Tex *next_at;
} Tex;
typedef struct {Tex *ce;uint32_t cgen;int used;Tex *selected;} TexSlot;
static Tex *texture_registration_source(const TexSlot *s){return s->selected;}
static int source_access_matches(const Tex *e,RcRegistryHandle handle){
 return e->source_access.owner.handle.id==handle.id&&e->source_access.owner.handle.index==handle.index;
}
static void texture_registration_source_dirty(Tex *e,void (*visit)(int,void *),void *arg){
 for(unsigned id=0;id<12;id++)if(e->source_registrations>>id&1)visit((int)id,arg);
}
static Tex objects[4];static TexSlot tslots[12];static int ntslots=12;
static uint32_t agen[4]={1,1,1,1},slot_info[12],*sinfo=slot_info;
static int32_t next_slots[12],*slot_next=next_slots,bucket_head[4],bucket_ghead[4];
static struct {int32_t gnext;struct {int32_t head;} pub;} grp[2];
static int by_addr;
static uint32_t agen_idx(uint64_t addr){return (uint32_t)(addr%4);}
static Tex *nvn_hm_get(void *map,const uint64_t *addr){
 (void)map;for(unsigned i=0;i<4;i++)if(objects[i].pub.addr==*addr)return &objects[i];return NULL;
}
static int inject_change;
static uint32_t changes_with_injection(RcRegistry *r,uint64_t *cursor,RcRegistryChange *out,uint32_t n,int *overflow){
 uint32_t got=rc_registry_changes(r,cursor,out,n,overflow);
 if(inject_change){inject_change=0;OK(rc_registry_cpu_changed(r,(RcSourceRange){1,220,1}));}
 return got;
}
#define rc_registry_changes changes_with_injection
''' + body + r'''
#undef rc_registry_changes
static unsigned visits[12];
static void visit(int id,void *arg){assert(arg==visits);assert(id>=0&&id<12);visits[id]++;}
static void clear_visits(void){memset(visits,0,sizeof visits);}
static void expected(unsigned bits){for(unsigned id=0;id<12;id++)assert(visits[id]==((bits>>id)&1));}
static void attach(unsigned slot,Tex *e){tslots[slot]=(TexSlot){.ce=e,.cgen=1,.used=1};slot_info[slot]=agen_idx(e->pub.addr);}
int main(void){
 registry=rc_registry_create((RcRegistryConfig){16,8,2});assert(registry);
 RcRegistrySource leases[3]={0};
 for(unsigned i=0;i<3;i++){
  uint64_t addr=i==2?200:100,bytes=i==1?32:64;
  OK(rc_registry_source_acquire(registry,7+i,(RcSourceRange){1,addr,bytes},&leases[i]));
  objects[i]=(Tex){.source_access={.owner=leases[i]},.t={7+i,{bytes}},.pub={addr}};
 }
 objects[0].next_at=&objects[1];
 for(unsigned i=0;i<12;i++)next_slots[i]=-1;
 for(unsigned i=0;i<4;i++)bucket_head[i]=bucket_ghead[i]=-1;
 attach(0,&objects[0]);attach(1,&objects[0]);attach(2,&objects[1]);attach(3,&objects[2]);
 attach(4,&objects[0]);tslots[4].cgen=0; // Stale address generation is handled by slot-token history.
 attach(8,&objects[0]); // Texture namespace can exceed the native descriptor table.
 bucket_head[0]=0;next_slots[0]=2;next_slots[2]=3;next_slots[3]=8;
 bucket_ghead[0]=0;grp[0].gnext=-1;grp[0].pub.head=1;next_slots[1]=4;
 uint64_t cursor=rc_registry_change_position(registry);
 uint64_t schedule_cursor=rc_registry_schedule_change_position(registry);
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,10}));
 RcRegistryReadObservation old,fresh;
 OK(rc_registry_observe_at(registry,leases[0].handle,1,(RcRegistryPosition){1,2},&old));
 uint64_t ordered=rc_registry_ordered_epoch(registry),legacy=rc_registry_authority_epoch(registry);
 RcRegistryWrite future;
 OK(rc_registry_write_begin_at(registry,(RcSourceRange){1,150,1},RC_REGISTRY_BUFFER,(RcRegistryPosition){1,20},&future));
 assert(rc_registry_read_observation_current(registry,&old));
 OK(rc_registry_read_revalidate(registry,&old,1,(RcRegistryPosition){1,2},&fresh));
 assert(fresh.sealed_through.ordinal==10&&rc_registry_ordered_epoch(registry)==ordered);
 assert(rc_registry_authority_epoch(registry)!=legacy);
 uint64_t event_cursor=cursor;RcRegistryChange event;int overflow=0;
 assert(!rc_registry_changes(registry,&event_cursor,&event,1,&overflow)&&!overflow);
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(0);
 assert(vk_texture_schedule_dirty_slots_locked(&schedule_cursor,visit,visits));expected(7);
 clear_visits();OK(rc_registry_write_cancel(registry,future));
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(0);
 assert(vk_texture_schedule_dirty_slots_locked(&schedule_cursor,visit,visits));expected(0);
 assert(rc_registry_ordered_epoch(registry)==ordered);
 OK(rc_registry_declare_through(registry,(RcRegistryPosition){1,30}));
 assert(rc_registry_read_revalidate(registry,&old,1,(RcRegistryPosition){1,12},&fresh)==RC_REGISTRY_STALE);
 // An unpositioned writer can affect the captured prefix and must still notify every matching association.
 OK(rc_registry_write_begin(registry,(RcSourceRange){1,150,1},RC_REGISTRY_BUFFER,&future));
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(7);
 clear_visits();OK(rc_registry_write_cancel(registry,future));
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(0);
 // Multiple events deduplicate one Tex while distinct overlapping source registrations notify both associations.
 clear_visits();OK(rc_registry_cpu_changed(registry,(RcSourceRange){1,110,1}));
 OK(rc_registry_cpu_changed(registry,(RcSourceRange){1,115,1}));
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(7);
 // A new event arriving during the read is left after the captured cursor end, even when registry history is live.
 clear_visits();OK(rc_registry_cpu_changed(registry,(RcSourceRange){1,150,1}));
 uint64_t end=rc_registry_change_position(registry),before=rc_registry_authority_epoch(registry);inject_change=1;
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(7);
 assert(cursor==end&&rc_registry_change_position(registry)==end+1&&rc_registry_authority_epoch(registry)!=before);
 clear_visits();assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(8);
 // A new lease can retain the same slot token/image while an outstanding pool receipt names the old lease.
 // The old event must invalidate that logical association; handle mismatch is not a reason to skip it.
 clear_visits();RcRegistryHandle retired=leases[0].handle;before=rc_registry_authority_epoch(registry);
 OK(rc_registry_source_release(registry,&leases[0]));
 assert(rc_registry_authority_epoch(registry)!=before);
 OK(rc_registry_source_acquire(registry,7,(RcSourceRange){1,100,64},&leases[0]));
 assert(leases[0].handle.id!=retired.id);objects[0].source_access.owner.handle=leases[0].handle;
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(7);
 clear_visits();
 OK(rc_registry_cpu_changed(registry,(RcSourceRange){1,150,1}));
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(7);
 // An old event can arrive after entry_of has replaced this Tex's guest object and range at the same address.
 // Its outstanding descriptor receipt still needs invalidation even though serial, size and handle differ.
 clear_visits();OK(rc_registry_source_release(registry,&leases[0]));
 OK(rc_registry_source_acquire(registry,17,(RcSourceRange){1,100,48},&leases[0]));
 objects[0].source_access.owner.handle=leases[0].handle;objects[0].t.serial=17;objects[0].t.L.size=48;
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(7);
 // Canonical offsets are unrelated to guest addresses. One overlapping range can route independent associations.
 RcRegistrySourceAccess canonical[2]={0};RcSourceRange identity={27,9,64};
 OK(rc_registry_source_access_acquire_associated(registry,19,identity,100,&canonical[0]));
 OK(rc_registry_source_access_acquire_associated(registry,19,identity,200,&canonical[1]));
 assert(canonical[0].owner.handle.id!=canonical[1].owner.handle.id);
 clear_visits();OK(rc_registry_cpu_changed(registry,identity));
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(15);
 clear_visits();OK(rc_registry_source_access_release(registry,&canonical[0]));
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(7);
 RcRegistryRecord surviving;OK(rc_registry_get(registry,canonical[1].owner.handle,&surviving));assert(surviving.current);
 clear_visits();OK(rc_registry_cpu_changed(registry,identity));
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(8);
 // Selected secondary-pool readers bypass the address-bucket cache and use the exact source handle.
 objects[2].source_access=canonical[1];objects[2].t.serial=19;objects[2].source_registrations=1u<<8;
 tslots[3].selected=&objects[2];tslots[8].selected=&objects[2];
 clear_visits();OK(rc_registry_cpu_changed(registry,(RcSourceRange){1,220,1}));
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(0);
 clear_visits();OK(rc_registry_cpu_changed(registry,identity));OK(rc_registry_cpu_changed(registry,identity));
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(1u<<8);
 schedule_cursor=rc_registry_schedule_change_position(registry);
 clear_visits();OK(rc_registry_write_begin_at(registry,identity,RC_REGISTRY_BUFFER,(RcRegistryPosition){1,40},&future));
 assert(vk_texture_schedule_dirty_slots_locked(&schedule_cursor,visit,visits));expected(1u<<8);
 OK(rc_registry_write_cancel(registry,future));
 // History loss is explicit, and its cursor cannot accidentally acknowledge events newer than entry.
 clear_visits();for(unsigned i=0;i<20;i++)OK(rc_registry_cpu_changed(registry,(RcSourceRange){1,140,1}));
 end=rc_registry_change_position(registry);assert(!vk_texture_source_dirty_slots_locked(&cursor,visit,visits));
 assert(cursor==end);expected(0);
 assert(vk_texture_source_dirty_slots_locked(&cursor,visit,visits));expected(0);
 cursor=end+1;assert(!vk_texture_source_dirty_slots_locked(&cursor,visit,visits)&&cursor==end);
 // Poison invalidates even an empty drain; a matching cursor never grants usable authority.
 rc_registry_fail_closed(registry);assert(!vk_texture_source_dirty_slots_locked(&cursor,visit,visits));
 rc_registry_destroy(registry);
 puts("source changes: generic revisions, same-address identity replacement, bounded drain races and overflow passed");
}
'''
replacement_start = source.index("    if (e->t.serial != t->serial || e->t.L.size != t->L.size) {")
replacement_end = source.index("    touch(e);", replacement_start)
replacement = source[replacement_start:replacement_end]
identity_fixture = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#include "resource_core_registry.c"
#define OK(x) assert((x)==RC_REGISTRY_OK)
static RcRegistry *registry;
typedef struct {uint64_t serial;struct {uint64_t size;} L;} tex_t;
typedef struct {tex_t t;struct {uint64_t addr;} key;RcRegistrySourceAccess source_access;
 void *source_spans;unsigned source_count;} Tex;
static uint64_t n_bumps,bump_count[4],bsig[4],slot_info[2]={1u<<16,2u<<16},*sinfo=slot_info;
static uint32_t tex_gen=1,slots_gen=1,bepoch[4],agen[4]={1,1,1,1},agen_top[4]={1,1,1,1};
static struct {uint64_t sig;uint32_t b,epoch,gen,valid;} gs[8192];
static uint32_t agen_idx(uint64_t addr){return (uint32_t)(addr%4);}
#define DIRTY_BUCKET (1u<<31)
static unsigned dirty_count,discard_count,release_count;
static void dirty_push(uint32_t event){assert(event==(DIRTY_BUCKET|0));dirty_count++;}
static void content_source_discard(Tex *e){(void)e;assert(dirty_count>discard_count);discard_count++;}
static void source_registration_release(Tex *e){
 assert(dirty_count>release_count);OK(rc_registry_source_release(registry,&e->source_access.owner));release_count++;
}
''' + "\n".join(function(source, name) for name in ("gs_at", "tex_gen_bump")) + function(
    source[source.index("uint64_t vk_texture_slot_token("):], "vk_texture_slot_token") + r'''
static void replace_source(Tex *e,const tex_t *t){
''' + replacement + r'''
}
int main(void){
 registry=rc_registry_create((RcRegistryConfig){8,8,2});assert(registry);
 Tex e={.t={7,{64}},.key={100}};RcRegistrySource other={0};
 OK(rc_registry_source_acquire(registry,7,(RcSourceRange){1,100,64},&e.source_access.owner));
 OK(rc_registry_source_acquire(registry,7,(RcSourceRange){1,100,64},&other));
 assert(e.source_access.owner.handle.id==other.handle.id);
 e.source_spans=malloc(8);assert(e.source_spans);e.source_count=1;
 uint64_t epoch=rc_registry_authority_epoch(registry),head=rc_registry_change_position(registry);
 uint64_t token0=vk_texture_slot_token(0),token1=vk_texture_slot_token(1);
 tex_t next={17,{48}};replace_source(&e,&next);
 assert(rc_registry_authority_epoch(registry)==epoch&&rc_registry_change_position(registry)==head);
 assert(vk_texture_slot_token(0)!=token0&&vk_texture_slot_token(1)!=token1);
 assert((vk_texture_slot_token(0)>>32)==(token0>>32));
 assert(e.t.serial==17&&e.t.L.size==48&&!e.source_spans&&!e.source_count&&!e.source_access.owner.id);
 assert(dirty_count==1&&discard_count==1&&release_count==1);
 // Identical source identity does not generate invalidation work.
 token0=vk_texture_slot_token(0);replace_source(&e,&next);
 assert(vk_texture_slot_token(0)==token0&&dirty_count==1);
 // A size-only replacement is independently covered, also with a remaining owner and no source event.
 RcRegistrySource other_new={0};
 OK(rc_registry_source_acquire(registry,17,(RcSourceRange){1,100,48},&e.source_access.owner));
 OK(rc_registry_source_acquire(registry,17,(RcSourceRange){1,100,48},&other_new));
 epoch=rc_registry_authority_epoch(registry);head=rc_registry_change_position(registry);next.L.size=32;
 replace_source(&e,&next);
 assert(rc_registry_authority_epoch(registry)==epoch&&rc_registry_change_position(registry)==head);
 assert(vk_texture_slot_token(0)!=token0&&dirty_count==2&&release_count==2);
 assert(n_bumps==2&&bump_count[0]==2&&slots_gen==3&&gs[gs_at(0)].gen==agen[0]);
 rc_registry_destroy(registry);
 puts("source replacement: actual entry_of branch invalidates descriptor tokens without a shared-lease event");
}
'''
env = dict(os.environ, PATH=str(CC.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="texture-source-changes-", dir=ROOT / "build/test") as tmp:
    for name, text in (("changes", fixture), ("identity", identity_fixture)):
        path = Path(tmp) / (name + ".c")
        path.write_text(text)
        exe = path.with_suffix(".exe")
        subprocess.run([str(CC), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-Wno-unused-function", "-fsanitize=undefined", "-fsanitize-trap=all",
                        "-I", str(ROOT / "runtime"), str(path), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)

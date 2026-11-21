"""Exercise production direct-group association ownership and installation transitions without Vulkan."""
from pathlib import Path
import os, re, subprocess, sys, tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bind_recipe_check import function
ROOT = Path(__file__).resolve().parents[1]
CC = Path(os.environ['CC'])
source = Path(os.getenv('POOL_ASSOCIATION_DRAW_SOURCE', ROOT/'runtime/vk_exec_draw.c')).read_text()
def extract(name):
    m = re.search(r'^static (?:void|int) '+name+r'\([^;{]*\)\s*\{', source, re.M)
    assert m, name
    return function(source[m.start():], name)

prefix = r'''
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define VK_POOL_TEX 8
#define VK_POOL_KINDS 2
#define VK_NULL_HANDLE 0
#define RT_TRACE_PERF 1
static unsigned rt_trace;
typedef uint64_t VkImage;
typedef uint64_t VkImageView;
typedef struct {uint64_t registration;int native_slot;uint8_t canonical,refused;} VkTextureReadKey;
typedef struct {int unused;} VkExec;
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
typedef struct {uint64_t incarnation;int head,n;uint32_t association;} VkSlotGroup;
static VkSlotGroup groups[4];
static int membership[VK_POOL_TEX],next_slot[VK_POOL_TEX],finalized[VK_POOL_TEX];
static int group_finalized[4];
static uint32_t registrations[VK_POOL_TEX],address_generation[4]={1,1,1,1};
static uint32_t holds[32];static unsigned hold_acquires,hold_releases;
static VkImage selected_image[4]={100,200,300,400};
static VkImageView selected_view[4][2]={{1000,1001},{2000,2001},{3000,3001},{4000,4001}};
static uint32_t selected_entry[4]={1,2,3,4};
static int direct[4]={1,1,1,1},unstable,fail_fill;
static uint64_t generation=1;
static unsigned resolutions,native_writes,queued_writes;
static struct {
 VkImage image[VK_POOL_TEX*VK_POOL_KINDS];VkImageView view[VK_POOL_TEX*VK_POOL_KINDS];
 uint64_t tok[VK_POOL_TEX],registration_scope,registration_pool;
 struct {uint32_t stamp,fmask,selected_kinds;VkImage selected_image;VkTextureReadKey selected;} ps[VK_POOL_TEX];
} PL;
static int vk_texture_pool_primary(uint64_t owner){return owner==0;}
static uint64_t pool_diag_image_changes[VK_POOL_TEX];
static uint32_t pool_defer[VK_POOL_TEX],slot_holds[VK_POOL_TEX];
static int pool_retry;
static void vk_pool_retry_remove(int *retry,uint32_t id){assert(retry==&pool_retry);(void)id;}
static void pool_unres(uint32_t id,int got){(void)id;assert(got);}
static VkSlotGroup *vk_texture_group_locked(int group){assert(group>=0&&group<4);return &groups[group];}
static int vk_texture_slot_group_locked(int id){return membership[id];}
static int vk_texture_group_next_locked(int id){return next_slot[id];}
static uint64_t vk_texture_slot_token(int id){
 int g=membership[id];return ((uint64_t)registrations[id]<<32)|(g>=0?address_generation[g]:1);
}
static int vk_texture_group_member_locked(int group,uint64_t incarnation,int id){
 return id>=0&&id<VK_POOL_TEX&&finalized[id]==group_finalized[group]&&membership[id]==group&&groups[group].incarnation==incarnation;
}
static int vk_texture_group_direct_locked(int group,uint64_t incarnation,int id,VkImage expected,uint32_t *entry){
 *entry=0;if(!vk_texture_group_member_locked(group,incarnation,id)||!direct[group]||unstable||expected!=selected_image[group])return 0;
 *entry=selected_entry[group];return 1;
}
static void vk_texture_pool_hold_locked(uint32_t old,uint32_t next){
 if(next){assert(next<32);holds[next]++;hold_acquires++;}
 if(old){assert(old<32&&holds[old]);holds[old]--;hold_releases++;}
}
static void pool_hold(uint32_t id,uint32_t entry){
 if(slot_holds[id]!=entry){vk_texture_pool_hold_locked(slot_holds[id],entry);slot_holds[id]=entry;}
}
static void pool_write(uint32_t kind,uint32_t id,VkImageView view,uint64_t sampler){
 assert(kind<VK_POOL_KINDS&&id<VK_POOL_TEX&&PL.view[kind*VK_POOL_TEX+id]==view&&!sampler);queued_writes++;
}
static void native_flush(void){native_writes+=queued_writes;queued_writes=0;}
typedef struct {uint64_t generation,authority,token,legacy_authority;uint32_t kinds,source,native,source_epoch;RcRegistryPosition origin,next;} PoolMemberReceipt;
static PoolMemberReceipt pool_member[VK_POOL_TEX];
#include "vk_pool_frontier.h"
static VkPoolFrontier pool_frontier;
static struct {int count;} pool_frontier_state;
static uint32_t pool_source_members[(VK_POOL_TEX+31)/32],pool_legacy_members[(VK_POOL_TEX+31)/32];
static uint32_t pool_legacy_count,pool_legacy_total;
static uint64_t pool_member_generation(void){return generation;}
'''
start = source.index('typedef struct {\n    uint64_t incarnation,version,generation;')
end = source.index('static void pool_member_invalidate(', start)
fixture = prefix + source[start:end]
fixture += '\n'.join(extract(n) for n in ('pool_member_invalidate','pool_member_schedule','pool_set_image'))
fixture += r'''
static int pool_refill(VkExec *ex,uint32_t id,uint32_t kinds,uint64_t token,int spent){
 (void)ex;(void)token;(void)spent;resolutions++;pool_member_invalidate(id);
 if(fail_fill)return 0;
 int group=membership[id];assert(group>=0);
 for(unsigned k=0;k<VK_POOL_KINDS;k++)if(kinds>>k&1)pool_set_image(k,id,selected_image[group],selected_view[group][k]);
 pool_hold(id,selected_entry[group]);return 1;
}
'''
fixture += '\n'.join(extract(n) for n in ('pool_association_current','pool_association_member',
 'pool_association_install','pool_association_install_existing','pool_association_refresh'))
fixture += r'''
static void register_slot(int id,int group){
 pool_association_detach((uint32_t)id);
 int old=membership[id];
 if(old>=0){int *link=&groups[old].head;while(*link!=id){assert(*link>=0);link=&next_slot[*link];}*link=next_slot[id];groups[old].n--;}
 membership[id]=group;registrations[id]++;finalized[id]=0;next_slot[id]=-1;
 if(group>=0){int *link=&groups[group].head;while(*link>=0)link=&next_slot[*link];*link=id;groups[group].n++;}
}
static void prefix_count(int count){
 pool_frontier_state.count=count;vk_pool_frontier_reset(&pool_frontier);pool_association_recount(count);
 for(int id=0;id<count;id++){pool_association_member((uint32_t)id,1);pool_member_schedule((uint32_t)id);}
}
// The texture side rekeys without editing pool ownership or its registration serial. Old installed sets still own
// their old physical images; association checks must notice that the selector group changed before reusing them.
static void finalize_registration(int id,int group){
 int old=membership[id];int *link=&groups[old].head;
 while(*link!=id){assert(*link>=0);link=&next_slot[*link];}*link=next_slot[id];groups[old].n--;
 finalized[id]=1;membership[id]=group;next_slot[id]=-1;group_finalized[group]=1;
 link=&groups[group].head;while(*link>=0)link=&next_slot[*link];*link=id;groups[group].n++;
 address_generation[old]++;address_generation[group]++;
}
static void assert_installed(int id,int group,unsigned kinds){
 uint32_t owner=pool_association_slot[id].owner;assert(owner&&pool_association_member((uint32_t)id,kinds));
 for(unsigned k=0;k<VK_POOL_KINDS;k++)if(kinds>>k&1){
  assert(PL.image[k*VK_POOL_TEX+id]==selected_image[group]);assert(PL.view[k*VK_POOL_TEX+id]==selected_view[group][k]);
 }
 assert(pool_association_slot[id].registration==registrations[id]);
}
int main(void){
 VkExec ex={0};for(int i=0;i<VK_POOL_TEX;i++)membership[i]=next_slot[i]=-1;
 for(int g=0;g<4;g++){groups[g].head=-1;groups[g].incarnation=(uint64_t)g+1;}
 pool_frontier_state.count=4;register_slot(0,0);register_slot(1,0);register_slot(2,0);
 assert(pool_association_refresh(&ex,0,3,4,0));uint32_t owner=groups[0].association;
 assert(owner&&resolutions==1&&holds[1]==1&&!slot_holds[0]&&!slot_holds[1]&&!slot_holds[2]);
 for(int id=0;id<3;id++)assert_installed(id,0,3);native_flush();
 unsigned writes=native_writes;uint64_t version=pool_associations[owner-1]->version;
 for(unsigned i=0;i<100;i++)assert(pool_association_refresh(&ex,0,3,4,0));
 assert(resolutions==1&&native_writes==writes&&!queued_writes);
 // A selector generation transition resolves once and renews every alias without rewriting their descriptors.
 address_generation[0]++;assert(pool_association_refresh(&ex,0,3,4,0));native_flush();
 assert(resolutions==2&&native_writes==writes&&pool_associations[owner-1]->version==version&&holds[1]==1);
 // Physical replacement updates every reachable alias and drops coverage of inactive kinds.
 selected_image[0]=101;selected_view[0][0]=1010;selected_view[0][1]=1011;address_generation[0]++;
 assert(pool_association_refresh(&ex,0,1,4,0));assert(pool_associations[owner-1]->kinds==1);
 for(int id=0;id<3;id++)assert_installed(id,0,1);assert(!pool_association_current(owner,2));
 assert(pool_association_refresh(&ex,0,3,4,0));for(int id=0;id<3;id++)assert_installed(id,0,3);native_flush();
 // A new alias installs exactly one descriptor pair from the group certificate without another resolution.
 register_slot(3,0);unsigned before=resolutions;assert(pool_association_install_existing(3,3));
 assert(resolutions==before);assert_installed(3,0,3);native_flush();
 // Shrink, change, grow: inactive linked aliases retain old native contents and must be scheduled for install.
 prefix_count(2);selected_image[0]=102;selected_view[0][0]=1020;address_generation[0]++;
 assert(pool_association_refresh(&ex,0,1,2,0));assert(PL.image[2]==101);native_flush();
 prefix_count(4);assert(pool_association_frontier.count==1&&pool_associations[owner-1]->install_pending);
 assert(pool_association_refresh(&ex,0,1,4,0));for(int id=0;id<4;id++)assert_installed(id,0,1);native_flush();
 assert(!pool_association_frontier.count);
 // The last reachable alias departing releases the group's activity hold even if high aliases remain linked.
 prefix_count(1);register_slot(0,-1);assert(!groups[0].association&&!holds[1]);
 assert(!pool_association_slot[1].owner&&!pool_association_slot[2].owner&&!pool_association_slot[3].owner);
 // Reused group index/incarnation cannot certify an old association or keep its entry hold.
 prefix_count(4);assert(pool_association_refresh(&ex,0,1,4,0));owner=groups[0].association;
 groups[0].incarnation++;groups[0].association=0;
 assert(pool_association_member(1,1)<0);assert(pool_association_member(2,1)<0);assert(pool_association_member(3,1)<0);
 assert(!pool_associations[owner-1]&&!holds[1]);
 // A changed selector cannot join its old key. Source-dependent and unstable selections dissolve to individual proofs.
 finalized[2]=1;assert(pool_association_refresh(&ex,0,1,4,0));assert(!pool_association_slot[2].owner);
 owner=groups[0].association;direct[0]=0;address_generation[0]++;
 assert(!pool_association_refresh(&ex,0,1,4,0)&&!groups[0].association&&!pool_associations[owner-1]);
 direct[0]=1;unstable=1;assert(!pool_association_refresh(&ex,0,1,4,0));unstable=0;
 finalized[2]=0;assert(pool_association_refresh(&ex,0,1,4,0));
 // Serial-specific finalization rekeys active and inactive registrations. Only the live finalized DIRECT
 // selection can form a new owner; an inactive alias cannot claim installation merely by joining that group.
 uint32_t serial=registrations[1];prefix_count(2);
 finalize_registration(1,1);finalize_registration(2,1);
 assert(registrations[1]==serial&&pool_association_member(1,1)<0);
 assert(!pool_association_refresh(&ex,1,1,2,0)&&!groups[1].association);
 prefix_count(4);assert(pool_association_refresh(&ex,1,1,4,0));
 assert_installed(1,1,1);assert_installed(2,1,1);assert(membership[3]==0&&!finalized[3]);
 unsigned resolved=resolutions;for(int i=0;i<100;i++)assert(pool_association_member(1,1));assert(resolutions==resolved);
 // Replacing the selected live image updates both aliases, while retirement/source dependence dissolves ownership.
 selected_image[1]=201;selected_view[1][0]=2010;selected_entry[1]=5;address_generation[1]++;
 assert(pool_association_refresh(&ex,1,1,4,0));assert_installed(1,1,1);assert_installed(2,1,1);
 direct[1]=0;address_generation[1]++;assert(!pool_association_refresh(&ex,1,1,4,0)&&!groups[1].association);
 assert(holds[5]==(unsigned)(slot_holds[1]==5)+(unsigned)(slot_holds[2]==5)); // Only individual fallback activity remains.
 direct[1]=1;assert(pool_association_refresh(&ex,1,1,4,0));
 register_slot(1,0);assert(!pool_association_slot[1].owner&&!finalized[1]);
 // Ordinary ordered selection owns this member's physical binding independently of an optional alias group.
 assert(pool_association_refresh(&ex,0,1,4,0));
 pool_association_detach(1);
 PL.ps[1].selected.registration=registrations[1];
 PL.ps[1].selected_kinds=1;
 pool_set_image(0,1,444,4440);
 assert(!pool_association_install_existing(1,1));
 selected_image[0]=105;selected_view[0][0]=1050;address_generation[0]++;
 assert(pool_association_refresh(&ex,0,1,4,0));
 assert(PL.image[1]==444&&PL.view[1]==4440&&!pool_association_slot[1].owner);
 assert_installed(3,0,1);
 pool_association_clear();assert(!groups[0].association);for(int id=0;id<VK_POOL_TEX;id++)assert(!pool_association_slot[id].owner);
 for(int id=0;id<VK_POOL_TEX;id++)pool_hold((uint32_t)id,0);
 for(unsigned i=0;i<32;i++)assert(!holds[i]);assert(hold_acquires==hold_releases);
 assert(!pool_legacy_count&&!pool_legacy_total&&pool_association_created&&pool_association_renewed);
 assert(pool_association_changed&&pool_association_installed);
 puts("pool associations: exact installation, unchanged group renewal, kind replacement, prefix growth, membership reuse and ownership release passed");
}
'''
env=dict(os.environ,PATH=str(CC.parent)+os.pathsep+os.environ.get('PATH',''))
with tempfile.TemporaryDirectory(prefix='pool-association-',dir=ROOT/'build/test') as tmp:
    src=Path(tmp)/'check.c';src.write_text(fixture);exe=src.with_suffix('.exe')
    subprocess.run([str(CC),'-std=gnu11','-O1','-Wall','-Wextra','-Werror','-Wno-unused-function','-Wno-unused-variable',
                    '-Wno-undefined-internal','-fsanitize=undefined','-fsanitize-trap=all','-I',str(ROOT/'runtime'),str(src),'-o',str(exe)],env=env,check=True)
    subprocess.run([str(exe)],env=env,check=True)

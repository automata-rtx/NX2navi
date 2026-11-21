"""Exercise production pool member receipts and boundary-only refresh without Vulkan."""
from pathlib import Path
import os, re, subprocess, tempfile, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from bind_recipe_check import function
ROOT=Path(__file__).resolve().parents[1]
CC=Path(os.environ["CC"])
source=Path(os.getenv("POOL_PUBLICATION_DRAW_SOURCE", ROOT/"runtime/vk_exec_draw.c")).read_text()
def extract(name):
    m=re.search(r"^(?:static )?(?:void|int|unsigned|uint32_t|uint64_t|VkImageView) "+name+r"\([^;{]*\)\s*\{",source,re.M)
    assert m,name
    return function(source[m.start():],name)
prefix=r'''
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "replay_work_trace.h"
#define VK_POOL_TEX 4
#define VK_POOL_KINDS 2
#define RT_TRACE_PERF 1
static unsigned rt_trace;
static uint32_t pool_incompatible_comparison_kinds(uint32_t id,uint32_t kinds){(void)id;(void)kinds;return 0;}
static int diagnostic_groups[4]={0,0,-1,-1};
static int vk_texture_slot_group_locked(int id){return id>=0&&id<4?diagnostic_groups[id]:-1;}
typedef uintptr_t VkImage;
typedef uintptr_t VkImageView;
typedef struct {VkImageView view;} ViewKey;
static struct {int klen;} view_of;
static void *nvn_hm_get(void *map,const ViewKey *key){assert(map==&view_of);return (void *)key->view;}
typedef struct {VkImage feedback;} VkExec;
typedef struct {uint64_t incarnation;uint32_t association;int head;} VkSlotGroup;
static VkSlotGroup groups[2];
static VkSlotGroup *vk_texture_group_locked(int group){assert(group>=0&&group<2);return &groups[group];}
static int vk_texture_group_next_locked(int id){(void)id;return -1;}
static void vk_texture_pool_hold_locked(uint32_t old,uint32_t next){(void)old;(void)next;}
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
typedef struct {uint64_t registration;int native_slot,canonical,unbacked,projected;} VkTextureReadKey;
typedef struct {VkTextureReadKey selected;uint32_t selected_kinds;} PoolSlot;
typedef struct {int proved;} VkTextureAliasRead;
typedef struct {uint64_t mapping;} RcInterval;
typedef int VkTextureAliasPhase;
typedef struct {unsigned gate;} RcRegistryNativeImageFailure;
typedef struct {VkTextureAliasPhase phase; int status; RcInterval receiver; uint64_t registration,material_serial;
    VkImage image; RcRegistryNativeImageFailure coverage; const char *layout_gate; } VkTextureAliasFailure;
enum { VK_TEXTURE_ALIAS_SAMPLE_FORCE = 1, VK_TEXTURE_ALIAS_SAMPLE_LINEAR_RGBA8_VIEW = 2,
       VK_TEXTURE_ALIAS_SAMPLE_ONLY = 4 };
typedef enum { VK_TEXTURE_SAMPLE_READY, VK_TEXTURE_SAMPLE_PREPARE, VK_TEXTURE_SAMPLE_FALLBACK, VK_TEXTURE_SAMPLE_UNAVAILABLE }
    VkTextureSampleStatus;
static void vk_texture_alias_sample_report_layout_locked(int logical,const VkTextureAliasFailure *failure) {
    (void)logical;(void)failure;
}
enum {VK_TEXTURE_ALIAS_ORDER,VK_TEXTURE_ALIAS_MAPPING,VK_TEXTURE_ALIAS_LAYOUT,VK_TEXTURE_ALIAS_COVERAGE,
      VK_TEXTURE_ALIAS_OWNER,VK_TEXTURE_ALIAS_CAPACITY};
enum {RC_REGISTRY_UNSUPPORTED,RC_REGISTRY_CAPACITY,RC_REGISTRY_BLOCKED};
enum {VK_EXEC_PREPARATION_ORDER,VK_EXEC_PREPARATION_MAPPING,VK_EXEC_PREPARATION_ATTACHMENT,
      VK_EXEC_PREPARATION_PLACEMENT,VK_EXEC_PREPARATION_PRESSURE};
enum {VK_EXEC_PREPARATION_POOL_IMAGE=4,VK_EXEC_GRAPHICS_UNAVAILABLE};
typedef int VkExecPreparationReason;
typedef struct {int owner,result,detail;VkExecPreparationReason reason;uint32_t role,slot;RcInterval range;
    const char *stage;uint64_t registration,material_serial;VkImage image;RcRegistryNativeImageFailure coverage;
    struct {VkTextureAliasFailure alias;uint64_t registration;RcRegistryPosition read;} sampled;} VkExecPreparationFailure;
typedef struct {VkImage images[8],native[8];VkTextureAliasRead *aliases[8];} TextureImagePublication;
typedef struct {
    struct {uint32_t slot,kinds;} entries[VK_POOL_TEX];
    uint32_t count;
    uint8_t domain;
} VkExecPoolIndices;
typedef struct PoolAliases PoolAliases;
typedef struct {uint32_t kinds;const VkExecPoolIndices *indices;const VkExec *ex;PoolAliases *aliases;
    RcRegistryPosition read;} GraphicsPoolRestore;
static GraphicsPoolRestore *graphics_pool_restore;
static int pool_domain_comparison_supported(GraphicsPoolRestore *request,uint32_t id,uint32_t k,const ViewKey *key,uint64_t reg) {
 (void)request;(void)id;(void)k;(void)key;(void)reg;return 1;
}
static PoolSlot selected_slots[VK_POOL_TEX];
static struct {VkImage image[8];VkImageView view[8];uint32_t kinds;uint64_t registration_scope,registration_pool;
               int cur,wanted_reconcile,finite_reconcile;uint8_t set_registration_gone[2],registration_gone[VK_POOL_TEX],set_rebuild[2];
               PoolSlot *ps;uint64_t tok[VK_POOL_TEX],set_synced[2],logn,registration_target,registration_publication;
               TextureImagePublication *publication[2];uint32_t publication_updates;}
    PL={.kinds=1,.ps=selected_slots};
enum {POOL_SOURCE_REFUSED=2};
enum { VK_TEXTURE_REGISTRATION_CURRENT=0, VK_TEXTURE_REGISTRATION_EXACT=1, VK_TEXTURE_REGISTRATION_GONE=-2,
           VK_TEXTURE_REGISTRATION_ABSENT=-1 };
enum {VK_TEXTURE_READ_EXACT_DETAIL=8,VK_IMAGE_ASPECT_COLOR_BIT=1};
static int history_slot=-1,history_status;
static unsigned history_queries,token_queries;
static VkImage history_image;
typedef struct {VkImage image;unsigned aspect;} VkTexImage;
typedef struct {VkImageView view;} view_t;
typedef struct {const VkTexImage *image;view_t view;int has_view;uint64_t registration;int projected;} VkTextureSample;
static VkTextureSampleStatus vk_texture_select_sample_locked(int slot,unsigned flags,VkTextureSample *out){
    assert(slot>=0&&slot<VK_POOL_TEX&&flags==VK_TEXTURE_READ_EXACT_DETAIL&&out);
    return VK_TEXTURE_SAMPLE_FALLBACK; // This fixture's resident cases use the legacy GPU carrier below.
}
static VkTexImage gpu_image[VK_POOL_TEX];
static VkImageView gpu_base_view[VK_POOL_TEX];
static int gpu_status[VK_POOL_TEX], installed_receiver[VK_POOL_TEX];
static unsigned gpu_queries, gpu_holds, view_queries, projected_queries;
static int projected_current;
static int in_fbl(const VkExec *ex,VkImage image){return ex&&ex->feedback==image;}
static int vk_texture_pool_gpu_sample_locked(int logical,VkImage expected,VkTextureSample *out){
    gpu_queries++;
    if (gpu_status[logical]<=0) return gpu_status[logical];
    assert(gpu_image[logical].image==expected);
    *out=(VkTextureSample){.image=&gpu_image[logical],.view={gpu_base_view[logical]},.has_view=1};
    return 1;
}
static int vk_texture_projected_sample_receipt_locked(VkTextureReadKey key,VkImage expected,VkTextureSample *out){
    assert(key.projected && key.registration==901 && key.native_slot==1 && expected==gpu_image[1].image);
    projected_queries++;
    if (!projected_current) return 0;
    *out=(VkTextureSample){.image=&gpu_image[1],.view={gpu_base_view[1]},.has_view=1};
    return 1;
}
static int vk_texture_pool_current_view_locked(const TextureImagePublication *set,uint32_t descriptor,
                                               int logical,VkImage expected,VkTexImage *image,view_t *view,int *has_view){
    view_queries++;
    if (!installed_receiver[logical] || gpu_image[logical].image!=expected || set->images[descriptor]!=expected) return 0;
    *image=gpu_image[logical];*view=(view_t){gpu_base_view[logical]};*has_view=1;
    return 1;
}
static VkTextureAliasRead *vk_texture_publication_alias_locked(const TextureImagePublication *p,
    uint32_t at,VkImage expected) {
    return p&&p->images[at]==expected?p->aliases[at]:NULL;
}
static int vk_texture_alias_published_current_locked(int logical,VkTextureAliasRead *alias,RcRegistryPosition read) {
    assert(logical>=0&&logical<VK_POOL_TEX&&read.scene);
    return alias->proved;
}
enum {VK_EXEC_ALIAS_READ_LIMIT=8};
typedef struct {VkTextureAliasRead *read;VkTextureSample sample;VkTexImage image;uint32_t slot,kinds;} PoolAliasMember;
struct PoolAliases {PoolAliasMember members[VK_EXEC_ALIAS_READ_LIMIT];uint32_t count;int refused;
    VkExecPreparationFailure failure;};
typedef int VkResult;
enum {VK_SUCCESS};
static VkResult pool_aliases_append(PoolAliases *aliases,PoolAliasMember member){(void)aliases;(void)member;abort();}
static PoolAliasMember *pool_alias_member(PoolAliases *aliases,uint32_t slot){
    if(aliases) for(uint32_t i=0;i<aliases->count;i++) if(aliases->members[i].slot==slot)return &aliases->members[i];
    return NULL;
}
static int refuse_alias;
static VkTextureSampleStatus vk_texture_ordered_sample_select_locked(int logical,RcRegistryPosition read,
    unsigned flags,unsigned options,VkTextureSample *sample,VkTextureAliasRead **alias,VkTexImage *image,
    VkTextureAliasFailure *failure) {
    assert(logical==1&&read.scene==66&&read.ordinal==4&&!flags&&alias&&sample&&image&&refuse_alias);
    assert(options==(VK_TEXTURE_ALIAS_SAMPLE_ONLY|VK_TEXTURE_ALIAS_SAMPLE_FORCE));
    *failure=(VkTextureAliasFailure){.phase=VK_TEXTURE_ALIAS_COVERAGE,.status=RC_REGISTRY_BLOCKED,
        .receiver={71},.registration=82,.material_serial=93,.image=202,.coverage={10}};
    return VK_TEXTURE_SAMPLE_UNAVAILABLE;
}
static const RcInterval *vk_texture_alias_read_intervals(const VkTextureAliasRead *alias,uint32_t *count){
    (void)alias; *count=0; return NULL;
}
static void vk_texture_alias_read_cancel(VkTextureAliasRead *alias){assert(!alias);}
static int vk_texture_publication_contains_native_locked(const TextureImagePublication *p,uint32_t at,VkImage expected){
    return p->native[at]==expected;
}
static int vk_texture_pool_gpu_hold_locked(TextureImagePublication *p,uint32_t at,const VkTextureSample *sample){
    if (p->images[at]!=sample->image->image) return 0;
    p->native[at]=p->images[at];
    gpu_holds++;
    return 1;
}
#define VK_NULL_HANDLE 0
static int flat3d_pool_consuming;
static VkTexImage history_child;
static unsigned history_child_queries;
static uint16_t pool_kind_type(int k){return k==1?4:3;}
static int res_view_type(uint16_t type){return type;}
static int dummy_index(int type,int depth){assert(!depth);return type==4;}
enum { DV_DREF_2D = 2 };
static struct {struct {VkImageView view;} img[3];} Z;
static VkImageView pool_absent_view(uint32_t kind){return Z.img[kind==10?DV_DREF_2D:dummy_index(res_view_type(pool_kind_type(kind)),0)].view;}
static VkTextureSampleStatus vk_texture_history_flat3d_locked(int logical,const VkTexImage **image,void *plan){
 assert(logical==history_slot&&!plan);history_child_queries++;
 *image=history_child.image?&history_child:NULL;
 return *image?VK_TEXTURE_SAMPLE_READY:VK_TEXTURE_SAMPLE_PREPARE;
}
static int vk_texture_history_flat3d_installed_locked(int logical,const TextureImagePublication *set,
                                                       uint32_t descriptor,VkImage image){
 assert(logical==history_slot&&descriptor==VK_POOL_TEX+(unsigned)logical);
 return set->images[descriptor]==image;
}
static uint64_t vk_texture_registration_scope_token(void){return PL.registration_scope;}
static uint64_t vk_exec_texture_pool(void){return PL.registration_pool;}
static int vk_texture_registration_expected_locked(int id,VkImage *image){
 history_queries++;
 if(image)*image=id==history_slot?history_image:0;
 return id==history_slot?history_status:VK_TEXTURE_REGISTRATION_CURRENT;
}
static int active_slots=4,race_mapping,race_epoch;
static void vk_texture_lock(void){}
static void vk_texture_unlock(void){}
static int pool_visible_slots(void){return active_slots;}
static RcRegistryPosition position={1,2};
static uint64_t authority=1,legacy_delta,mapping=1,protection=1;
static uint64_t source_head;static unsigned source_events[64],source_drains;
static uint64_t schedule_head;static unsigned schedule_events[64],schedule_drains;
static int schedule_overflow,schedule_race,position_open=1,position_race;static unsigned schedule_race_bits;
static void schedule_change(unsigned bits){assert(schedule_head<64);schedule_events[schedule_head++]=bits;legacy_delta++;}
static int source_overflow,source_race;static unsigned source_race_bits;
static void source_change(unsigned bits){assert(source_head<64);source_events[source_head++]=bits;authority++;}
static int vk_texture_source_dirty_slots_locked(uint64_t *cursor,void(*visit)(int,void*),void *arg){
 uint64_t end=source_head;unsigned bits=0;source_drains++;
 if(source_overflow){source_overflow=0;*cursor=end;return 0;}
 while(*cursor<end)bits|=source_events[(*cursor)++];
 for(int id=0;id<4;id++)if(bits>>id&1)visit(id,arg);
 if(source_race){source_race=0;source_change(source_race_bits);}
 return authority!=0;
}
static int vk_texture_schedule_dirty_slots_locked(uint64_t *cursor,void(*visit)(int,void*),void *arg){
 uint64_t end=schedule_head;unsigned bits=0;schedule_drains++;
 if(schedule_overflow){schedule_overflow=0;*cursor=end;return 0;}
 while(*cursor<end)bits|=schedule_events[(*cursor)++];
 for(int id=0;id<4;id++)if(bits>>id&1)visit(id,arg);
 if(schedule_race){schedule_race=0;schedule_change(schedule_race_bits);}
 return authority!=0;
}
static uint32_t slots=1,epoch=1,samplers=1,vk_pool_kinds_used=1;
enum {VK_TEXTURE_BINDING_DIRECT=1,VK_TEXTURE_BINDING_SOURCE=2,VK_TEXTURE_BINDING_LEGACY=4};
static uint64_t tokens[4]={1,1,1,1},dirty_serial=1;
static unsigned dirty_bits;static int dirty_overflow;
static int proof_type[4]={2,2,2,2};
static int extra_proof_type[4]={1,1,1,1};
static uint64_t vk_texture_slot_token(int id){return tokens[id];}
static int pool_native_slot(int id){return id+(PL.registration_pool?12:0);}
static int vk_texture_pool_logical_locked(uint64_t pool,int native){
 assert(pool==PL.registration_pool);return native-(pool?12:0);
}
static uint64_t pool_slot_token(int id){token_queries++;return vk_texture_slot_token(id);}
static uint64_t vk_texture_dirty_pos_locked(void){return dirty_serial;}
static int vk_texture_dirty_slots_locked(uint64_t *cursor,void(*fn)(int,void*),void(*g)(int,void*),void *arg){
 (void)g;if(dirty_overflow){*cursor=dirty_serial;dirty_overflow=0;return 0;}
 if(*cursor!=dirty_serial)for(int id=0;id<4;id++)if(dirty_bits>>id&1)fn(id,arg);
 *cursor=dirty_serial;return 1;
}
static unsigned validations[4];static uint64_t edges[4]={4,10,0,0};static int stale[4],race_legacy;
static RcRegistryPosition vk_exec_resource_position(void){return position;}
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b){
 if(a.scene!=b.scene)return a.scene<b.scene?-1:1;
 return a.ordinal==b.ordinal?0:(a.ordinal<b.ordinal?-1:1);
}
static void *rc_registry_runtime(void){return NULL;}
static uint64_t rc_registry_ordered_epoch(void *p){(void)p;return authority;}
static uint64_t rc_registry_authority_epoch(void *p){(void)p;return authority?authority+legacy_delta:0;}
static int rc_registry_position_valid(RcRegistryPosition p){return p.scene&&p.ordinal;}
static int rc_registry_position_open(void *p,RcRegistryPosition at){
 (void)p;(void)at;if(position_race){legacy_delta++;position_race=0;}return position_open;
}
static uint64_t mem_mapping_serial(void){return mapping;}
static uint64_t mem_protection_serial(void){return protection;}
static uint32_t vk_texture_slots_gen(void){return slots;}
static uint32_t vk_texture_epoch(void){return epoch;}
static uint32_t vk_sampler_gen(void){return samplers;}
static int vk_texture_binding_schedule_locked(int id,VkImage image,RcRegistryPosition *next){
 if(PL.registration_pool){assert(id>=12);id-=12;}
 assert(image==PL.image[id]||image==PL.image[VK_POOL_TEX+id]);validations[id]++;if(race_mapping){mapping++;race_mapping=0;}
 if(race_epoch){epoch++;race_epoch=0;}
 if(race_legacy){legacy_delta++;race_legacy=0;}if(stale[id])return 0;
 *next=(RcRegistryPosition){edges[id]?position.scene:0,edges[id]};
 return image==PL.image[id]?proof_type[id]:extra_proof_type[id];
}
static int texture_view_key(const VkTexImage *source,const void *view,int has_view,uint16_t type,int storage,ViewKey *key){
 if(!source||!has_view||storage)return 0;
 key->view=((const view_t *)view)->view+(type==4);
 return 1;
}
static int vk_texture_published_alias_schedule_locked(const TextureImagePublication *set,uint32_t descriptor,
    int logical,VkTextureReadKey receiver,VkImage expected,const VkTexImage **source,void *view,int *has_view,RcRegistryPosition *until){
 (void)set;(void)descriptor;(void)logical;(void)receiver;(void)expected;(void)source;(void)view;(void)has_view;(void)until;return 0;
}
static unsigned selected_checks;
static int installed_hole_current;
static int vk_texture_sample_unbacked_current_locked(VkTextureReadKey key) {
    return key.unbacked && key.registration==812 && installed_hole_current;
}
static int vk_texture_resolved_schedule_locked(VkTextureReadKey key,VkImage image,RcRegistryPosition *next){
 assert(key.registration==812&&key.native_slot>=0&&key.native_slot<4);selected_checks++;
 return vk_texture_binding_schedule_locked(key.native_slot,image,next);
}
static unsigned published_checks[8];
static int published_permission=1;
static int vk_texture_publication_contains_image_locked(const TextureImagePublication *set,uint32_t descriptor,VkImage image){
 return image&&set->images[descriptor]==image;
}
static int vk_texture_published_sample_schedule_locked(const TextureImagePublication *set,uint32_t descriptor,
                                                       VkTextureReadKey key,VkImage image,RcRegistryPosition *next){
 assert(key.registration==812&&key.native_slot==0&&key.canonical&&descriptor<8);
 published_checks[descriptor]++;
 if(!published_permission||set->images[descriptor]!=image)return 0;
 if(next)*next=(RcRegistryPosition){position.scene,descriptor==0?31:29};
 return VK_TEXTURE_BINDING_SOURCE;
}
'''
start=source.index("typedef struct {\n    uint64_t generation,authority,token,legacy_authority;")
end=source.index("static void pool_order_publish(",start)
order_declaration=re.search(r"static struct \{[^\n]+\} pool_order;",source).group(0)
selected_start=source.index("enum {\n    POOL_EXACT_READY")
selected_end=source.index("static unsigned pool_selected_member_status(",selected_start)
member_source=source[start:end]
for helper in ("pool_domain_comparison_supported", "pool_comparison_format", "pool_alias_depth_before_draw",
               "pool_alias_depth_required", "pool_alias_depth_select_locked"):
    member_source=member_source.replace(extract(helper), "")
member_source=member_source.replace(
    "static int pool_prepare_depth_alias_locked(GraphicsPoolRestore *, uint32_t, uint32_t, const VkTextureSample *);",
    "static int pool_prepare_depth_alias_locked(GraphicsPoolRestore *r,uint32_t id,uint32_t kinds,const VkTextureSample *s)"
    "{(void)r;(void)id;(void)kinds;(void)s;return 0;}"
)
member=prefix+order_declaration+extract("pool_request_kinds")+member_source+extract("pool_association_current")+extract("pool_association_member")+extract("pool_order_publish")+extract("pool_order_current_for")+extract("pool_order_current")+extract("pool_required_previous_kinds")+source[selected_start:selected_end]+extract("pool_selected_member_status")+r'''
static uint64_t pool_ready_generation,pool_ready_authority;
'''+extract("vk_exec_pool_generation")+r'''
static void publish_order(void){pool_order_publish();pool_ready_authority=rc_registry_authority_epoch(NULL);}
static void exact_selected_members(void) {
 TextureImagePublication physical={0};
 VkExec ex={0};
 VkExecPoolIndices indexed={.entries={{1,2}},.count=1};
 GraphicsPoolRestore request={.kinds=3,.indices=&indexed,.ex=&ex,.read={66,4}};
 unsigned idx=VK_POOL_TEX+1;
 uint32_t failed;
 RcRegistryPosition prior_position=position;
 position=(RcRegistryPosition){66,4};
 PL.publication[0]=&physical;
 PL.image[0]=201;
 PL.image[idx]=202;
 PL.view[idx]=301;
 physical.images[idx]=202;
 gpu_image[1].image=202;
 gpu_base_view[1]=300;
 gpu_status[1]=installed_receiver[1]=view_of.klen=1;
 tokens[1]=UINT64_C(0x100000007);
 PL.tok[1]=UINT64_C(0x100000006);
 graphics_pool_restore=&request;
 assert(pool_request_kinds(&request,0)==0&&pool_request_kinds(&request,1)==2);
 RcRegistryPosition edge;
 assert(!pool_gpu_binding(1,0,202,&edge)&&!gpu_queries);
 assert(pool_selected_member_status(&ex,1,2,1,&failed)==POOL_EXACT_READY);
 assert(physical.native[idx]==202&&gpu_queries==1&&gpu_holds==1&&view_queries==1);
 assert(PL.tok[1]==UINT64_C(0x100000006)); // The scoped check never blesses the other kinds.
 assert(pool_gpu_binding(1,1,202,&edge)==1&&edge.scene==position.scene&&edge.ordinal==position.ordinal+1);
 VkTextureAliasRead reconstructed={.proved=1};
 physical.aliases[idx]=&reconstructed;
 gpu_status[1]=0;
 assert(pool_gpu_binding(1,1,202,&edge)==1&&edge.ordinal==position.ordinal+1);
 // The retained comparison alias is indexed by the bound descriptor even when another kind
 // of this logical slot uses an independently canonical or projected provider.
 const PoolSlot prior=selected_slots[1];
 selected_slots[1]=(PoolSlot){.selected={.registration=812,.native_slot=0,.canonical=1},.selected_kinds=3};
 published_permission=0;
 assert(pool_selected_schedule(&selected_slots[1],idx,&edge)==VK_TEXTURE_BINDING_SOURCE);
 assert(edge.scene==position.scene&&edge.ordinal==position.ordinal+1);
 published_permission=1;
 selected_slots[1].selected.canonical=0;
 selected_slots[1].selected.projected=1;
 unsigned before_projection=projected_queries;
 assert(pool_gpu_binding(1,1,202,&edge)==1&&projected_queries==before_projection);
 selected_slots[1]=prior;
 reconstructed.proved=0;
 assert(pool_gpu_binding(1,1,202,&edge)<0); // The old VkImage identity does not renew changed guest bytes.
 PoolAliases aliases={0};
 request.aliases=&aliases;refuse_alias=1;
 assert(pool_gpu_binding(1,1,202,&edge)<0 && aliases.refused && !aliases.count);
 assert(aliases.failure.owner==VK_EXEC_PREPARATION_POOL_IMAGE && aliases.failure.slot==1 &&
        aliases.failure.role==1 && aliases.failure.range.mapping==71 && aliases.failure.reason==VK_EXEC_PREPARATION_ORDER &&
        aliases.failure.registration==82 && aliases.failure.material_serial==93 && aliases.failure.coverage.gate==10);
 request.aliases=NULL;refuse_alias=0;
 physical.aliases[idx]=NULL;
 gpu_status[1]=1;
 PL.view[idx]++;assert(pool_selected_member_status(&ex,1,2,1,&failed)==POOL_EXACT_TOKEN&&failed==1);PL.view[idx]--;
 ex.feedback=202;assert(pool_selected_member_status(&ex,1,2,1,&failed)==POOL_EXACT_NATIVE);ex.feedback=0;
 gpu_status[1]=0;assert(pool_selected_member_status(&ex,1,2,1,&failed)==POOL_EXACT_NATIVE);gpu_status[1]=1;
 tokens[1]=UINT64_C(0x200000007);
 assert(pool_selected_member_status(&ex,1,2,1,&failed)==POOL_EXACT_TOKEN&&failed==UINT32_MAX);
 PL.tok[1]=pool_slot_token(1);
 PL.ps[1].selected=(VkTextureReadKey){.registration=901,.native_slot=1,.projected=1};
 PL.ps[1].selected_kinds=2;
 unsigned legacy_queries=gpu_queries;
 projected_current=1;
 assert(pool_gpu_binding(1,1,202,&edge)==1&&projected_queries==1&&gpu_queries==legacy_queries);
 assert(physical.native[idx]==202&&edge.ordinal==position.ordinal+1);
 projected_current=0;
 assert(pool_gpu_binding(1,1,202,&edge)<0&&projected_queries==2&&gpu_queries==legacy_queries);
 // A finite selected-kind reconciliation must re-prove its projected view without a legacy receiver.
 PL.finite_reconcile=1;
 const unsigned old_view_queries=view_queries;
 installed_receiver[1]=0;
 projected_current=1;
 assert(pool_selected_member_status(&ex,1,2,1,&failed)==POOL_EXACT_READY);
 PL.view[idx]++;
 assert(pool_selected_member_status(&ex,1,2,1,&failed)==POOL_EXACT_NATIVE&&failed==1);
 PL.view[idx]--;
 projected_current=0;
 assert(pool_selected_member_status(&ex,1,2,1,&failed)==POOL_EXACT_NATIVE&&failed==1);
 assert(view_queries==old_view_queries&&gpu_queries==legacy_queries);
 installed_receiver[1]=1;
 PL.finite_reconcile=0;
 // The outgoing indexed set renews the existing projected view directly. A
 // different view at the same VkImage or a lost actual owner cannot be carried.
 pool_transfer_images=&physical;
 projected_current=1;
 assert(pool_keep_selected(1,2)&&gpu_queries==legacy_queries);
 PL.view[idx]++;
 assert(!pool_keep_selected(1,2));PL.view[idx]--;
 physical.images[idx]=physical.native[idx]=0;
 assert(!pool_keep_selected(1,2));
 physical.images[idx]=physical.native[idx]=202;
 projected_current=0;
 assert(!pool_keep_selected(1,2)&&gpu_queries==legacy_queries);
 PL.ps[1].selected.projected=0;
 physical.aliases[idx]=&reconstructed;reconstructed.proved=1;
 assert(pool_keep_selected(1,2));
 reconstructed.proved=0;assert(!pool_keep_selected(1,2));
 physical.aliases[idx]=NULL;
 pool_transfer_images=NULL;
 PL.ps[1]=(PoolSlot){0};
 history_slot=1;history_status=VK_TEXTURE_REGISTRATION_ABSENT;
 PL.tok[1]=0;PL.image[idx]=0;
 Z.img[1].view=310;PL.view[idx]=310;
 assert(pool_selected_member_status(&ex,1,2,1,&failed)==POOL_EXACT_READY);
 PL.view[idx]=311;assert(pool_selected_member_status(&ex,1,2,1,&failed)==POOL_EXACT_ABSENT_IMAGE);
 graphics_pool_restore=NULL;
 PL.publication[0]=NULL;
 history_slot=-1;
 PL.image[0]=PL.image[idx]=PL.view[idx]=PL.tok[1]=0;
 gpu_status[1]=installed_receiver[1]=view_of.klen=0;
 tokens[1]=1;
 position=prior_position;
}
int main(void){
 exact_selected_members();
 VkPoolFrontier oracle={0};uint32_t random=1;uint8_t present[VK_POOL_TEX]={0};uint64_t deadlines[VK_POOL_TEX]={0};
 for(unsigned k=0;k<100000;k++){
  random=random*1664525u+1013904223u;unsigned id=(random>>4)%VK_POOL_TEX;
  if(random&1){deadlines[id]=(random>>10)%17;present[id]=1;vk_pool_frontier_set(&oracle,id,deadlines[id]);}
  else{present[id]=0;vk_pool_frontier_remove(&oracle,id);}
  unsigned count=0;uint64_t least=UINT64_MAX;
  for(unsigned j=0;j<VK_POOL_TEX;j++){assert(!!oracle.at[j]==present[j]);if(present[j]){count++;if(deadlines[j]<least)least=deadlines[j];}}
  assert(count==oracle.count);if(count)assert(oracle.deadline[oracle.ids[0]]==least);
  for(unsigned j=0;j<count;j++){assert(oracle.at[oracle.ids[j]]==j+1);if(j)assert(!vk_pool_frontier_less(&oracle,oracle.ids[j],oracle.ids[(j-1)/2]));}
 }
 PL.image[0]=11;PL.image[1]=12;
 PL.ps[0]=(PoolSlot){.selected={.registration=812,.native_slot=0},.selected_kinds=1};
 assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(1,1));
 assert(validations[0]==1&&validations[1]==1&&selected_checks==1);
 PL.registration_gone[0]=POOL_SOURCE_REFUSED;
 assert(!pool_slot_binding_current(0,1)&&selected_checks==1);
 PL.registration_gone[0]=0;
 PL.ps[0]=(PoolSlot){0};
 position.ordinal=3;assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(1,1));
 assert(validations[0]==1&&validations[1]==1);
 position.ordinal=4;stale[0]=1;
 assert(!pool_slot_binding_current(0,1)&&pool_slot_binding_current(1,1));
 assert(validations[0]==2&&validations[1]==1); // Only the crossed member is queried again.
 stale[0]=0;edges[0]=8;assert(pool_slot_binding_current(0,1));
 assert(pool_slot_binding_current(1,1)&&validations[1]==1);
 mapping++;assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(1,1));
 assert(validations[0]==4&&validations[1]==2);
 protection++;assert(pool_slot_binding_current(1,1)&&validations[1]==3);
 PL.image[1]=17;pool_member[1].generation=0;
 assert(pool_slot_binding_current(1,1)&&validations[1]==4);
 position.scene=2;edges[1]=0;assert(pool_slot_binding_current(1,1)&&validations[1]==5);
 // A descriptor containing only a stand-in has no source image to revalidate. It remains guarded by the same
 // registration/mapping selector guards, and a real replacement must obtain its own proof.
 PL.image[2]=0;stale[2]=1;
 for(unsigned i=0;i<1000;i++){position.ordinal++;assert(pool_slot_binding_current(2,1));}
 assert(!validations[2]);
 mapping++;assert(pool_slot_binding_current(2,1)&&!validations[2]);
 PL.image[2]=19;slots++;tokens[2]++;stale[2]=0;
 assert(pool_slot_binding_current(2,1)&&validations[2]==1);
 authority=0;assert(!pool_slot_binding_current(1,1));
 assert(pool_member_hits && pool_member_checks);
 // External changes rebuild once. An unprovable member never invalidates the independent members' schedule.
 authority=1;position=(RcRegistryPosition){3,1};edges[0]=edges[1]=edges[2]=0;
 stale[0]=stale[1]=stale[2]=0;stale[3]=1;PL.image[3]=20;
 assert(pool_frontier_prepare(1,4));uint64_t rebuilds=pool_frontier_rebuilds;
 unsigned other=validations[0]+validations[1]+validations[2];
 assert(pool_frontier.count==1 && pool_frontier.ids[0]==3 && !pool_frontier.deadline[3]);
 for(unsigned i=0;i<1000;i++){position.ordinal++;assert(pool_frontier_prepare(1,4));}
 assert(pool_frontier_rebuilds==rebuilds && validations[0]+validations[1]+validations[2]==other);
 // Actual descriptor replacement re-enters the frontier even with unchanged global guards.
 pool_member_invalidate(1);assert(pool_frontier.at[1] && !pool_frontier.deadline[1]);
 assert(pool_slot_binding_current(1,1));assert(!pool_frontier.at[1]);
 // Backwards and new-scene readers revoke source permission without rebuilding the selector frontier.
 position.ordinal=1;assert(pool_frontier_prepare(1,4)&&pool_frontier_rebuilds==rebuilds);
 assert(pool_frontier.count==4&&!pool_member[0].generation&&!pool_member[1].generation&&!pool_member[2].generation);
 position.scene++;assert(pool_frontier_prepare(1,4)&&pool_frontier_rebuilds==rebuilds);
 assert(pool_frontier_prepare(1,3)&&pool_frontier_rebuilds==++rebuilds&&!pool_frontier.at[3]);
 mapping++;assert(pool_frontier_prepare(1,3)&&pool_frontier_rebuilds==++rebuilds);
 active_slots=3;publish_order();assert(pool_order.scene==position.scene&&!pool_order.next_scene);
 PL.set_registration_gone[PL.cur]=1;publish_order();assert(!pool_order.scene);
 PL.set_registration_gone[PL.cur]=0;publish_order();assert(pool_order.scene==position.scene);
 edges[0]=position.ordinal+2;pool_member_invalidate(0);assert(pool_slot_binding_current(0,1));
 publish_order();assert(pool_order.scene==position.scene&&pool_order.next_ordinal==edges[0]);
 position.ordinal=edges[0];publish_order();assert(!pool_order.scene);
 edges[0]=0;pool_member_invalidate(0);assert(pool_slot_binding_current(0,1));
 publish_order();assert(pool_order.scene);
 // Unrelated registry changes preserve every existing source receipt after an exact drain.
 proof_type[1]=VK_TEXTURE_BINDING_DIRECT;pool_member_invalidate(1);assert(pool_slot_binding_current(1,1));
 unsigned direct_checks=validations[1];rebuilds=pool_frontier_rebuilds;authority++;
 assert(pool_frontier_prepare(1,3)&&pool_frontier_rebuilds==rebuilds);
 assert(!pool_frontier.at[0] && !pool_frontier.at[1] && !pool_frontier.at[2]);
 unsigned source_checks=validations[0]+validations[2];
 assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(2,1));
 assert(validations[0]+validations[2]==source_checks);
 // A matching generic source revision invalidates only its own slot, even with unchanged content/image/token.
 source_change(1);assert(pool_frontier_prepare(1,3)&&pool_frontier_rebuilds==rebuilds);
 assert(pool_frontier.at[0] && !pool_frontier.at[1] && !pool_frontier.at[2]);
 assert(pool_slot_binding_current(1,1)&&validations[1]==direct_checks);
 assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(2,1));publish_order();assert(pool_order.scene);
 assert(validations[0]+validations[2]==source_checks+1);
 // Even an emitted slot ID cannot invalidate a physically exact GPU receipt.
 source_change(2);assert(pool_frontier_prepare(1,3)&&!pool_frontier.at[1]);
 assert(pool_slot_binding_current(1,1)&&validations[1]==direct_checks);
 // A new registration at even the same image must invalidate dependency classification via its own token.
 tokens[1]++;slots++;dirty_serial++;dirty_bits|=2;proof_type[1]=VK_TEXTURE_BINDING_SOURCE;
 assert(pool_frontier_prepare(1,3)&&pool_frontier_rebuilds==rebuilds&&pool_frontier.at[1]);
 assert(pool_slot_binding_current(1,1)&&validations[1]==direct_checks+1);
 source_change(2);assert(pool_frontier_prepare(1,3)&&pool_frontier.at[1]);
 assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(1,1)&&pool_slot_binding_current(2,1));
 // A hard source event can shorten the supplied horizon even when the prior test receipt had no cutoff.
 edges[0]=position.ordinal+3;source_change(1);assert(pool_frontier_prepare(1,3));
 assert(pool_slot_binding_current(0,1));publish_order();assert(pool_order.next_ordinal==edges[0]);
 position.ordinal=edges[0];stale[0]=1;assert(!pool_slot_binding_current(0,1));
 stale[0]=0;edges[0]=0;source_change(1);assert(pool_frontier_prepare(1,3));
 assert(pool_slot_binding_current(0,1));
 // Before the drain, a receipt from the older epoch must be checked even in an out-of-frontier caller.
 source_change(4);unsigned before_unacked=validations[2];stale[2]=1;
 assert(!pool_slot_binding_current(2,1)&&validations[2]==before_unacked+1);
 stale[2]=0;assert(pool_frontier_prepare(1,3)&&pool_slot_binding_current(2,1));
 // Events arriving after the captured drain end stay pending and cannot advance the epoch acknowledgement.
 uint64_t ack=pool_source_authority;source_change(1);source_race=1;source_race_bits=4;
 assert(!pool_frontier_prepare(1,3)&&pool_source_authority==ack&&pool_source_cursor+1==source_head);
 assert(pool_frontier_prepare(1,3)&&pool_source_cursor==source_head);
 assert(pool_frontier.at[0]&&pool_frontier.at[2]);
 assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(2,1));
 // Lost source history invalidates SOURCE receipts once; direct bindings and unrelated later epochs survive.
 proof_type[1]=VK_TEXTURE_BINDING_DIRECT;pool_member_invalidate(1);assert(pool_slot_binding_current(1,1));
 direct_checks=validations[1];source_overflow=1;authority++;
 assert(pool_frontier_prepare(1,3)&&pool_frontier.at[0]&&pool_frontier.at[2]&&!pool_frontier.at[1]);
 uint64_t invalidations=pool_frontier_source_invalidations;unsigned drains=source_drains;
 assert(pool_frontier_prepare(1,3)&&pool_frontier_source_invalidations==invalidations&&source_drains==drains);
 assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(1,1)&&pool_slot_binding_current(2,1));
 assert(validations[1]==direct_checks);
 dirty_overflow=1;assert(pool_frontier_prepare(1,3)&&pool_frontier_rebuilds==rebuilds+1);
 assert(pool_frontier_source_invalidations && pool_frontier_slot_invalidations);
 assert(pool_frontier_reason[0] && pool_frontier_reason[5] && pool_frontier_reason[7]);
 race_mapping=1;mapping++;publish_order();assert(!pool_order.scene);
 publish_order();assert(pool_order.scene);
 // A standalone descriptor lookup can certify a SOURCE outside the currently scheduled count. An event
 // must invalidate that receipt before count growth, rather than let the new global ack bless its old proof.
 assert(pool_frontier_state.count==3);stale[3]=0;proof_type[3]=VK_TEXTURE_BINDING_SOURCE;edges[3]=0;
 pool_member_invalidate(3);assert(pool_slot_binding_current(3,1));
 unsigned outside_checks=validations[3];uint64_t outside_token=tokens[3];
 stale[3]=1;source_change(8);assert(pool_frontier_prepare(1,4));
 assert(tokens[3]==outside_token&&!pool_member[3].generation&&validations[3]==outside_checks+1);
 // Shrinking the scheduling count must not erase the source membership of a retained receipt. A later
 // standalone fast hit and lost event history still require that excluded member to be invalidated.
 stale[3]=0;assert(pool_slot_binding_current(3,1));assert(pool_frontier_prepare(1,3));
 outside_checks=validations[3];assert(pool_slot_binding_current(3,1)&&validations[3]==outside_checks);
 stale[3]=1;source_overflow=1;authority++;assert(pool_frontier_prepare(1,4));
 assert(!pool_member[3].generation&&validations[3]==outside_checks+1);
 // Schedule proofs include the actual known producer deadline. Later declarations revoke the aggregate until
 // both notification rings are drained, but an unrelated declaration does not revalidate these members.
 position=(RcRegistryPosition){20,2};active_slots=3;
 for(unsigned id=0;id<4;id++){stale[id]=0;edges[id]=0;proof_type[id]=VK_TEXTURE_BINDING_DIRECT;pool_member_invalidate(id);}
 proof_type[0]=VK_TEXTURE_BINDING_SOURCE;edges[0]=5;publish_order();
 pool_ready_generation=900;pool_ready_authority=rc_registry_authority_epoch(NULL);
 assert(pool_order.next_ordinal==5&&!pool_order.legacy_authority&&vk_exec_pool_generation()==900);
 unsigned finite_checks=validations[0],finite_drains=source_drains;
 legacy_delta++;position.ordinal=4;
 assert(!vk_exec_pool_generation());publish_order();
 assert(vk_exec_pool_generation()==900&&pool_slot_binding_current(0,1));
 assert(validations[0]==finite_checks&&source_drains==finite_drains+1);
 position.ordinal=5;assert(!vk_exec_pool_generation());
 edges[0]=9;assert(pool_slot_binding_current(0,1));publish_order();
 assert(pool_order.next_ordinal==9&&vk_exec_pool_generation()==900);
 legacy_delta++;assert(!vk_exec_pool_generation());publish_order();assert(vk_exec_pool_generation()==900);
 position.ordinal=9;assert(!vk_exec_pool_generation());
 // The same-operation bridge has an additional legacy dependency. A later declaration at the same query
 // position must invalidate both the aggregate and its member, but not unrelated finite/direct receipts.
 position.ordinal=10;edges[0]=11;proof_type[0]=VK_TEXTURE_BINDING_SOURCE|VK_TEXTURE_BINDING_LEGACY;
 pool_member_invalidate(0);assert(pool_slot_binding_current(0,1));publish_order();
 assert(pool_legacy_count==1&&pool_legacy_total==1);
 assert(pool_order.legacy_authority==rc_registry_authority_epoch(NULL)&&vk_exec_pool_generation()==900);
 unsigned legacy_checks=validations[0],unrelated_checks=validations[1]+validations[2];
 legacy_delta++;assert(!vk_exec_pool_generation());stale[0]=1;
 assert(pool_frontier_prepare(1,3)&&pool_frontier.at[0]&&!pool_member[0].generation);
 assert(!pool_slot_binding_current(0,1)&&validations[0]==legacy_checks+1);
 assert(pool_slot_binding_current(1,1)&&pool_slot_binding_current(2,1));
 assert(validations[1]+validations[2]==unrelated_checks);
 stale[0]=0;assert(pool_slot_binding_current(0,1));publish_order();assert(vk_exec_pool_generation()==900);
 position.ordinal=11;assert(!vk_exec_pool_generation()); // The legacy flag does not remove the operation edge.
 position.ordinal=12;edges[0]=13;pool_member_invalidate(0);race_legacy=1;
 assert(!pool_slot_binding_current(0,1)); // A declaration racing proof acquisition cannot publish old permission.
 assert(pool_slot_binding_current(0,1));publish_order();assert(vk_exec_pool_generation()==900);
 // Replacing the bridge with a finite proof removes the conditional global dependency immediately.
 proof_type[0]=VK_TEXTURE_BINDING_SOURCE;edges[0]=30;pool_member_invalidate(0);
 assert(pool_slot_binding_current(0,1));publish_order();
 assert(!pool_legacy_count&&!pool_legacy_total&&!pool_order.legacy_authority);
 legacy_delta++;assert(!vk_exec_pool_generation());publish_order();assert(vk_exec_pool_generation()==900);
 // A retained bridge outside a shrunken active frontier still loses permission on a legacy change.
 proof_type[3]=VK_TEXTURE_BINDING_SOURCE|VK_TEXTURE_BINDING_LEGACY;edges[3]=13;
 assert(pool_slot_binding_current(3,1)&&pool_legacy_total==1&&!pool_legacy_count);
 legacy_delta++;assert(pool_frontier_prepare(1,3)&&!pool_member[3].generation);
 stale[3]=1;assert(pool_frontier_prepare(1,4)&&pool_legacy_count==1&&!pool_member[3].generation);
 // Reclamation clears receipt values before rebuilding the frontier. Retained membership bits/counts may
 // conservatively survive, but each rescheduled bit must be removed exactly once, including after shrink.
 assert(pool_frontier_prepare(1,3)&&!pool_legacy_count&&pool_legacy_total==1);
 memset(pool_member,0,sizeof pool_member);pool_frontier_state.generation=0;
 assert(pool_frontier_prepare(1,3)&&!pool_legacy_count&&pool_legacy_total==1);
 assert(pool_frontier_prepare(1,4)&&!pool_legacy_count&&!pool_legacy_total);
 active_slots=4;publish_order();assert(!pool_order.scene); // Failed cleared member still blocks publication.
 stale[3]=0;assert(pool_slot_binding_current(3,1)&&pool_legacy_count==1&&pool_legacy_total==1);
 // The actual generation-wrap branch also clears receipts. Replacing the old legacy classification during
 // that rebuild cannot underflow a count or leave a publication without its remaining dependencies.
 pool_member_guard.generation=UINT64_MAX;mapping++;proof_type[3]=VK_TEXTURE_BINDING_DIRECT;
 assert(pool_frontier_prepare(1,4)&&pool_member_guard.generation==1);
 assert(!pool_legacy_count&&!pool_legacy_total);publish_order();assert(pool_order.scene);
 // Soft schedule additions have their own bounded history. They invalidate only affected SOURCE schedules,
 // without erasing unrelated byte proofs; an event after drain-entry cannot be acknowledged prematurely.
 edges[3]=0;pool_member_invalidate(3);assert(pool_slot_binding_current(3,1));
 proof_type[2]=VK_TEXTURE_BINDING_SOURCE;edges[2]=0;pool_member_invalidate(2);assert(pool_slot_binding_current(2,1));
 unsigned unaffected=validations[2],schedule_before=schedule_drains;
 edges[0]=position.ordinal+2;schedule_change(1);assert(pool_frontier_prepare(1,4));
 assert(pool_frontier.at[0]&&!pool_frontier.at[2]&&schedule_drains==schedule_before+1);
 assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(2,1)&&validations[2]==unaffected);
 ack=pool_source_authority;schedule_change(1);schedule_race=1;schedule_race_bits=4;
 assert(!pool_frontier_prepare(1,4)&&pool_source_authority==ack&&pool_schedule_cursor+1==schedule_head);
 assert(pool_frontier_prepare(1,4)&&pool_frontier.at[0]&&pool_frontier.at[2]);
 assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(2,1));
 schedule_overflow=1;legacy_delta++;assert(pool_frontier_prepare(1,4));
 assert(pool_frontier.at[0]&&pool_frontier.at[2]&&!pool_frontier.at[1]&&!pool_frontier.at[3]);
 assert(pool_slot_binding_current(0,1)&&pool_slot_binding_current(2,1));publish_order();
 assert(vk_exec_pool_generation()==900);position_open=0;assert(!vk_exec_pool_generation());
 publish_order();assert(!pool_order.scene);position_open=1;publish_order();assert(vk_exec_pool_generation()==900);
 position_race=1;assert(!vk_exec_pool_generation()); // The final open-position check cannot hide a new topology event.
 publish_order();assert(vk_exec_pool_generation()==900);
 // Diagnostic attribution is bounded to the same refresh, exact current group, kinds and address generation.
 // Tagged cache collisions lose coverage rather than inventing a representative skip.
 rt_trace=RT_TRACE_PERF;pool_diag_sync=100;
 tokens[1]=(tokens[1]&UINT64_C(0xffffffff00000000))|(uint32_t)tokens[0];
 pool_diag_skip(0,1,tokens[0]);assert(pool_diag_skipped(0,1)&&pool_diag_skipped(1,1));
 assert(!pool_diag_skipped(2,1)&&!pool_diag_skipped(1,3));
 pool_member[1].token=tokens[1]-1;uint64_t tagged=pool_diag.after_skip;
 pool_frontier_dirty_slot(1,NULL);assert(pool_diag.invalidated==1&&pool_diag.grouped==1&&pool_diag.after_skip==tagged+1);
 tokens[1]++;assert(!pool_diag_skipped(1,1));
 pool_diag_skip(POOL_DIAG_GROUPS,1,tokens[0]);assert(pool_diag.cache_replaced==1&&!pool_diag_skipped(0,1));
 pool_diag_skip(0,1,tokens[0]);pool_diag_sync++;assert(!pool_diag_skipped(0,1));
 rt_trace=0;pool_diag_skip(0,1,tokens[0]);assert(!pool_diag_skipped(0,1));
 // Scope changes revoke only read permission. Use the actual receipt/frontier functions, including a real
 // installed group record, so an old-scene physical selector is not re-resolved through a test substitute.
 memset(pool_member,0,sizeof pool_member);memset(pool_source_members,0,sizeof pool_source_members);
 memset(pool_legacy_members,0,sizeof pool_legacy_members);pool_legacy_count=pool_legacy_total=0;
 memset(&pool_frontier_state,0,sizeof pool_frontier_state);vk_pool_frontier_reset(&pool_frontier);
 position=(RcRegistryPosition){40,10};active_slots=4;
 for(unsigned id=0;id<4;id++){proof_type[id]=VK_TEXTURE_BINDING_DIRECT;stale[id]=0;edges[id]=0;PL.image[id]=id+101;}
 proof_type[1]=VK_TEXTURE_BINDING_SOURCE;stale[2]=1;
 assert(pool_frontier_prepare(1,4));rebuilds=pool_frontier_rebuilds;
 uint64_t selector_generation=pool_member_generation();unsigned first_direct=validations[0],first_source=validations[1];
 // A SOURCE bit outside the active count survives shrink, scope invalidation, and later growth.
 active_slots=3;assert(pool_frontier_prepare(1,3));rebuilds=pool_frontier_rebuilds;
 proof_type[3]=VK_TEXTURE_BINDING_SOURCE;tokens[3]++;pool_member_invalidate(3);
 assert(pool_slot_binding_current(3,1)&&(pool_source_members[0]&8));unsigned high_source=validations[3];
 PoolAssociation *a=calloc(1,sizeof *a);assert(a);
 uint64_t token=vk_texture_slot_token(0);groups[0]=(VkSlotGroup){.incarnation=1,.association=1,.head=0};
 *a=(PoolAssociation){.incarnation=1,.version=1,.generation=selector_generation,.group=0,.representative=0,
  .registration=(uint32_t)(token>>32),.agen=(uint32_t)token,.kinds=1,.head=1,.count=1,.active=1};
 pool_associations[0]=a;pool_association_next=1;
 pool_association_slot[0].owner=1;pool_association_slot[0].registration=(uint32_t)(token>>32);pool_association_slot[0].installed=1;
 pool_member_schedule(0);
 epoch++;position=(RcRegistryPosition){41,2};
 assert(pool_member_generation()==selector_generation&&pool_association_member(0,1));
 stale[1]=1;assert(!pool_slot_binding_current(1,1)&&validations[1]==first_source+1); // No frontier call required.
 assert(pool_frontier_prepare(1,3)&&pool_frontier_rebuilds==rebuilds);
 assert(pool_association_member(0,1)&&validations[0]==first_direct);
 assert(pool_frontier.at[1]&&pool_frontier.at[2]&&!pool_frontier.deadline[2]); // Unresolved work survives.
 assert(!pool_member[3].generation&&(pool_source_members[0]&8));
 stale[1]=0;assert(pool_slot_binding_current(1,1));unsigned renewed_source=validations[1];
 assert(pool_slot_binding_current(3,1)&&validations[3]==high_source+1);
 publish_order();assert(!pool_order.scene); // The unchanged unresolved DIRECT-looking entry still blocks readiness.
 stale[2]=0;assert(pool_slot_binding_current(2,1));publish_order();assert(pool_order.scene);
 // A later-to-earlier query inside the same scene cannot reuse a forward read proof, including outside refresh.
 position.ordinal=20;pool_member_invalidate(1);assert(pool_slot_binding_current(1,1));renewed_source=validations[1];
 assert(pool_frontier_prepare(1,3));position.ordinal=10;stale[1]=1;
 assert(!pool_slot_binding_current(1,1)&&validations[1]==renewed_source+1);
 assert(pool_frontier_prepare(1,3)&&pool_frontier_rebuilds==rebuilds&&pool_association_member(0,1));
 stale[1]=0;assert(pool_slot_binding_current(1,1));
 // Reclassification updates the retained bitset. A direct interval survives scope changes; a later SOURCE
 // interval at the same physical image must again enforce the new epoch and source-event acknowledgement.
 proof_type[1]=VK_TEXTURE_BINDING_DIRECT;tokens[1]++;pool_member_invalidate(1);assert(pool_slot_binding_current(1,1));
 assert(!(pool_source_members[0]&2));unsigned reclassified=validations[1];epoch++;position.scene++;
 assert(pool_frontier_prepare(1,3)&&pool_frontier_rebuilds==rebuilds);
 assert(pool_slot_binding_current(1,1)&&validations[1]==reclassified);
 proof_type[1]=VK_TEXTURE_BINDING_SOURCE;tokens[1]++;pool_member_invalidate(1);assert(pool_slot_binding_current(1,1));
 assert(pool_source_members[0]&2);reclassified=validations[1];epoch++;stale[1]=1;
 assert(!pool_slot_binding_current(1,1)&&validations[1]==reclassified+1);
 stale[1]=0;race_epoch=1;assert(!pool_slot_binding_current(1,1)); // Epoch changes during acquisition refuse publication.
 assert(pool_slot_binding_current(1,1));
 assert(pool_frontier_prepare(1,3)&&!pool_member[3].generation);high_source=validations[3];stale[3]=1;
 assert(pool_frontier_prepare(1,4)&&!pool_member[3].generation&&validations[3]==high_source+1);
 assert(pool_frontier.at[3]);stale[3]=0;assert(pool_slot_binding_current(3,1));
 // A physical selector token event still queues the group, even though scene/epoch changes no longer do.
 tokens[0]++;dirty_serial++;dirty_bits|=1;assert(pool_frontier_prepare(1,4));
 assert(!a->generation&&pool_association_frontier.count==1);publish_order();assert(!pool_order.scene);
 pool_association_clear();
 // A newly requested view kind can introduce a source dependency outside the active prefix, even if the
 // previously installed kind was DIRECT. Its receipt must join the retained source set before acknowledgement.
 active_slots=3;proof_type[3]=VK_TEXTURE_BINDING_DIRECT;pool_member_invalidate(3);
 assert(pool_slot_binding_current(3,1)&&!(pool_source_members[0]&8));
 PL.kinds=vk_pool_kinds_used=3;PL.image[VK_POOL_TEX+3]=501;extra_proof_type[3]=VK_TEXTURE_BINDING_SOURCE;
 assert(pool_slot_binding_current(3,3)&&(pool_source_members[0]&8));
 assert(pool_frontier_prepare(3,3));high_source=validations[3];epoch++;position.scene++;
 assert(pool_frontier_prepare(3,3)&&!pool_member[3].generation);stale[3]=1;
 assert(!pool_slot_binding_current(3,3)&&validations[3]==high_source+1);
 assert(pool_frontier_prepare(3,4)&&!pool_member[3].generation&&pool_frontier.at[3]);
 history_slot=3;history_status=VK_TEXTURE_REGISTRATION_EXACT;history_image=PL.image[3]=55;
 PL.image[VK_POOL_TEX+3]=0;
 unsigned checked=validations[3];
 assert(pool_slot_binding_current(3,3)&&validations[3]==checked);
 assert(!pool_frontier.at[3]);
 history_child.image=84;flat3d_pool_consuming=1;
 assert(!pool_slot_binding_current(3,3)); // Exact child not installed in selected set.
 PL.image[VK_POOL_TEX+3]=history_image;
 assert(!pool_slot_binding_current(3,3)); // The same old base is an incompatible 3D descriptor.
 PL.image[VK_POOL_TEX+3]=history_child.image;
 assert(pool_slot_binding_current(3,3)&&history_child_queries==4&&validations[3]==checked);
 TextureImagePublication archived_set={.images={[VK_POOL_TEX+3]=54}};
 PL.publication[0]=&archived_set;
 assert(!pool_slot_binding_current(3,3));
 archived_set.images[VK_POOL_TEX+3]=history_child.image;
 assert(pool_slot_binding_current(3,3));
 PL.publication[0]=NULL;
 PL.image[VK_POOL_TEX+3]=0;flat3d_pool_consuming=0;
 PL.image[3]=56;assert(!pool_slot_binding_current(3,3)); // Current or unrelated native set cannot satisfy old proof.
 history_status=VK_TEXTURE_REGISTRATION_GONE;history_image=0;PL.image[3]=0;
 assert(pool_slot_binding_current(3,3)&&validations[3]==checked); // Descriptor maintenance alone can finish.
 history_slot=-1;PL.registration_pool=39;PL.image[3]=55;stale[3]=0;pool_member[3].generation=0;
 assert(pool_slot_binding_current(3,1)&&validations[3]==checked+1);
 vk_texture_pool_logical_locked(PL.registration_pool,15);
 PL.registration_pool=0;PL.image[0]=11;PL.tok[0]=tokens[0];proof_type[0]=VK_TEXTURE_BINDING_SOURCE;
 PL.ps[0]=(PoolSlot){.selected={.registration=812,.native_slot=0,.canonical=1},.selected_kinds=1};
 unsigned old_history_queries=history_queries,old_token_queries=token_queries,old_selected_checks=selected_checks;
 slots++;authority++; // A global change is not a replacement of this selected source.
 assert(pool_slot_binding_current(0,1)&&selected_checks==old_selected_checks+1);
 assert(history_queries==old_history_queries&&token_queries==old_token_queries);
 stale[0]=1; // A changed mapping or provider proof may not restart legacy history/slot discovery.
 assert(!pool_slot_binding_current(0,1)&&selected_checks==old_selected_checks+2);
 assert(history_queries==old_history_queries&&token_queries==old_token_queries);
 stale[0]=0;
 assert(pool_slot_binding_current(0,3)&&history_queries==old_history_queries+1);
 TextureImagePublication installed={.images={11,0,0,0,67}}, wrong={.images={11,0,0,0,56}};
 PL.ps[0].selected_kinds=3;PL.image[VK_POOL_TEX]=67;
 PL.publication[0]=&installed;PL.publication[1]=&wrong;
 stale[0]=1;old_selected_checks=selected_checks;
 RcRegistryPosition edge={0};
 assert(pool_slot_binding_receipt(0,3,&edge)&&edge.ordinal==29&&selected_checks==old_selected_checks);
 assert(published_checks[0]==1&&published_checks[VK_POOL_TEX]==1&&pool_keep_selected(0,3));
 PL.publication_updates=1;
 assert(!pool_keep_selected(0,3)&&!pool_slot_binding_receipt(0,3,&edge));
 PL.publication_updates=0;PL.logn=2;
 assert(!pool_keep_selected(0,3)&&!pool_slot_binding_receipt(0,3,&edge));
 PL.cur=1;PL.set_synced[1]=2;
 assert(!pool_keep_selected(0,3)); // Indexed mismatch in the set actually selected.
 PL.cur=0;PL.set_synced[0]=2;
 assert(pool_keep_selected(0,3));
 // A new native set can carry the outgoing exact image, while the actual member and source still match.
 PL.cur=1;PL.set_synced[1]=0;pool_transfer_images=&installed;
 GraphicsPoolRestore request={.kinds=1};graphics_pool_restore=&request;
 assert(pool_keep_selected(0,3)&&pool_slot_binding_receipt(0,3,&edge)&&edge.ordinal==29);
 PL.image[VK_POOL_TEX]=56;
 assert(!pool_keep_selected(0,3)&&pool_transfer_selected_kinds(0,3)==1);
 PL.image[VK_POOL_TEX]=67;published_permission=0;
 assert(!pool_keep_selected(0,3)&&!pool_slot_binding_receipt(0,3,&edge));
 assert(pool_required_previous_kinds(0,3)==1&&pool_required_previous_kinds(1,3)==0);
 PL.image[0]=44;assert(!pool_required_previous_kinds(0,3));PL.image[0]=11;
 published_permission=1;pool_transfer_images=NULL;
 assert(!pool_required_previous_kinds(0,3));
 assert(!pool_keep_selected(0,3)); // Once the transfer ends, ownership must exist on the selected set.
 PL.publication[1]=&installed;PL.set_synced[1]=PL.logn;
 assert(pool_keep_selected(0,3));
 // Exact native set installed after catch-up. Reproduce a stale prior receipt and an absent receipt,
 // with a current selected registration in both cases.
 pool_association_clear();
 graphics_pool_restore=NULL;
 memset(pool_member,0,sizeof pool_member);
 memset(pool_source_members,0,sizeof pool_source_members);
 memset(pool_legacy_members,0,sizeof pool_legacy_members);
 pool_legacy_count=pool_legacy_total=0;
 memset(&pool_frontier_state,0,sizeof pool_frontier_state);
 memset(&PL,0,sizeof PL);
 memset(selected_slots,0,sizeof selected_slots);
 PL.ps=selected_slots;
 PL.kinds=vk_pool_kinds_used=1;
 PL.registration_publication=PL.registration_target=901;
 PL.set_synced[0]=PL.logn=13;
 PL.publication[0]=&installed;
 memset(&installed,0,sizeof installed);
 active_slots=4;position=(RcRegistryPosition){700,4};
 dirty_bits=dirty_overflow=race_mapping=race_epoch=race_legacy=0;
 for(unsigned id=0;id<4;id++){
  proof_type[id]=VK_TEXTURE_BINDING_DIRECT;stale[id]=0;edges[id]=0;
  installed.images[id]=PL.image[id]=301+id;PL.tok[id]=tokens[id];
 }
 selected_slots[0]=(PoolSlot){.selected={.registration=812,.native_slot=0},.selected_kinds=1};
 selected_slots[1]=(PoolSlot){.selected={.registration=812,.native_slot=1},.selected_kinds=1};
 assert(pool_frontier_prepare(1,4));
 assert(pool_member[0].token==tokens[0]&&pool_member[0].origin.scene==700);
 position=(RcRegistryPosition){701,44};
 pool_member_invalidate(0);
 memset(&pool_member[1],0,sizeof pool_member[1]);
 vk_pool_frontier_set(&pool_frontier,1,0);
 proof_type[1]=VK_TEXTURE_BINDING_SOURCE;edges[1]=49;
 unsigned stale_checks=validations[0], missing_checks=validations[1], steady_checks=validations[2];
 publish_order();
 assert(!pool_order.scene); // An optional publication has no active required-use acquisition.
 GraphicsPoolRestore required={.kinds=1};graphics_pool_restore=&required;
 publish_order();
 assert(pool_order_current()&&pool_order.scene==701&&pool_order.ordinal==44&&pool_order.next_ordinal==49);
 assert(pool_member[0].origin.scene==701&&pool_member[0].token==tokens[0]);
 assert(pool_member[1].origin.scene==701&&pool_member[1].token==tokens[1]&&pool_member[1].source);
 assert(validations[0]==stale_checks+1&&validations[1]==missing_checks+1&&validations[2]==steady_checks);

 position.ordinal=49;stale[1]=1;
 publish_order();
 assert(!pool_order.scene&&pool_frontier.deadline[1]==0); // No permission after a real writer.
 stale[1]=0;edges[1]=0;
 installed.images[1]=0;
 publish_order();
 assert(!pool_order.scene&&pool_frontier.at[1]); // Retained descriptor owner must actually match.
 installed.images[1]=PL.image[1];
 ++tokens[1];
 publish_order();
 assert(!pool_order.scene&&pool_frontier.at[1]); // Only a refresh can repair a changed slot identity.
 PL.tok[1]=tokens[1];
 publish_order();assert(pool_order_current());
 vk_pool_kinds_used=3;
 assert(pool_order_current_for(1)&&!pool_order_current_for(2)&&!pool_order_current());
 vk_pool_kinds_used=1;
 PL.finite_reconcile=1;
 publish_order();
 assert(!pool_order_current()); // This exact table must never be accepted by a whole-kind consumer.
 PL.finite_reconcile=0;
 graphics_pool_restore=NULL;
 puts("pool member receipts: exact installed stale or missing renewal, finite source edges, and owner changes passed");
}
'''
# Existing fixture supplies unrelated Vulkan/pool services; production refresh and dirty-member functions remain exact.
base=(ROOT/"tests/pool_retry_check.c").read_text().split('#include "pool_under_test.h"')[0]
base='#include <stdlib.h>\n#include "replay_work_trace.h"\n'+base
base=base.replace("uint64_t tok[VK_POOL_TEX]", "uint64_t authority,registration_scope,registration_pool; uint64_t tok[VK_POOL_TEX]")
base=base.replace("int ready,all", "int registration_refresh,registration_live,finite_reconcile,split_samplers; VkSampler comparison[VK_POOL_SAMP]; uint64_t registration_publication,registration_target; int ready,all")
base=base.replace("typedef struct { int head,n; uint32_t stamp,fmask; } VkSlotGroup;",
 "typedef struct { int head,n; uint32_t stamp,fmask; uint64_t incarnation; } VkSlotGroup;")
base=base.replace("static int vk_texture_pool_fresh_locked(void){return slot_count;}",
 "static unsigned global_refreshes; static int vk_texture_pool_fresh_locked(void){global_refreshes++;return slot_count;}")
base=base.replace("VkSampler samp; } Z={1}", "VkSampler samp,samp_shadow; } Z={.samp=1,.samp_shadow=2}")
base=base.replace("static VkSampler vk_sampler_by_id_locked", "#define VK_POOL_COMPARISON_BINDING 18\nstatic VkSampler vk_sampler_pool_by_id_locked(int id,int comparing){(void)id;(void)comparing;abort();}\nstatic VkSampler vk_sampler_by_id_locked")
base=base.replace("static int pool_fill(VkExec *ex", "static int slot_current[VK_POOL_TEX],persistent_failed[VK_POOL_TEX];\nstatic int pool_fill(VkExec *ex")
base=base.replace("fills[id]++;", "fills[id]++;slot_current[id]=!persistent_failed[id];")
base+=r'''
#define RT_TRACE_PERF 1
static unsigned rt_trace;
static unsigned registration_delta;
static int history_lost;
static void pool_association_detach(uint32_t id){(void)id;}
static int vk_texture_registration_changes_locked(uint64_t pool,uint64_t from,uint64_t to,void(*visit)(int,void*),void *arg){
 assert(!pool&&from!=to&&from&&to);
 if(history_lost)return 0;
 for(int id=0;id<VK_POOL_TEX;id++)if(registration_delta>>id&1)visit(id,arg);
 return 1;
}
static int vk_texture_slot_group_locked(int id){(void)id;return -1;}
static uint64_t pool_slot_token(int id){return vk_texture_slot_token(id);}
static int pool_native_slot(int id){return id;}
static int pool_visible_slots(void){return vk_texture_slot_count_locked();}
static int vk_texture_pool_primary(uint64_t pool){return pool==0;}
static int vk_texture_pool_logical_locked(uint64_t pool,int id){
 int logical=id-(pool?12:0);return logical>=0&&logical<VK_POOL_TEX?logical:-1;
}
static void vk_texture_current_registration_locked(int id,void *view,void *hv){
 vk_texture_by_id_locked(id,view,hv);
}
static uint64_t pool_member_refills;
typedef struct {uint32_t group;uint64_t incarnation;} PoolAssociation;
static PoolAssociation *pool_associations[VK_POOL_TEX];
static void pool_association_release(uint32_t owner){(void)owner;assert(0);}
static int pool_association_refresh(VkExec *ex,int group,uint32_t kinds,int count,int spent){
 (void)ex;(void)group;(void)kinds;(void)count;(void)spent;return 0;
}
static int pool_association_install_existing(uint32_t id,uint32_t kinds){(void)id;(void)kinds;return 0;}
static void *rc_registry_runtime(void){return NULL;}
static uint64_t rc_registry_authority_epoch(void *r){(void)r;return 1;}
static int pool_order_current_for(uint32_t kinds){(void)kinds;return 0;}
static int pool_slot_binding_current(uint32_t id,uint32_t kinds){(void)kinds;return slot_current[id];}
#include "vk_pool_frontier.h"
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
static VkPoolFrontier pool_frontier;
static VkPoolFrontier pool_association_frontier;
static uint32_t pool_association_work[VK_POOL_TEX];
static uint32_t pool_frontier_work[VK_POOL_TEX];static uint64_t pool_frontier_due;
static RcRegistryPosition refresh_position={1,2};
static RcRegistryPosition vk_exec_resource_position(void){return refresh_position;}
static int rc_registry_position_valid(RcRegistryPosition p){return p.scene&&p.ordinal;}
static int pool_frontier_prepare(uint32_t kinds,int count){
 (void)kinds;for(int id=0;id<count;id++)if(!slot_current[id])vk_pool_frontier_set(&pool_frontier,id,0);return 1;
}
static int pool_slot_binding_receipt(uint32_t id,uint32_t kinds,void *edge){(void)edge;return pool_slot_binding_current(id,kinds);}
static int selected_ready[VK_POOL_TEX];
static int pool_keep_selected(uint32_t id,uint32_t kinds){(void)kinds;return selected_ready[id];}
typedef struct {
    struct {uint32_t slot,kinds;} entries[VK_POOL_TEX];
    uint32_t count;
    uint8_t domain;
} VkExecPoolIndices;
typedef struct {const VkExecPoolIndices *indices;VkExec *ex;uint32_t kinds;} GraphicsPoolRestore;
static GraphicsPoolRestore *graphics_pool_restore;
static int pool_need_restore(uint32_t id,uint32_t kinds){(void)id;(void)kinds;return 0;}
static uint32_t pool_incompatible_comparison_kinds(uint32_t id,uint32_t kinds){(void)id;(void)kinds;return 0;}
typedef struct {uint64_t image;} VkTexImage;
typedef struct {const VkTexImage *image;} VkTextureSample;
typedef enum {VK_TEXTURE_SAMPLE_READY,VK_TEXTURE_SAMPLE_ABSENT,VK_TEXTURE_SAMPLE_GONE,VK_TEXTURE_SAMPLE_UNBACKED,
 VK_TEXTURE_SAMPLE_FALLBACK} VkTextureSampleStatus;
enum {VK_TEXTURE_READ_REQUIRED=1,POOL_MISSING_HISTORY=1,POOL_SOURCE_REFUSED};
static unsigned pending_alias,alias_plans,selected_resolves,selected_installs;
static const VkTexImage selected_image={.image=29};
static int pool_need_alias_locked(GraphicsPoolRestore *request,uint32_t id,uint32_t kind,int force) {
 assert(request==graphics_pool_restore&&id==2&&kind==1&&!force);
 alias_plans++;
 return pending_alias ? -1 : 0;
}
static VkTextureSampleStatus vk_texture_resolve_sample_locked(int id,unsigned flags,VkTextureSample *sample) {
 assert(id==2&&flags==VK_TEXTURE_READ_REQUIRED);
 selected_resolves++;
 sample->image=&selected_image;
 return VK_TEXTURE_SAMPLE_READY;
}
static int pool_install_selected_member(VkExec *ex,uint32_t slot,uint32_t kinds,const VkTextureSample *sample) {
 assert(ex&&slot==2&&kinds==2&&sample&&sample->image==&selected_image);
 selected_installs++;
 return 1;
}
static void pool_member_invalidate(uint32_t id){(void)id;assert(0);}
static void pool_sample_availability(uint32_t id,uint8_t reason){(void)id;(void)reason;assert(0);}
static uint32_t pool_required_previous_kinds(uint32_t id,uint32_t kinds){(void)id;(void)kinds;return 0;}
static void pool_member_schedule(uint32_t id){if(slot_current[id])vk_pool_frontier_remove(&pool_frontier,id);else vk_pool_frontier_set(&pool_frontier,id,0);}
'''
base+=source[source.index('#define POOL_DIAG_GROUPS '):source.index('#include "vk_pool_frontier.h"')]
base+="\n".join(extract(n) for n in ("pool_request_kinds","pool_refill","pool_dirty_slot","pool_dirty_group","pool_registration_slot",
                                    "pool_refresh_registration","pool_prepare_selected_member","pool_sync_selected_entries","pool_refresh_kinds","pool_sync_entries"))
base+=r'''
static void pool_sync(VkExec *ex){pool_sync_entries(ex);}
int main(void){
 VkExec ex={0};PL.ready=1;PL.authority=1;PL.kinds=PL.gen=PL.epoch=PL.sgen=1;
 rt_trace=RT_TRACE_PERF;
 PL.dirty_pos=1;vk_pool_kinds_used=gen=ep=sg=1;slot_count=VK_POOL_TEX;walk_ok=1;
 for(int i=0;i<VK_POOL_TEX;i++){slot_current[i]=1;PL.tok[i]=tokens[i]=1;}
 PoolDirty selected_dirty={&ex,1,VK_POOL_TEX,0,0};
 selected_ready[0]=1;tokens[0]=2; // Unrelated compatibility generation changed, current exact selection is valid.
 pool_dirty_slot(0,&selected_dirty);
 assert(!fills[0]&&pool_refill(&ex,0,1,2,1)&&!fills[0]);
 selected_ready[0]=0;tokens[0]=1;
 slot_current[3]=0;refresh_position=(RcRegistryPosition){0};
 pool_sync_entries(&ex);assert(!fills[3]&&!pool_member_refills);
 // A maintenance update cannot consume the pending read boundary. The next real consumer must repair it.
 refresh_position=(RcRegistryPosition){1,2};pool_sync_entries(&ex);
 assert(!global_refreshes && !PL.n_full && pool_member_refills==1 && fills[3]==1);
 for(int i=0;i<VK_POOL_TEX;i++)if(i!=3)assert(!fills[i]);
 // An unrelated authority event validates source receipts; it does not repeat global legacy discovery.
 PL.authority=2;pool_sync_entries(&ex);assert(!global_refreshes && fills[3]==1);
 // The legacy memory epoch still requests its original memory discovery path.
 ep++;pool_sync_entries(&ex);assert(global_refreshes==1);
 slot_current[3]=0;persistent_failed[3]=1;unsigned before=fills[3];uint64_t due=pool_frontier_due;
 for(unsigned k=0;k<100;k++)pool_sync_entries(&ex);
 assert(fills[3]==before+100 && pool_frontier_due==due+100 && pool_frontier.at[3]);
 unsigned total_fills=0;for(unsigned id=0;id<VK_POOL_TEX;id++)total_fills+=fills[id];
 assert(pool_diag.fills==total_fills&&pool_diag.attempts>=pool_diag.fills);
 // Image preparation succeeds even when the subsequent authority proof stays false: this is the repeated
 // successful-but-unchanged refill chain, distinct from an unfinished allocation/upload attempt.
 assert(pool_diag.unchanged==pool_diag.fills&&!pool_diag.changed&&!pool_diag.incomplete);
 assert(!pool_diag.skip_attempts&&!pool_diag.skip_fills);
 persistent_failed[3]=0;slot_current[3]=1;
 memset(fills,0,sizeof fills);
 PL.registration_refresh=1;PL.registration_publication=10;PL.registration_target=7;registration_delta=2;
 pool_sync_entries(&ex);
 assert(fills[1]==1&&!fills[0]&&!fills[2]&&!fills[3]&&!PL.wanted_reconcile);
 history_lost=1;PL.registration_refresh=1;
 pool_refresh_registration(&ex,1,VK_POOL_TEX,0);
 assert(PL.wanted_reconcile); // Lost history calls the existing complete rebuild through the selected horizon.
 // A retired or different-pool native selector has no visible member. Route a valid secondary selector.
 PL.registration_pool=2;PoolDirty dirty={&ex,1,VK_POOL_TEX,0,0};
 unsigned missing_fills=fills[0]+fills[1]+fills[2]+fills[3];
 pool_dirty_slot(2,&dirty);pool_dirty_slot(16,&dirty);
 assert(fills[0]+fills[1]+fills[2]+fills[3]==missing_fills && !dirty.retry);
 slot_current[1]=0;
 pool_dirty_slot(13,&dirty);
 assert(fills[1]==2&&fills[0]+fills[1]+fills[2]+fills[3]==missing_fills+1);
 // An authenticated subset updates only its actual slot/kind pair, even when broad reconciliation is due.
 memset(fills,0,sizeof fills);
 memset(filled,0,sizeof filled);
 PL.registration_pool=0;
 PL.wanted_reconcile=1;
 vk_pool_kinds_used=3;
 const VkExecPoolIndices indices={.entries={{2,2}},.count=1};
 GraphicsPoolRestore selected={.indices=&indices,.ex=&ex,.kinds=2};
 graphics_pool_restore=&selected;
 unsigned refreshes_before=global_refreshes;
 pool_sync_entries(&ex);
 assert(PL.finite_reconcile && !PL.wanted_reconcile && selected_resolves==1 && selected_installs==1 && alias_plans==1);
 assert(!fills[0] && !fills[1] && !fills[2] && !fills[3] && global_refreshes==refreshes_before);
 pending_alias=1;
 PL.wanted_reconcile=1;
 pool_sync_entries(&ex);
 assert(!PL.wanted_reconcile && alias_plans==2 && selected_resolves==1 && selected_installs==1);
 graphics_pool_restore=NULL;
 (void)pool_sync_calls;(void)lock_calls;(void)unresolved_calls;
 puts("pool refresh: logical boundary refills only the expired member without global dependency traversal");
}
'''
env=dict(os.environ,PATH=str(CC.parent)+os.pathsep+os.environ.get("PATH",""))
with tempfile.TemporaryDirectory(prefix="pool-receipts-",dir=ROOT/"build/test") as tmp:
 for name,text in (("member",member),("refresh",base)):
  p=Path(tmp)/(name+".c");p.write_text(text);exe=p.with_suffix(".exe")
  subprocess.run([str(CC),"-std=gnu11","-O1","-Wall","-Wextra","-Werror","-Wno-unused-function","-Wno-unused-variable",
   "-fsanitize=undefined","-fsanitize-trap=all","-I",str(ROOT/"runtime"),str(p),"-o",str(exe)],env=env,check=True)
  subprocess.run([str(exe)],env=env,check=True)

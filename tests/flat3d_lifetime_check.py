"""Check the production derived-image version, layer, recording and retirement policy."""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_content import GENERATED_MARKER

SOURCE = (ROOT / "runtime/vk_texture.c").read_text()


def function(name):
    match = re.search(r"^(?:static )?[^\n]*\b" + name + r"\([^;]*?\)\s*\{", SOURCE, re.M)
    if not match:
        raise RuntimeError(name)
    begin = SOURCE.index("{", match.start())
    depth, end = 1, begin + 1
    while depth:
        depth += (SOURCE[end] == "{") - (SOURCE[end] == "}")
        end += 1
    return SOURCE[match.start():end]


PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "resource_core_registry.h"
typedef uint64_t VkImage,VmaAllocation,VkCommandBuffer;
typedef struct { uint64_t image,addr; uint32_t format,type,view_type,samples,width,height,depth,levels,layers,
    aspect,usage,bound,block_views,fb_usage,fb_learned,gone,gpu_owned; } VkTexImage;
typedef struct {uint32_t width,height,depth;} Extent;
typedef struct {int32_t x,y,z;} Offset;
typedef struct {uint32_t aspectMask,mipLevel,baseArrayLayer,layerCount;} Subresource;
typedef struct {uint32_t aspectMask,baseMipLevel,levelCount,baseArrayLayer,layerCount;} SubRange;
typedef struct {int sType,imageType;uint32_t format;Extent extent;uint32_t mipLevels,arrayLayers,samples,tiling,usage,initialLayout;} VkImageCreateInfo;
typedef struct {int usage;} VmaAllocationCreateInfo;
typedef struct {int sType;uint64_t srcStageMask,dstStageMask,srcAccessMask,dstAccessMask;
 int oldLayout,newLayout;uint32_t srcQueueFamilyIndex,dstQueueFamilyIndex;VkImage image;SubRange subresourceRange;} VkImageMemoryBarrier2;
typedef struct {int sType;uint32_t imageMemoryBarrierCount;const VkImageMemoryBarrier2*pImageMemoryBarriers;} VkDependencyInfo;
typedef struct {Subresource srcSubresource;Offset srcOffset;Subresource dstSubresource;Offset dstOffset;Extent extent;} VkImageCopy;
enum {VK_IMAGE_TYPE_2D=2,VK_IMAGE_TYPE_3D=3,VK_IMAGE_VIEW_TYPE_3D=3,VK_IMAGE_ASPECT_COLOR_BIT=1,
 VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO=1,VK_SAMPLE_COUNT_1_BIT=1,VK_IMAGE_TILING_OPTIMAL=1,
 VK_IMAGE_USAGE_SAMPLED_BIT=1,VK_IMAGE_USAGE_TRANSFER_DST_BIT=2,VK_IMAGE_LAYOUT_UNDEFINED=0,
 VK_IMAGE_LAYOUT_GENERAL=1,VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE=1,VK_SUCCESS=0,
 VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2=2,VK_PIPELINE_STAGE_2_NONE=0,VK_PIPELINE_STAGE_2_COPY_BIT=1,
 VK_ACCESS_2_TRANSFER_READ_BIT=1,VK_ACCESS_2_TRANSFER_WRITE_BIT=2,VK_QUEUE_FAMILY_IGNORED=~0u,
 VK_REMAINING_MIP_LEVELS=~0u,VK_STRUCTURE_TYPE_DEPENDENCY_INFO=3};
#define VK_NULL_HANDLE 0
#define FLAT3D_MAX (1024u*1024u)
#define ALL_RW 8,16
#define VK_IMAGE_USAGE_TRANSFER_SRC_BIT 4
#define VK_NOT_READY 1
typedef struct {uint32_t format,type;} ImgKey;
#include "vk_texture_material.h"
typedef struct TextureUseNode {TextureMaterialUse use;} TextureUseNode;
static TextureMaterialLedger material_ledger;
/* Production owner declaration. */
typedef struct Tex {VkTexImage pub;struct {uint64_t addr;} key;uint64_t content_revision;
 uint32_t n_up,material_writable;int storage;TextureFlat3D *flat3d;TextureMaterial *material;struct Tex *next_at;} Tex;
typedef struct {int native_slot;uint64_t registration;int canonical;} VkTextureReadKey;
typedef struct {int dummy;} TexSlot;
typedef struct {uint64_t size;} VmaAllocationInfo;
static struct {int vma;} context;
static TexSlot slot;
static int by_addr,tex_deferred,turn_due,fail_allocation,no_legacy_lookup;
static Tex *owner;
static volatile int32_t n_flat3d;
static unsigned destroyed_unused;
static RcRegistryPosition position={1,2};
static uint64_t next_image=100,expected_cb=77,scene=1,completed;
static unsigned allocations,copies,barriers,generation,purges,retired_images,retired_nodes,destructions;
static VkImage last_source,last_destination;
static uint32_t last_layer,last_mips;
static struct {TextureFlat3D *node;uint64_t scene;} retired[64];
static void *nvn_hm_get(const int *map,const uint64_t *address){
 assert(map==&by_addr&&!no_legacy_lookup);return owner&&owner->pub.addr==*address?owner:NULL;
}
static const TexSlot *texture_read_key_slot(VkTextureReadKey key){
 return key.registration==19&&key.native_slot==4 ? &slot:NULL;
}
static Tex *texture_read_key_source(const TexSlot *s,VkImage image){
 return s==&slot&&owner->pub.image==image ? owner:NULL;
}
static RcRegistryPosition handed;
RcRegistry *rc_registry_runtime(void){return NULL;}
int rc_registry_position_handed(RcRegistry *registry,RcRegistryPosition p){
 (void)registry;return p.scene<handed.scene||(p.scene==handed.scene&&p.ordinal<=handed.ordinal);
}
static void img_key_of(const VkImageCreateInfo *ci,ImgKey *key){*key=(ImgKey){ci->format,ci->imageType};}
static void vmaGetAllocationInfo(int vma,VmaAllocation allocation,VmaAllocationInfo *out){
 assert(vma==context.vma&&allocation);out->size=4096;
}
static void resource_core_vma_destroy_image(int vma,VkImage image,VmaAllocation allocation){
 assert(vma==context.vma&&image==allocation);
 destroyed_unused++;
}
static TextureMaterialInstance *mapped[256];
static TextureMaterialUse copy_uses[2];
static void content_use_image_add(TextureMaterialInstance *i){assert(!mapped[i->value.image]);mapped[i->value.image]=i;}
void vk_texture_use_images(const VkImage *images,size_t count){
 assert(count==2&&images[0]!=images[1]);
 for(size_t i=0;i<count;i++){
  assert(mapped[images[i]]);
  TextureMaterialInstance *m=mapped[images[i]];
  TextureMaterialUse *prior=m->use_head?&m->use_head->use:NULL;
  RcLifetimeResult result=prior?texture_material_use_fork(prior,&copy_uses[i]):
      texture_material_use_begin(m->material,&copy_uses[i]);
  assert(result==RC_LIFETIME_OK);
 }
}
static uint64_t content_retire(TextureMaterial *m){return texture_material_retire(m).bytes;}
static int from_storage(const Tex *e){return e->storage;}
static uint32_t mip(uint32_t side,unsigned level){return side>>level ? side>>level:1;}
static int format_compressed(unsigned f){return f==99;}
static int rt_knob(const char*n,const char*d){(void)n;(void)d;return 1;}
static RcRegistryPosition vk_exec_resource_position(void){return position;}
static int vk_exec_turn_due(void){return turn_due;}
static typeof(context) *vk_ctx(void){return &context;}
static void rt_fatal(const char*s){fputs(s,stderr);abort();}
static void log_limited(volatile int32_t*n,const char*s,...){(void)n;(void)s;}
static void tex_gen_bump(uint64_t address){assert(address==owner->pub.addr);generation++;}
static int texture_image_allocate(void*c,const VkImageCreateInfo*ci,const VmaAllocationCreateInfo*ai,
 VkImage*out,VmaAllocation*allocation,int required,const char*reason,void*extra){
 (void)required;(void)reason;(void)extra;assert(c==&context&&ai->usage==VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
 assert(ci->imageType==VK_IMAGE_TYPE_3D&&ci->extent.depth==1&&ci->arrayLayers==1&&ci->mipLevels==3);
 allocations++;if(fail_allocation)return -1;*out=++next_image;*allocation=*out;return VK_SUCCESS;
}
static void vkCmdPipelineBarrier2(VkCommandBuffer cb,const VkDependencyInfo*di){
 assert(cb==expected_cb&&di->imageMemoryBarrierCount==1&&di->pImageMemoryBarriers->oldLayout==VK_IMAGE_LAYOUT_UNDEFINED);barriers++;
}
static void barrier(VkCommandBuffer cb,int a,int b,int c,int d){(void)a;(void)b;(void)c;(void)d;assert(cb==expected_cb);barriers++;}
static void vkCmdCopyImage(VkCommandBuffer cb,VkImage source,int sl,VkImage destination,int dl,
 uint32_t count,const VkImageCopy*regions){
 assert(cb==expected_cb&&sl==VK_IMAGE_LAYOUT_GENERAL&&dl==VK_IMAGE_LAYOUT_GENERAL&&source!=destination);
 assert(count==3);for(uint32_t i=0;i<count;i++){
 assert(regions[i].srcSubresource.mipLevel==i&&regions[i].srcSubresource.baseArrayLayer==regions[0].srcSubresource.baseArrayLayer);
 assert(regions[i].dstSubresource.mipLevel==i&&!regions[i].dstSubresource.baseArrayLayer&&regions[i].extent.depth==1);
 assert(regions[i].extent.width==(32u>>i)&&regions[i].extent.height==(16u>>i));}
 copies++;last_source=source;last_destination=destination;last_layer=regions[0].srcSubresource.baseArrayLayer;last_mips=count;
}
static void vk_exec_image_gone(VkImage image){assert(image>100);purges++;}
static void vk_defer_image(VkImage image,VmaAllocation allocation){assert(image==allocation);retired_images++;}
static void vk_defer_free(void*pointer){
 TextureFlat3D*copy=pointer;assert(copy->image.gone);retired[retired_nodes++]=(typeof(retired[0])){copy,scene};
}
static void complete(uint64_t fence){
 completed=fence;for(unsigned i=0;i<retired_nodes;i++)if(retired[i].node&&retired[i].scene<=completed){free(retired[i].node);retired[i].node=NULL;destructions++;}
}
'''

HISTORY_PREFIX = r'''
enum {VK_TEXTURE_REGISTRATION_CURRENT,VK_TEXTURE_REGISTRATION_EXACT,VK_TEXTURE_REGISTRATION_GONE,VK_TEXTURE_REGISTRATION_ABSENT};
typedef enum {VK_TEXTURE_SAMPLE_READY,VK_TEXTURE_SAMPLE_FALLBACK,VK_TEXTURE_SAMPLE_GONE,
    VK_TEXTURE_SAMPLE_ABSENT,VK_TEXTURE_SAMPLE_PREPARE,VK_TEXTURE_SAMPLE_UNAVAILABLE} VkTextureSampleStatus;
typedef struct VkTextureRegistrationScope {unsigned refs;} VkTextureRegistrationScope;
typedef struct VkTextureHistorical3D VkTextureHistorical3D;
typedef struct {VkTexImage exact;RcRegistryHandle handle;TextureFlat3D *flat3d;uint32_t layer;} TextureRegistration;
typedef struct {int selected;} RcVmaDependency;
typedef struct {int required,producer;RcVmaDependency *dependency;} RcVmaRequest;
typedef int VkResult;
enum {RC_BUDGET_TEXTURE=1};
static TextureRegistration old_record;
static VkTextureRegistrationScope old_scope={1},new_scope={1},*selected_scope=&old_scope;
static TextureUseNode archive_owner;
#define archive_use archive_owner.use
struct TextureImagePublication {TextureMaterialUse *selected;};
static TextureImagePublication archived_base={&archive_use};
static struct {unsigned nodes;} texture_uses;
static int lock_depth,old_record_valid=1,archived_owner_valid=1;
static void lock(void){lock_depth++;}
static void unlock(void){assert(lock_depth);lock_depth--;}
static uint32_t texture_registration_exact_layer(const TextureRegistration *record){return record->layer;}
static VkTextureRegistrationScope *vk_exec_texture_registration_scope(void){return selected_scope;}
static VkTextureRegistrationScope *vk_texture_registration_scope_retain(VkTextureRegistrationScope *scope){
 assert(scope==selected_scope&&scope->refs);scope->refs++;return scope;
}
static void vk_texture_registration_scope_end(VkTextureRegistrationScope *scope){assert(scope->refs>1);scope->refs--;}
static int texture_registration_resolve(int logical,const TextureRegistration **record,int *native){
 assert(!native&&logical==6);
 *record=&old_record;
 return selected_scope==&old_scope?(old_record_valid&&archived_owner_valid?VK_TEXTURE_REGISTRATION_EXACT:VK_TEXTURE_REGISTRATION_GONE):
      VK_TEXTURE_REGISTRATION_CURRENT;
}
static int texture_registration_resolve_owner(int logical,const TextureRegistration **record,int *native,
                                                const TextureUseNode **retained) {
 int status=texture_registration_resolve(logical,record,native);
 *retained=status==VK_TEXTURE_REGISTRATION_EXACT?&archive_owner:NULL;
 return status;
}
static void texture_use_scene_material_locked(TextureMaterialInstance *instance,const TextureMaterialUse *previous) {
 if(previous){
    assert(instance==archive_use.instance && previous->instance==instance && previous!=&archive_use);
    assert(texture_material_use_fork(previous,&copy_uses[0])==RC_LIFETIME_OK);
 }else{
    assert(instance!=archive_use.instance);
    assert(texture_material_use_begin(instance->material,&copy_uses[1])==RC_LIFETIME_OK);
 }
}
static const TextureMaterialInstance *texture_publication_image_locked(const TextureImagePublication *set,uint32_t descriptor,VkImage expected){
 if(set==&archived_base)assert(descriptor==old_record.handle.index);
 else assert(descriptor==51);
 return set->selected->instance&&set->selected->instance->value.image==expected?set->selected->instance:NULL;
}
static RcLifetimeResult texture_use_retain_instance(TextureUseNode *target,TextureMaterialInstance *source,
                                                     const TextureMaterialUse *previous){
 assert(previous==&archive_use&&source==archive_use.instance);
 texture_uses.nodes++;
 return texture_material_use_fork(&archive_use,&target->use);
}
static void texture_use_unlink(TextureUseNode *node){assert(texture_uses.nodes&&node->use.instance==archive_use.instance);}
static void content_defer_instance(TextureMaterialInstance *source){assert(source);}
static VkResult resource_core_vma_create_image(const RcVmaRequest *request,int vma,const VkImageCreateInfo *recipe,
 const VmaAllocationCreateInfo *placement,VkImage *out,VmaAllocation *allocation,void *info,void *decision){
 assert(vma==context.vma&&request->required&&request->producer==RC_BUDGET_TEXTURE&&!info&&!decision);
 *out=0;*allocation=0;request->dependency->selected=1;
 if(fail_allocation)return VK_NOT_READY;
 return texture_image_allocate(&context,recipe,placement,out,allocation,1,NULL,NULL);
}
void vk_texture_history_flat3d_cancel(VkTextureHistorical3D *);
'''

TESTS = r'''
static unsigned count_nodes(Tex *e){unsigned count=0;for(TextureFlat3D*p=e->flat3d;p;p=p->next)count++;return count;}
int main(void){
 Tex source={.pub={.image=7,.addr=1000,.format=1,.type=VK_IMAGE_TYPE_2D,.samples=1,.width=32,.height=16,
 .depth=1,.levels=3,.layers=2,.aspect=VK_IMAGE_ASPECT_COLOR_BIT},.key.addr=1000,.storage=1,.n_up=1};
 owner=&source;VkTexImage captured=source.pub; // Not a Tex pointer; resolve ownership by exact address+image.
 assert(!vk_texture_flat3d_locked(&captured,0,0,0)&&!tex_deferred&&!allocations);
 assert(!vk_texture_flat3d_locked(&captured,0,0,2)&&tex_deferred&&!allocations);tex_deferred=0;
 const VkTexImage *a=vk_texture_flat3d_locked(&captured,0,expected_cb,1);assert(a&&a->image!=7&&copies==1&&barriers==3);
 assert(last_source==7&&last_destination==a->image&&last_layer==0&&last_mips==3&&generation==1);
 assert(a->type==VK_IMAGE_TYPE_3D&&a->layers==1&&a->depth==1&&!a->gone);
 assert(vk_texture_flat3d_locked(&captured,0,0,1)==a&&copies==1);
 const VkTexImage *b=vk_texture_flat3d_locked(&captured,1,expected_cb,1);assert(b&&b!=a&&last_layer==1&&count_nodes(&source)==2);
 position.ordinal=4;assert(vk_texture_flat3d_locked(&captured,0,0,2)==a&&!retired_nodes); // Stable CPU version spans operations.
 source.n_up++;assert(!vk_texture_flat3d_locked(&captured,0,0,0)&&!tex_deferred);
 fail_allocation=1;assert(!vk_texture_flat3d_locked(&captured,0,expected_cb,1)&&tex_deferred&&!retired_nodes&&!a->gone);
 fail_allocation=tex_deferred=0;scene=2;
 const VkTexImage *a2=vk_texture_flat3d_locked(&captured,0,expected_cb,1);
 assert(a2!=a&&a->gone&&retired_nodes==1&&purges==1&&retired_images==1&&!destructions&&count_nodes(&source)==2);
 complete(1);assert(!destructions&&a->image>100);complete(2);assert(destructions==1);
 // Opening the consuming pass announces its first GPU write after the preflight CPU snapshot.
 source.pub.gpu_owned=1;source.content_revision=1;
 assert(vk_texture_flat3d_locked(&captured,0,0,1)==a2);
 assert(!vk_texture_flat3d_locked(&captured,0,0,2)&&tex_deferred);tex_deferred=0;
 const VkTexImage *gpu=vk_texture_flat3d_locked(&captured,0,expected_cb,1);assert(gpu!=a2&&source.flat3d->ordered);
 source.content_revision++;assert(vk_texture_flat3d_locked(&captured,0,0,1)==gpu); // Pass-begin notification only.
 assert(!vk_texture_flat3d_locked(&captured,0,0,2)&&tex_deferred);tex_deferred=0;
 const VkTexImage *written=vk_texture_flat3d_locked(&captured,0,expected_cb,1);assert(written!=gpu); // Next internal dispatch re-preflights.
 position.ordinal=6;assert(!vk_texture_flat3d_locked(&captured,0,0,1)&&tex_deferred);tex_deferred=0;
 unsigned before=copies;turn_due=1;assert(!vk_texture_flat3d_locked(&captured,0,expected_cb,1)&&tex_deferred&&copies==before);
 turn_due=tex_deferred=0;const VkTexImage *later=vk_texture_flat3d_locked(&captured,0,expected_cb,1);assert(later!=written);
 // Storage writes keep legacy gpu_owned false but still require an operation-scoped derivative.
 source.pub.gpu_owned=0;source.material_writable=1;source.content_revision++;
 const VkTexImage *storage=vk_texture_flat3d_locked(&captured,1,expected_cb,1);assert(storage!=b&&source.flat3d->next->ordered);
 position.ordinal=8;assert(!vk_texture_flat3d_locked(&captured,1,0,2)&&tex_deferred);tex_deferred=0;
 // Replaced physical source cannot be mistaken for the captured value at the same address.
 source.pub.image=8;assert(!vk_texture_flat3d_locked(&captured,0,expected_cb,1)&&!tex_deferred);
 captured=source.pub;assert(vk_texture_flat3d_locked(&captured,0,expected_cb,1)&&last_source==8);
 captured.type=VK_IMAGE_TYPE_3D;assert(!vk_texture_flat3d_locked(&captured,0,0,2)&&!tex_deferred);
 captured=source.pub;assert(!vk_texture_flat3d_locked(&captured,2,0,2)&&!tex_deferred);
 captured.format=99;assert(!vk_texture_flat3d_locked(&captured,0,0,2)&&!tex_deferred);
 // The actual image_texels tail changes the version only if a copy or ASTC decode was recorded.
 uint64_t revision=source.content_revision;
 texels_write_version(&source,1,0);assert(source.content_revision==revision);
 texels_write_version(&source,1,-1);assert(source.content_revision==revision);
 texels_write_version(&source,1,1);assert(source.content_revision==revision+1);
 texels_write_version(&source,0,0);assert(source.content_revision==revision+2);
 while(source.flat3d){TextureFlat3D*p=source.flat3d;source.flat3d=p->next;flat3d_retire(p);}
 complete(scene);assert(destructions==retired_nodes&&retired_images==retired_nodes&&purges==retired_nodes);

 TextureMaterial *parent=malloc(sizeof *parent);
 assert(texture_material_init(parent,&material_ledger,source.pub.image,source.pub.image,&(ImgKey){1,2},4096,1));
 parent->ready=1;content_use_image_add(parent->resident.instance);source.material=parent;
 source.content_revision=source.material_writable=0;source.storage=1;tex_deferred=0;
 no_legacy_lookup=1;scene=3;position=(RcRegistryPosition){3,4};captured=source.pub;
 VkTextureReadKey exact={.native_slot=4,.registration=19,.canonical=1};
 VkTextureReadKey wrong={.native_slot=4,.registration=20,.canonical=1};
 assert(!vk_texture_flat3d_selected_locked(&captured,wrong,0,expected_cb,1)&&tex_deferred);
 tex_deferred=0;
 const VkTexImage *derived=vk_texture_flat3d_selected_locked(&captured,exact,1,expected_cb,1);
 assert(derived&&last_layer==1&&copy_uses[0].instance==parent->resident.instance);
 TextureMaterial *child=source.flat3d->material;
 assert(child&&child->resident.image==derived->image&&!source.flat3d->allocation);
 assert(child->lifetime.recovery==RC_LIFETIME_RECOVERY_NONE&&copy_uses[1].instance==child->resident.instance);
 TextureMaterialUse sealed={0};
 assert(texture_material_use_fork(&copy_uses[1],&sealed)==RC_LIFETIME_OK);
 RcLifetimeCompletion done={0},early={0};
 rc_lifetime_completion_add(&done,TEXTURE_MATERIAL_GRAPHICS_DOMAIN,3);
 rc_lifetime_completion_add(&early,TEXTURE_MATERIAL_GRAPHICS_DOMAIN,2);
 for(int i=0;i<2;i++)assert(texture_material_use_submit(&copy_uses[i],0,&done)==RC_LIFETIME_OK);
 assert(texture_material_use_submit(&sealed,0,&done)==RC_LIFETIME_OK);
 position=(RcRegistryPosition){4,2};
 assert(!vk_texture_flat3d_selected_locked(&captured,exact,1,0,1)&&tex_deferred);
 handed=(RcRegistryPosition){3,4};tex_deferred=0;
 assert(vk_texture_flat3d_selected_locked(&captured,exact,1,0,1)==derived&&!tex_deferred);
 ++source.n_up;
 assert(!vk_texture_flat3d_selected_locked(&captured,exact,1,0,1)&&tex_deferred);
 TextureMaterialInstance *old_parent=copy_uses[0].instance,*old_child=copy_uses[1].instance;
 flat3d_retire(source.flat3d);source.flat3d=NULL;
 assert(texture_material_release(parent,1));content_retire(parent);
 source.material=NULL;source.pub.image++;
 assert(!vk_texture_flat3d_selected_locked(&captured,exact,1,0,1));
 assert(!texture_material_instance_defer_ready(old_parent)&&!texture_material_instance_defer_ready(old_child));
 assert(texture_material_use_complete(&copy_uses[1],0,&early)==RC_LIFETIME_BUSY);
 for(int i=0;i<2;i++)assert(texture_material_use_complete(&copy_uses[i],0,&done)==RC_LIFETIME_OK);
 assert(texture_material_instance_defer_ready(old_parent)&&!texture_material_instance_defer_ready(old_child));
 assert(texture_material_use_complete(&sealed,0,&done)==RC_LIFETIME_OK);
 assert(texture_material_instance_defer_ready(old_child));
 complete(scene);

 VkImage original=9;
 parent=malloc(sizeof *parent);
 assert(texture_material_init(parent,&material_ledger,original,original,&(ImgKey){1,2},4096,1));
 parent->ready=1;content_use_image_add(parent->resident.instance);
 assert(texture_material_use_begin(parent,&archive_use)==RC_LIFETIME_OK);
 // The old parent is detached and survives exclusively through its archive's physical owner.
 parent->resident.instance->use_head=&archive_owner;
 assert(texture_material_release(parent,1));content_retire(parent);
 old_record=(TextureRegistration){.exact=source.pub,.handle={99,6},.layer=1};
 old_record.exact.image=original;old_record.exact.usage|=VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
 source.pub.image=13;scene=5;position=(RcRegistryPosition){5,8};
 const VkTexImage *archived_child=NULL;
 assert(vk_texture_history_flat3d_locked(6,&archived_child,NULL)==VK_TEXTURE_SAMPLE_PREPARE&&!archived_child);
 VkTextureHistorical3D *plan=NULL;
 assert(vk_texture_history_flat3d_locked(6,NULL,&plan)==VK_TEXTURE_SAMPLE_PREPARE&&plan);
 assert(old_scope.refs==2&&texture_uses.nodes==1);
 assert(vk_texture_history_flat3d_current(plan));
 archived_owner_valid=0;
 assert(!vk_texture_history_flat3d_current(plan));
 assert(!vk_texture_history_flat3d_record_locked(plan,expected_cb));
 archived_owner_valid=1;
 RcVmaDependency dependency={0};
 before=copies;fail_allocation=1;
 assert(vk_texture_history_flat3d_prepare(plan,&dependency)==VK_NOT_READY&&dependency.selected&&copies==before);
 assert(!vk_texture_history_flat3d_record_locked(plan,expected_cb)&&copies==before);
 fail_allocation=0;
 assert(vk_texture_history_flat3d_prepare(plan,&dependency)==VK_SUCCESS&&copies==before&&!old_record.flat3d);
 selected_scope=&new_scope;
 assert(!vk_texture_history_flat3d_current(plan));
 assert(!vk_texture_history_flat3d_record_locked(plan,expected_cb)&&copies==before);
 vk_texture_history_flat3d_cancel(plan);plan=NULL;
 assert(old_scope.refs==1&&!texture_uses.nodes&&destroyed_unused==1);
 selected_scope=&old_scope;
 assert(vk_texture_history_flat3d_locked(6,NULL,&plan)==VK_TEXTURE_SAMPLE_PREPARE&&plan);
 assert(vk_texture_history_flat3d_prepare(plan,&dependency)==VK_SUCCESS);
 assert(vk_texture_history_flat3d_record_locked(plan,expected_cb)&&copies==before+1&&last_source==original);
 assert(vk_texture_history_flat3d_record_locked(plan,expected_cb)&&copies==before+1);
 vk_texture_history_flat3d_cancel(plan);
 assert(vk_texture_history_flat3d_locked(6,&archived_child,NULL)==VK_TEXTURE_SAMPLE_READY&&archived_child->image==last_destination);
 TextureImagePublication installed={&copy_uses[1]};
 assert(vk_texture_history_flat3d_installed_locked(6,&installed,51,archived_child->image));
 assert(!vk_texture_history_flat3d_installed_locked(6,&archived_base,old_record.handle.index,archived_child->image));
 position=(RcRegistryPosition){6,1};
 assert(vk_texture_history_flat3d_locked(6,&archived_child,NULL)==VK_TEXTURE_SAMPLE_PREPARE);
 handed=(RcRegistryPosition){5,8};
 assert(vk_texture_history_flat3d_locked(6,&archived_child,NULL)==VK_TEXTURE_SAMPLE_READY);
 old_record_valid=0;
 assert(vk_texture_history_flat3d_locked(6,NULL,NULL)==VK_TEXTURE_SAMPLE_GONE);
 flat3d_retire(old_record.flat3d);
 assert(texture_material_use_cancel(&archive_use)==RC_LIFETIME_OK);
 assert(!texture_material_instance_defer_ready(copy_uses[0].instance));
 for(int i=0;i<2;i++)assert(texture_material_use_cancel(&copy_uses[i])==RC_LIFETIME_OK);
 complete(scene);
 assert(old_scope.refs==1&&!texture_uses.nodes);
 puts("flat3d: ordinary exact parent and child survive address reuse and true completion; compatibility layer and ordered versions passed");
}
'''


def main():
    start = SOURCE.index("typedef struct TextureFlat3D {")
    owner = SOURCE[start:SOURCE.index("} TextureFlat3D;", start) + len("} TextureFlat3D;")]
    prefix = PREFIX.replace("/* Production owner declaration. */", owner)
    body = "\n".join(function(name) for name in ("flat3d_retire", "flat3d_current", "flat3d_material_attach",
        "flat3d_recipe", "flat3d_record", "texture_flat3d_prepare", "vk_texture_flat3d_locked",
        "vk_texture_flat3d_selected_locked"))
    history_start = SOURCE.index("struct VkTextureHistorical3D {")
    history = SOURCE[history_start:SOURCE.index("static const VkTexImage *texture_by_slot_locked(", history_start)]
    # Every actual image writer must distinguish versions even when the scene fence is unchanged.
    for name in ("gpu_written_prepared", "vk_texture_storage_bound_locked"):
        assert "++e->content_revision" in function(name), name
    texels = function("image_texels")
    start = texels.index("    if (!agpu || done > 0) {")
    end = texels.index("\n    }", start) + len("\n    }")
    assert texels.index("vkCmdCopyBufferToImage") < start < texels.index("vk_upload_end()")
    version = ("static void resource_image_written(uint64_t image,uint64_t addr){(void)image;(void)addr;}\n"
               "static void texels_write_version(Tex *e,int agpu,int done){\n" + texels[start:end] + "\n}\n")
    cc = Path(os.environ.get("CC", "C:/Users/Andrew/AppData/Local/Programs/2NX2Navi/toolchain/msys64/ucrt64/bin/clang.exe"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    work = ROOT / "build/test"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flat3d-lifetime-", dir=work) as directory:
        code = Path(directory) / "check.c"
        marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
        code.write_text("// " + marker + "\n" + prefix + body + HISTORY_PREFIX + history + version + TESTS)
        exe = code.with_suffix(".exe")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                        "-fsanitize-trap=all", "-I", str(ROOT / "runtime"), str(code), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

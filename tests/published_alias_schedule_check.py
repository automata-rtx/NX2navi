"""Check the native bindless alias receipt against the receiver view and supplying material."""
from pathlib import Path
import os
import subprocess
import tempfile

from ordered_graphics_feedback_check import function

ROOT = Path(__file__).resolve().parents[1]
contents = {name: (ROOT / 'runtime' / name).read_text() for name in ('vk_texture.c', 'vk_exec_draw.c')}
prefix=r'''
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#define VK_POOL_TEX 2
#define VK_POOL_KINDS 2
typedef uintptr_t VkImage,VkImageView;
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
typedef struct {uint64_t registration;int native_slot,canonical;} VkTextureReadKey;
typedef struct {uint32_t format,swizzle_r,swizzle_g,swizzle_b,swizzle_a;} nvn_texdesc;
typedef struct {uint32_t min_layer,set_swizzle;uint8_t swz[4];} view_t;
typedef struct {VkImage image;uint32_t format,layers;uint8_t swz[4];} VkTexImage;
typedef struct {uint64_t serial,addr;uint32_t layout;nvn_texdesc d;struct {uint64_t size;} L;} tex_t;
typedef struct {uint64_t addr;uint32_t layout,format;} TexKey;
typedef struct TextureMaterial TextureMaterial;
typedef struct {VkImage image;TextureMaterial *material;} TextureMaterialInstance;
struct TextureMaterial {
    struct {TextureMaterialInstance *instance;} resident;
    int ready,retired,input;
    void *provider;
};
typedef struct {
    tex_t t; TexKey key; VkTexImage pub; TextureMaterial *material;
    int released,retiring,superseded,source_canonical,uploaded,up_retry,mutable;
} Tex;
typedef struct {tex_t descriptor;view_t view;int has_view;Tex *source;} TextureRegistration;
typedef struct {TextureRegistration *registration;Tex *ce;tex_t t;view_t v;int has_view;} TexSlot;
typedef struct {TextureMaterialInstance *entries[VK_POOL_TEX*VK_POOL_KINDS];} TextureImagePublication;
typedef struct {VkTextureReadKey selected;} PoolSlot;
typedef struct {VkImage image;uint32_t format,swz,kind,layer;} ViewKey;
static struct {VkImage image[VK_POOL_TEX*VK_POOL_KINDS];VkImageView view[VK_POOL_TEX*VK_POOL_KINDS];} PL;
static struct {int klen;} view_of={1};
enum {VK_TEXTURE_BINDING_SOURCE=2,VK_TEXTURE_REGISTRATION_CURRENT=0,VK_TEXTURE_REGISTRATION_EXACT=1};
enum {NVN_TEXTURE_SWIZZLE_R=2,NVN_TEXTURE_SWIZZLE_G,NVN_TEXTURE_SWIZZLE_B,NVN_TEXTURE_SWIZZLE_A};
static RcRegistryPosition read;
static Tex supplier;
static TextureRegistration receiver;
static TexSlot slot;
static uint64_t live_registration=160;
static int current_generation=1,native_slot=8,historical,installed_ready=1,source_open=1;
static RcRegistryPosition supplier_edge;
static TextureImagePublication current_set;
static const TextureImagePublication *pool_transfer_images;
static const TextureImagePublication *pool_member_installed_images(void){return installed_ready?&current_set:NULL;}
static RcRegistryPosition vk_exec_resource_position(void){return read;}
static int rc_registry_position_valid(RcRegistryPosition at){return at.scene&&at.ordinal;}
static int rc_registry_position_compare(RcRegistryPosition a,RcRegistryPosition b){
 return a.scene<b.scene?-1:a.scene>b.scene?1:a.ordinal<b.ordinal?-1:a.ordinal>b.ordinal?1:0;
}
static const TexSlot *texture_read_key_slot(VkTextureReadKey key){
 return key.native_slot==native_slot&&key.registration==live_registration?&slot:NULL;
}
static int texture_registration_resolve(int logical,const TextureRegistration **history,int *native){
 assert(logical==1);*native=native_slot;*history=historical?&receiver:NULL;return historical;
}
static int texture_registration_ordinary(const tex_t *t){return t->layout!=0;}
static Tex *texture_binding_entry(const TexSlot *s,int native){
 assert(s==&slot&&native==native_slot);return current_generation?s->ce:NULL;
}
static int source_private_sample(const Tex *e){assert(e==&supplier);return !e->mutable;}
static void make_key(const tex_t *t,TexKey *key){*key=(TexKey){t->addr,t->layout,t->d.format};}
static int same_but_swizzle(const TexKey *a,const TexKey *b){
 return a->addr==b->addr&&a->layout==b->layout&&a->format==b->format;
}
static const TextureMaterialInstance *texture_publication_image_locked(
    const TextureImagePublication *images,uint32_t descriptor,VkImage expected){
 const TextureMaterialInstance *instance=images->entries[descriptor];
 return instance&&instance->image==expected?instance:NULL;
}
static int rc_input_exact(const int *input){return *input==9;}
static int vk_texture_binding_receipt_locked(int native,VkImage expected,RcRegistryPosition *next){
 assert(native==native_slot&&expected==supplier.pub.image);*next=supplier_edge;
 return source_open?VK_TEXTURE_BINDING_SOURCE:0;
}
static int vk_texture_resolved_schedule_locked(VkTextureReadKey key,VkImage expected,RcRegistryPosition *next){
 (void)key;(void)expected;(void)next;return 0; // The serial-equality path does not certify an alias.
}
static int vk_texture_published_sample_schedule_locked(const TextureImagePublication *set,uint32_t descriptor,
    VkTextureReadKey key,VkImage expected,RcRegistryPosition *next){
 (void)set;(void)descriptor;(void)key;(void)expected;(void)next;return 0;
}
static uint16_t pool_kind_type(uint32_t kind){return kind?0x103:3;}
static int texture_view_key(const VkTexImage *source,const view_t *view,int has_view,uint16_t type,int storage,ViewKey *out){
 assert(!storage);
 const uint8_t *sw=has_view&&view->set_swizzle?view->swz:source->swz;
 *out=(ViewKey){.image=source->image,.format=source->format,.kind=type,.layer=has_view?view->min_layer:0,
   .swz=(uint32_t)sw[0]|(uint32_t)sw[1]<<8|(uint32_t)sw[2]<<16|(uint32_t)sw[3]<<24};
 return 1;
}
static void *nvn_hm_get(void *map,const ViewKey *key){
 assert(map==&view_of);
 if (key->image!=42||key->format!=9||key->layer!=1||key->swz!=0x05050505) return NULL;
 return (void *)(uintptr_t)(key->kind==3?108:key->kind==0x103?109:0);
}
'''
checks=r'''
int main(void){
 TextureMaterial material={.ready=1,.input=9,.provider=&material};
 TextureMaterialInstance instance={42,&material};
 TextureMaterialInstance same_handle_other_version={42,&material};
 material.resident.instance=&instance;
 supplier=(Tex){.t={.serial=239,.addr=5000,.layout=7,.d.format=9,.L.size=1024},
    .pub={.image=42,.format=9,.layers=2,.swz={2,3,4,5}},
    .material=&material,.source_canonical=1,.uploaded=1};
 make_key(&supplier.t,&supplier.key);
 receiver=(TextureRegistration){.descriptor=supplier.t,.view.min_layer=1,.has_view=1};
 receiver.descriptor.serial=247;
 receiver.descriptor.d.swizzle_r=receiver.descriptor.d.swizzle_g=receiver.descriptor.d.swizzle_b=receiver.descriptor.d.swizzle_a=5;
 slot=(TexSlot){.registration=&receiver,.ce=&supplier,.t=receiver.descriptor,.v=receiver.view,.has_view=1};
 current_set.entries[1]=current_set.entries[3]=&instance;
 PL.image[1]=PL.image[3]=42;PL.view[1]=108;PL.view[3]=109;
 PoolSlot selected={.selected={160,8,0}};
 RcRegistryPosition until={0};read=(RcRegistryPosition){701,44};supplier_edge=(RcRegistryPosition){701,90};
 assert(pool_selected_schedule(&selected,1,&until)==VK_TEXTURE_BINDING_SOURCE&&until.scene==701&&until.ordinal==45);
 assert(pool_selected_schedule(&selected,3,&until)==VK_TEXTURE_BINDING_SOURCE&&until.ordinal==45);
 // Distinct array view type and receiver swizzle must both match the actual supplied VkImageView.
 PL.view[3]=108;
 assert(!pool_selected_schedule(&selected,3,&until));PL.view[3]=109;
 slot.t.d.swizzle_r=2;
 assert(!pool_selected_schedule(&selected,1,&until));slot.t.d.swizzle_r=5;
 current_set.entries[1]=&same_handle_other_version;
 assert(!pool_selected_schedule(&selected,1,&until));current_set.entries[1]=&instance;
 source_open=0;assert(!pool_selected_schedule(&selected,1,&until));source_open=1;
 historical=1;assert(!pool_selected_schedule(&selected,1,&until));historical=0;
 live_registration++;assert(!pool_selected_schedule(&selected,1,&until));live_registration--;
 current_generation=0;assert(!pool_selected_schedule(&selected,1,&until));current_generation=1;
 receiver.descriptor.layout++;assert(!pool_selected_schedule(&selected,1,&until));receiver.descriptor.layout--;
 receiver.descriptor.addr++;assert(!pool_selected_schedule(&selected,1,&until));receiver.descriptor.addr--;
 supplier.mutable=1;assert(!pool_selected_schedule(&selected,1,&until));supplier.mutable=0;
 supplier_edge=(RcRegistryPosition){701,44};
 assert(!pool_selected_schedule(&selected,1,&until)); // The supplying proof already expired at a real writer.
 puts("bindless alias receipt: two object identities, independently owned material, receiver swizzle and shape, exact source permission passed");
}
'''
src=prefix
src+='\n'+function(contents['vk_texture.c'],'slot_result_view')
src+='\n'+function(contents['vk_texture.c'],'vk_texture_published_alias_schedule_locked')
src+='\n'+function(contents['vk_exec_draw.c'],'pool_selected_schedule')
src+=checks
clang = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="pool-alias-receipt-") as folder:
    source = Path(folder) / "alias.c"
    source.write_text(src)
    exe = source.with_suffix(".exe")
    subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                    "-fsanitize-trap=all", str(source), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

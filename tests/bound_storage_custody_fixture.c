#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t VkImage;
typedef uint32_t VkImageUsageFlags;
typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct { uint64_t address, mapping; } RcInterval;
typedef struct { uint64_t addr; } TexKey;
typedef struct { uint32_t aspectMask, baseMipLevel, levelCount, baseArrayLayer, layerCount; } VkImageSubresourceRange;
typedef struct { uint32_t shape; uint16_t pad1; } view_t;
enum {VK_IMAGE_TYPE_2D=1,VK_IMAGE_TYPE_3D=2,VK_IMAGE_VIEW_TYPE_3D=2,VK_IMAGE_ASPECT_COLOR_BIT=1,
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT=1,VK_IMAGE_USAGE_TRANSFER_DST_BIT=2,VK_IMAGE_USAGE_STORAGE_BIT=8,
    VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED=-2,VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED=-1,
    VK_TEXTURE_STORAGE_CAPTURE_BOUND=3, NVN_TEXTURE_TARGET_3D=2};
typedef struct {uint64_t image,addr; uint32_t type,view_type,usage,aspect,depth,layers,bound,mirrored,skip,fb_learned;} VkTexImage;
typedef struct {uint64_t serial; struct {size_t size;int layers;} L; struct {unsigned visual,target,depth;} d;} tex_t;
typedef struct { struct {uint64_t id;} handle; tex_t descriptor; } TextureRegistration;
typedef struct Tex {VkTexImage pub; tex_t t; TexKey key; uint64_t content_revision;
    void *bound_to,*material; int pin_owned,bound_alias_whole,released,overwritten;
    uint64_t image_device_size; uint32_t image_device_count,image_mapped_hold_count;
    RcInterval *image_device_intervals;
} Tex;
typedef struct {Tex *ce; const TextureRegistration *record;} TexSlot;
typedef struct {TexSlot *slot; Tex *entry; int associated;} TextureSlotSelection;
typedef struct {unsigned id;} VkMemoryImageHold;
typedef struct {VkImage source,clone;} BoundRelocation;
typedef struct {RcRegistryPosition read; RcInterval *destination_intervals; uint64_t destination,destination_size;
    uint32_t count,owners,destination_count; BoundRelocation images[1];} VkTextureBoundRelocation;
typedef struct VkTextureStorageTarget VkTextureStorageTarget;

static Tex image_entry, other_entry;
static TexSlot slot;
static const RcRegistryPosition read={4,18};
static VkMemoryImageHold held[4];
static int refs[4], released, copies, moves, cut, mapping_current, lock_depth, proof_current;
static uint32_t effective_view;
static void lock(void) {++lock_depth;}
static void unlock(void) {assert(lock_depth);--lock_depth;}
static RcRegistryPosition vk_exec_resource_position(void){return read;}
static uint64_t vk_exec_texture_pool(void){return 13;}
static int nvn_gpu_only(uint64_t addr){return addr==0x8000000000ULL;}
#define nvn_tex_is_linear(d) ((d)->visual != 0)
static Tex *texture_binding_entry(TexSlot *s,int native){assert(native==4);return s->ce;}
static Tex *texture_registration_source(const TexSlot *s){assert(s==&slot);return NULL;}
static void texture_slot_selected_view(const TextureSlotSelection *s,view_t *view,int *has){
    assert(s->slot==&slot && s->entry==&image_entry);*view=(view_t){.shape=effective_view};*has=1;
}
static int storage_target_identity_current(const Tex *e) {return e && !e->released && !e->overwritten && e->pub.image;}
static int storage_target_volume_view(const Tex *e,const view_t *view,int has,VkImageSubresourceRange *mip) {
    if (e->pub.type!=VK_IMAGE_TYPE_3D)return 1;
    if (!has || view->shape>1)return 0;
    *mip=(VkImageSubresourceRange){1,0,1,0,1};return 1;
}
static int by_key;
static Tex *nvn_hm_get(int *key,const TexKey *t){assert(key==&by_key && t->addr==image_entry.key.addr);return &image_entry;}
static int image_alias_guest_mapping_current(uint64_t address,uint64_t bytes,const RcInterval *m,unsigned count) {
    return mapping_current && address==image_entry.pub.addr && bytes==image_entry.t.L.size && count==1 && m->mapping==31;
}
static VkMemoryImageHold *vk_mem_image_hold(VkImage image){assert(image==1||image==2);refs[image]++;held[image].id=image;return &held[image];}
static void vk_mem_image_release(VkMemoryImageHold *h){assert(refs[h->id]);--refs[h->id];}
static void bound_relocation_release(VkTextureBoundRelocation *plan){
    if(!plan)return;
    free(plan->destination_intervals);free(plan);++released;
}
static void vk_texture_bound_relocation_cancel(VkTextureBoundRelocation *p){bound_relocation_release(p);}
static void nvn_texture_pin_release(const tex_t *p) { assert(!p); }
static int bound_relocation_capture_one(BoundRelocation *plan,const Tex *e){
    assert(e==&image_entry);plan->source=e->pub.image;return 1;
}
static int bound_relocation_current_locked(const VkTextureBoundRelocation *p){
    assert(lock_depth && p->read.ordinal==read.ordinal && p->count==1);
    return p->images[0].source==image_entry.pub.image && mapping_current && proof_current;
}
static int vk_texture_bound_relocation_commit(VkTextureBoundRelocation *p){
    assert(lock_depth && copies && p->images[0].clone==2);
    if(!cut)return 0;
    ++moves;image_entry.pub.image=2;image_entry.pub.bound=0;image_entry.bound_to=NULL;return 1;
}
#include "target.inc"
static TexSlot *storage_target_registration_locked(const VkTextureStorageTarget *target){
    assert(lock_depth);return slot.record && slot.record->handle.id==target->registration &&
        slot.record->descriptor.serial==target->registration_serial ? &slot : NULL;
}
static Tex *storage_target_current_locked(const VkTextureStorageTarget *target,int issued) {
    (void)issued;assert(target->bound_relocated && !target->bound_plan);
    return target->hold && slot.ce==&image_entry && target->image==image_entry.pub.image ? &image_entry:NULL;
}
#include "bound.inc"

static void test_one(unsigned type) {
    RcInterval mapping={0x8000000000ULL,31};
    TextureRegistration record={.handle={52},.descriptor={.serial=90,.L={0x4000,1}}};
    TextureSlotSelection existing={&slot,&image_entry,0};
    view_t view={.shape=1};
    image_entry=(Tex){.pub={.image=1,.addr=mapping.address,.type=type,.view_type=2,.usage=11,.aspect=1,
        .depth=type==VK_IMAGE_TYPE_3D?8:1,.layers=1,.bound=1},
        .t={.serial=91,.L={0x4000,1},.d={0,type==VK_IMAGE_TYPE_3D?2:1,type==VK_IMAGE_TYPE_3D?8:1}},
        .key={mapping.address},.content_revision=7,.bound_to=&mapping,.bound_alias_whole=1,
        .image_device_size=0x4000,.image_device_count=1,.image_mapped_hold_count=1,.image_device_intervals=&mapping};
    slot=(TexSlot){&image_entry,&record};
    effective_view=1;mapping_current=proof_current=1;copies=cut=0;
    VkTextureStorageTarget *target=NULL;
    const char *reason=NULL;
    lock();
    image_entry.bound_alias_whole=0;
    assert(storage_target_capture_bound_locked(0,4,&record,&existing,&image_entry.key,&view,1,&target,&reason)==VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
    assert(target==NULL && strstr(reason,"original-contents"));
    image_entry.bound_alias_whole=1;
    image_entry.released=1;
    assert(storage_target_capture_bound_locked(0,4,&record,&existing,&image_entry.key,&view,1,&target,&reason)==VK_TEXTURE_STORAGE_CAPTURE_UNSUPPORTED);
    image_entry.released=0;
    mapping_current=0;
    assert(storage_target_capture_bound_locked(0,4,&record,&existing,&image_entry.key,&view,1,&target,&reason)==VK_TEXTURE_STORAGE_CAPTURE_INVALIDATED);
    mapping_current=1;
    assert(storage_target_capture_bound_locked(0,4,&record,&existing,&image_entry.key,&view,1,&target,&reason)==VK_TEXTURE_STORAGE_CAPTURE_BOUND);
    assert(target->registration_serial==90 && target->serial==91 && target->mapping[0].mapping==31);
    assert((target->volume_mip.aspectMask!=0)==(type==VK_IMAGE_TYPE_3D));
    assert(refs[1]==1 && target->bound_plan && !reason);
    unlock();
    vk_texture_storage_target_release(target);
    assert(!refs[1] && image_entry.pub.image==1);
    target=NULL;
    lock();
    assert(storage_target_capture_bound_locked(0,4,&record,&existing,&image_entry.key,&view,1,&target,&reason)==VK_TEXTURE_STORAGE_CAPTURE_BOUND);
    unlock();
    VkTextureBoundRelocation *plan=vk_texture_storage_target_bound_plan(target);
    plan->images[0].clone=2;copies=1;
    VkTexImage out={0};
    assert(!vk_texture_storage_target_bound_commit(target,&out));
    assert(image_entry.pub.image==1 && refs[1]==1 && target->bound_plan==plan);
    cut=1;slot.ce=&other_entry;
    assert(!vk_texture_storage_target_bound_commit(target,&out));
    slot.ce=&image_entry;
    record.handle.id++;
    assert(!vk_texture_storage_target_bound_commit(target,&out));
    record.handle.id--;
    effective_view++;
    assert(!vk_texture_storage_target_bound_commit(target,&out));
    effective_view--;
    assert(vk_texture_storage_target_bound_commit(target,&out));
    assert(out.image==2 && !out.bound && !target->bound_plan && refs[1]==0 && refs[2]==1 && !lock_depth);
    assert(!vk_texture_storage_target_bound_commit(target,&out));
    vk_texture_storage_target_release(target);
    assert(!refs[2]);
}

int main(void) {
    test_one(VK_IMAGE_TYPE_2D);
    test_one(VK_IMAGE_TYPE_3D);
    assert(released==4 && moves==2);
    puts("Bound 2D and 3D selected custody: same-key serial pair, whole-content gate, remap, release, view and binding changes, known cancel, completed clone and once-only adoption passed");
}

import os
from pathlib import Path
import runpy
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
runtime = Path(os.environ.get("STORAGE_TARGET_SOURCE", ROOT / "runtime"))
function = runpy.run_path(str(ROOT / "tests/ordered_graphics_feedback_check.py"))["function"]
texture = (runtime / "vk_texture.c").read_text(encoding="utf-8")
start = texture.index("struct VkTextureStorageTarget {")
end = texture.index("\n};", start) + len("\n};")
declaration = texture[start:end]

prefix = r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "resource_core_registry.h"
typedef uint64_t VkImage, TexKey, tex_t;
typedef struct {uint32_t aspectMask,baseMipLevel,levelCount,baseArrayLayer,layerCount;} VkImageSubresourceRange;
typedef struct {uint32_t pad1;} view_t;
typedef struct {int retired;} VkMemoryImageHold;
typedef struct VkTextureStorageTarget VkTextureStorageTarget;
typedef struct TextureStorageCpuLoad TextureStorageCpuLoad;
typedef struct {int provisional;} VkTextureBoundRelocation;
typedef struct {TexKey key;uint64_t serial;struct {uint64_t size;} L;} Descriptor;
typedef struct {struct {uint64_t id;} handle;Descriptor descriptor;} TextureRegistration;
typedef struct Tex Tex;
typedef struct {const TextureRegistration *registration;Tex *ce,*owned;int finalized;} TexSlot;
typedef struct {TexSlot *slot;Tex *entry;int associated;} TextureSlotSelection;
struct Tex {
    TexKey key;
    struct { VkImage image; uint64_t addr; int gone, bound; } pub;
    struct { uint64_t serial; struct {uint64_t size;} L; } t;
    uint64_t content_revision, sample_general_revision;
    int released, retiring, superseded, overwritten;
    int pin_detached, pin_owned, image_device_count;
    void *alloc, *bound_to, *material, *pending_write;
    Tex *image_host_backing;
    const RcInterval *image_private_mapping;
    uint32_t image_private_mapping_count;
};
static Tex entries[2];
static TextureRegistration records[] = {{.handle={101}},{.handle={102}}};
static TexSlot tslots[] = {{.registration=&records[0]},{.registration=&records[1]}};
static RcRegistryPosition read={9,22};
static VkMemoryImageHold holds[2];
static uint64_t pool=77;
static int locked;
static const RcInterval *active_mapping[2];
static uint32_t active_mapping_count[2];
static const int by_key;
enum {VK_TEXTURE_REGISTRATION_CURRENT=1};
static void lock(void){assert(!locked++);}
static void unlock(void){assert(locked--==1);}
static uint64_t vk_exec_texture_pool(void){return pool;}
static RcRegistryPosition vk_exec_resource_position(void){return read;}
static int vk_mem_image_hold_retired(const VkMemoryImageHold *hold){assert(locked);return hold->retired;}
static int texture_registration_resolve(int logical,const TextureRegistration **history,int *native) {
    assert(locked && logical>=0 && logical<2 && history && native);
    *native=logical;
    *history=tslots[logical].registration;
    return *history ? VK_TEXTURE_REGISTRATION_CURRENT : 0;
}
static Tex *nvn_hm_get(const int *map,const TexKey *key) {
    assert(locked && map==&by_key && *key<2);
    return &entries[*key];
}
static void make_key(const Descriptor *desc,TexKey *key){assert(locked);*key=desc->key;}
static int same_but_swizzle(const TexKey *a,const TexKey *b){return *a==*b;}
static int __attribute__((unused)) storage_word_alias_initialized(Tex *entry,VkImageSubresourceRange plane,
    RcRegistryPosition read) {
    (void)entry; (void)plane; (void)read; assert(0); return 0;
}
static Tex *texture_registration_source(const TexSlot *slot){return slot->owned;}
static Tex *texture_binding_entry(const TexSlot *slot,int native){
    assert(locked&&native>=0&&native<2&&slot==&tslots[native]);
    return slot->owned?slot->owned:slot->ce;
}
static void texture_slot_selected_view(const TextureSlotSelection *selected,view_t *view,int *has_view){
    assert(locked&&selected->slot&&selected->entry);
    *view=(view_t){0}; *has_view=0;
}
static int mem_backing_current(const Tex *entry){
    assert(locked);
    return entry && active_mapping[entry->key] && active_mapping_count[entry->key];
}
static const RcInterval *mem_backing_intervals(const Tex *entry,uint32_t *count){
    assert(locked && entry && count);
    *count=active_mapping_count[entry->key];
    return active_mapping[entry->key];
}
static Tex *nvn_texture_pin_attachment(const tex_t *pin) {
    assert(locked && pin && *pin < 2);
    return &entries[*pin];
}
static int image_alias_guest_mapping_current(uint64_t address,uint64_t size,const RcInterval *mapping,uint32_t count) {
    assert(locked);
    const unsigned index=address==entries[1].pub.addr;
    assert(entries[index].pub.addr==address && entries[index].t.L.size==size);
    if (count!=active_mapping_count[index]) return 0;
    for (uint32_t i=0;i<count;++i) {
        RcInterval current=active_mapping[index][i];
        if (current.address!=mapping[i].address || current.mapping!=mapping[i].mapping ||
            !rc_range_equal(current.range,mapping[i].range)) return 0;
    }
    return 1;
}
'''

checks = r'''
static VkTextureStorageTarget *make_target(unsigned i,const RcInterval *mapping,uint32_t count) {
    const unsigned pool_index=i;
    Tex *entry=&entries[i];
    const uint64_t address=mapping[0].address;
    const uint64_t size=count==2?64:32;
    *entry=(Tex){.key=i,.pub={.image=30+i,.addr=address},.t={44+i,{size}},.content_revision=55+i,
                 .alloc=entry, .image_private_mapping=mapping,.image_private_mapping_count=count};
    tslots[i].ce=entry;
    records[i].descriptor=(Descriptor){.key=i,.serial=entry->t.serial,.L.size=entry->t.L.size};
    active_mapping[i]=mapping;
    active_mapping_count[i]=count;
    VkTextureStorageTarget *target=calloc(1,sizeof *target+count*sizeof *mapping);
    assert(target);
    *target=(VkTextureStorageTarget){.key=i,.read=read,.image=entry->pub.image,.hold=&holds[i],.pool=pool,
        .registration=records[i].handle.id,.serial=entry->t.serial,.revision=entry->content_revision,
        .registration_key=i,.registration_serial=entry->t.serial,.size=size,.logical=i,.native=pool_index,.mapping_count=count};
    memcpy(target->mapping,mapping,count*sizeof *mapping);
    return target;
}
int main(void) {
    const RcInterval a[2]={{0x10000,11,{21,64,32}},{0x10020,12,{23,256,32}}};
    const RcInterval adjacent[2]={{0x20000,31,{21,96,32}},{0x20020,32,{24,256,32}}};
    const RcInterval overlapping_alias[2]={{0x20000,81,{22,64,32}},{0x20020,82,{23,287,32}}};
    const RcInterval remap[2]={{0x20000,91,{21,96,32}},{0x20020,32,{24,256,32}}};
    const RcInterval invalid[2]={{0x20000,31,{21,96,32}},{0x20020,32,{24,UINT64_MAX,32}}};
    VkTextureStorageTarget *first=make_target(0,a,2);
    VkTextureStorageTarget *second=make_target(1,adjacent,2);
    assert(vk_texture_storage_targets_disjoint(first,second));
    assert(vk_texture_storage_targets_disjoint(second,first));
    assert(!vk_texture_storage_targets_disjoint(first,first));
    assert(!vk_texture_storage_targets_disjoint(first,NULL));
    entries[1].alloc=entries[0].alloc;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    entries[1].alloc=&entries[1];
    entries[1].pub.image=second->image=first->image;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    entries[1].pub.image=second->image=31;
    entries[1].bound_to=&entries[1];
    assert(!vk_texture_storage_targets_disjoint(first,second));
    entries[1].bound_to=NULL;
    holds[0].retired=1;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    holds[0].retired=0;
    read.ordinal+=2;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    read.ordinal-=2;
    records[1].handle.id++;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    records[1].handle.id--;
    entries[1].content_revision++;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    entries[1].content_revision--;
    active_mapping[1]=remap;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    active_mapping[1]=adjacent;
    assert(vk_texture_storage_targets_disjoint(first,second));
    free(second);
    second=make_target(1,overlapping_alias,2);
    assert(first->mapping[1].address!=second->mapping[1].address &&
           first->mapping[1].mapping!=second->mapping[1].mapping);
    assert(!vk_texture_storage_targets_disjoint(first,second));
    free(second);
    second=make_target(1,invalid,2);
    assert(!vk_texture_storage_targets_disjoint(first,second));
    // The same held target contract also covers exact host backing, without a GPU alias recipe.
    free(second);
    second=make_target(1,adjacent,2);
    entries[1].sample_general_revision=second->host_layout=701;
    entries[1].image_host_backing=&entries[1];
    entries[1].image_private_mapping_count=0;
    assert(vk_texture_storage_targets_disjoint(first,second));
    VkTextureBoundRelocation pending={.provisional=1};
    first->bound_plan=&pending;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    first->bound_plan=NULL;
    assert(vk_texture_storage_targets_disjoint(first,second));
    active_mapping[1]=remap;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    active_mapping[1]=adjacent;
    entries[1].sample_general_revision++;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    entries[1].sample_general_revision=second->host_layout;
    entries[1].pending_write=&entries[1];
    assert(!vk_texture_storage_targets_disjoint(first,second));
    entries[1].pending_write=NULL;
    tex_t pin=1;
    second->host_pin=&pin;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    entries[1].pin_owned=1;
    assert(vk_texture_storage_targets_disjoint(first,second));
    pin=0;
    assert(!vk_texture_storage_targets_disjoint(first,second));
    assert(!locked);
    free(second);
    free(first);
    puts("private storage targets: normalized alias overlap, disjoint adjacency, native owner and stale read controls passed");
}
'''
names = ("storage_target_identity_current", "storage_host_mapping", "storage_target_registration_locked",
         "storage_target_current_locked",
         "vk_texture_storage_targets_disjoint")
source = prefix + declaration + "\n".join(function(texture, name) for name in names) + checks
compiler = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="storage-target-disjoint-") as folder:
    path = Path(folder) / "check.c"
    exe = path.with_suffix(".exe")
    path.write_text(source, encoding="utf-8")
    subprocess.run([str(compiler), "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-O1",
                    "-fsanitize=undefined", "-fsanitize-trap=all", f"-I{runtime}", str(path), "-o", str(exe)],
                   env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

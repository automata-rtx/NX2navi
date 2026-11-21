
from pathlib import Path
import os
import runpy
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
runtime = Path(os.environ.get("DIRECT_GPU_SAMPLE_SOURCE", ROOT / "runtime"))
function = runpy.run_path(str(ROOT / "tests/ordered_graphics_feedback_check.py"))["function"]
texture = (runtime / "vk_texture.c").read_text(encoding="utf-8")
memory = (runtime / "vk_memory.c").read_text(encoding="utf-8")
prefix = r'''

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "resource_core_registry.h"

typedef uint64_t VkImage,VkDeviceMemory;
typedef struct {VkDeviceMemory deviceMemory;uint64_t offset,size;} VmaAllocationInfo;
typedef VmaAllocationInfo *VmaAllocation;
typedef struct {VkDeviceMemory memory;uint64_t offset,size;} VkImageMemoryPlacement;
typedef struct {uint32_t plane,layers,pad1;} view_t;
typedef struct {VkImage image;uint64_t addr;int gone;} VkTexImage;
typedef struct {view_t view;int historical,native_slot,has_view;uint64_t registration;} VkTextureSample;
typedef struct {VkImage image;unsigned readers;int detached,binding_pending;
    VkDeviceMemory pool_memory;uint64_t pool_begin,pool_end;} VkMemoryImageHold;
typedef struct {VkMemoryImageHold *raw;} VkTexturePresentSource;
typedef struct {
    VkTexImage pub;VmaAllocation alloc,bound_to;const void *image_host_backing;
    uint64_t image_device_size;
    uint32_t image_device_count,image_private_mapping_count;
    const RcInterval *image_device_intervals,*image_private_mapping;
} Tex;
typedef struct {struct {uint64_t id;} handle;} TextureRegistration;
typedef struct {TextureRegistration *registration;const Tex *ce;int used,finalized;uint64_t pool_owner;} TexSlot;

enum { ORIGINAL_SLOT=10671, };
static const int ntslots=ORIGINAL_SLOT+1;
static Tex original,replacement;
static TexSlot tslots[ORIGINAL_SLOT+1];
static TextureRegistration registration={{37727}};
static RcRegistryPosition position={14514,42750};
static RcInterval private_mapping={UINT64_C(0x800193f800), 1349, {1346, 0x193f800, 0xf3c000}};
static RcInterval bound_mapping={UINT64_C(0x800193f800), 19, {9, 0x300000, 0xfa0000}};
static VmaAllocationInfo allocator_input={11,0x2000,0xfa0000};
static VkMemoryImageHold raw={.image=620,.readers=1};
static const VkTexturePresentSource hold={.raw=&raw};
static int native_current,private_current,host_current,slot_receipt_checks,cmd_lock;
static const struct {int vma;} ctx={0};
static const typeof(ctx) *vk_ctx(void) {return &ctx;}
static void vmaGetAllocationInfo(int vma,VmaAllocation allocation,VmaAllocationInfo *info) {
    assert(!vma&&allocation);
    *info=*allocation;
}
static RcRegistryPosition vk_exec_resource_position(void) {return position;}
static uint64_t vk_exec_texture_pool(void) {return 777;}
static void plat_rmutex_lock(int *lock) {assert(!(*lock)++);}
static void plat_rmutex_unlock(int *lock) {assert((*lock)--==1);}
static int vk_texture_transfer_read_current(const VkTexturePresentSource *actual,VkImage image) {
    return actual==&hold&&actual->raw->image==image&&actual->raw->readers;
}
static const Tex *texture_binding_entry(const TexSlot *slot,int native) {
    assert(slot==&tslots[ORIGINAL_SLOT]&&native==ORIGINAL_SLOT);
    slot_receipt_checks++;
    return slot->ce;
}
static int mem_backing_current(const void *backing) {return host_current&&backing==&original;}
static const RcInterval *mem_backing_intervals(const void *backing,uint32_t *count) {
    assert(mem_backing_current(backing));
    *count=1;
    return &private_mapping;
}
static int vk_mem_image_alias_current(VkImage image,uint64_t address,uint64_t size,VmaAllocation allocation,
    const RcInterval *mapping,uint32_t count) {
    assert(image==original.pub.image&&address==original.pub.addr&&allocation==original.bound_to);
    return native_current&&size==bound_mapping.range.size&&mapping==&bound_mapping&&count==1;
}
static int image_alias_placement_current(const Tex *entry) {
    assert(entry==&original&&!entry->bound_to);
    return private_current&&entry->image_private_mapping==&private_mapping&&entry->image_private_mapping_count==1;
}
static void slot_result_view(const TexSlot *slot,const Tex *entry,view_t *view,int *has_view) {
    assert(slot==&tslots[ORIGINAL_SLOT]&&entry==&original);
    *view=(view_t){2,3,44};
    *has_view=1;
}
'''

cases = r'''

static void no_footprint(VkTextureSample sample,RcRegistryPosition read,VkImage image) {
    VkImageMemoryPlacement placement={0};
    uint32_t count=200;
    assert(!vk_texture_native_sample_intervals_locked(read,&sample,&hold,image,&placement,&count));
    assert(!count);
}

int main(void) {
    const VkImage selected=620;
    original=(Tex){.pub={.image=selected,.addr=private_mapping.address},.alloc=&allocator_input,
        .image_private_mapping=&private_mapping,.image_private_mapping_count=1};
    tslots[ORIGINAL_SLOT]=(TexSlot){.registration=&registration,.ce=&original,.used=1,.pool_owner=777};
    VkTextureSample sample={.native_slot=ORIGINAL_SLOT,.has_view=1,.view={2,3,81}};
    VkImageMemoryPlacement placement={0};
    uint32_t count=0;

    // This opaque D16 array read has the original native slot and physical hold, but no
    // registration key. Permission to use the held VkImage is already checked above this adapter.
    private_current=1;
    assert(vk_texture_native_sample_intervals_locked(position,&sample,&hold,selected,&placement,&count)==&private_mapping);
    assert(count==1&&placement.memory==11&&placement.offset==0x2000&&placement.size==0xfa0000&&!sample.registration);

    VmaAllocationInfo adjacent={placement.memory,placement.offset+placement.size,0xf0000};
    assert(texture_placement_disjoint(&adjacent,placement));
    adjacent.offset--;
    assert(!texture_placement_disjoint(&adjacent,placement));
    adjacent.deviceMemory=17;
    assert(texture_placement_disjoint(&adjacent,placement));
    assert(!texture_placement_disjoint(NULL,placement));
    placement.size=UINT64_MAX;
    assert(!texture_placement_disjoint(&adjacent,placement));

    sample.registration=37728;
    no_footprint(sample,position,selected);
    sample.registration=0;
    private_current=0;
    no_footprint(sample,position,selected);
    private_current=1;
    sample.view.layers=2;
    no_footprint(sample,position,selected);
    sample.view.layers=3;
    sample.native_slot=ORIGINAL_SLOT-1;
    no_footprint(sample,position,selected);
    sample.native_slot=ORIGINAL_SLOT;
    int lookups=slot_receipt_checks;
    no_footprint(sample,(RcRegistryPosition){14514,42749},selected);
    sample.historical=1;
    no_footprint(sample,position,selected);
    sample.historical=0;
    raw.readers=0;
    no_footprint(sample,position,selected);
    raw.readers=1;
    no_footprint(sample,position,selected+1);
    assert(slot_receipt_checks==lookups);

    replacement=original;
    replacement.pub.image=selected+1;
    tslots[ORIGINAL_SLOT].ce=&replacement;
    no_footprint(sample,position,selected);
    tslots[ORIGINAL_SLOT].ce=&original;

    // Host-backed standalone image also has an exact retained native allocation and
    // a pinned normalized mapping. No raw producer or typed layout is needed here.
    original.image_private_mapping_count=0;
    original.image_host_backing=&original;
    host_current=1;
    assert(vk_texture_native_sample_intervals_locked(position,&sample,&hold,selected,&placement,&count)==&private_mapping);
    host_current=0;
    no_footprint(sample,position,selected);

    // Bound image holds the full Vulkan interval including native padding. Other image
    // suballocations in the same VkDeviceMemory are evaluated by actual offsets.
    raw.pool_memory=13;
    raw.pool_begin=0x400000;
    raw.pool_end=raw.pool_begin+bound_mapping.range.size;
    original.alloc=NULL;
    original.bound_to=&allocator_input;
    original.image_device_size=bound_mapping.range.size;
    original.image_device_intervals=&bound_mapping;
    original.image_device_count=1;
    native_current=1;
    assert(bound_mapping.range.size!=private_mapping.range.size);
    assert(vk_texture_native_sample_intervals_locked(position,&sample,&hold,selected,&placement,&count)==&bound_mapping);
    assert(count==1&&placement.memory==13&&placement.offset==raw.pool_begin&&placement.size==bound_mapping.range.size);
    assert(texture_placement_disjoint(&(VmaAllocationInfo){13,raw.pool_end,0xf0000},placement));
    assert(!texture_placement_disjoint(&(VmaAllocationInfo){13,raw.pool_end-1,0xf0000},placement));
    original.image_device_size=private_mapping.range.size;
    no_footprint(sample,position,selected);
    original.image_device_size=bound_mapping.range.size;
    native_current=0;
    no_footprint(sample,position,selected);
    native_current=1;
    raw.pool_end--;
    no_footprint(sample,position,selected);
    assert(!cmd_lock);
    puts("Original sampled placement: unkeyed D16, opaque and bound GPU and host footprints, physical overlap, remap and exact receipt controls passed");
}
'''

names = ("texture_image_placement", "vk_texture_native_sample_intervals_locked", "texture_placement_disjoint")
source = prefix + function(memory, "vk_mem_image_hold_placement") + \
    "\n".join(function(texture, name) for name in names) + cases
compiler = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="direct-gpu-sample-footprint-") as folder:
    path = Path(folder) / "check.c"
    exe = path.with_suffix(".exe")
    path.write_text(source, encoding="utf-8")
    subprocess.run([str(compiler), "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-O1",
                    "-fsanitize=undefined", "-fsanitize-trap=all", f"-I{runtime}", str(path), "-o", str(exe)],
                   env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True)

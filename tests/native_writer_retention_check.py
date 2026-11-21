"""Check the actual renderer ownership transfer and bounded detached-writer collection."""
import os
from pathlib import Path
import subprocess
import tempfile

from ordered_graphics_feedback_check import function

ROOT = Path(__file__).resolve().parents[1]
CC = os.environ["CC"]
STUBS = r"""
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint64_t VkImage;
typedef void *VmaAllocation;
typedef struct {VkImage image; uint64_t addr;} VkTexImage;
typedef struct {int key;} nvn_texdesc;
typedef struct {int format;} TexVkFormat;
typedef struct {uint64_t id; uint32_t index;} RcRegistryWrite;
typedef struct {uint64_t address, mapping; struct {uint64_t backing,offset,size;} range;} RcInterval;
typedef struct {struct {RcRegistryWrite hold;} binding; RcInterval interval;} RcRegistryNativeImageFragment;
typedef struct {int live;} RcBudgetTicket;
typedef struct RcRegistry RcRegistry;
typedef int RcRegistryResult;
typedef struct {uint64_t id; uint32_t slot;} RcBackingId;
typedef struct NativeTextureOutput NativeTextureOutput;
typedef struct Tex {
    VkTexImage pub;
    struct {nvn_texdesc d; struct {uint64_t size;} L;} t;
    TexVkFormat f;
    uint64_t bytes;
    VmaAllocation alloc;
    RcInterval *image_private_mapping;
    uint32_t image_private_mapping_count;
    RcRegistryWrite *image_mapped_holds;
    uint32_t image_mapped_hold_count;
    int bound_to,material,pin_owned,alias_fragment_owner;
    NativeTextureOutput *native_output;
} Tex;
typedef struct {int klen;} nvn_hmap;
enum {RC_BUDGET_TEXTURE, VK_SUCCESS=0, RC_REGISTRY_OK=0, RC_REGISTRY_BLOCKED=4};

static NativeTextureOutput *index[80];
static int metadata_live, holders_disposed, views_retired, vma_queued, allow_supersede, authority_calls;
static uint64_t last_image;
static int allocation[80];
static void *nvn_hm_put(nvn_hmap *map,const VkImage *image,NativeTextureOutput *owner){
    assert(map->klen == sizeof(*image) && *image < 80);
    NativeTextureOutput *old = index[*image];
    index[*image]=owner;
    return old;
}
static NativeTextureOutput *nvn_hm_get(nvn_hmap *map,const VkImage *image){
    assert(map->klen == sizeof(*image) && *image < 80);
    return index[*image];
}
static void nvn_hm_del(nvn_hmap *map,const VkImage *image){
    assert(nvn_hm_get(map,image));index[*image]=NULL;
}
static int vk_replay_metadata_allocate(size_t bytes,int clear,int kind,RcBudgetTicket *ticket,void **out) {
    assert(clear && kind==RC_BUDGET_TEXTURE);
    *out=calloc(1,bytes); assert(*out);
    metadata_live++;
    ticket->live=1;
    return VK_SUCCESS;
}
static void vk_replay_metadata_free(void *object,RcBudgetTicket *ticket) {
    assert(ticket->live && metadata_live);
    metadata_live--;free(object);
}
static void vk_exec_image_gone(VkImage im){assert(im);views_retired++;}
static int vk_mem_image_held(VkImage image) { assert(image); return 0; }
static void vk_exec_texture_publications_request_reclaim(void) { abort(); }
static int vk_mem_device_backing_reclaimed(RcBackingId *identity) { assert(identity->id == 12); return 0; }
static RcRegistryResult rc_registry_native_image_discard_reclaimed(RcRegistry *r,uint64_t image,uint64_t backing,
    const RcRegistryWrite *holds,uint32_t count) {
    (void)r; (void)image; (void)backing; (void)holds; (void)count; abort();
}
static void (*destroyed_callback[80])(void *);
static void vk_defer_image_notify(VkImage im,VmaAllocation alloc,uint64_t bytes,void (*destroyed)(void *),void *arg){
    assert(destroyed && !arg && !destroyed_callback[im]);
    destroyed_callback[im]=destroyed;
    assert(alloc==&allocation[im] && bytes==8192);
    vma_queued++;last_image=im;
}
static RcRegistry *rc_registry_runtime(void){return NULL;}
static int rc_registry_native_image_discard_superseded(RcRegistry *r,uint64_t image,
                                                       const RcRegistryWrite *holds,uint32_t count){
    (void)r;
    assert(holds && count==1 && holds->id==image*10);
    authority_calls++;
    if (image!=1 && (image!=19 || !allow_supersede)) return RC_REGISTRY_BLOCKED;
    if (!allow_supersede) return RC_REGISTRY_BLOCKED;
    holders_disposed++;return RC_REGISTRY_OK;
}
static void rt_fatal(const char *text){(void)text;abort();}
"""

CASES = r"""
static Tex attached(VkImage image,int published) {
    Tex e={.pub={image,UINT64_C(0x8000000000)+image*8192},
        .t={{3},{8192}},.f={63},.bytes=8192,.alloc=&allocation[image],.image_mapped_hold_count=1};
    e.image_mapped_holds=malloc(sizeof(*e.image_mapped_holds));
    assert(e.image_mapped_holds);
    *e.image_mapped_holds=(RcRegistryWrite){.id=image*10,.index=(uint32_t)image};
    e.image_private_mapping=malloc(sizeof(*e.image_private_mapping));
    assert(e.image_private_mapping);
    *e.image_private_mapping=(RcInterval){e.pub.addr,7,{12,image*8192,8192}};
    e.image_private_mapping_count=1;
    assert(!e.native_output && !e.alias_fragment_owner);
    assert(published==0 || published==1);
    return e;
}

static void capture(Tex *e,int published) {
    assert(native_texture_output_reserve(e));
    assert(native_texture_output_reserve(e));
    e->native_output->possible=1;
    if (published) native_texture_output_publish(e);
}

int main(void) {
    Tex old=attached(1,1);
    capture(&old,1);
    const VkImage physical=old.pub.image;
    NativeTextureOutput *retained=old.native_output;
    RcRegistryWrite *tickets=old.image_mapped_holds;
    assert(native_texture_output_transfer(&old));
    assert(!old.native_output && !old.alloc && !old.image_mapped_holds && !old.alias_fragment_owner);
    assert(nvn_hm_get(&alias_fragment_owners,&physical)==retained && !retained->entry &&
           retained->image.image==physical && retained->descriptor.key==3 && retained->format.format==63 &&
           retained->guest_bytes==8192 && retained->holds==tickets && retained->allocation==&allocation[1]);
    assert(!native_texture_output_transfer(&old) && metadata_live==1);
    assert(!old.image_private_mapping && !old.image_private_mapping_count && retained->mapping_count==1);
    RcRegistryNativeImageFragment f={{tickets[0]},retained->mapping[0]};
    assert(native_texture_output_contains(retained,&f));
    f.interval.address+=16; f.interval.range.offset+=16; f.interval.range.size-=16;
    assert(native_texture_output_contains(retained,&f));
    f.interval.range.offset++;
    assert(!native_texture_output_contains(retained,&f));
    f.interval.range.offset--;
    f.interval.mapping++;
    assert(!native_texture_output_contains(retained,&f));
    native_texture_output_collect(32);
    assert(vma_queued==0 && holders_disposed==0 && authority_calls==1);

    for (VkImage image=2;image<=19;image++) {
        Tex e=attached(image, image!=19);
        capture(&e, image!=19);
        assert(native_texture_output_transfer(&e));
    }
    assert(!index[19] && metadata_live==19); // An unknown write is never a sampled positive source.
    allow_supersede=1; // Canonical query now proves complete newer coverage for 1 and 19 only.
    native_texture_output_collect(4);
    assert(holders_disposed==1 && vma_queued==1 && last_image==1 && !index[1] &&
           metadata_live==18 && views_retired==1 && !native_outputs_destroyed);
    destroyed_callback[1](NULL); // Physical destruction, not submitting a deferred retirement, counts.
    assert(native_outputs_destroyed==1);
    native_texture_output_collect(4);
    assert(holders_disposed==1 && vma_queued==1);
    native_texture_output_collect(32);
    assert(holders_disposed==2 && vma_queued==2 && last_image==19 && metadata_live==17);
    assert(index[2] && !index[2]->entry && index[2]->image.image==2);
    assert(native_texture_output_reserve(&old)==0); // A detached cache node cannot donate again.

    for (VkImage image=2;image<19;image++) {
        NativeTextureOutput *p=index[image];
        native_texture_output_retire(p);
    }
    assert(metadata_live==0 && vma_queued==19 && views_retired==19);
    puts("Native image authority survives cache transfer; rotating reclamation consumes only superseded owners");
}
"""

def main():
    source = (ROOT / "runtime/vk_texture.c").read_text(encoding="utf-8")
    decl = source[source.index("struct NativeTextureOutput {"):source.index("static NativeTextureOutput *native_texture_output_create")]
    parts = ("native_texture_output_create", "native_texture_output_drop", "native_texture_output_reserve",
             "native_texture_output_publish", "native_texture_output_unpublish", "native_texture_output_transfer",
             "native_texture_output_destroyed", "native_texture_output_retire", "native_texture_output_collect",
             "native_texture_output_contains")
    with tempfile.TemporaryDirectory(prefix="native-image-owner-") as temp:
        path = Path(temp)
        c = path/"test.c"
        c.write_text(STUBS+decl+"\n".join(function(source, part) for part in parts)+CASES, encoding="utf-8")
        executable = path/("test.exe" if os.name=="nt" else "test")
        subprocess.run([CC,"-std=c11","-Wall","-Wextra","-Werror","-fsanitize=undefined",
                        "-fsanitize-trap=all",str(c),"-o",str(executable)],check=True)
        subprocess.run([str(executable)],check=True)

if __name__=="__main__":
    main()

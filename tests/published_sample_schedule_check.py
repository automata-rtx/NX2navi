"""Exercise the installed bindless member's scoped ordinary-source proof."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from ordered_graphics_feedback_check import function


def source():
    path = ROOT / "runtime"
    texture = (path / "vk_texture.c").read_text()
    uses = (path / "vk_texture_uses.h").read_text()
    return PREFIX + function(uses, "texture_publication_owner_locked") + "\n" + function(uses, "texture_publication_image_locked") + "\n" + function(
        texture, "vk_texture_published_sample_schedule_locked") + CASES


PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef uint64_t VkImage;
typedef struct { uint64_t scene, ordinal; } RcRegistryPosition;
typedef struct { uint64_t registration; int native_slot, canonical; } VkTextureReadKey;
typedef struct { uint64_t generation; } TextureSourceStamp;
typedef struct { TextureSourceStamp stamp; } TextureSourceObservation;
typedef struct { int ready, retired, input; void *provider; } TextureMaterial;
typedef struct { TextureMaterial *material; struct { VkImage image; } value; } TextureMaterialInstance;
typedef struct { struct { TextureMaterialInstance *instance; } use; uint32_t publication_refs; } TextureUseNode;
typedef struct { uint32_t entry_limit, entry_capacity; struct {TextureUseNode *use;} entries[3]; } TextureImagePublication;
typedef struct {
    struct { uint64_t serial; } t;
    struct { VkImage image; } pub;
    TextureMaterial *material;
    int released, retiring, superseded, source_canonical;
    TextureSourceStamp source_stamp;
} Tex;
typedef struct { struct { uint64_t serial; } descriptor; } Registration;
typedef struct { int finalized; Registration *registration; } TexSlot;
enum { VK_TEXTURE_BINDING_SOURCE=2, TEX_SOURCE_CLEAN=1, TEX_SOURCE_DIRTY=2 };
static int proof_stage, allowed, state, observed, committed, scheduled;
static RcRegistryPosition read;
static Tex live;
static Registration registration={.descriptor.serial=37};
static TexSlot slot={.registration=&registration};
static RcRegistryPosition vk_exec_resource_position(void) { return read; }
static int rc_registry_position_valid(RcRegistryPosition p) { return p.scene && p.ordinal; }
static const TexSlot *texture_read_key_slot(VkTextureReadKey key) {
    return key.canonical && key.registration==195 && key.native_slot==8 ? &slot : NULL;
}
static Tex *texture_registration_source(const TexSlot *s) { assert(s==&slot);return &live; }
static uint32_t texture_publication_entry_at(const TextureImagePublication *set,uint32_t index) {
    assert(index<set->entry_capacity);return index;
}
static int source_private_sample(const Tex *e) { assert(e==&live);return 1; }
static int source_effects_read_allowed(const Tex *e) { assert(e==&live);return allowed; }
static int rc_input_exact(const int *input) { return *input==9; }
static int source_observe(Tex *e,const TextureSourceStamp *stamp,TextureSourceObservation *obs) {
    assert(e==&live&&stamp==&live.source_stamp);observed++;
    obs->stamp=(TextureSourceStamp){.generation=4};return state;
}
static void source_commit(Tex *e,const TextureSourceStamp *stamp,int same) {
    assert(e==&live&&stamp->generation==4&&same);committed++;e->source_stamp=*stamp;
}
static int source_permission_current(Tex *e) { assert(e==&live);return proof_stage; }
static int texture_source_stamp_valid(const TextureSourceStamp *stamp) { return stamp->generation!=0; }
static int texture_binding_schedule(Tex *e,VkImage expected,RcRegistryPosition *next,int proof) {
    assert(e==&live&&expected==42&&proof==VK_TEXTURE_BINDING_SOURCE);scheduled++;
    *next=(RcRegistryPosition){read.scene,read.ordinal+4};return proof;
}
static int vk_texture_resolved_schedule_locked(VkTextureReadKey key,VkImage expected,RcRegistryPosition *next) {
    assert(key.registration==195&&expected==42&&live.pub.image==42);
    return texture_binding_schedule(&live,expected,next,VK_TEXTURE_BINDING_SOURCE);
}
'''

CASES = r'''
int main(void) {
    TextureMaterial m={.ready=1,.input=9,.provider=&m}, newer={.ready=1,.input=9,.provider=&newer};
    TextureMaterialInstance a={&m,{42}},b={&newer,{43}};
    TextureUseNode held={.use.instance=&a,.publication_refs=1},replacement={.use.instance=&b,.publication_refs=1};
    TextureImagePublication set={.entry_limit=3,.entry_capacity=3,.entries={{&held},{&replacement},{0}}};
    live=(Tex){.t.serial=37,.source_canonical=1,.material=&m,.source_stamp.generation=2};
    proof_stage=allowed=1;state=TEX_SOURCE_CLEAN;read=(RcRegistryPosition){19,24};
    VkTextureReadKey key={.registration=195,.native_slot=8,.canonical=1};
    RcRegistryPosition next;
    assert(!vk_texture_published_sample_schedule_locked(&set,1,key,42,&next) && !observed && !next.scene);
    assert(!vk_texture_published_sample_schedule_locked(&set,1,key,43,&next) && !observed);
    assert(!vk_texture_published_sample_schedule_locked(&set,2,key,42,&next) && !observed);
    live.pub.image=42;
    assert(vk_texture_published_sample_schedule_locked(&set,0,key,42,&next)==VK_TEXTURE_BINDING_SOURCE);
    assert(scheduled==1&&!observed&&next.scene==19&&next.ordinal==28);
    live.pub.image=0;
    assert(vk_texture_published_sample_schedule_locked(&set,0,key,42,&next)==VK_TEXTURE_BINDING_SOURCE);
    assert(scheduled==2&&observed==1&&committed==1&&!live.pub.image&&next.ordinal==28);
    proof_stage=0;
    assert(!vk_texture_published_sample_schedule_locked(&set,0,key,42,&next)&&!next.scene&&scheduled==2);
    proof_stage=1;state=TEX_SOURCE_DIRTY;
    assert(!vk_texture_published_sample_schedule_locked(&set,0,key,42,&next)&&!next.scene&&scheduled==2&&committed==2);
    state=TEX_SOURCE_CLEAN;live.material=&newer;
    assert(!vk_texture_published_sample_schedule_locked(&set,0,key,42,&next)&&scheduled==2);
    live.material=&m;slot.finalized=1;
    assert(!vk_texture_published_sample_schedule_locked(&set,0,key,42,&next)&&scheduled==2);
    slot.finalized=0;key.registration++;
    assert(!vk_texture_published_sample_schedule_locked(&set,0,key,42,&next)&&scheduled==2);
    puts("published sample schedule: exact installed member, detached material and finite source proof passed");
}
'''

cc = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="published-sample-") as directory:
    c = Path(directory) / "check.c"
    exe = c.with_suffix(".exe")
    c.write_text(source())
    subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-fsanitize=undefined", "-fsanitize-trap=all", str(c), "-o", str(exe)], env=env, check=True)
    subprocess.run([str(exe)], env=env, check=True, timeout=20)

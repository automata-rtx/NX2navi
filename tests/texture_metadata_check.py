"""Check production shader metadata selection without texture preparation or guest-byte access."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "runtime/vk_texture.c").read_text()


def function(name, source=SOURCE):
    match = re.search(r"^(?:static )?[^\n]*\b" + name + r"\([^;]*?\)\s*\{", source, re.M)
    if not match:
        raise RuntimeError(name)
    start = source.index("{", match.start())
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef int VkImageType;
typedef int VkImageViewType;
enum { VK_IMAGE_TYPE_1D, VK_IMAGE_TYPE_2D, VK_IMAGE_TYPE_3D };
enum { VK_IMAGE_VIEW_TYPE_1D, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_3D,
       VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_1D_ARRAY, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY };
enum { NVN_TEXTURE_TARGET_1D, NVN_TEXTURE_TARGET_2D, NVN_TEXTURE_TARGET_3D, NVN_TEXTURE_TARGET_1D_ARRAY,
       NVN_TEXTURE_TARGET_2D_ARRAY, NVN_TEXTURE_TARGET_2D_MULTISAMPLE, NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY,
       NVN_TEXTURE_TARGET_RECTANGLE, NVN_TEXTURE_TARGET_CUBEMAP, NVN_TEXTURE_TARGET_CUBEMAP_ARRAY, NVN_TEXTURE_TARGET_BUFFER };
enum { NVN_TEXTURE_SWIZZLE_R=2, NVN_TEXTURE_SWIZZLE_G, NVN_TEXTURE_SWIZZLE_B, NVN_TEXTURE_SWIZZLE_A };
typedef struct { uint32_t target,format,width,height,depth,levels,samples,swizzle_r,swizzle_g,swizzle_b,swizzle_a; } nvn_texdesc;
typedef struct { uint32_t layers; } nvn_texlayout;
typedef struct { void *mem; nvn_texdesc d; int has_layout; uint8_t layout[8]; int mirrored; uint64_t serial; struct { uint64_t size; } L; } tex_t;
typedef struct { uint64_t addr; nvn_texdesc d; int has_layout; uint8_t layout[8]; } TexKey;
typedef struct { uint64_t image,addr; int format,type,view_type; uint32_t samples,width,height,depth,levels,layers;
    int gpu_owned,mirrored,gone,skip; uint8_t swz[4]; } VkTexImage;
typedef struct { int set_target,set_format,set_swizzle; uint32_t target,format; uint8_t swz[4]; } view_t;
enum {VK_TEXTURE_REGISTRATION_ABSENT=-1,VK_TEXTURE_REGISTRATION_GONE=-2};
typedef struct { tex_t descriptor; VkTexImage exact; view_t view,exact_view; int has_view,has_exact_view;
    struct Tex *source; } TextureRegistration;
static uint64_t vk_exec_texture_pool(void){return 0;}
static int vk_texture_pool_slot_locked(uint64_t pool,int logical){assert(pool==0);return logical;}
static int texture_registration_resolve(int id,const TextureRegistration **record,int *native){
 assert(!native);(void)id;*record=NULL;return 0;
}
static const VkTexImage *historical_sample;
static int vk_texture_registration_image_locked(int id,const VkTexImage **image,view_t *v,int *hv,uint64_t *size){
    (void)id;
    if (!historical_sample) return 0;
    *image=historical_sample;*v=(view_t){.set_target=1,.target=NVN_TEXTURE_TARGET_2D_ARRAY};
    *hv=1;*size=512;return 1;
}
typedef struct TextureMaterialInstance { int live; } TextureMaterialInstance;
typedef struct { uint64_t image,bytes; TextureMaterialInstance *instance; } TextureMaterialResidency;
typedef struct TextureMaterial { TextureMaterialResidency resident; int retained_byte,reclaim;
    struct Tex *associations; } TextureMaterial;
typedef struct RcRegistryReadVersion { uint64_t id; } RcRegistryReadVersion;
typedef struct Tex { VkTexImage pub; tex_t t; TexKey key;
    int released,superseded,overwritten,uploaded,gpu_dirty,up_retry,retiring;
    uint64_t checked,checked_authority,upload_mapping,upload_protection;
    int material_prepared,chash_ok,source_canonical,source_copy,source_blocked,source_digest_valid;
    TextureMaterial *material;
    RcRegistryReadVersion material_version;
    struct Tex *next_at,*material_next; } Tex;
typedef struct { tex_t t; int used,finalized,has_view; view_t v; Tex *ce; uint32_t cgen;
    uint64_t absent_mapping,absent_protection; int cwant; TextureRegistration *registration; } TexSlot;
static Tex *texture_registration_source(const TexSlot *slot){return slot->registration?slot->registration->source:NULL;}
typedef struct { int vk; } TexVkFormat;
static TexSlot tslots[2]; static int ntslots=2;
static uint32_t agen[1]={1},sinfo[2];
static int by_key,by_addr,context=1,format_calls,deferred=7;
static Tex *key_owner,*address_head;
static struct { int n; } holds;
static int pool_ctx,create_budget=-1,create_deferred,resolve_held,resolve_stale;
static uint64_t n_stale_kept,n_stale,n_stale_hm,submit_epoch;
static struct { uint64_t evicted,evicted_bytes; } material_stats;
static unsigned restores,destroys,creations,refreshed;
static unsigned native_byte[128];
static unsigned raw_mapped_byte;
#define VK_NULL_HANDLE 0
#define tex_deferred deferred
static Tex *sample_choice,*sample_exact;
static unsigned locks,touches,captures,promotions;static uint64_t captured_image;
static void lock(void){locks++;}
static void unlock(void){assert(locks);locks--;}
static int from_storage(const Tex *e){return e->pub.mirrored;}
static void touch(Tex *e){assert(e==sample_choice || e==sample_exact);touches++;}
static void vk_texture_use_images(const uint64_t *images,size_t count){assert(count==1);captures++;captured_image=*images;}
static const VkTexImage *vk_texture_by_id_locked(int id,view_t *view,int *hv){
    (void)id;*view=(view_t){0};*hv=0;return sample_choice?&sample_choice->pub:NULL;
}
static const VkTexImage *vk_texture_exact_by_id_locked(int id,view_t *view,int *hv){
    (void)id;(void)view;(void)hv;promotions++;return sample_exact?&sample_exact->pub:NULL;
}
static void *vk_ctx(void){return context?&context:NULL;}
static int tex_mirrored(const tex_t *t){return t->mirrored;}
static int rt_knob(const char *name,const char *fallback){(void)name;(void)fallback;return 1;}
static void *nvn_hm_get(const void *map,const void *key){
    if(map==&by_key)return key_owner && !memcmp(&key_owner->key,key,sizeof(TexKey))?key_owner:NULL;
    if (map == (void *)&holds) return NULL;
    assert(map==&by_addr);return address_head && address_head->key.addr==*(const uint64_t *)key?address_head:NULL;
}
static int tex_vk_format(int format,TexVkFormat *out){format_calls++;out->vk=format==99?4:format;return format!=0;}
static int tex_vk_format_supported(const TexVkFormat *format){return format->vk!=98;}
static int nvn_fmt_bpb(int format,int *w,int *h){*w=*h=format==97?4:1;return 4;}
static void image_owner_replaced(Tex *entry,uint64_t image){(void)entry;(void)image;}
static void tex_gen_bump(uint64_t address){assert(address);agen[0]++;}
static TextureMaterialResidency texture_material_take_residency(TextureMaterial *material){
    TextureMaterialResidency old=material->resident;
    material->resident=(TextureMaterialResidency){0};
    return old;
}
static int content_defer_instance_mode(TextureMaterialInstance *instance,int completed){
    assert(instance && instance->live && completed);instance->live=0;destroys++;return 1;
}
static uint64_t vk_exec_recording_fence(void){return 0;}
static void texture_material_note_use(TextureMaterial *material,uint64_t fence,uint64_t epoch){
    (void)material;(void)fence;(void)epoch;
}
static int up_wait(Tex *entry){(void)entry;return 0;}
static int vk_exec_turn_due(void){return 0;}
static int content_early_prepare(Tex *entry){(void)entry;refreshed++;return 0;}
static int content_write_begin(Tex *entry,int content){(void)entry;(void)content;assert(0);return 0;}
static int content_restore(Tex *entry,TextureMaterial *material,void *stamp){
    assert(!stamp && entry->released && material && !entry->pub.image && !material->resident.image);
    assert(restores < 4);
    entry->pub.image=material->resident.image=120 + ++restores;
    native_byte[entry->pub.image]=material->retained_byte;
    entry->uploaded=1;
    return 1;
}
static int alias_has_gpu_content(const Tex *entry){return entry->pub.gpu_owned;}
static int follow_size_ok(const Tex *entry,const TexKey *key){return entry->key.d.width==key->d.width;}
static int slot_storage_readable(TexSlot *slot){(void)slot;return raw_mapped_byte != 0;}
static int want_take(void){return 1;}
static Tex *entry_of_mode(const tex_t *texture,int private_output,uint32_t *installed_generation){
    (void)texture;assert(!private_output && !installed_generation);creations++;return NULL;
}
static void retire_cancel(Tex *entry){entry->retiring=0;}
static int terrain_heightmap(const tex_t *texture){(void)texture;return 0;}
static void nvn_tex_layout(const nvn_texdesc *d,nvn_texlayout *layout){
    layout->layers=d->target==NVN_TEXTURE_TARGET_CUBEMAP?6:d->target==NVN_TEXTURE_TARGET_CUBEMAP_ARRAY?6*d->depth:d->depth;
}
'''

MAIN = r'''
static void finalized_residency(void) {
    view_t view;
    int has_view, destroyed=0;
    VkTexImage metadata;
    TexSlot *a=&tslots[1];
    *a=(TexSlot){.used=1,.finalized=1,.t={.mem=(void *)(uintptr_t)8192,.mirrored=1,.serial=31,.L.size=64,
       .d={.target=NVN_TEXTURE_TARGET_2D,.width=16,.height=4,.samples=1}}};
    TextureMaterialInstance native_a={1};
    TextureMaterial material_a={.retained_byte=17,.resident={81,64,&native_a}};
    Tex original={.material=&material_a,.t=a->t,.released=1,.uploaded=1,.pub={.image=81,.format=5}};
    material_a.associations=&original;
    make_key(&a->t,&original.key);
    key_owner=&original;
    address_head=&original;
    native_byte[81]=17;
    raw_mapped_byte=201; // Reused address contains different bytes from the immutable previous material.
    assert(vk_texture_metadata_locked(1,&metadata,&view,&has_view)==1 && metadata.format==5);
    assert(slot_resolve(a,&view,&has_view,0,NULL)==&original);
    assert(content_detach_residency_mode(&material_a,&destroyed)==64 && destroyed && !original.pub.image);
    assert(vk_texture_metadata_locked(1,&metadata,&view,&has_view)==1 && metadata.format==5 && !restores);
    assert(slot_resolve(a,&view,&has_view,0,NULL)==&original && original.pub.image==121);
    assert(native_byte[121]==17 && native_byte[121]!=raw_mapped_byte && restores==1);

    TextureMaterialInstance native_b={1};
    TextureMaterial material_b={.retained_byte=73,.resident={82,64,&native_b}};
    Tex replacement={.material=&material_b,.key=original.key,.t=a->t,.released=1,.uploaded=1,
                     .pub={.image=82,.format=9}};
    material_b.associations=&replacement;
    replacement.t.serial=32;
    key_owner=address_head=&replacement;
    native_byte[82]=73;
    const int warm_metadata=vk_texture_metadata_locked(1,&metadata,&view,&has_view);
    const Tex *warm_owner=slot_resolve(a,&view,&has_view,0,NULL);
    const uint64_t warm_image=warm_owner?warm_owner->pub.image:0;
    assert(content_detach_residency_mode(&material_b,&destroyed)==64 && destroyed && !replacement.pub.image);
    const int cold_metadata=vk_texture_metadata_locked(1,&metadata,&view,&has_view);
    const Tex *cold_owner=slot_resolve(a,&view,&has_view,0,NULL);
    if (warm_metadata || warm_owner || cold_metadata || cold_owner || restores != 1)
        fprintf(stderr,"wrong finalized owner: warm metadata=%d owner=%llu, cold metadata=%d owner=%llu, restores=%u\n",
                warm_metadata,(unsigned long long)warm_image,
                cold_metadata,(unsigned long long)(cold_owner?cold_owner->pub.image:0),restores);
    assert(!warm_metadata && !warm_owner && !cold_metadata && !cold_owner && resolve_stale && restores==1);
    assert(!replacement.pub.image && !creations && !refreshed && !create_deferred);

    // A different logical registration for the actual object can still get that exact material back.
    a->t.serial=32;
    assert(vk_texture_metadata_locked(1,&metadata,&view,&has_view)==1 && metadata.format==9);
    assert(slot_resolve(a,&view,&has_view,0,NULL)==&replacement && native_byte[replacement.pub.image]==73);
    assert(restores==2 && destroys==2 && !creations && !refreshed);
    *a=(TexSlot){0};
    key_owner=address_head=NULL;
}

int main(void){
    finalized_residency();
    VkTexImage metadata;view_t view;int has_view;uint64_t source_size;
    assert(vk_texture_metadata_locked(-1,&metadata,&view,&has_view)==0);
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==0);
    TexSlot *s=&tslots[0];s->used=1;s->t.mem=(void *)(uintptr_t)4096;s->t.mirrored=1;s->t.serial=10;
    s->t.d=(nvn_texdesc){.target=NVN_TEXTURE_TARGET_2D,.format=99,.width=64,.height=32,.depth=1,.levels=4,.samples=1};
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1);
    assert(metadata.view_type==VK_IMAGE_VIEW_TYPE_2D && metadata.format==4 && !metadata.image && !metadata.gpu_owned);
    assert(!s->ce && deferred==7); // Cold shader selection neither creates an entry nor alters ordered retry state.
    const unsigned targets[]={NVN_TEXTURE_TARGET_1D,NVN_TEXTURE_TARGET_1D_ARRAY,NVN_TEXTURE_TARGET_3D,
        NVN_TEXTURE_TARGET_2D_ARRAY,NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY,NVN_TEXTURE_TARGET_CUBEMAP,NVN_TEXTURE_TARGET_CUBEMAP_ARRAY};
    const int kinds[]={VK_IMAGE_VIEW_TYPE_1D,VK_IMAGE_VIEW_TYPE_1D_ARRAY,VK_IMAGE_VIEW_TYPE_3D,
        VK_IMAGE_VIEW_TYPE_2D_ARRAY,VK_IMAGE_VIEW_TYPE_2D_ARRAY,VK_IMAGE_VIEW_TYPE_CUBE,VK_IMAGE_VIEW_TYPE_CUBE_ARRAY};
    for(unsigned i=0;i<sizeof targets/sizeof *targets;i++){
        s->t.d.target=targets[i];s->t.d.depth=6;s->t.d.height=s->t.d.width;
        s->t.d.samples=targets[i]==NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY?4:1;
        assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1 && metadata.view_type==kinds[i]);
    }
    s->t.d.target=NVN_TEXTURE_TARGET_CUBEMAP;s->t.d.height=32;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1 && metadata.view_type==VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    s->t.d.target=NVN_TEXTURE_TARGET_BUFFER;assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==0);
    s->t.d.target=NVN_TEXTURE_TARGET_2D;s->t.d.depth=1;
    context=0;assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==-1);context=1;
    s->t.d.format=98;assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==0);s->t.d.format=99;
    Tex own={.pub={.image=55,.format=8,.type=VK_IMAGE_TYPE_2D,.view_type=VK_IMAGE_VIEW_TYPE_2D,.samples=1,
        .width=32,.height=16,.levels=2,.layers=1},.uploaded=1};
    own.t=s->t;make_key(&s->t,&own.key);key_owner=&own;
    int queries=format_calls;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1 && metadata.format==8 && metadata.levels==2);
    assert(format_calls==queries && !metadata.image); // Actual reduced/substituted representation wins over prediction.
    s->ce=&own;s->cgen=agen[0];own.released=1;s->finalized=1;TextureMaterial owned_material={0};own.material=&owned_material;own.pub.image=0;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1 && metadata.format==8);
    s->ce=(Tex *)(uintptr_t)1;s->cgen=0; // Slot reincarnation: stale cache pointers must never be dereferenced.
    s->finalized=0;own.released=0;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1);
    TextureRegistration selected={.descriptor=s->t,.source=&own,.has_view=1,
                                  .view={.set_target=1,.target=NVN_TEXTURE_TARGET_2D_ARRAY}};
    key_owner=NULL;s->registration=&selected;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1 && metadata.format==8);
    assert(format_calls==queries && !metadata.image && has_view && view.target==NVN_TEXTURE_TARGET_2D_ARRAY);
    s->registration=NULL;s->ce=NULL;
    Tex alias=own;alias.key.d.swizzle_r=NVN_TEXTURE_SWIZZLE_A;alias.pub.format=16;alias.pub.gpu_owned=1;
    address_head=&alias;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1 && metadata.format==16);
    s->finalized=1;s->has_view=1;s->v.set_format=1;s->v.format=16;
    alias.pub.mirrored=1;alias.gpu_dirty=1;own.released=1;own.uploaded=1;key_owner=&own;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==-1); // Guest overwrite may select a different integer width.
    own.pub.format=16;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1 && metadata.format==16);
    alias.pub.view_type=VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==-1);
    s->v.set_target=1;s->v.target=NVN_TEXTURE_TARGET_2D;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1 && has_view && view.set_target);
    // An exact deterministic alias supersedes unrelated ambiguous non-exact candidates.
    alias.pub.mirrored=0;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==1);
    address_head=NULL;key_owner=NULL;
    assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==0); // Finalized CPU slot without retained content.
    s->finalized=0;s->t.mem=NULL;assert(vk_texture_metadata_locked(0,&metadata,&view,&has_view)==0);
    assert(deferred==7);
    // The actual sampled capture retains read ownership and never creates a writable incarnation. Read-only
    // storage-image dependencies can use this same physical instance after their writable peers are prepared.
    deferred=0;own.pub=(VkTexImage){.image=77,.mirrored=1};own.uploaded=1;own.t.L.size=4096;sample_choice=&own;
    assert(vk_texture_effect_sample(0,0,&metadata,&view,&has_view,&source_size)==1 && metadata.image==77);
    assert(source_size==4096 && captured_image==77 && captures==1 && touches==1 && !locks);
    VkTexImage retained={.image=37,.mirrored=1};historical_sample=&retained;
    assert(vk_texture_effect_sample(0,0,&metadata,&view,&has_view,&source_size)==1 && metadata.image==37);
    assert(source_size==512 && has_view && view.target==NVN_TEXTURE_TARGET_2D_ARRAY && touches==1 && !locks);
    historical_sample=NULL;
    own.uploaded=0;assert(vk_texture_effect_sample(0,0,&metadata,&view,&has_view,&source_size)==0 && !metadata.image && !source_size);
    own.pub.gpu_owned=1;assert(vk_texture_effect_sample(0,0,&metadata,&view,&has_view,&source_size)==1);
    own.pub.gpu_owned=0;own.pub.mirrored=0;assert(vk_texture_effect_sample(0,0,&metadata,&view,&has_view,&source_size)==1);
    deferred=1;assert(vk_texture_effect_sample(0,0,&metadata,&view,&has_view,&source_size)==-1 && !metadata.image);deferred=0;
    own.pub.skip=1;alias.pub=(VkTexImage){.image=88};sample_exact=&alias;
    assert(vk_texture_effect_sample(0,1,&metadata,&view,&has_view,&source_size)==1 && metadata.image==88 && captured_image==88);
    assert(promotions==1 && captures==4 && !locks);
    sample_choice=NULL;assert(vk_texture_effect_sample(0,0,&metadata,&view,&has_view,&source_size)==0);
    // A copied image can outlive the owner that purged its cached views. Only a current exact physical owner
    // may create a new cached view; retained copies of retired instances use their consumer's deferred view.
    own.pub=(VkTexImage){.image=77,.addr=4096};own.key.addr=4096;address_head=&own;
    metadata=own.pub;assert(vk_texture_image_live_locked(&metadata));
    own.pub.gone=1;assert(!vk_texture_image_live_locked(&metadata));
    own.pub.gone=0;own.pub.image=99;assert(!vk_texture_image_live_locked(&metadata));
    alias.pub=metadata;own.next_at=&alias;assert(vk_texture_image_live_locked(&metadata));
    address_head=NULL;assert(!vk_texture_image_live_locked(&metadata));
    assert(!vk_texture_image_live_locked(NULL));metadata.image=0;assert(!vk_texture_image_live_locked(&metadata));
    puts("texture metadata: cold types/formats, retained representations, alias ambiguity and slot reincarnation passed");
}
'''


def main():
    names = ("umax", "make_key", "same_but_swizzle", "follow_size_matches", "slot_finalized_owner", "stale_kept", "slot_result_view",
             "content_require_resident", "slot_resolve",
             "shader_metadata_of", "shader_metadata_cold", "shader_metadata_equal", "shader_metadata_merge",
             "vk_texture_metadata_locked", "vk_texture_effect_sample", "vk_texture_image_live_locked")
    compiler = Path(os.environ.get("CC", "C:/Users/Andrew/AppData/Local/Programs/2NX2Navi/toolchain/msys64/ucrt64/bin/clang.exe"))
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
    work = ROOT / "build/test"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="texture-metadata-", dir=work) as directory:
        code = Path(directory) / "check.c"
        content = (ROOT / "runtime/vk_texture_content.h").read_text()
        detach = function("content_detach_residency_mode", content)
        code.write_text(PREFIX + detach + "\n".join(function(n) for n in names) + MAIN)
        executable = code.with_suffix(".exe")
        subprocess.run([str(compiler), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", str(code), "-o", str(executable)], env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)


if __name__ == "__main__":
    main()

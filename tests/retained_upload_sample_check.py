
"""Check exact prior upload permission across its own output declaration."""
from pathlib import Path
import os
import runpy
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
runtime = Path(os.environ.get("RETAINED_UPLOAD_RUNTIME", ROOT / "runtime"))
headers = Path(os.environ.get("RETAINED_UPLOAD_HEADERS", ROOT / "runtime"))
texture = (runtime / "vk_texture.c").read_text(encoding="utf8")
function = runpy.run_path(str(ROOT / "tests/ordered_graphics_feedback_check.py"))["function"]

PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define RC_REGISTRY_SINGLE_THREADED 1
#define rc_registry_runtime crc_registry_fixture_runtime
#include "resource_core_registry.c"
#undef rc_registry_runtime

typedef uint64_t VkImage;
typedef struct MemBackingCapture {RcInterval parts[2];} MemBackingCapture;
typedef struct RetainedTextureUpload RetainedTextureUpload;
typedef struct {
    VkImage image;
    uint64_t addr;
    uint32_t aspect, type, levels, samples;
    int skip,mirrored,bound,gone;
} VkTexImage;
typedef struct {
    uint64_t serial;
    uint32_t pool_flags;
    struct {uint64_t size;} L;
} tex_t;
typedef struct Tex {
    VkTexImage pub;
    tex_t t;
    MemBackingCapture *image_host_backing;
    RetainedTextureUpload *retained_sample_upload;
    uint64_t content_revision, sample_upload_sequence;
    uint32_t n_up, wgen, wt_n;
    void *alloc;
    int source_canonical, uploaded, up_retry, writable;
} Tex;
typedef struct {
    uint64_t registration;
    int native_slot;
    uint8_t canonical,projected;
} VkTextureReadKey;
typedef struct {
    struct {tex_t descriptor;} *registration;
} TexSlot;
typedef struct {
    void *raw, *pin;
    RetainedTextureUpload *upload;
    MemBackingCapture *sample_mapping;
    RcRegistryPosition sample_read;
    uint64_t sample_pool, sample_publication, sample_registration, sample_address, sample_size;
    uint64_t sample_mapping_stamp, sample_protection_stamp;
    uint32_t sample_wgen, sample_writes;
    const char *sample_upload_gate;
} VkTexturePresentSource;
enum { RC_BACKING_OK, VK_IMAGE_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT=1, MATERIAL_LEVELS=10,
       NVN_MEMORY_POOL_FLAGS_PHYSICAL=2, NVN_MEMORY_POOL_FLAGS_VIRTUAL=4, TEX_SOURCE_CLEAN=0 };
static RcRegistry *registry;
static Tex entry;
static TexSlot slot;
static RcRegistryPosition read={22,40};
static uint64_t publication=343, pool=9, observed_upload=12, mapping_stamp=1, protection_stamp=1;
static unsigned lock_depth, mapping_holds;
static int cache_current=1, backing_current=1, dirty, stopped, device_lost, recorder_failed;
static const MemBackingCapture actual={{{0x90000,7,{19,0x6000,32}},{0x90020,8,{19,0x8000,32}}}};
static void lock(void){lock_depth++;}
static void unlock(void){assert(lock_depth--);}
#define rt_fatal(...) abort()
static RcRegistry *rc_registry_runtime(void){return registry;}
static const TexSlot *texture_read_key_slot(VkTextureReadKey k){
    assert(lock_depth);
    return k.registration==70&&k.native_slot==4&&cache_current ? &slot:NULL;
}
static Tex *texture_binding_entry(const TexSlot *s,int i){assert(lock_depth&&s==&slot&&i==4);return &entry;}
static int source_private_sample(const Tex *t){return t->pub.mirrored&&!t->pub.gone&&!t->writable&&!t->pub.bound;}
static RcRegistryPosition vk_exec_resource_position(void){return read;}
static uint64_t vk_exec_texture_pool(void){return pool;}
static uint64_t vk_texture_registration_publication(void){return publication;}
static uint64_t vk_upload_dependency(void){return observed_upload;}
static int vk_sync_stopping(void){return stopped;}
static int vk_device_is_lost(void){return device_lost;}
static int vk_exec_recording_failed(void){return recorder_failed;}
static uint64_t mem_mapping_serial(void){return mapping_stamp;}
static uint64_t mem_protection_serial(void){return protection_stamp;}
static int mem_backing_current(const MemBackingCapture *p){return p&&backing_current;}
static int mem_mapping_range_unchanged(uint64_t addr,uint64_t bytes,uint64_t stamp){
    return addr==0x90000&&bytes==64&&stamp==mapping_stamp;
}
static int mem_protection_range_unchanged(uint64_t addr,uint64_t bytes,uint64_t stamp){
    return addr==0x90000&&bytes==64&&stamp==protection_stamp;
}
static int mem_backing_capture(uint64_t addr,uint64_t bytes,MemBackingCapture **p){
    assert(addr==0x90000&&bytes==64&&!*p);
    *p=malloc(sizeof **p);assert(*p);**p=actual;mapping_holds++;
    return RC_BACKING_OK;
}
static void mem_backing_release(MemBackingCapture **p){if (*p) {assert(mapping_holds--);free(*p);*p=NULL;}}
static const RcInterval *mem_backing_intervals(const MemBackingCapture *p,uint32_t *count){*count=2;return p->parts;}
static int vk_mem_changed(uint64_t addr,uint64_t bytes,uint32_t observed,uint32_t *now){
    assert(addr==0x90000&&bytes==64&&observed==92);
    *now=observed+dirty;return dirty;
}
'''

CASES = r'''
int main(void) {
    registry=rc_registry_create((RcRegistryConfig){16,16,4});
    assert(registry);
    slot.registration=calloc(1,sizeof *slot.registration);
    assert(slot.registration);
    slot.registration->descriptor=(tex_t){.serial=5,.L.size=64};
    entry=(Tex){.pub={.image=31,.addr=0x90000,.type=VK_IMAGE_TYPE_2D,.aspect=1,.mirrored=1,.levels=1,.samples=1},
        .t=slot.registration->descriptor,.image_host_backing=(MemBackingCapture*)&actual,.content_revision=7,
        .sample_upload_sequence=11,.n_up=4,.wgen=92,.wt_n=1,.alloc=&entry,.uploaded=1};
    VkTextureReadKey key={.registration=70,.native_slot=4};
    VkTexturePresentSource source={.raw=&entry};
    lock();
    assert(rc_registry_declare_through(registry,read)==RC_REGISTRY_OK);
    vk_texture_retain_compat_sample_locked(&source,key,31);
    assert(source.upload && source.upload->references==1 && mapping_holds==1);
    RcRegistryWrite own_odd, normalized;
    const RcRegistryPosition successor={read.scene,read.ordinal+1};
    assert(rc_registry_write_guard_begin_reason_at(registry,RC_REGISTRY_UNKNOWN,successor,
        RC_REGISTRY_GUARD_COMMAND_OPAQUE,&own_odd)==RC_REGISTRY_OK);
    assert(rc_registry_write_begin_at(registry,actual.parts[1].range,RC_REGISTRY_BUFFER,successor,&normalized)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,successor)==RC_REGISTRY_OK);
    cache_current=0; // issue uses the already held incarnation, not a following slot entry
    assert(vk_texture_retained_compat_sample_current_locked(&source,70,31));
    assert(!vk_texture_retained_compat_sample_current_locked(&source,71,31));
    assert(!vk_texture_retained_compat_sample_current_locked(&source,70,32));
    publication++;
    assert(!vk_texture_retained_compat_sample_current_locked(&source,70,31));
    publication--;
    entry.n_up++;
    assert(!vk_texture_retained_compat_sample_current_locked(&source,70,31));
    entry.n_up--;
    entry.content_revision++;
    assert(!vk_texture_retained_compat_sample_current_locked(&source,70,31));
    entry.content_revision--;
    dirty=1;
    assert(!vk_texture_retained_compat_sample_current_locked(&source,70,31));
    dirty=0;
    mapping_stamp++;
    assert(!vk_texture_retained_compat_sample_current_locked(&source,70,31));
    mapping_stamp--;
    stopped=1;
    assert(!vk_texture_retained_compat_sample_current_locked(&source,70,31));
    stopped=0;
    observed_upload=10; // Original recording is no longer in the live upload timeline.
    assert(!strcmp(retained_compat_upload_issue_gate(&source,70,31,NULL,NULL),"upload-sequence"));
    assert(!vk_texture_retained_compat_sample_current_locked(&source,70,31));
    observed_upload=12;
    assert(vk_texture_retained_compat_sample_current_locked(&source,70,31));

    // After COW detaches this VkImage, only the part's already-created view may use it.
    RetainedTextureUpload *old=source.upload;
    old->references++;
    retained_texture_upload_detach(&entry);
    assert(!entry.retained_sample_upload&&!entry.sample_upload_sequence);
    assert(!strcmp(retained_compat_upload_issue_gate(&source,70,31,NULL,NULL),"detached-before-view"));
    assert(!vk_texture_retained_compat_sample_current_locked(&source,70,31));
    retained_texture_upload_release(source.upload);
    assert(old->references==1&&!old->entry);
    retained_texture_upload_cancel(old,0);
    mem_backing_release(&source.sample_mapping);
    assert(!mapping_holds);
    unlock();

    // Holding only a VkImage never certifies content across an earlier unresolved global writer.
    rc_registry_destroy(registry);
    registry=rc_registry_create((RcRegistryConfig){16,16,4});
    RcRegistryWrite earlier;
    assert(rc_registry_write_guard_begin_reason_at(registry,RC_REGISTRY_UNKNOWN,
        (RcRegistryPosition){22,37},RC_REGISTRY_GUARD_COMMAND_OPAQUE,&earlier)==RC_REGISTRY_OK);
    assert(rc_registry_declare_through(registry,read)==RC_REGISTRY_OK);
    source=(VkTexturePresentSource){.raw=&entry};
    entry.sample_upload_sequence=11;
    cache_current=1;
    lock();
    vk_texture_retain_compat_sample_locked(&source,key,31);
    assert(!source.upload&&!mapping_holds);
    assert(!strcmp(source.sample_upload_gate,"normalized-range"));
    unlock();
    free(slot.registration);
    rc_registry_destroy(registry);
    puts("Retained upload: own odd write, original slot departure, earlier opaque blocker, reset/generation and cancellation passed");
}
'''

upload = texture[texture.index("struct RetainedTextureUpload {"):texture.index("static int image_owner_replaced(")]
checks = ("retained_texture_sample_range_gate", "vk_texture_retain_compat_sample_locked", "retained_compat_upload_issue_gate",
          "vk_texture_retained_compat_sample_current_locked")
code = PREFIX + upload + "\n".join(function(texture, name) for name in checks) + CASES
compiler = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="retained-upload-") as folder:
    source = Path(folder) / "check.c"
    executable = source.with_suffix(".exe")
    source.write_text(code, encoding="utf8")
    subprocess.run([str(compiler), "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-O1", "-fsanitize=undefined",
                    "-fsanitize-trap=all", f"-I{headers}", str(source), "-o", str(executable)],
                   check=True, env=env)
    subprocess.run([str(executable)], check=True, env=env)

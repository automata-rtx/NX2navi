"""Exercise the exact registered native sample selector and its finite read receipt."""
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
#include <string.h>

typedef uint64_t VkImage;
typedef int VkFormat;
typedef struct {uint64_t scene,ordinal;} RcRegistryPosition;
typedef struct {int id;} RcInterval;
typedef struct {unsigned id,index;} RcRegistryWrite;
typedef struct {int known_unhanded;} TextureUseBatch;
typedef struct VkTextureAliasRead VkTextureAliasRead;
typedef struct {uint64_t serial; struct {int target,format,width,height,samples,levels;} d;
                struct {uint64_t size; uint32_t layers;} L;} tex_t;
typedef struct {int set_format,set_target,set_levels,set_layers,set_dsmode,set_swizzle;
                int format,target,base_level,nlevels,min_layer,nlayers,dsmode,swz[4];} view_t;
typedef struct {VkImage image; int gpu_owned,gone,usage,type,view_type,skip,format; uint32_t aspect;
                uint32_t width,height,levels,layers,depth,samples,scale;} VkTexImage;
typedef struct {uint32_t aspectMask,baseMipLevel,levelCount,baseArrayLayer,layerCount;} VkImageSubresourceRange;
typedef int TexKey;
typedef struct {tex_t descriptor; struct {uint64_t id;} handle; view_t view; int has_view;} TextureRegistration;
typedef struct Tex {
    VkTexImage pub;
    tex_t t;
    TexKey key;
    int released,retiring,superseded,overwritten;
    unsigned image_private_mapping_count,image_mapped_hold_count;
    const RcInterval *image_private_mapping;
    const RcRegistryWrite *image_mapped_holds;
    unsigned alias_full_count;
    const RcInterval *alias_full_demand;
    void *bound_to;
    struct {int vk;} f;
} Tex;
typedef struct {TextureRegistration *registration; Tex *entry; view_t view; int finalized;} TexSlot;
typedef struct {uint64_t registration; int native_slot,canonical,projected;} VkTextureReadKey;
typedef struct {int recorded;const char *stage;RcRegistryPosition position;} VkTextureSampleRefusal;
typedef struct {const VkTexImage *image; view_t view; int has_view,native_slot,historical;
                uint64_t registration; int canonical,projected;} VkTextureSample;
typedef enum {VK_TEXTURE_SAMPLE_READY, VK_TEXTURE_SAMPLE_PREPARE,VK_TEXTURE_SAMPLE_FALLBACK,
              VK_TEXTURE_SAMPLE_ABSENT,VK_TEXTURE_SAMPLE_GONE} VkTextureSampleStatus;
enum {VK_TEXTURE_REGISTRATION_CURRENT=0,VK_TEXTURE_REGISTRATION_EXACT=1,VK_TEXTURE_REGISTRATION_GONE=-2,
      VK_TEXTURE_REGISTRATION_ABSENT=-1,VK_TEXTURE_READ_REQUIRED=1,VK_TEXTURE_READ_FORCE_COMPAT=2,
      VK_TEXTURE_READ_EXACT_DETAIL=8,
      VK_TEXTURE_BINDING_SOURCE=2,VK_IMAGE_USAGE_SAMPLED_BIT=1,VK_IMAGE_TYPE_2D=2,VK_IMAGE_VIEW_TYPE_2D=3,
      NVN_TEXTURE_TARGET_2D=4,VK_IMAGE_ASPECT_COLOR_BIT=5,NVN_TEXTURE_TARGET_2D_ARRAY=6,
      VK_IMAGE_VIEW_TYPE_2D_ARRAY=7,VK_IMAGE_ASPECT_DEPTH_BIT=8,VK_FORMAT_UNDEFINED=0,
      NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH=0,VK_NULL_HANDLE=0};

static TextureRegistration record = {.descriptor = {81,{4,63,1280,720,1,1},{0x3c0000,1}},
    .handle = {12058}, .view = {.set_format=1,.format=63}, .has_view = 1};
static RcInterval meaningful[] = {{1},{2}};
static const RcRegistryWrite arc[] = {{104032,2101}};
static TextureUseBatch active_part;
typedef struct {RcRegistryPosition read;int host_layout,has_view,bound_relocated;view_t view;VkImageSubresourceRange volume_mip;} VkTextureStorageTarget;
static unsigned published,marked,dispositions;
static int initialized,selected_storage=1,opaque_written=1;
static Tex donor = {.pub = {.image=71,.gpu_owned=1,.usage=1,.type=2,.view_type=3,.aspect=5,.format=63,
                            .width=1280,.height=720,.levels=1,.layers=1,.depth=1,.samples=1},
    .t = {81,{4,63,1280,720,1,1},{0x3c0000,1}}, .key=63,
    .image_private_mapping_count=1,.image_mapped_hold_count=1,.image_private_mapping=meaningful,.image_mapped_holds=arc,
    .alias_full_count=2, .alias_full_demand=meaningful, .f={63}};
static TexSlot tslots[1] = {{&record,&donor,{.set_format=1,.format=63},0}};
static RcRegistryPosition at = {5936,1042};
static unsigned legacy_selects,legacy_receipts,coverage_checks;
static int covering=1, native_continuation, mapped=1, retired=0, historical_status;
static struct {unsigned calls;} binding_path;

static RcRegistryPosition vk_exec_resource_position(void){return at;}
static void *rc_registry_runtime(void){return NULL;}
static int rc_registry_position_valid(RcRegistryPosition read){return read.scene && read.ordinal;}
static int rc_registry_native_image_covers(void *r,uint64_t image,const RcInterval *d,uint32_t n,RcRegistryPosition p){
    (void)r;
    coverage_checks++;
    return covering && image == 71 && d == meaningful && n == 2 && p.scene == 5936 && p.ordinal == 1042;
}
static int rc_registry_native_image_resident_covers(void *r,uint64_t image,const RcRegistryWrite *holds,
    uint32_t hold_count,const RcInterval *d,uint32_t n,RcRegistryPosition p) {
    (void)r;
    return native_continuation && image==71 && holds==arc && hold_count==1 &&
           d==meaningful && n==2 && p.scene==5936 && p.ordinal==1042;
}
static Tex *image_alias_sample_entry_locked(const TexSlot *slot,int native){assert(native==0);return slot->entry;}
static const RcInterval *image_alias_plane_demand(Tex *entry,const VkImageSubresourceRange *range,uint32_t *count){
    assert(entry==&donor && range->aspectMask==5 && range->levelCount==1 && range->layerCount==1);
    *count=2;
    return initialized ? meaningful : NULL;
}
static int image_alias_source_current(const Tex *e){return e==&donor && mapped;}
static VkFormat alias_depth_format(int guest){(void)guest;return VK_FORMAT_UNDEFINED;}
static int image_alias_load_current(Tex *entry,const VkImageSubresourceRange *planes,RcRegistryPosition read){
    (void)entry;(void)planes;(void)read;assert(0 && "the color test uses a color projection");return 0;
}
static Tex *storage_target_current_locked(const VkTextureStorageTarget *target,int issued) {
    assert(issued); return target && selected_storage ? &donor : NULL;
}
static TextureUseBatch *texture_use_active(uint64_t scene,int create) {
    assert(scene==5936 && !create); return &active_part;
}
static TextureUseBatch *texture_use_part_retain(TextureUseBatch *p) { assert(p==&active_part);return p; }
static int texture_use_part_known_unhanded(const void *p) { return ((const TextureUseBatch *)p)->known_unhanded; }
static void texture_use_part_release(void *p) { assert(p==&active_part); }
static void resource_effects_retain_part_disposition(void *p,int (*unhanded)(const void *),void *owner,
    void (*release)(void *,void *,int)) {
    assert(p==&active_part && owner==p && !unhanded(p) && published);
    dispositions++;
    release(p,owner,0);
}
static int vk_texture_storage_target_written(const VkTextureStorageTarget *target,VkTextureAliasRead *seed,
    const VkImageSubresourceRange *planes) {
    assert(target && !seed && !planes);
    published++;
    return opaque_written;
}
static int storage_word_format(int guest,const void *format) {
    return guest==63 && format==&donor.f;
}
static int storage_word_view_plane(const view_t *view,int has_view,const VkTexImage *image,
    VkImageSubresourceRange *plane) {
    (void)image;
    *plane=(VkImageSubresourceRange){5,0,1,0,1};
    return has_view && view->set_format && view->format==98;
}
static int last_projection_layout;
static int resource_effects_image_preserved_intervals(uint64_t image,const RcInterval *mapping,
    const RcRegistryWrite *holds,uint32_t count,RcRegistryPosition output,int exact_projection_layout) {
    last_projection_layout=exact_projection_layout;
    assert(image==71 && mapping==meaningful && holds==arc && count==1 && published && dispositions);
    assert(output.scene==5936 && output.ordinal==1043);
    marked++;
    return 1;
}
static void make_key(const tex_t *d,TexKey *key){*key=d->d.format;}
static int same_but_swizzle(const TexKey *a,const TexKey *b){return *a==*b;}
static void slot_result_view(const TexSlot *s,const Tex *e,view_t *v,int *hv){
    assert(e==&donor);*v=s->view;*hv=1;
}
static const TexSlot *texture_read_key_slot(VkTextureReadKey key){
    return !retired && key.native_slot == 0 && key.registration==record.handle.id ? tslots : NULL;
}
static int texture_registration_resolve(int logical,const TextureRegistration **history,int *native){
    assert(logical==4688);*history=NULL;*native=0;return VK_TEXTURE_REGISTRATION_CURRENT;
}
static int texture_history_sample_locked(int logical,VkTextureSample *sample,int *native,int recording){
    const TextureRegistration *history;
    (void)sample;(void)recording;
    const int current=texture_registration_resolve(logical,&history,native);
    return historical_status ? historical_status : current;
}
static VkTextureSampleStatus vk_texture_sample_query_locked(int native,VkTextureSample *out){
    (void)native;(void)out;return VK_TEXTURE_SAMPLE_FALLBACK;
}
static VkTextureSampleStatus texture_sample_prepare_locked(int native,VkTextureSample *out,int optional,VkTextureSampleRefusal *refused){
    (void)native;(void)out;(void)optional;(void)refused;assert(0);return VK_TEXTURE_SAMPLE_GONE;
}
static void vk_texture_sample_use_locked(const VkTextureSample *out){(void)out;assert(0);}
static const VkTexImage *texture_by_slot_locked(int native,view_t *view,int *hv,VkTextureSampleStatus *status,VkTextureSampleRefusal *refused){
    (void)refused;
    assert(native==0 && *status==VK_TEXTURE_SAMPLE_FALLBACK);
    legacy_selects++;slot_result_view(tslots,&donor,view,hv);return &donor.pub;
}
static void sample_refusal_record(VkTextureSampleRefusal *refused,const TexSlot *slot,const Tex *e,const char *gate){
    (void)refused;(void)slot;(void)e;(void)gate;assert(0 && "the color test uses a color projection");
}
static uint64_t texture_slot_registration(int native){assert(native==0);return record.handle.id;}
static int texture_registration_source_attach(TexSlot *s,Tex *e){(void)s;(void)e;assert(0);return 0;}
static int texture_read_key_compatibility(const TexSlot *s,int n){return s==tslots&&n==0;}
static int vk_texture_binding_receipt_locked(int id,VkImage expected,RcRegistryPosition *edge){
    (void)edge;legacy_receipts++;return id==0&&expected==71;
}
static int vk_texture_binding_schedule_locked(int id,VkImage expected,RcRegistryPosition *edge){
    return vk_texture_binding_receipt_locked(id,expected,edge);
}
static Tex *texture_read_key_image(const TexSlot *s,VkImage e){(void)s;(void)e;return NULL;}
static void pool_mapping_source(const Tex *entry) { (void)entry; }
static int source_binding_receipt(Tex *e,RcRegistryPosition *p){(void)e;(void)p;return 0;}
static int texture_binding_schedule(Tex *e,VkImage i,RcRegistryPosition *p,int proof){
    (void)e;(void)i;(void)p;return proof;
}
"""

CASE = r"""
static void check_preserving_storage_write(void) {
    const VkTextureStorageTarget target={.read={5936,1042}};
    const VkImageSubresourceRange planes={5,0,1,0,1}, wrong_aspect={8,0,1,0,1};
    const VkImageSubresourceRange sibling_mip={5,1,1,0,1}, sibling_layer={5,0,1,1,1};
    initialized=1;
    assert(vk_texture_storage_target_written_preserving(&target,&planes));
    assert(published==1 && marked==1 && dispositions==1);
    initialized=0;
    assert(vk_texture_storage_target_written_preserving(&target,&planes));
    assert(published==2 && marked==1 && dispositions==1);
    initialized=1;
    assert(vk_texture_storage_target_written_preserving(&target,&wrong_aspect));
    assert(published==3 && marked==1 && dispositions==1);
    assert(vk_texture_storage_target_written_preserving(&target,&sibling_mip));
    assert(vk_texture_storage_target_written_preserving(&target,&sibling_layer));
    assert(published==5 && marked==1 && dispositions==1);
    VkTextureStorageTarget relocated=target;
    relocated.bound_relocated=1;
    assert(vk_texture_storage_target_written_preserving(&relocated,&planes));
    assert(published==6 && marked==1 && dispositions==1);
    opaque_written=0;
    assert(!vk_texture_storage_target_written_preserving(&target,&planes));
    assert(published==7 && marked==1 && dispositions==1);
    opaque_written=1;
    assert(!last_projection_layout);
    VkTextureStorageTarget words=target;
    words.has_view=1;
    words.view.set_format=1;
    words.view.format=98;
    assert(vk_texture_storage_target_written_preserving(&words,&planes) && last_projection_layout);
    assert(storage_target_written_preserving(&words,&planes,0) && !last_projection_layout);
    donor.pub.levels=2;
    assert(vk_texture_storage_target_written_preserving(&words,&planes) && !last_projection_layout);
    donor.pub.levels=1;
}

static void remains_legacy(VkTextureSample *sample) {
    assert(vk_texture_select_sample_locked(4688,0,sample)==VK_TEXTURE_SAMPLE_FALLBACK && !sample->projected);
    assert(vk_texture_resolve_sample_locked(4688,VK_TEXTURE_READ_REQUIRED,sample)==VK_TEXTURE_SAMPLE_FALLBACK &&
           !sample->projected && !sample->registration);
}

int main(void){
    VkTextureSample sample;
    assert(vk_texture_resolve_sample_locked(4688,VK_TEXTURE_READ_FORCE_COMPAT|VK_TEXTURE_READ_REQUIRED,&sample)
           ==VK_TEXTURE_SAMPLE_FALLBACK && !sample.projected);
    assert(legacy_selects==1 && coverage_checks==0);
    assert(vk_texture_select_sample_locked(4688,VK_TEXTURE_READ_EXACT_DETAIL,&sample)==VK_TEXTURE_SAMPLE_READY);
    assert(sample.projected && sample.image==&donor.pub);
    assert(vk_texture_resolve_sample_locked(4688,VK_TEXTURE_READ_REQUIRED|VK_TEXTURE_READ_EXACT_DETAIL,&sample)
           ==VK_TEXTURE_SAMPLE_READY && sample.projected && sample.image==&donor.pub);
    assert(legacy_selects==1);
    unsigned examined=coverage_checks;
    assert(vk_texture_select_sample_locked(4688,VK_TEXTURE_READ_FORCE_COMPAT|VK_TEXTURE_READ_EXACT_DETAIL,
                                           &sample)==VK_TEXTURE_SAMPLE_FALLBACK && !sample.projected);
    assert(coverage_checks==examined);
    historical_status=VK_TEXTURE_REGISTRATION_GONE;
    assert(vk_texture_select_sample_locked(4688,VK_TEXTURE_READ_EXACT_DETAIL,&sample)==VK_TEXTURE_SAMPLE_GONE &&
           !sample.projected && !sample.image);
    historical_status=2;  // Model unresolved history even though this native slot now has a complete current image.
    assert(vk_texture_select_sample_locked(4688,VK_TEXTURE_READ_EXACT_DETAIL,&sample)==VK_TEXTURE_SAMPLE_FALLBACK &&
           !sample.projected && !sample.image);
    assert(coverage_checks==examined);
    historical_status=0;
    assert(vk_texture_select_sample_locked(4688,0,&sample)==VK_TEXTURE_SAMPLE_READY);
    assert(vk_texture_resolve_sample_locked(4688,VK_TEXTURE_READ_REQUIRED,&sample)==VK_TEXTURE_SAMPLE_READY);
    VkTextureSampleRefusal refused = {.recorded=1};
    assert(vk_texture_resolve_sample_report_locked(4688,VK_TEXTURE_READ_REQUIRED,&sample,&refused)
           ==VK_TEXTURE_SAMPLE_READY && !refused.recorded);
    assert(legacy_selects==1 && sample.projected && !sample.canonical && sample.image==&donor.pub);
    const VkTextureReadKey key = {sample.registration,sample.native_slot,0,1};
    VkTexImage earlier = *sample.image;
    assert(key.registration==12058 && sample.view.set_format && sample.view.format==63);
    assert(vk_texture_projected_sample_receipt_locked(key,71,&sample) && sample.projected && sample.image == &donor.pub);
    RcRegistryPosition edge = {0};
    assert(vk_texture_resolved_receipt_locked(key,71,&edge)==VK_TEXTURE_BINDING_SOURCE);
    assert(edge.scene==5936 && edge.ordinal==1043);
    assert(vk_texture_resolved_schedule_locked(key,71,&edge)==VK_TEXTURE_BINDING_SOURCE);
    assert(!vk_texture_resolved_receipt_locked(key,70,NULL));
    assert(!vk_texture_resolved_receipt_locked(key,0,NULL));

    covering=0;  // Later overlapping authoritative writer, or only a partial projected winner.
    assert(!vk_texture_resolved_receipt_locked(key,71,NULL) && !vk_texture_resolved_schedule_locked(key,71,NULL));
    remains_legacy(&sample);
    native_continuation=1;
    assert(vk_texture_resolved_receipt_locked(key,71,&edge)==VK_TEXTURE_BINDING_SOURCE);
    assert(vk_texture_select_sample_locked(4688,VK_TEXTURE_READ_EXACT_DETAIL,&sample)==VK_TEXTURE_SAMPLE_READY);
    assert(sample.image==&donor.pub && sample.view.format==63 && sample.registration==12058);
    const int formats[]={37,81,98,122}; // RGBA8, RG16, R32 integer and packed RGB.
    for (size_t i=0;i<sizeof formats/sizeof *formats;i++) {
        record.descriptor.d.format=donor.t.d.format=donor.key=formats[i];
        tslots[0].view.format=donor.pub.format=donor.f.vk=formats[i];
        assert(vk_texture_select_sample_locked(4688,VK_TEXTURE_READ_EXACT_DETAIL,&sample)==VK_TEXTURE_SAMPLE_READY);
        assert(sample.image==&donor.pub && sample.view.format==formats[i] && sample.registration==12058);
    }
    record.descriptor.d.format=donor.t.d.format=donor.key=63;
    tslots[0].view.format=donor.pub.format=donor.f.vk=63;
    donor.image_mapped_hold_count=0;
    remains_legacy(&sample);
    donor.image_mapped_hold_count=1;
    tslots[0].view.format=98;
    remains_legacy(&sample);
    tslots[0].view.format=63;
    at.ordinal+=2;
    remains_legacy(&sample);
    at.ordinal-=2;
    native_continuation=0;
    covering=1;
    donor.alias_full_count=0; remains_legacy(&sample); donor.alias_full_count=2;
    donor.alias_full_count=1; remains_legacy(&sample); donor.alias_full_count=2;
    donor.released=1; remains_legacy(&sample); donor.released=0;
    donor.overwritten=1; remains_legacy(&sample); donor.overwritten=0;
    donor.retiring=1; remains_legacy(&sample); donor.retiring=0;
    donor.superseded=1; remains_legacy(&sample); donor.superseded=0;
    donor.pub.gone=1; remains_legacy(&sample); donor.pub.gone=0;
    donor.t.serial++; remains_legacy(&sample); donor.t.serial--;
    donor.pub.samples=2; remains_legacy(&sample); donor.pub.samples=1;
    donor.pub.skip=1;
    assert(vk_texture_select_sample_locked(4688,VK_TEXTURE_READ_EXACT_DETAIL,&sample)==VK_TEXTURE_SAMPLE_FALLBACK &&
           !sample.projected);
    assert(vk_texture_resolve_sample_locked(4688,VK_TEXTURE_READ_REQUIRED|VK_TEXTURE_READ_EXACT_DETAIL,&sample)
           ==VK_TEXTURE_SAMPLE_FALLBACK && !sample.projected && !sample.registration);
    donor.pub.skip=0;
    donor.pub.width--; remains_legacy(&sample); donor.pub.width++;
    tslots[0].view.format=56; remains_legacy(&sample); tslots[0].view.format=63;
    tslots[0].view.set_levels=1; tslots[0].view.base_level=1;
    remains_legacy(&sample); tslots[0].view.set_levels=0;
    mapped=0; remains_legacy(&sample); mapped=1;
    at.ordinal=1044; remains_legacy(&sample); at.ordinal=1042;
    assert(!legacy_receipts && !memcmp(&earlier,&donor.pub,sizeof earlier));

    record.handle.id++;
    assert(!vk_texture_resolved_receipt_locked(key,71,NULL));
    assert(vk_texture_resolve_sample_locked(4688,VK_TEXTURE_READ_REQUIRED,&sample)==VK_TEXTURE_SAMPLE_READY);
    assert(sample.registration==12059);
    retired=1;
    assert(!vk_texture_projected_sample_receipt_locked(key,71,&sample) && !legacy_receipts);
    check_preserving_storage_write();
    puts("current GPU sampled image used once, ordered whole-view proof preserved, stale and incomplete refusals");
}
"""


def main():
    source = (ROOT / "runtime/vk_texture.c").read_text(encoding="utf-8")
    selected = "\n".join(function(source, name) for name in (
        "image_alias_sample_entry_stale", "image_alias_resident_current", "texture_projected_color_current",
        "texture_projected_single_plane_current", "storage_preserving_part_release",
        "storage_target_written_preserving", "vk_texture_storage_target_written_preserving",
        "texture_projected_sample_report_locked", "texture_projected_sample_locked",
        "vk_texture_projected_sample_receipt_locked", "vk_texture_select_sample_locked",
        "vk_texture_resolve_sample_report_locked", "vk_texture_resolve_sample_locked", "vk_texture_resolved_receipt_locked",
        "vk_texture_resolved_schedule_locked"))
    with tempfile.TemporaryDirectory(prefix="projected-sample-") as temp:
        directory = Path(temp)
        c = directory / "test.c"
        c.write_text(STUBS + selected + CASE, encoding="utf-8")
        executable = directory / ("test.exe" if os.name == "nt" else "test")
        subprocess.run([CC, "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                        "-fsanitize-trap=all", str(c), "-o", str(executable)], check=True)
        subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()


"""Exercise the actual direct/motion manifest consumer against the real material lifetime service."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
from check_content import GENERATED_MARKER
from bind_recipe_check import function

PREFIX = r'''
#define main existing_scene_checks
#include "resource_core_scene_check.c"
#undef main
typedef struct { TextureImagePublication *images; } CbRes;
typedef unsigned VkDescriptorSet;
typedef int VkPipeInfo;
typedef struct { TextureImagePublication *bind_manifest; CbRes *res; int flat3d_pool_ready; } DrawCtx;
typedef struct { DrawCtx *dc; } VkExec;
enum { MAX_DESC=1024 };
static int deferred, failed, pool_will_defer, flat3d_pool_consuming;
static int vk_texture_deferred(void) { return deferred; }
static int vk_exec_recording_failed(void) { return failed; }
static int graphics_pool_bind(VkExec *ex,const VkPipeInfo *info,VkDescriptorSet *out) {
    (void)ex; (void)info; (void)out; return 0;
}
static VkDescriptorSet pool_sync_draw(VkExec *ex) {
    assert(flat3d_pool_consuming == ex->dc->flat3d_pool_ready);
    deferred = pool_will_defer;
    return deferred || failed ? 0 : 23;
}
static CbRes *cb_res(VkExec *ex) { assert(ex->dc->res);return ex->dc->res; }
typedef struct VkTexturePresentSource VkTexturePresentSource;
static const Tex *nvn_texture_pin_attachment(const tex_t *pin) {
    return pin == &pinned_texture && pinned_refs ? &pinned_owner : NULL;
}
'''

TEST = r'''
static void selected_sample_handoff(void) {
    TextureMaterialLedger ledger = {0};
    TextureMaterial material;
    init(&material, &ledger, 1003, 7);
    TextureMaterialInstance *instance = material.resident.instance;
    VkTexturePresentSource managed = {0}, raw = {0};
    lock();
    content_use_image_remove(instance);
    assert(!content_use_image_find(1003));
    assert(texture_use_retain_instance(&managed.material, instance, NULL) == RC_LIFETIME_OK);
    native_owner.retired = 0;
    raw.raw = vk_mem_image_hold(8001);
    raw.pin = nvn_texture_pin_retain_attachment(pinned_texture.serial, &pinned_owner);
    native_owner.retired = 1;
    assert(retire(&material) == instance);
    recording_fence = 0;
    assert(!vk_texture_record_selected_sample_locked(&managed, 1003));
    assert(!vk_texture_record_selected_sample_locked(&raw, 8001));
    recording_fence = 120;
    assert(!vk_texture_record_selected_sample_locked(&managed, 1004));
    assert(!vk_texture_record_selected_sample_locked(&raw, 8002));
    assert(instance->lifetime.captures == 1 && native_owner.refs == 1 && pinned_refs == 1);
    for (int i = 0; i < 2; ++i) {
        assert(vk_texture_record_selected_sample_locked(&managed, 1003));
        assert(vk_texture_record_selected_sample_locked(&raw, 8001));
    }
    assert(instance->lifetime.captures == 2 && native_owner.refs == 2 && pinned_refs == 2);
    unlock();
    TextureUseBatch *first = vk_texture_uses_seal(120, 7);
    assert(first && first->head->use.instance == instance && first->native->hold == raw.raw);

    recording_fence = 121;
    lock();
    assert(vk_texture_record_selected_sample_locked(&managed, 1003));
    assert(vk_texture_record_selected_sample_locked(&raw, 8001));
    texture_use_unlink(&managed.material);
    assert(texture_material_use_cancel(&managed.material.use) == RC_LIFETIME_OK);
    texture_uses.nodes--;
    vk_mem_image_release(raw.raw);
    nvn_texture_pin_release(raw.pin);
    unlock();
    TextureUseBatch *second = vk_texture_uses_seal(121, 7);
    assert(second && second->id != first->id && second->head->use.instance == instance);
    vk_texture_uses_handed(first);
    vk_texture_uses_abandon(second);
    vk_texture_uses_complete(119, 7);
    assert(instance->lifetime.captures == 2 && native_owner.refs == 2 && pinned_refs == 2);
    vk_texture_uses_complete(120, 7);
    assert(!instance->lifetime.captures && !native_owner.refs && !pinned_refs);
    destroy(instance, &material);
    assert(!texture_uses.nodes && !texture_uses.batches && !locked);
    puts("selected descriptor source: exact detached material and native hold, distinct recording generations and cancellation passed");
}

int main(void) {
    assert(existing_scene_checks()==0);
    TextureMaterialLedger ledger={0};TextureMaterial a,b;
    init(&a,&ledger,1001,0);init(&b,&ledger,1002,0);
    TextureMaterialInstance *ia=a.resident.instance,*ib=b.resident.instance;
    CbRes native={0};DrawCtx dc={.res=&native,.flat3d_pool_ready=1};VkExec ex={.dc=&dc};
    recording_fence=80;
    binding_manifest_begin(&ex);binding_manifest_select(&ex,0,1001);
    assert(ia->lifetime.captures==1);
    assert(retire(&a)==ia); // A source can lose residency after selection and before descriptor installation.
    VkDescriptorSet pool=0;
    assert(binding_manifest_prepare_pool(&ex,NULL,&pool)&&pool==23&&ia->lifetime.captures==1);
    binding_manifest_publish(&ex);
    assert(ia->lifetime.captures==2 && !dc.bind_manifest->head && !dc.bind_manifest->entry_count);
    TextureUseBatch *first=vk_texture_uses_seal(80,0);assert(first);
    binding_manifest_begin(&ex);binding_manifest_select(&ex,0,1001);binding_manifest_publish(&ex);
    TextureUseBatch *second=vk_texture_uses_seal(80,0);
    assert(second&&second!=first&&second->id!=first->id&&ia->lifetime.captures==3);
    // A partially resolved direct or motion command is abandoned. Its builder must keep no dormant physical
    // capture, while earlier immutable native packets and both recorded parts retain their own obligations.
    binding_manifest_begin(&ex);binding_manifest_select(&ex,9,1002);
    assert(ib->lifetime.captures==1);assert(retire(&b)==ib);
    pool_will_defer=1;
    assert(!binding_manifest_prepare_pool(&ex,NULL,&pool)&&deferred&&!pool&&!flat3d_pool_consuming);
    pool_will_defer=deferred=0;
    assert(!ib->lifetime.captures&&!dc.bind_manifest->head&&!dc.bind_manifest->pending);
    assert(ia->lifetime.captures==3);destroy(ib,&b);
    failed=1;
    binding_manifest_begin(&ex);binding_manifest_select(&ex,3,1001);
    assert(!binding_manifest_prepare_pool(&ex,NULL,&pool)&&!pool&&ia->lifetime.captures==3);
    failed=0;
    vk_texture_publication_release(dc.bind_manifest);dc.bind_manifest=NULL;
    vk_texture_publication_release(native.images);native.images=NULL;
    assert(ia->lifetime.captures==2);
    vk_texture_uses_handed(first);vk_texture_uses_abandon(second);
    vk_texture_uses_complete(80,0);assert(!ia->lifetime.captures);destroy(ia,&a);
    assert(!texture_uses.nodes&&!texture_uses.batches&&!texture_uses.publications&&!locked);
    puts("binding manifest: exact direct/motion handoff, split batches and canceled preparation passed");
    selected_sample_handoff();
    return 0;
}
'''


def main():
    draw = (ROOT / "runtime/vk_exec_draw.c").read_text()
    texture = (ROOT / "runtime/vk_texture.c").read_text()
    begin = texture.index("struct VkTexturePresentSource {")
    present_source = texture[begin:texture.index("\n};", begin) + 3]
    selected_sample = function(texture, "vk_texture_record_selected_sample_locked")
    helpers = "\n".join(function(draw, name) for name in (
        "binding_manifest_begin", "binding_manifest_select", "binding_manifest_publish", "binding_manifest_cancel",
        "binding_manifest_prepare_pool"))
    bind = function(draw, "bind_resources")
    motion = function(draw, "mo_desc")
    assert bind.index("binding_manifest_prepare_pool(ex, info, &pool)") < bind.index("prepared_publish_images(")
    for body in (bind, motion):
        assert "binding_manifest_select(" in body and "binding_manifest_publish(ex)" in body
        assert "binding_manifest_cancel(ex)" in body
        assert "vk_texture_publication_add_and_use" not in body
    for displaced in ("tex_info_lookup(", "binding_images_publish(", "binding_images_prepared(", "bind_sig(", "cx->TM["):
        assert displaced not in draw, displaced
    cc = Path(os.environ.get("CC", r"C:\Users\Andrew\AppData\Local\Programs\2NX2Navi\toolchain\msys64\ucrt64\bin\clang.exe"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    parent = ROOT / "build/test"
    parent.mkdir(parents=True, exist_ok=True)
    marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
    with tempfile.TemporaryDirectory(prefix="binding-manifest-", dir=parent) as temporary:
        source = Path(temporary) / "check.c"
        exe = source.with_suffix(".exe")
        source.write_text("\n".join(("// " + marker, PREFIX, present_source, selected_sample, helpers, TEST)),
                          encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(ROOT / "tests"),
                        str(source), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

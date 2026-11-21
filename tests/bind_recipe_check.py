"""CPU-only recipe and prepared-instance checks; no GPU, game data or downloads.

CC selects Clang. BIND_RECIPE_INCLUDE_DIRS supplies Vulkan/VMA header paths when necessary.
The validity fixture extracts production functions verbatim because vk_exec_draw.c also defines device entry points
that PE linkers retain. It does not replace these functions with a parallel implementation.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    import re
    match = re.search(r"^(?:static )?[^\n]*\b" + name + r"\(", source, re.M)
    if not match:
        raise RuntimeError(f"Missing production function {name}")
    start = match.start()
    body = source.index("{", match.end())
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + "\n"


def run_checks(work):
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    env = dict(os.environ)
    cc_path = Path(shutil.which(cc) or cc).resolve()
    env["PATH"] = str(cc_path.parent) + os.pathsep + env.get("PATH", "")
    draw = (ROOT / "runtime/vk_exec_draw.c").read_text()
    prepared = (ROOT / "runtime/vk_bind_prepared.h").read_text()
    prefix = draw[:draw.index("static void prepared_stats_flush")]
    declarations = '''
static uint64_t test_authority_epoch = 1;
RcRegistry *rc_registry_runtime(void) {return NULL;}
uint64_t rc_registry_authority_epoch(RcRegistry *r) {(void)r;return test_authority_epoch;}

static _Thread_local int twin_dispatch;
static int td_mode(void);
static void shader_packet_stats_collect(void) {} // telemetry is outside the resource-validity fixture
static void pool_stats_collect(void) {}
static void prepared_residency_stats_collect(void) {}
static void effect_stats_collect(void) {}
static int slot_still(uint32_t *, uint64_t, uint64_t, uint32_t);
static int registered_slot_locked(int logical) { return logical; }
static CbRes *cb_res(VkExec *);
static void count_use(DrawCtx *, int, int, uint64_t);
static void prepared_support(VkExec *, const VkBindRecipeOp *, PreparedBindings *, int);
static void prepared_uniforms(VkExec *, const VkBindRecipeOp *, PreparedBindings *, int);
static void prepared_storage(VkExec *, const VkBindRecipeOp *, PreparedBindings *, int);
static void prepared_images(VkExec *, const VkBindRecipeOp *, PreparedBindings *, int);
static void prepared_texels(VkExec *, const VkBindRecipeOp *, PreparedBindings *, int);
static void *prepared_test_calloc(size_t, size_t);
#define calloc prepared_test_calloc
#ifdef PREP_TEST_RESET_TRACE
static void *prepared_test_memset(void *, int, size_t);
#define memset prepared_test_memset
#endif
'''
    # Device resource resolution is stubbed by the fixture. Cache decisions, dirty notifications, proof collection,
    # ordered traversal, accounting and lifetime logic below are extracted verbatim from the production sources.
    selected = prepared[:prepared.index("// Included")] + "\n".join(function(draw, n) for n in (
        "prepared_stats_flush", "vk_exec_draw_prepared_stats_flush", "vk_exec_draw_prepared_stats_collect",
        "vk_exec_draw_state_adopted", "vk_exec_draw_new_cb", "in_fbl", "bind_handles",
        "descriptor_packet_equal", "compute_bindings_disturbed"))
    selected += "\n" + prepared[prepared.index("typedef void (*PreparedOp)"):prepared.index("typedef void (*PreparedOp)") +
                            prepared[prepared.index("typedef void (*PreparedOp)"):].index(";") + 1]
    start = prepared.index("enum { PREP_LIVE")
    selected += "\n" + prepared[start:prepared.index(";", start) + 1]
    selected += "\n" + "\n".join(function(prepared, n) for n in (
        "same_binding_interface", "shared_binding_recipe",
        "prepared_watch", "prepared_watch_recipe", "prepared_bucket", "prepared_hot", "prepared_directory_init",
        "prepared_fixed_bytes", "prepared_total_bytes", "prepared_forget", "prepared_evict", "prepared_make_room",
        "prepared_same_epoch", "prepared_demote", "prepared_dormant_find", "prepared_dormant_admit", "prepared_promote",
        "prepared_dispose", "prepared_instance", "prepared_publish_images", "prepared_buffer",
        "prepared_handle", "prepared_buffer_valid", "prepared_buffer_end", "prepared_op_check", "prepared_changed",
        "prepared_stamp", "prepared_stamp_equal",
        "prepared_refresh_record", "prepared_sample_matches", "prepared_texture_reusable", "prepared_texture_receipt",
        "prepared_texture_valid", "prepared_group_guard", "prepared_group_valid", "prepared_qualified_sample",
        "prepared_group_current", "prepared_op_inputs_equal",
        "prepared_census_adjust", "prepared_renew_member", "prepared_try_renew", "prepared_seal", "prepared_resolve"))
    # The fixture contains runtime source only; use the repository marker for files emitted by tools.
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from check_content import GENERATED_MARKER
    marker = GENERATED_MARKER.decode() if isinstance(GENERATED_MARKER, bytes) else GENERATED_MARKER
    (work / "prepared_state_under_test.h").write_text(f"// {marker}\n" + prefix + declarations + "\n" + selected)
    images = function(prepared, "prepared_images").replace("prepared_images(", "actual_prepared_images(")
    images = images.replace("prepared_check(", "texture_check(").replace("prepared_sample_matches(", "texture_sample_matches(")
    matches = function(prepared, "prepared_sample_matches").replace("prepared_sample_matches(", "texture_sample_matches(")
    matches = matches.replace("slot_still(", "texture_slot_still(")
    still = function(draw, "slot_still").replace("slot_still(", "texture_slot_still(")
    texture = (ROOT / "runtime/vk_texture.c").read_text()
    sampled = "\n".join(function(texture, name) for name in (
        "vk_texture_resolve_sample_report_locked", "vk_texture_resolve_sample_locked"))
    assert "static void tex_info_lookup(" not in draw and "} TD[" not in draw
    (work / "prepared_images_under_test.h").write_text(f"// {marker}\n" + still + "\n" + sampled + matches + images)
    includes = [ROOT / "runtime", work, cc_path.parent.parent / "include/vma"]
    includes += [Path(p) for p in env.get("BIND_RECIPE_INCLUDE_DIRS", "").split(os.pathsep) if p]
    flags = [arg for p in includes if p.is_dir() for arg in ("-I", str(p))]
    compiled_mode = int(env.get("NX2NAVI_TEST_COMPILED_RESOLVE", "0"))
    assert compiled_mode in (0, 1)
    flags += [f"-DNX2NAVI_PREP_COMPILED_RESOLVE={compiled_mode}"]
    for name in ("bind_recipe_check", "prepared_bindings_check", "prepared_texture_check", "prepared_residency_check", "compiled_members_check"):
        exe = work / (name + (".exe" if os.name == "nt" else ""))
        mode = 1 if name == "prepared_residency_check" else 0
        subprocess.check_call([cc, "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Wno-unused-function",
            f"-DNX2NAVI_PREP_DORMANT={mode}",
            "-fsanitize=undefined", "-fsanitize-trap=all",
            "-Wno-unused-variable", "-Wno-missing-field-initializers", *flags,
            str(ROOT / f"tests/{name}.c"), str(ROOT / "runtime/vk_bind_recipe.c"), "-o", str(exe)], env=env)
        subprocess.check_call([str(exe)], env=env)


def main():
    parent = ROOT / "build/test"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bind-recipe-", dir=parent) as temporary:
        run_checks(Path(temporary))


if __name__ == "__main__":
    main()

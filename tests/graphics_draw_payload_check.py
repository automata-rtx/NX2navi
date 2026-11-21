"""CPU-only checks of the exact native graphics scratch layout and scene custody."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ordered_graphics_feedback_check import function

RUNTIME = Path(os.environ.get("GRAPHICS_TEST_RUNTIME", HERE.parent / "runtime"))


def section(source, first, following):
    start = source.index(first)
    return source[start:source.index(following, start + len(first))]


def main():
    cc = os.environ.get("CC") or shutil.which("clang")
    if not cc:
        raise SystemExit("Clang is required (set CC)")
    clang = Path(shutil.which(cc) or cc).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", clang.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
    draw = Path(os.environ.get("DRAW_SOURCE", RUNTIME / "vk_exec_draw.c")).read_text(encoding="utf-8")
    defines = "\n".join(line for line in draw.splitlines() if line.startswith(
        ("#define MAX_BINDINGS ", "#define CHUNK_BYTES ", "#define SUPPORT_BYTES ", "#define NVN_STAGE_")))
    owners = section(draw, "// ---- resources of one command buffer", "typedef struct {\n    VkPipelineLayout layout;")
    unused = section(owners, "static GraphicsUboMerge *graphics_ubo_merge_find(", "struct VkDrawScratchCredit {")
    owners = owners.replace(unused, "")
    owners = owners.replace(function(owners, "descriptor_pool_opaque_bytes"), "")
    provider = section(draw, "static unsigned scratch_detach_spares(", "static void cb_finish(")
    functions = "\n".join(function(draw, name) for name in (
        "cb_finish", "cb_done", "cb_cancel", "cb_res", "nvn_stage_of", "graphics_payload_align",
        "graphics_payload_demand_at", "graphics_payload_demand", "graphics_payload_fit", "graphics_payload_view", "c0_table", "c0_ssbo_base",
        "vk_c0_fill", "support_in", "support_fill", "support_descriptor", "update_current", "scratch_slab_create", "update_slab_create",
        "scratch_placement_fits", "update_publish", "scratch_slabs_useful", "vk_exec_draw_update_prefix_completed",
    ))
    executor = (RUNTIME / "vk_exec.c").read_text(encoding="utf-8")
    overflow = section(draw, "struct GraphicsUboOverflow {", "static void graphics_ubo_overflow_release(")
    graphics = overflow + "\n".join(function(draw, name) for name in (
        "fixed_peak_add", "graphics_ubo_sealed", "graphics_ubo_sealed_gaps", "ubo_merge_budget_peak", "graphics_slab_adopt", "ubo_merges_initialize", "graphics_payload_initialize",
        "graphics_scratch_claim",
        "graphics_payload_plan", "graphics_scratch_failure", "graphics_scratch_wait",
        "graphics_index_needs_widening", "graphics_native_index_type",
        "graphics_argument_select", "graphics_held_argument_ready",
        "graphics_byte_index_select", "graphics_widen_current", "graphics_widen_bytes",
        "graphics_widen_capture", "graphics_widen_import_current", "graphics_widen_import",
    ))
    graphics += "\n" + function(executor, "vk_exec_read_device_indices")
    graphics += """
typedef struct { int unused; } RawStorageInput;
static int prepared_buffer_requires_image(const VkMemSpan *span,uint64_t address,uint64_t bytes,RcRegistryPosition read) {
    (void)span;(void)address;(void)bytes;(void)read;return 0;
}
static const RawStorageInput *prepared_buffer_argument(const DrawDescriptorGrant *grant,uint64_t address,uint64_t bytes) {
    (void)grant;(void)address;(void)bytes;return NULL;
}
static int prepared_buffer_argument_record(VkExec *ex,const DrawDescriptorGrant *grant,const RawStorageInput *input,VkMemSpan *arg) {
    (void)ex;(void)grant;(void)input;(void)arg;abort();
}
static VkExecHostInputResult graphics_widen_projected(VkExec *ex,const VkMemSpan *span,VkMemSpanCapture *hold,
    uint64_t address,uint64_t bytes,RcRegistryPosition read,uint16_t *output,int restart) {
    (void)ex;(void)span;(void)hold;(void)address;(void)bytes;(void)read;(void)output;(void)restart;
    return VK_EXEC_HOST_INPUT_ABSENT;
}
"""
    graphics += "\n" + "\n".join(function(draw, name) for name in (
        "graphics_widen_native", "graphics_widen_prepare", "vk_exec_draw_graphics_scratch_prepare",
        "mfb_payload_phase", "mfb_materialize_span", "mfb_claim_payload", "mfb_scratch_cohort_ready",
        "mfb_late_scratch_ready", "vk_exec_draw_graphics_scratch_finish",
        "vk_exec_draw_graphics_scratch_spare_ready", "vk_exec_draw_graphics_scratch_prior_useful",
        "graphics_widen_bind", "graphics_prepare_byte_indices", "op_draw_idx",
    ))

    mfb_types = section(draw, "typedef struct MfbCs MfbCs;", "static void mfb_task_pointer_release(")
    phase_first = draw.index("typedef struct {\n    const GfxShader *shader;\n    VkPipeline pipeline;")
    mfb_types = draw[phase_first:draw.index("typedef struct MfbCs MfbCs;", phase_first)] + mfb_types
    with tempfile.TemporaryDirectory(prefix="graphics-draw-payload-") as directory:
        work = Path(directory)
        (work / "mfb_payload_types_under_test.h").write_text(mfb_types, encoding="utf-8")
        (work / "graphics_payload_types_under_test.h").write_text(defines + "\n" + owners, encoding="utf-8")
        (work / "graphics_payload_under_test.h").write_text(
            provider + "\n" + functions + "\n" + graphics, encoding="utf-8")
        for compiled in (0, 1):
            exe = work / ("check.exe" if os.name == "nt" else "check")
            subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=undefined", "-fsanitize-trap=undefined",
                            f"-DNX2NAVI_PREP_COMPILED_RESOLVE={compiled}",
                            "-I", str(RUNTIME),
                            "-I", str(vma), "-I", str(work), str(HERE / "graphics_draw_payload_check.c"),
                            str(RUNTIME / "vk_bind_recipe.c"), "-o", str(exe)], env=env, check=True)
            subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

from argparse import ArgumentParser
import ast
import os
from pathlib import Path
import subprocess
import tempfile

from pass_end_lock_check import function

TESTS = Path(__file__).resolve().parent
RUNTIME = TESTS.parent / "runtime"


def without_includes(source):
    return "\n".join(line for line in source.splitlines() if not line.startswith("#include"))


def fixture(runtime):
    adapter = ast.parse((TESTS / "resource_command_effects_check.py").read_text(encoding="utf-8"))
    prefix = next(ast.literal_eval(item.value) for item in adapter.body if isinstance(item, ast.Assign)
                  and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name)
                  and item.targets[0].id == "prefix")
    def read(name):
        return (runtime / name).read_text(encoding="utf-8")
    host = read("resource_host_write.c")
    host = host[host.index("ResourceHostWrite resource_host_write_prepare("):]
    host = host.replace("resource_host_write_prepare(", "actual_host_prepare(")
    host = host.replace("resource_host_write_dispose(", "actual_host_dispose(")
    texture = read("vk_texture.c")
    start = texture.index("struct VkTextureProjectedTransfer {")
    layout = texture[start:texture.index("\n};", start) + len("\n};")] + "\n"
    host_volume = "static int first_mirrored_transfer_candidate(" in texture
    host_color = "static int mirrored_color_transfer_capture(" in texture
    if host_color:
        layout = texture[texture.index("typedef struct MirroredColorWrite {"):start] + layout
    ranges = read("vk_texture_alias_bytes.h")
    layout_start = ranges.index("enum { VK_TEXTURE_ALIAS_MAX_PLANES")
    current = function(texture, "projected_transfer_current_locked").replace(
        "projected_transfer_current_locked(", "actual_projected_transfer_current_locked(", 1)
    compare = (function(texture, "image_alias_write_offset_compare")
               if "static int image_alias_write_offset_compare(" in texture else "")
    trace = ("texture_trace_placement", "image_alias_plane_result") if "image_alias_plane_result(" in texture else ()
    host_current = "".join(function(texture, name) for name in (
        "first_mirrored_transfer_candidate",
        *(("mirrored_transfer_stamp_current",) if host_color else ()),
        "first_mirrored_transfer_cpu_current",
        "first_mirrored_transfer_cpu_capture", "first_mirrored_transfer_cpu_cancel", "projected_transfer_mapping_equal",
    )) if host_volume else ""
    if host_color:
        host_current += "\n".join(function(texture, name) for name in (
            "mirrored_color_warm_candidate", "mirrored_color_current",
        )) + "\n"
    projection = host_current + current + compare + "\n".join(function(texture, name) for name in (
        "image_alias_project_write", "image_hold_mapped", "vk_texture_projected_transfer_cancel",
        "projected_transfer_alloc", "projected_transfer_trim", "projected_transfer_address_compare",
        "projected_transfer_union", "projected_transfer_prior_part", "projected_transfer_prior_source",
        "projected_transfer_rect", "projected_transfer_copy_boxes", "projected_transfer_self_copy_geometry",
        "projected_transfer_sources_current", "vk_texture_projected_self_copy_capture",
        "vk_texture_projected_self_copy_regions", "vk_texture_projected_transfer_declare",
        "vk_texture_projected_transfer_begin", "vk_texture_projected_transfer_written",
        "image_alias_placement_current", "image_alias_whole_range", "image_alias_plane_demand",
        *trace, "image_alias_private_planes_written_locked", "image_alias_planes_written_locked",
        "vk_texture_alias_clear_written", "image_alias_load_current", "storage_volume_shape",
        "storage_volume_mip", "storage_target_volume_view",
    ))
    replay = read("vk_exec.c")
    clear = function(replay, "clear_color")
    if "clear_image_fragment(" in clear:
        clear = function(replay, "clear_image_fragment") + clear
    attachment_planes = ("attachment_load_planes" if "attachment_load_planes(" in replay else "attachment_depth_planes")
    projection += function(replay, "whole_target") + clear + "\n".join(function(replay, name) for name in (
        attachment_planes, "resource_target_pinned", "resource_attachment_prepare", "resource_attachment_written",
        "vk_exec_targets_prepare", "vk_exec_targets_written",
    ))
    template = Path(__file__).with_name("projected_image_mappings_fixture.c").read_text(encoding="utf-8")
    header = read("vk_exec.h")
    binding_end = header.index("} VkExecTargetBinding;") + len("} VkExecTargetBinding;")
    binding = header[header.rfind("typedef struct {", 0, binding_end):binding_end]
    test = template.replace("/* Alias plane layout */", ranges[layout_start:]).replace(
        "/* Private target layout */", layout).replace("/* Target binding */", binding).replace(
        "/* Production functions */", projection)
    xfer = read("vk_exec_xfer.c")
    sub = xfer[xfer.index("typedef struct {\n    const VkTexImage *img;"):
               xfer.index("// Resolves tex, its view and a copy region")]
    pitch_end = xfer.index("} BufferImagePitch;") + len("} BufferImagePitch;")
    pitch = xfer[xfer.rindex("typedef struct {", 0, pitch_end):pitch_end]
    types = "\n".join((sub, pitch))
    host_helpers = ("transfer_volume_layout", "vk_texture_host_volume_transfer",
                    "first_mirrored_transfer_capture") if host_volume else ()
    before = "\n".join(function(texture, name) for name in (
        *host_helpers,
        *(("vk_texture_host_color_transfer", "mirrored_color_transfer_capture") if host_color else ()),
        "vk_texture_alias_region_transfer", "vk_texture_alias_volume_transfer", "alias_depth_format",
        "vk_texture_alias_depth_clear_candidate", "vk_texture_attachment_base_view", "vk_texture_alias_depth_transfer",
        "vk_texture_projected_transfer_capture", "vk_texture_projected_transfer_image", "vk_texture_transfer_read_prove"))
    transfer = "\n".join(function(xfer, name) for name in (
        "sub_region", "sub_format", "sub_layers", "sub_offset", "sub_extent", "buffer_image_pitch", "buffer_layout",
        "projected_b2t_capture", "projected_transfer_done", "projected_transfer_canceled", "projected_b2t_record"))
    transfer = transfer.replace("warn_once(", "report_volume_warning(")
    transfer = transfer.replace("vk_gpu_gap_note(", "report_transfer_gap(")
    transfer = transfer.replace("vk_exec_required_graphics_unavailable(", "report_transfer_failure(")
    test = test.replace("static int volume_warnings,", types + "\nstatic int volume_warnings,").replace(
        "/* VOLUME PRODUCTION */", "enum { COPY_PITCH_OK, COPY_PITCH_INVALID, COPY_PITCH_OVERFLOW };\n" + before + transfer)
    if host_volume:
        part_end = texture.index("} FirstMirroredWrite;") + len("} FirstMirroredWrite;")
        part_start = texture.rfind("typedef struct {", 0, part_end)
        part = texture[part_start:part_end] + "\n" + "\n".join(function(texture, name) for name in (
            *(("cancel_fresh_mirrored_authority", "mirrored_color_write_dispose",
               "mirrored_color_write_recorded") if host_color else ()),
            "first_mirrored_write_dispose", "first_mirrored_write_recorded",
        ))
        if host_color:
            compare_first = texture.index("typedef struct {\n    const uint8_t *before, *current;")
            compare_last = texture.index("} MirroredColorComparison;", compare_first) + len("} MirroredColorComparison;")
            part += "\n" + texture[compare_first:compare_last] + "\n" + "\n".join(
                function(texture, name) for name in ("compare_mirrored_color_range", "mirrored_color_storage_changed"))
            test = "#define TEST_HOST_COLOR 1\n" + test
        test = "#define TEST_HOST_VOLUME 1\n" + test.replace("/* HOST PART PRODUCTION */", part)
    if attachment_planes == "attachment_depth_planes":
        test = "#define OLD_ATTACHMENT_LOAD 1\n" + test
    return "\n".join((prefix, host, without_includes(read("resource_command_effects.c")),
                      without_includes(read("resource_image_write.c")), test))


def main():
    parser = ArgumentParser(description="Exercise sparse projected transfers using real command effects and registry")
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--host-volume", action="store_true")
    parser.add_argument("--host-color", action="store_true")
    args = parser.parse_args()
    compiler = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="projected-mappings-") as tmp:
        source = Path(tmp) / "check.c"
        source.write_text(fixture(args.runtime), encoding="utf-8")
        exe = source.with_suffix(".exe")
        subprocess.run(
            [str(compiler), "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function", "-O1", "-fsanitize=undefined",
             "-fsanitize-trap=all", "-DRC_REGISTRY_SINGLE_THREADED", "-DRC_BACKING_SINGLE_THREADED",
             f"-I{RUNTIME}", f"-I{compiler.parent.parent / 'include/vma'}", str(source),
             str(args.runtime / "resource_core_registry.c"), str(args.runtime / "resource_core_backing.c"),
             str(args.runtime / "tex_deswizzle.c"), str(args.runtime / "nvn_tex.c"), "-o", str(exe)],
            env=env, check=True,
        )
        mode = ["--host-color"] if args.host_color else ["--host-volume"] if args.host_volume else []
        return subprocess.run([str(exe), *mode], env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify raw shader storage binds exact mixed image and physical buffer authority."""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from pass_end_lock_check import function

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--pointer-baseline", action="store_true")
    args = parser.parse_args()
    draw = (args.runtime / "vk_exec_draw.c").read_text(encoding="utf-8")
    if args.baseline or args.pointer_baseline:
        code = function(draw, "bind_range")
        if args.pointer_baseline:
            code += "\n" + function(draw, "raw_storage_fixed")
    else:
        texture = (args.runtime / "vk_texture.c").read_text(encoding="utf-8")
        begin = draw.index("typedef struct {\n    const VkPipeBinding *binding;\n    p_bind_buf input;")
        end = draw.index("// NX2NAVI_PCSS", begin)
        effects = (args.runtime / "resource_command_effects.c").read_text(encoding="utf-8")
        effects = "\n".join(line for line in effects.splitlines() if not line.startswith("#include"))
        host = (args.runtime / "resource_host_write.c").read_text(encoding="utf-8")
        code = "\n".join((effects, function(host, "resource_host_interval_slice"),
                          function(draw, "readonly_storage_pointer"),
                          function(draw, "vk_exec_effects_read_allowed"),
                          function(draw, "vk_exec_effects_read_allowed_range"),
                          function(texture, "image_alias_compose_buffer"), function(texture, "vk_texture_alias_buffer_copy"),
                          function(draw, "indexed_draw_reached_bytes"),
                          draw[begin:end], function(draw, "graphics_widen_bytes"),
                          function(draw, "graphics_widen_projected")))
        for caller in ("graphics_draw_select", "prepare_draw_phase"):
            assert "graphics_buffers_prepare(" in function(draw, caller), caller
        index = function(draw, "op_draw_idx")
        assert index.index("prepared_buffer_argument_record(") < index.index("vkCmdBindIndexBuffer(")
        assert "prepared_buffer_requires_image(" in index
        for caller in ("vk_exec_dispatch_prepare_native_inputs", "vk_exec_dispatch_prepare_compat_inputs"):
            assert "dispatch_buffers_prepare(" in function(draw, caller), caller
        issued = function(draw, "dispatch_indirect_emit")
        chosen = issued.index("prepared_buffer_argument_record(")
        assert chosen < issued.index("buffer_scratch_indirect_c0(ex, &ip)")
        assert chosen < issued.index("vkCmdDispatchIndirect(ex->cb, ip.buffer, ip.offset)")
        assert "dispatch_indirect_requires_image(" in function(draw, "vk_exec_dispatch_compat_inputs_select")
        widened = function(draw, "graphics_widen_native")
        assert widened.index("graphics_widen_projected(") < widened.index("vk_exec_read_device_indices(")
        mfb = function(draw, "mfb_draw")
        assert mfb.index("graphics_mesh_count_use(") < mfb.index("vk_meshfb_record(")
        assert "graphics_mesh_count_capture(" in function(draw, "mfb_draw_prepare")
        bind = function(draw, "bind_resources")
        assert bind.index("raw_storage_binding(ex,") < bind.index("int prep_mode =")
        assert bind.index("raw_storage_bind(ex,") < bind.index("ssbo_info(ex, st, b, 0)")
    template = Path(__file__).with_name("raw_storage_image_source_fixture.c").read_text(encoding="utf-8")
    prefix = "#define BASELINE 1\n" if args.baseline or args.pointer_baseline else ""
    if args.pointer_baseline:
        prefix += "#define POINTER_BASELINE 1\n"
    source = prefix + template.replace("/* Production functions */", code)
    compiler = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="raw-storage-source-") as folder:
        path = Path(folder) / "check.c"
        exe = path.with_suffix(".exe")
        path.write_text(source, encoding="utf-8")
        subprocess.run([str(compiler), "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function", "-O1",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-DRC_REGISTRY_SINGLE_THREADED",
                        f"-I{args.runtime}", f"-I{ROOT / 'runtime'}", str(path),
                        str(args.runtime / "resource_core_registry.c"), "-o", str(exe)], env=env, check=True)
        return subprocess.run([str(exe)], env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())

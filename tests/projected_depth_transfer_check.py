"""Check the ordinary projected D32 copy path using the production transfer planner."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from pass_end_lock_check import function

ROOT = Path(__file__).resolve().parents[1]


def fixture(runtime):
    texture = (runtime / "vk_texture.c").read_text(encoding="utf-8")
    xfer = (runtime / "vk_exec_xfer.c").read_text(encoding="utf-8")
    sub = xfer[xfer.index("typedef struct {\n    const VkTexImage *img;"):
               xfer.index("// Resolves tex, its view and a copy region")]
    names = ["sub_region", "sub_format", "sub_layers", "sub_offset", "sub_extent",
             "image_copy_same_layers", "image_copy_same_box", "image_copy_overlap", "image_copy_shape",
             "image_copy_blit_supported", "image_copy_geometry", "image_copy_prepare_overlap",
             "texture_copy_stage_source", "image_copy_record", "vk_exec_texture_to_texture_alias_candidate"]
    if "static VkImageAspectFlags projected_t2t_aspect(" in xfer:
        names.append("projected_t2t_aspect")
    names.append("projected_t2t_capture")
    shape = ["vk_texture_alias_region_transfer", "vk_texture_alias_depth_clear_candidate", "vk_texture_attachment_base_view"]
    if "int vk_texture_host_color_transfer(" in texture:
        shape.append("vk_texture_host_color_transfer")
    if "static VkFormat alias_depth_format(" in texture:
        shape.insert(0, "alias_depth_format")
    if "int vk_texture_alias_depth_transfer(" in texture:
        shape.append("vk_texture_alias_depth_transfer")
    code = "\n".join([sub] + [function(texture, name) for name in shape] + [function(xfer, name) for name in names])
    if "vk_texture_host_color_transfer" in shape:
        code = "#define TEST_HOST_COLOR 1\n" + code
    return Path(__file__).with_name("projected_depth_transfer_fixture.c").read_text(encoding="utf-8").replace(
        "/* Production functions */", code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime")
    parser.add_argument("--host-color", action="store_true")
    args = parser.parse_args()
    clang = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="projected-depth-transfer-") as tmp:
        source = Path(tmp) / "check.c"
        exe = source.with_suffix(".exe")
        source.write_text(fixture(args.runtime), encoding="utf-8")
        subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        "-fsanitize=undefined", "-fsanitize-trap=all", f"-I{ROOT / 'runtime'}",
                        f"-I{clang.parent.parent / 'include/vma'}", str(source), "-o", str(exe)], env=env, check=True)
        mode = ["--host-color"] if args.host_color else []
        return subprocess.run([str(exe), *mode], env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())

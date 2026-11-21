"""Validate selected depth LOAD geometry and footprint using real NVN tiling."""
import os
from pathlib import Path
import subprocess
import tempfile

from pass_end_lock_check import function

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"


def main():
    source = (RUNTIME / "vk_texture.c").read_text(encoding="utf-8")
    production = "\n".join(function(source, name) for name in (
        "alias_depth_format", "vk_texture_alias_depth_attachment_range", "image_alias_layout_for",
        "image_alias_whole_range", "image_alias_plane_demand", "image_alias_load_current",
    ))
    fixture = Path(__file__).with_name("depth_attachment_planes_fixture.c").read_text(encoding="utf-8")
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="depth-attachment-planes-") as tmp:
        path = Path(tmp) / "check.c"
        exe = path.with_suffix(".exe")
        path.write_text(fixture.replace("/* Production functions */", production), encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        "-fsanitize=undefined", "-fsanitize-trap=all", f"-I{RUNTIME}",
                        f"-I{cc.parent.parent / 'include/vma'}", str(path), str(RUNTIME / "nvn_tex.c"),
                        str(RUNTIME / "tex_deswizzle.c"), "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

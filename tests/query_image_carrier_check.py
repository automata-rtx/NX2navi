"""Verify the query-only image receipt, record position, and native lifetime without a Vulkan driver."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from present_source_check import ROOT, function


def main():
    cc = Path(os.environ.get("CC") or shutil.which("clang") or "").resolve()
    if not cc.is_file():
        raise SystemExit("Clang is required (set CC)")
    runtime = Path(os.environ.get("QUERY_IMAGE_CHECK_RUNTIME", ROOT / "runtime"))
    texture = (runtime / "vk_texture.c").read_text(encoding="utf-8")
    registrations = (runtime / "vk_texture_registrations.h").read_text(encoding="utf-8")
    sync = (runtime / "vk_sync.c").read_text(encoding="utf-8")
    start = texture.index("struct VkTextureQueryImage {")
    end = texture.index("int vk_texture_binding_receipt_locked", start)
    definitions = "\n".join((
        *(function(registrations, name) for name in (
            "texture_registration_key", "texture_registration_ordinary",
            "texture_registration_resolve_owner", "texture_registration_resolve",
        )),
        *(function(texture, name) for name in (
            "aspect_of", "same_but_swizzle", "follow_size_matches", "shader_metadata_cold", "sample_geometry_view",
            "sample_geometry_mapping", "sample_registered_geometry_mapping", "sample_geometry_exact_format",
            "sample_geometry_effective_equal", "sample_geometry_native_shape",
            "sample_association_view_locked", "sample_association_agrees",
            "sample_registered_geometry_candidates", "sample_association_mapping", "sample_registered_geometry_locked",
            "vk_texture_sample_registered_geometry_locked", "vk_texture_sample_associated_geometry_locked",
            "vk_texture_sample_historical_geometry_locked",
            "vk_texture_sample_geometry_current_locked",
        )),
        texture[start:end],
    ))
    retirement = "\n".join(function(sync, name) for name in ("pending_disposable", "pending_dispose"))
    template = Path(__file__).with_name("query_image_carrier_fixture.c").read_text(encoding="utf-8")
    vma = Path(os.environ.get("VMA_INCLUDE", cc.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="query-image-carrier-") as temp:
        fixture = Path(temp) / "check.c"
        exe = Path(temp) / ("check.exe" if os.name == "nt" else "check")
        fixture.write_text(template.replace("/* Extract carrier */", definitions)
                           .replace("/* Extract retirement */", retirement), encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(runtime), "-I", str(ROOT / "runtime"),
                        "-I", str(vma), str(fixture), str(ROOT / "runtime" / "nvn_tex.c"), "-o", str(exe)],
                       env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    main()

"""Exercise exact sampled-plane admission beside retained attachment LOAD without a GPU."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

TESTS = Path(__file__).resolve().parent
RUNTIME = TESTS.parent / "runtime"
sys.path.insert(0, str(TESTS))
from ordered_graphics_feedback_check import function

GROUPS = {
    "projected": ("texture", (
        "alias_depth_format", "image_alias_layout_for", "image_alias_exact_layout", "image_alias_plane_demand",
        "image_alias_whole_range", "image_alias_load_current", "image_alias_resident_current", "texture_projected_color_current",
        "texture_projected_single_plane_current", "texture_projected_sample_report_locked", "texture_projected_sample_locked",
    )),
    "alias": ("texture", ("image_alias_sample_select_locked",)),
    "resolve": ("texture", (
        "texture_by_slot_locked", "texture_slot_registration", "texture_read_key_compatibility",
        "vk_texture_resolved_receipt_locked", "vk_texture_select_sample_locked",
        "vk_texture_ordered_sample_select_locked", "vk_texture_resolve_sample_report_locked",
    )),
    "sample": ("draw", (
        "vk_exec_graphics_failure_set", "graphics_sample_shape_current", "graphics_sample_admit",
        "graphics_prepared_sample_current", "graphics_sample", "graphics_sample_failure",
        "graphics_sample_selection_failure", "graphics_retained_sample", "graphics_keep_sample", "graphics_stage_samples", "graphics_drop_geometry_samples", "graphics_compatibility_begin",
    )),
    "prepare": ("draw", ("vk_exec_graphics_draw_prepare",)),
}


def extract(runtime, expect_blocked):
    source = {
        "texture": (runtime / "vk_texture.c").read_text(encoding="utf-8"),
        "draw": (runtime / "vk_exec_draw.c").read_text(encoding="utf-8"),
    }
    geometry = "static int graphics_try_geometry_sample(" in source["draw"]
    code = f"#define TEST_QUERY_GEOMETRY {int(geometry)}\n" + TESTS.joinpath(
        "retained_projected_sample_fixture.c").read_text(encoding="utf-8")
    for group, (unit, names) in GROUPS.items():
        marker = f"/* Extract {group} */"
        assert code.count(marker) == 1, marker
        definitions = []
        for name in names:
            if not geometry and name == "graphics_drop_geometry_samples":
                continue
            if name == "graphics_sample_selection_failure" and name+"(" not in source[unit]:
                continue
            if (expect_blocked and name in {"texture_projected_single_plane_current", "image_alias_resident_current",
                                           "texture_projected_color_current"} and name not in source[unit]):
                continue
            definitions.append(function(source[unit], name))
        code = code.replace(marker, "\n".join(definitions))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-blocked", action="store_true",
                        help="Verify the exact refusal on a baseline without selected-plane admission")
    args = parser.parse_args()
    compiler = os.environ.get("CC") or shutil.which("clang")
    if not compiler:
        raise SystemExit("Clang is required (set CC)")
    cc = Path(shutil.which(compiler) or compiler).resolve()
    vma = Path(os.environ.get("VMA_INCLUDE", cc.parent.parent / "include" / "vma"))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="retained-projected-sample-") as tmp:
        source = Path(tmp) / "check.c"
        exe = Path(tmp) / ("check.exe" if os.name == "nt" else "check")
        source.write_text(extract(RUNTIME, args.expect_blocked), encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                        "-fsanitize=undefined", "-fsanitize-trap=all", f"-I{RUNTIME}", f"-I{TESTS}", f"-I{vma}",
                        str(source), str(RUNTIME / "nvn_tex.c"), str(RUNTIME / "tex_deswizzle.c"), "-o", str(exe)],
                       check=True, timeout=60, env=env)
        if args.expect_blocked:
            result = subprocess.run([str(exe)], timeout=20, env=env)
            if result.returncode != 42:
                raise SystemExit(f"Expected the baseline exact refusal (exit 42), got {result.returncode}")
            subprocess.run([str(exe), "--expect-blocked"], check=True, timeout=20, env=env)
        else:
            subprocess.run([str(exe)], check=True, timeout=20, env=env)


if __name__ == "__main__":
    main()

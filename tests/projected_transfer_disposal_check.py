"""Exercise projected image-transfer disposal through the real declaration and replay helpers."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from pass_end_lock_check import function

ROOT = Path(__file__).resolve().parents[1]


def fixture(runtime):
    sources = {name: (runtime / name).read_text(encoding="utf-8") for name in
               ("vk_texture.c", "vk_exec.c", "vk_exec_draw.c", "vk_exec_xfer.c")}
    texture = sources["vk_texture.c"]
    start = texture.index("struct VkTextureProjectedTransfer {")
    layout = texture[start:texture.index("\n};", start) + len("\n};")] + "\n"
    host_cancel = ("first_mirrored_transfer_cpu_cancel",) if "static void first_mirrored_transfer_cpu_cancel(" in texture else ()
    helpers = {
        "vk_texture.c": (
            *host_cancel, "vk_texture_projected_transfer_cancel", "vk_texture_projected_transfer_declare",
        ),
        "vk_exec_xfer.c": (
            "vk_exec_buffer_to_texture_cancel", "vk_exec_texture_to_texture_cancel",
            "vk_exec_projected_transfer_capture", "vk_exec_projected_transfer_current",
            "vk_exec_projected_transfer_cancel", "vk_exec_projected_transfer_record",
        ),
        "vk_exec_draw.c": ("vk_exec_effects_declare_projected_transfer",),
        "vk_exec.c": (
            "resource_assign", "resource_declare_projected_transfer",
            "projected_transfer_changed", "prepare_ordered_projected_transfer",
        ),
    }
    definitions = [function(sources[file], name) for file, names in helpers.items() for name in names]
    exec_source = sources["vk_exec.c"]
    if "resource_seal(ex)" in function(exec_source, "resource_declare_projected_transfer"):
        definitions.insert(0, function(exec_source, "resource_seal"))

    walk = function(exec_source, "walk")
    declare = "effect = resource_declare_projected_transfer(ex, &projected);"
    assert walk.count(declare) == 1
    begin = walk.index("                vk_exec_projected_transfer_record(ex, &projected);")
    end = walk.index("            } else if (exact_b2t || compat_b2t)", begin)
    replay = (
        "static void replay_tail(VkExec *ex, VkExecProjectedTransfer projected) {\n"
        f"    void *effect;\n    {declare}\n"
        "    begin_effect(effect);\n"
        f"{walk[begin:end]}"
        "    vk_exec_effects_release(effect);\n}\n"
    )
    template = Path(__file__).with_name("projected_transfer_disposal_fixture.c").read_text(encoding="utf-8")
    return (template.replace("/* Private target layout */", layout)
                    .replace("/* Production functions */", "\n".join(definitions))
                    .replace("/* Production replay tail */", replay))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime", help="Extract functions from this source directory")
    args = parser.parse_args()
    clang = Path(os.environ["CC"])
    vma = clang.parent.parent / "include/vma"
    env = dict(os.environ, PATH=str(clang.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="projected-disposal-") as directory:
        source = Path(directory) / "check.c"
        source.write_text(fixture(args.runtime), encoding="utf-8")
        binary = source.with_suffix(".exe")
        subprocess.run([str(clang), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all",
                        f"-I{ROOT / 'runtime'}", f"-I{vma}", str(source), "-o", str(binary)],
                       env=env, check=True)
        return subprocess.run([str(binary)], env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())

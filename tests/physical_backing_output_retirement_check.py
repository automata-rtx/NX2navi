"""Exercise physical pool death through the real detached-image and deferred-destruction paths."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


TESTS = Path(__file__).resolve().parent
RUNTIME = TESTS.parent / "runtime"


def definition(source, name):
    match = re.search(r"^(?:static )?[A-Za-z_][\w \t*]*\b" + name + r"\([^;{}]*\)\s*\{", source, re.M)
    if not match:
        raise ValueError(f"Missing production definition for {name}")
    depth, end = 1, match.end()
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


def main():
    compiler = os.environ.get("CC") or shutil.which("clang")
    if not compiler:
        raise SystemExit("Clang is required (set CC)")
    cc = Path(shutil.which(compiler) or compiler).resolve()
    memory = (RUNTIME / "vk_memory.c").read_text(encoding="utf-8")
    texture = (RUNTIME / "vk_texture.c").read_text(encoding="utf-8")
    begin = memory.index("enum { DEAD_IMAGE = 1")
    end = memory.index("} Dead;", begin) + len("} Dead;")
    dead_types = "typedef struct VkMemoryImageHold VkMemoryImageHold;\n" + memory[begin:end]
    memory_parts = ("pool_native_acquire", "vk_mem_device_backing_reclaimed", "destroy_rec", "image_hold_find_or_add",
                    "vk_mem_image_hold", "vk_mem_image_hold_retain_exact", "vk_mem_image_hold_retired",
                    "vk_mem_image_release", "defer_notify", "vk_defer_image_notify", "destroy_dead", "collect_locked_out",
                    "vk_mem_image_held")
    begin = texture.index("struct NativeTextureOutput {")
    end = texture.index("static NativeTextureOutput *native_texture_output_create", begin)
    texture_parts = ("native_texture_output_create", "native_texture_output_drop", "native_texture_output_reserve",
                     "native_texture_output_publish", "native_texture_output_unpublish", "native_texture_output_transfer",
                     "native_texture_output_destroyed", "native_texture_output_retire", "native_texture_output_collect")
    extract = ["static uint64_t reclaim_pending_bytes;"]
    extract.extend(definition(memory, name) for name in memory_parts)
    extract.append(texture[begin:end])
    extract.extend(definition(texture, name) for name in texture_parts)
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ["PATH"])
    with tempfile.TemporaryDirectory(prefix="physical-backing-image-") as temporary:
        build = Path(temporary)
        (build / "dead_types.h").write_text(dead_types, encoding="utf-8")
        (build / "extract.h").write_text("\n\n".join(extract), encoding="utf-8")
        executable = build / ("check.exe" if os.name == "nt" else "check")
        subprocess.run([str(cc), "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                        "-fsanitize-trap=all", "-DRC_REGISTRY_SINGLE_THREADED=1", "-DRC_BACKING_SINGLE_THREADED=1",
                        "-I" + str(build), "-I" + str(RUNTIME), str(TESTS / "physical_backing_output_retirement_fixture.c"),
                        str(RUNTIME / "resource_core_backing.c"), "-o", str(executable)], env=env, check=True)
        subprocess.run([str(executable)], env=env, check=True)


if __name__ == "__main__":
    main()

"""Check actual physical baseline capture and native alias rollback under Clang UBSan."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


TESTS = Path(__file__).resolve().parent


def definition(memory, name):
    match = re.search(r"^(?:static )?[A-Za-z_][\w \t*]*\b" + name + r"\([^;{}]*\)\s*\{", memory, re.M)
    if not match:
        raise ValueError(f"Missing definition for {name}")
    depth = 1
    end = match.end()
    while depth:
        depth += (memory[end] == "{") - (memory[end] == "}")
        end += 1
    return memory[match.start():end]


def main(root=TESTS.parent, memory_path=None, overlay=None):
    runtime = root / "runtime"
    overlay_runtime = overlay / "runtime" if overlay else runtime
    overlay_tests = overlay / "tests" if overlay else TESTS
    registry = overlay_runtime / "resource_core_registry.c"
    if not registry.exists():
        registry = runtime / "resource_core_registry.c"
    for filename, name in (("vk_memory.c", "IMAGE_BASELINE_TRACE"),
                           ("vk_memory.c", "GRAPHICS_PREP_TRACE"),
                           ("vk_exec_draw.c", "GRAPHICS_PREP_TRACE"),
                           ("vk_texture.c", "GRAPHICS_PREP_TRACE")):
        call_sites = [line for line in (runtime / filename).read_text(encoding="utf-8").splitlines() if name in line]
        assert call_sites and all(f'rt_knob("{name}", "0")' in line for line in call_sites), (filename, name)
    compiler = os.environ.get("CC") or shutil.which("clang")
    if not compiler:
        raise SystemExit("Clang is required (set CC)")
    cc = Path(shutil.which(compiler) or compiler).resolve()
    include = Path(os.environ.get("VMA_INCLUDE", cc.parent.parent / "include/vma"))
    memory_path = memory_path or overlay_runtime / "vk_memory.c"
    if not memory_path.exists():
        memory_path = runtime / "vk_memory.c"
    memory = memory_path.read_text(encoding="utf-8")
    start = memory.index("#define IMAGE_ALIAS_BASELINE_LIMIT ")
    end = memory.index("VkMemImageAliasPlacement vk_mem_bind_image_alias(", start)
    user_start = memory.index("int vk_mem_span_buffer_preserve_current(", start)
    user_end = memory.index("static int image_alias_baseline_current_locked(", user_start)
    baseline = memory[start:user_start] + memory[user_end:end]
    host = (runtime / "resource_host_write.c").read_text(encoding="utf-8")
    perf = (runtime / "perf.c").read_text(encoding="utf-8")
    origin = "\n".join((definition(perf, "rt_knob"), definition(host, "resource_host_interval_slice"), definition(memory, "physical_initial_fill_covers"),
                        definition(memory, "physical_origin_initialize"),
                        definition(memory, "pool_device_span"),
                        definition(memory, "pool_image_alias_provider_locked"), baseline,
                        definition(memory, "device_intervals"), definition(memory, "vk_mem_device_intervals"),
                        definition(memory, "vk_mem_device_writer_intervals"), definition(memory, "vk_mem_span_input_intervals")))
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="native-image-baseline-") as temporary:
        directory = Path(temporary)
        (directory / "origin_extract.h").write_text(origin, encoding="utf-8")
        (directory / "origin_bind_extract.h").write_text(definition(memory, "vk_mem_bind_image_alias"), encoding="utf-8")
        flags = []
        if overlay:
            replaced = [name for name in ("resource_core_registry.h", "vk.h") if (overlay_runtime / name).exists()]
            if replaced:
                headers = directory / "overlay.json"
                headers.write_text(json.dumps({"version": 0, "case-sensitive": False, "use-external-names": False,
                    "roots": [{"type": "file", "name": (runtime / name).as_posix(),
                        "external-contents": (overlay_runtime / name).as_posix()} for name in replaced]}))
                flags = ["-ivfsoverlay", str(headers)]
        exe = directory / ("check.exe" if os.name == "nt" else "check")
        subprocess.run([str(cc), "-std=c11", "-O1", "-fsanitize=undefined", "-fsanitize-trap=undefined", *flags,
                        f"-I{include}", f"-I{directory}", f"-I{root / 'tests'}", f"-I{overlay_runtime}", f"-I{runtime}",
                        "-DRC_REGISTRY_SINGLE_THREADED=1", "-DRC_BACKING_SINGLE_THREADED=1",
                        str(overlay_tests / "native_image_baseline_fixture.c"),
                        str(registry), str(runtime / "resource_core_backing.c"),
                        "-o", str(exe)], env=env, check=True)
        subprocess.run([str(exe)], env=env, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TESTS.parent)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    main(args.root, overlay=args.overlay)

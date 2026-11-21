"""Reproduce original-read B2T source selection with actual transfer and registry code."""
from pathlib import Path
import os
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / "runtime"
xfer = (runtime / "vk_exec_xfer.c").read_text(encoding="utf-8")
texture = (runtime / "vk_texture.c").read_text(encoding="utf-8")
execute = (runtime / "vk_exec.c").read_text(encoding="utf-8")
header = (runtime / "vk_exec.h").read_text(encoding="utf-8")


def section(source, first, last):
    start = source.index(first)
    return source[start:source.index(last, start)]


def function(source, name):
    match = re.search(r"^[^\n]*\b" + name + r"\([^;]*?\{", source, re.M)
    assert match, name
    return section(source, match.group(0), "\n}\n") + "\n}\n"


assert execute.index("prepare_compat_buffer_to_texture(ex, a, &b2t)") < execute.index(
    "vk_exec_buffer_to_texture_commit_compat(ex, a, &b2t)")
assert "CopyBufferToTexture reached replay without its ordered receipt" in xfer
assert "vk_texture_alias_buffer_capture_rows(copy->source_hold" in xfer
types = section(header, "typedef struct {\n    VkTextureWriteTarget *target;\n    VkMemSpan source;", "// Negative is")
texture_helpers = section(texture, "static uint32_t image_alias_buffer_rows(", "RcRegistryResult vk_texture_alias_buffer_capture(")
transfer = section(xfer, "typedef struct {\n    const VkTexImage *img;", "// A VRAM mirror supplies")
transfer += section(xfer, "static int buffer_to_texture_ready(", "int vk_exec_buffer_to_texture_alias_candidate(")
transfer += section(xfer, "int vk_exec_buffer_to_texture_source_current(", "static int b2t_probe_on;")
transfer += function(xfer, "vk_exec_buffer_to_texture_cancel")
template = Path(__file__).with_suffix(".c").read_text(encoding="utf-8")
combined = template.replace("/* TRANSFER TYPES */", types).replace("/* TRANSFER FUNCTIONS */", texture_helpers + transfer)
compiler = Path(os.environ["CC"])
env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))

with tempfile.TemporaryDirectory(prefix="ordered-buffer-texture-sources-") as tmp:
    source = Path(tmp) / "sources.c"
    source.write_text(combined, encoding="utf-8")
    binary = source.with_suffix(".exe")
    subprocess.run([str(compiler), "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                    "-fsanitize=undefined", "-fsanitize-trap=all", "-DRC_REGISTRY_SINGLE_THREADED",
                    f"-I{runtime}", str(source), str(runtime / "resource_core_registry.c"),
                    "-o", str(binary)], env=env, check=True)
    subprocess.run([str(binary)], env=env, check=True)

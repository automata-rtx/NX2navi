"""Exercise the common original-cut transfer input with real registry and byte composition."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent / "runtime"


def function(body, name):
    match = re.search(rf"^[^\n]*\b{name}\([^;]*?\{{", body, re.M)
    assert match, name
    start = match.start()
    return body[start:body.index("\n}\n", start) + 3] + "\n"


texture = (RUNTIME / "vk_texture.c").read_text()
transfers = (RUNTIME / "vk_exec_xfer.c").read_text()
commands = (RUNTIME / "vk_exec.c").read_text()
ordered = function(commands, "prepare_ordered_texture_to_texture")
assert ordered.index("prepare_bound_texture_relocation") < ordered.index("prepare_texture_to_texture_source")
assert ordered.index("prepare_texture_to_texture_source") < ordered.index("vk_exec_texture_to_texture_capture")
assert "vk_texture_transfer_input_current(&copy->input)" in function(transfers, "vk_exec_restore_copy_current")
assert "vk_texture_transfer_input_use(&copy->input)" in function(transfers, "vk_exec_restore_copy_commit")
assert "vk_texture_transfer_input_use(copy->input)" in function(transfers, "vk_exec_texture_to_texture_commit")
selected = "".join(function(texture, name) for name in (
    "texture_transfer_input_demand", "transfer_input_mapping", "vk_texture_transfer_input_capture",
    "vk_texture_transfer_input_current", "vk_texture_transfer_input_use", "vk_texture_transfer_input_cancel"))
start = transfers.index("typedef struct {\n    const VkTexImage *img;")
selected += transfers[start:transfers.index("typedef struct {\n    uint64_t row_bytes", start)]
pitch = transfers.index("typedef struct {\n    uint64_t row_bytes")
selected += transfers[pitch:transfers.index("enum { COPY_PITCH_OK", pitch)]
start = transfers.index("struct VkExecTextureBufferWrite {")
selected += transfers[start:transfers.index("\n};", start) + 4]
selected += "".join(function(transfers, name) for name in (
    "vk_exec_restore_copy_current", "restore_copy_capture_current", "vk_exec_restore_copy_commit"))
selected += "".join(function(transfers, name) for name in (
    "vk_exec_transfer_image_input_capture", "image_copy_same_layers", "image_copy_same_box", "image_copy_overlap",
    "image_copy_record", "image_copy_prepare_overlap", "vk_exec_texture_to_texture_alias_candidate", "vk_exec_texture_to_texture_compat"))
compiler = Path(os.environ["CC"]).resolve()
env = dict(os.environ, PATH=str(compiler.parent) + os.pathsep + os.environ.get("PATH", ""))
with tempfile.TemporaryDirectory(prefix="ordered-texture-source-") as scratch:
    source = Path(scratch) / "fixture.c"
    types_end = selected.index("int vk_exec_restore_copy_current(")
    types_begin = selected.index("typedef struct {\n    uint64_t row_bytes")
    write_end = selected.index("\n};", types_begin) + 4
    assert types_begin < write_end == types_end
    fixture = (HERE / "transfer_image_inputs_fixture.c").read_text()
    source.write_text(fixture.replace("/* BUFFER_TYPES */", selected[types_begin:types_end])
                             .replace("/* EXTRACT */", selected[:types_begin] + selected[types_end:]))
    program = Path(scratch) / "fixture.exe"
    result = subprocess.run([compiler, "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror",
        "-Wno-unused-function", "-fsanitize=undefined", "-fsanitize-trap=all", "-DRC_REGISTRY_SINGLE_THREADED",
        "-I" + str(RUNTIME), "-I" + str(compiler.parent.parent / "include/vma"),
        source, RUNTIME / "resource_core_registry.c", "-o", program], check=True, env=env)
    subprocess.run([program], check=True, env=env)

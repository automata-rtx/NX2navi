"""Preserve every mesh atomic and mark only executed, in-range storage words."""
import os
from pathlib import Path
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
function = runpy.run_path(str(Path(__file__).with_name("ordered_graphics_feedback_check.py")))["function"]


def text_words(value):
    data = value.encode() + b"\0"
    data += b"\0" * (-len(data) % 4)
    return struct.unpack(f"<{len(data) // 4}I", data)


def mesh(ordinary_store=False, invalid_scope=False):
    words = [0x07230203, 0x00010400, 0, 64, 0]

    def ins(op, *args):
        words.extend(((len(args) + 1) << 16 | op, *args))

    ins(17, 1)  # Shader
    ins(17, 5283)  # MeshShadingEXT
    ins(10, *text_words("SPV_EXT_mesh_shader"))
    ins(14, 0, 1)  # Logical GLSL450
    ins(15, 5365, 15, *text_words("main"), 9, 11)
    for mode in ((17, 1, 1, 1), (26, 1), (5270, 1), (5298,)):
        ins(16, 15, *mode)
    for operands in ((5, 6, 4), (6, 2), (9, 34, 0), (9, 33, 5)):
        ins(71, *operands)
    ins(72, 6, 0, 35, 0)
    ins(19, 1)  # void
    ins(33, 2, 1)
    ins(21, 3, 32, 0)
    ins(20, 4)
    ins(29, 5, 3)
    ins(30, 6, 5)
    ins(32, 7, 12, 6)
    ins(32, 8, 12, 3)
    ins(32, 10, 4, 3)
    ins(43, 3, 12, 0)
    ins(43, 3, 13, 1)
    ins(43, 3, 14, 2)
    ins(59, 7, 9, 12)
    ins(59, 10, 11, 4)
    ins(54, 1, 15, 0, 2)
    ins(248, 16)
    ins(5295, 12, 12)
    ins(230, 3, 17, 11, 13, 12, 12, 13, 12)  # Workgroup-only CAS, untouched.
    ins(171, 4, 18, 17, 12)
    ins(247, 20, 0)
    ins(250, 18, 19, 20)
    ins(248, 19)
    last = 17
    for site in range(5):
        index, pointer, result = range(21 + site * 3, 24 + site * 3)
        ins(128, 3, index, last, 13)
        ins(65, 8, pointer, 9, 12, index)
        ins(234, 3, result, pointer, 14 if invalid_scope and site == 2 else 13, 12, 12 if site == 0 else 13)
        last = result
    if ordinary_store:
        ins(62, pointer, last)
    ins(249, 20)
    ins(248, 20)
    ins(253)
    ins(56)
    return struct.pack(f"<{len(words)}I", *words)


def instructions(path):
    raw = path.read_bytes()
    words = struct.unpack(f"<{len(raw) // 4}I", raw)
    offset = 5
    while offset < len(words):
        length, operation = divmod(words[offset], 65536)
        assert length and offset + length <= len(words)
        yield operation, words[offset + 1:offset + length]
        offset += length


def check_transform(original, marked):
    before, after = list(instructions(original)), list(instructions(marked))
    marker, = (args[0] for op, args in after if op == 71 and args[1:] == (33, 31))
    assert (71, (marker, 34, 0)) in after
    old_entry, = (args for op, args in before if op == 15)
    new_entry, = (args for op, args in after if op == 15)
    assert new_entry == old_entry + (marker,)
    before_body = before[next(i for i, (op, _) in enumerate(before) if op == 54):]
    after_body = after[next(i for i, (op, _) in enumerate(after) if op == 54):]
    cursor = sites = 0
    for op, args in before_body:
        assert after_body[cursor] == (op, args)
        cursor += 1
        if op != 234:
            continue
        injected = after_body[cursor:cursor + 14]
        assert [opcode for opcode, _ in injected] == [124, 199, 194, 128, 196, 68, 68, 176, 176, 167, 169, 169, 65, 241]
        original_pointer, = (a for p, a in before_body if p == 65 and a[1] == args[2])
        assert injected[0][1][2] == original_pointer[4]
        assert injected[5][1][2] == original_pointer[2] and injected[6][1][2] == marker
        marker_write = injected[-1][1]
        assert marker_write[2] == injected[-2][1][1] and marker_write[3:5] == args[3:5]
        assert marker_write[5] == injected[-3][1][1]
        sites += 1
        cursor += len(injected)
    assert sites == 5 and cursor == len(after_body)


DRIVER = r'''
#include "rc_native_rmw_word.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    assert(argc == 4);
    FILE *input = fopen(argv[1], "rb");
    assert(input && !fseek(input, 0, SEEK_END));
    long bytes = ftell(input);
    assert(bytes > 0 && !(bytes % 4) && !fseek(input, 0, SEEK_SET));
    uint32_t *words = malloc((size_t)bytes);
    assert(words && fread(words, 1, bytes, input) == (size_t)bytes && !fclose(input));
    const size_t count = (size_t)bytes / 4;
    RcNativeRmwWord proof;
    size_t marked_count;
    assert(!rc_native_rmw_mesh_bitmap_mark(words, count, 5, &proof, &marked_count));
    assert(!proof.atomic_word && !marked_count);
    assert(!rc_native_rmw_word(words, count, &proof));
    uint32_t *marked = rc_native_rmw_mesh_bitmap_mark(words, count, 31, &proof, &marked_count);
    if (argv[3][0] == '0') {
        assert(!marked && !proof.atomic_word && !marked_count);
    } else {
        assert(marked && marked_count > count && proof.bitmap_sites == 5);
        assert(proof.storage_set == 0 && proof.storage_binding == 5 && proof.execution_model == 5365);
        FILE *out = fopen(argv[2], "wb");
        assert(out && fwrite(marked, 4, marked_count, out) == marked_count && !fclose(out));
        assert(!rc_native_rmw_mesh_bitmap(marked, marked_count, &proof));
    }
    free(marked);
    free(words);
}
'''

def check_indirect_mesh_order():
    draw = (RUNTIME / "vk_exec_draw.c").read_text(encoding="utf-8")
    replay = (RUNTIME / "vk_exec.c").read_text(encoding="utf-8")
    ring = function(draw, "mesh_ring_entry")
    assert "!ring || vk_exec_in_chunk()" in ring
    ahead = ring.index("if (!graphics_current &&")
    order = ring.index("vk_exec_barrier(ex)")
    ring_copy = ring.index("vkCmdCopyBuffer(")
    assert ahead < order < ring_copy < ring.index("VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT")
    activation = function(draw, "graphics_draw_activate")
    assert activation.index("mesh_ring_entry(") < activation.index("vk_exec_graphics_pass_activate(")
    compatibility = function(draw, "prepare_draw_phase")
    ring_use = compatibility.index("graphics_compat_mesh_ring_entry(")
    assert ring_use < compatibility.index("vk_exec_pass_begin(", ring_use)
    assert "BR_OTHER" in function(replay, "barrier_to")
    barrier = function(replay, "barrier_to_why")
    assert "ex->barrier_at == ex->recorded && !n" in barrier
    assert ".srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT" in barrier
    assert ".srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT" in barrier
    assert ".dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT" in barrier
    assert ".dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT" in barrier


def main():
    check_indirect_mesh_order()
    cc = Path(os.environ["CC"])
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ["PATH"])
    validator = shutil.which("spirv-val", path=env["PATH"])
    with tempfile.TemporaryDirectory(prefix="mesh-bitmap-") as directory:
        temp = Path(directory)
        driver, exe = temp / "driver.c", temp / "driver.exe"
        driver.write_text(DRIVER, encoding="utf-8")
        subprocess.run([str(cc), "-std=gnu11", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                        "-fsanitize-trap=all", "-I" + str(RUNTIME), str(driver),
                        str(RUNTIME / "rc_native_rmw_word.c"), str(RUNTIME / "resource_core_inventory.c"),
                        "-o", str(exe)], env=env, check=True)
        original, marked = temp / "original.spv", temp / "marked.spv"
        original.write_bytes(mesh())
        subprocess.run([str(exe), str(original), str(marked), "1"], env=env, check=True)
        check_transform(original, marked)
        if validator:
            for module in (original, marked):
                subprocess.run([validator, "--target-env", "vulkan1.3", str(module)], env=env, check=True)
        for rejected in (mesh(ordinary_store=True), mesh(invalid_scope=True)):
            original.write_bytes(rejected)
            subprocess.run([str(exe), str(original), str(marked), "0"], env=env, check=True)
        for archived in sys.argv[1:]:
            subprocess.run([str(exe), archived, str(marked), "1"], env=env, check=True)
            check_transform(Path(archived), marked)
            if validator:
                subprocess.run([validator, "--target-env", "vulkan1.3", str(marked)], env=env, check=True)
    print("Mesh storage bitmap: five exact sites, untouched control and Workgroup CAS, bounds and unsupported writes passed")


if __name__ == "__main__":
    main()

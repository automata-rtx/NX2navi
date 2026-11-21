"""Check shader overwrite qualification using original synthetic SPIR-V."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def instruction(opcode, *operands):
    return [(len(operands) + 1) << 16 | opcode, *operands]


MAXIMAL_RECONVERGENCE = b"SPV_KHR_maximal_reconvergence"
MAXIMALLY_RECONVERGES = (16, 18, 6023)


def make_module(*, shift=4, arrayed=0, conditional=False, image_operands=0, second_store=False,
                workgroup_size=False, cancel_overflow=False, overflow_decoration=None, image_format=0,
                formatless_capability=True, reads_storage=False, formatless_read_capability=False,
                shader_capability=True, extension=None, execution_modes=()):
    words = [0x07230203, 0x00010300, 0, 41, 0]
    emit = lambda opcode, *args: words.extend(instruction(opcode, *args))
    if shader_capability:
        emit(17, 1)                     # Shader
    if formatless_capability:
        emit(17, 56)                    # StorageImageWriteWithoutFormat
    if formatless_read_capability:
        emit(17, 55)                    # StorageImageReadWithoutFormat
    if overflow_decoration in (4469, 4470):
        name = b"SPV_KHR_no_integer_wrap_decoration\0"
        name += b"\0" * (-len(name) % 4)
        emit(10, *struct.unpack(f"<{len(name) // 4}I", name))
    if extension is not None:
        name = extension + b"\0"
        name += b"\0" * (-len(name) % 4)
        emit(10, *struct.unpack(f"<{len(name) // 4}I", name))
    emit(14, 0, 1)                      # Logical GLSL450
    emit(15, 5, 18, 0x6e69616d, 0, 12, 13)  # GLCompute main, two input built-ins
    emit(16, 18, 17, 16, 8, 1)          # LocalSize
    for mode in execution_modes:
        emit(*mode)
    emit(71, 12, 11, 26)                # WorkgroupId
    emit(71, 13, 11, 27)                # LocalInvocationId
    emit(71, 17, 34, 0)
    emit(71, 17, 33, 2)
    if workgroup_size:
        emit(71, 36, 11, 25)            # A competing WorkgroupSize built-in
    if overflow_decoration is not None:
        emit(71, 38, overflow_decoration)
    emit(19, 1)                         # void
    emit(21, 2, 32, 0)                  # uint
    emit(23, 3, 2, 2)
    emit(23, 4, 2, 3)
    emit(22, 5, 32)
    emit(23, 6, 5, 4)
    emit(32, 7, 1, 4)
    emit(32, 8, 1, 2)
    emit(25, 9, 5, 1, 0, arrayed, 0, 2, image_format)  # float 2D storage image
    emit(32, 10, 0, 9)
    emit(33, 11, 1)
    emit(59, 7, 12, 1)
    emit(59, 7, 13, 1)
    emit(43, 2, 14, shift)
    emit(43, 2, 15, 3)
    emit(46, 6, 16)
    emit(59, 10, 17, 0)
    if conditional:
        emit(20, 32)
        emit(41, 32, 33)
    if workgroup_size:
        emit(43, 2, 34, 16)
        emit(43, 2, 35, 8)
        emit(44, 4, 36, 34, 35, 35)
    if cancel_overflow:
        emit(43, 2, 37, 0xffffffff if overflow_decoration == 4470 else 0x7fffffff)
    emit(54, 1, 18, 0, 11)
    emit(248, 19)
    emit(61, 4, 20, 12)
    emit(61, 4, 21, 13)
    emit(81, 2, 22, 20, 0)
    emit(81, 2, 23, 20, 1)
    emit(81, 2, 24, 21, 0)
    emit(81, 2, 25, 21, 1)
    emit(196, 2, 26, 22, 14)
    emit(196, 2, 27, 23, 15)
    emit(128, 2, 28, 26, 24)
    emit(128, 2, 29, 27, 25)
    if cancel_overflow:
        emit(128, 2, 38, 28, 37)
        emit(130, 2, 39, 38, 37)
    emit(80, 3, 30, 39 if cancel_overflow else 28, 29)
    emit(61, 9, 31, 17)
    if reads_storage:
        emit(98, 6, 40, 31, 30)
    if conditional:
        emit(247, 35, 0)
        emit(250, 33, 34, 35)
        emit(248, 34)
    emit(99, 31, 30, 16, image_operands)
    if second_store:
        emit(99, 31, 30, 16)
    if conditional:
        emit(249, 35)
        emit(248, 35)
    emit(253)
    emit(56)
    return struct.pack(f"<{len(words)}I", *words)


DRIVER = r"""
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "rc_native_image_store.h"

int main(int argc, char **argv) {
    assert(argc > 2);
    for (int i = 1; i < argc; ++i) {
        unsigned char expected = argv[i++][0];
        FILE *f = fopen(argv[i], "rb");
        assert(f && !fseek(f, 0, SEEK_END));
        long bytes = ftell(f);
        assert(bytes > 0 && !fseek(f, 0, SEEK_SET));
        uint32_t *words = malloc((size_t)bytes);
        assert(words && fread(words, 1, (size_t)bytes, f) == (size_t)bytes);
        fclose(f);
        RcNativeImageStore plan;
        const RcNativeImageStoreResult answer = expected == '-' ? RC_NATIVE_IMAGE_STORE_NONE :
            expected == 'p' || expected == 'r' ? RC_NATIVE_IMAGE_STORE_BINDING : RC_NATIVE_IMAGE_STORE_XY;
        assert(rc_native_image_store(words, (size_t)bytes / 4, &plan) == answer);
        if (expected != '-') {
            assert(plan.set == 0 && plan.binding == 2);
            assert(plan.scalar == RC_NATIVE_IMAGE_FLOAT32);
            assert(!rc_native_image_store_format(&plan, 29, 1, 1)); // R8_SINT cannot receive float stores.
            assert(!rc_native_image_store_format(&plan, 39, 1, 1)); // R8_UINT cannot receive float stores.
            assert(rc_native_image_store_format(&plan, 15, 1, 1)); // R8_UNORM.
            if (expected == 't') {
                assert(plan.image_format == 15);
                assert(rc_native_image_store_format(&plan, 15, 0, 0));
                assert(!rc_native_image_store_format(&plan, 20, 1, 1));
            } else {
                assert(plan.image_format == 0);
                assert(!rc_native_image_store_format(&plan, 15, 0, 1));
            }
            assert(plan.reads_storage == (expected == 'r'));
            assert(rc_native_image_store_format(&plan, 15, 1, 0) == (expected != 'r'));
            const uint32_t groups[] = {54, 45, 1};
            assert(rc_native_image_store_dispatch_safe(&plan, groups, 860, 360, 1));
            assert(!rc_native_image_store_dispatch_safe(&plan, groups, 860, 360, 0));
            if (answer == RC_NATIVE_IMAGE_STORE_XY) {
                assert(plan.local[0] == 16 && plan.local[1] == 8 && plan.local[2] == 1);
                assert(rc_native_image_store_dispatch_safe(&plan, groups, 865, 361, 0));
                assert(rc_native_image_store_covers(&plan, groups, 864, 360, 0));
                assert(!rc_native_image_store_covers(&plan, groups, 860, 360, 0));
                assert(rc_native_image_store_covers(&plan, groups, 860, 360, 1));
                assert(!rc_native_image_store_covers(&plan, groups, 865, 360, 1));
                assert(!rc_native_image_store_covers(&plan, groups, 860, 361, 1));
                assert(!rc_native_image_store_dispatch_safe(&plan, (uint32_t[]){54, 45, 2}, 860, 360, 1));
                assert(!rc_native_image_store_dispatch_safe(&plan, (uint32_t[]){0, 45, 1}, 860, 360, 1));
                assert(!rc_native_image_store_dispatch_safe(&plan, (uint32_t[]){UINT32_MAX, 45, 1}, 860, 360, 1));
            } else {
                assert(!plan.local[0]);
                assert(!rc_native_image_store_dispatch_safe(&plan, groups, 865, 361, 0));
                assert(!rc_native_image_store_covers(&plan, groups, 860, 360, 1));
            }
            assert(!rc_native_image_store(words, (size_t)bytes / 4 - 1, &plan));
        } else assert(!plan.local[0]);
        free(words);
    }
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", action="append", default=[], type=Path)
    parser.add_argument("--binding", action="append", default=[], type=Path)
    parser.add_argument("--reject", action="append", default=[], type=Path)
    args = parser.parse_args()
    cases = [
        ("+", {}),
        ("+", {"extension": MAXIMAL_RECONVERGENCE, "execution_modes": (MAXIMALLY_RECONVERGES,)}),
        ("p", {"extension": MAXIMAL_RECONVERGENCE, "execution_modes": (MAXIMALLY_RECONVERGES,), "conditional": True}),
        ("p", {"extension": MAXIMAL_RECONVERGENCE, "execution_modes": (MAXIMALLY_RECONVERGES,), "shift": 3}),
        ("p", {"execution_modes": (MAXIMALLY_RECONVERGES,)}),
        ("p", {"extension": b"SPV_KHR_float_controls", "execution_modes": (MAXIMALLY_RECONVERGES,)}),
        ("p", {"extension": MAXIMAL_RECONVERGENCE + b"\0X", "execution_modes": (MAXIMALLY_RECONVERGES,)}),
        ("p", {"extension": MAXIMAL_RECONVERGENCE, "execution_modes": (MAXIMALLY_RECONVERGES,),
               "shader_capability": False}),
        ("p", {"extension": MAXIMAL_RECONVERGENCE, "execution_modes": (MAXIMALLY_RECONVERGES,) * 2}),
        ("p", {"extension": MAXIMAL_RECONVERGENCE, "execution_modes": ((16, 18),)}),
        ("p", {"extension": MAXIMAL_RECONVERGENCE, "execution_modes": ((16, 18, 6023, 0),)}),
        ("p", {"extension": MAXIMAL_RECONVERGENCE, "execution_modes": ((16, 19, 6023),)}),
        ("p", {"extension": MAXIMAL_RECONVERGENCE, "execution_modes": ((16, 18, 6024),)}),
        ("p", {"extension": MAXIMAL_RECONVERGENCE, "execution_modes": ((331, 18, 6023),)}),
        ("p", {"shift": 3}),
        ("-", {"shift": 32}),
        ("-", {"arrayed": 1}),
        ("p", {"conditional": True}),
        ("-", {"image_operands": 64}),
        ("p", {"second_store": True}),
        ("p", {"workgroup_size": True}),
        ("+", {"cancel_overflow": True}),
        ("-", {"cancel_overflow": True, "overflow_decoration": 4469}),
        ("-", {"cancel_overflow": True, "overflow_decoration": 4470}),
        ("p", {"cancel_overflow": True, "overflow_decoration": 0}),
        ("t", {"image_format": 15, "formatless_capability": False}),
        ("-", {"image_format": 39, "formatless_capability": False}),
        ("-", {"formatless_capability": False}),
        ("r", {"reads_storage": True, "formatless_read_capability": True}),
        ("-", {"reads_storage": True}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        source = work / "check.c"
        source.write_text(DRIVER)
        exe = work / ("check.exe" if os.name == "nt" else "check")
        subprocess.run([os.environ.get("CC", "clang"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-I" + str(ROOT / "runtime"), str(source),
                        str(ROOT / "runtime" / "rc_native_image_store.c"), "-o", str(exe)], check=True)
        command = [str(exe)]
        for i, (result, options) in enumerate(cases):
            path = work / f"module-{i}.spv"
            path.write_bytes(make_module(**options))
            command.extend((result, str(path)))
        for result, paths in (("+", args.accept), ("p", args.binding), ("-", args.reject)):
            for path in paths:
                command.extend((result, str(path.resolve())))
        subprocess.run(command, check=True)
    print("native compute image overwrite checks passed")


if __name__ == "__main__":
    main()

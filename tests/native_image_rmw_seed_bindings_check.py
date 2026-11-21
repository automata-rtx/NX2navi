"""Check the opt-in finite image atomic inventory with production source only."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
import struct
import subprocess
import tempfile

from native_image_seed_bindings_check import instruction


ROOT = Path(__file__).resolve().parents[1]
WRITE = 2
ATOMIC = 7


def atomic_module(mutation=None):
    words = [0x07230203, 0x00010500, 0, 80, 0]

    def emit(opcode, *args):
        words.extend(instruction(opcode, *args))

    emit(17, 1)  # Shader
    if mutation != "missing-gather-capability":
        emit(17, 25)  # ImageGatherExtended
    emit(14, 0, 1)
    emit(15, 5, 50, 0x6E69616D, 0, 12, 16, 32, 33)
    emit(16, 50, 17, 8, 8, 1)
    for variable, binding in [(12, 1), (16, 3), (32, 2), (33, 4)]:
        emit(71, variable, 34, int(variable == 16 and mutation == "wrong-set"))
        emit(71, variable, 33, 3 if variable == 12 and mutation == "descriptor-alias" else binding)
    if mutation in ("non-writable", "non-readable"):
        emit(71, 16, 24 if mutation == "non-writable" else 25)

    emit(19, 1)
    emit(21, 2, 32, 0)
    emit(21, 3, 32, 1)
    emit(23, 4, 3, 2)
    emit(23, 5, 2, 4)
    emit(22, 6, 32)
    emit(23, 7, 6, 2)
    emit(23, 8, 6, 4)
    emit(25, 9, 6, 1, 0, 0, 0, 1, 0)
    emit(27, 10, 9)
    emit(32, 11, 0, 10)
    emit(59, 11, 12, 0)
    emit(25, 13, 2, 1, 0, 0, 0, 2, 33)
    if mutation in ("unknown-format", "wrong-format", "array", "ms", "volume", "depth"):
        shape = {"unknown-format": (1, 0, 0, 0, 0), "wrong-format": (1, 0, 0, 0, 30),
                 "array": (1, 0, 1, 0, 33), "ms": (1, 0, 0, 1, 33),
                 "volume": (2, 0, 0, 0, 33), "depth": (1, 1, 0, 0, 33)}
        dim, depth, array, ms, fmt = shape[mutation]
        emit(25, 34, 2, dim, depth, array, ms, 2, fmt)
    emit(32, 14, 0, 34 if mutation in ("unknown-format", "wrong-format", "array", "ms", "volume", "depth") else 13)
    emit(32, 15, 7 if mutation == "wrong-pointer-storage" else 11,
         3 if mutation == "wrong-pointer-scalar" else 2)
    emit(59, 14, 255 if mutation == "invalid-variable-id" else 16, 0)
    emit(32, 31, 0, 13)
    emit(59, 31, 32, 0)
    emit(59, 31, 33, 0)
    emit(33, 17, 1)
    emit(43, 2, 18, 1)
    emit(43, 2, 19, 0)
    emit(43, 2, 20, 2)
    emit(43, 3, 21, 0xFFFFFFFF)
    emit(43, 3, 22, 0)
    emit(46, 4, 23)
    emit(46, 7, 24)
    emit(44, 4, 25, 21, 22)
    emit(43, 2, 26, 3 if mutation == "gather-array-count" else 4)
    emit(28, 27, 4, 26)
    emit(44, 27, 28, 25, 23, 23, 25)
    emit(46, 5, 29)
    emit(32, 30, 7, 2)

    emit(54, 1, 50, 0, 17)
    emit(248, 51)
    emit(59, 30, 52, 7, 18)
    emit(61, 2, 53, 52)
    emit(61, 10, 54, 12)
    emit(100, 9, 55, 54)
    emit(95, 8, 56, 55, 23, 18 if mutation == "fetch-dynamic-offset" else 10, 19,
         53 if mutation == "fetch-nonconstant-offset" else 25)
    emit(96, 8, 57, 54, 24, 26 if mutation == "gather-component" else 19,
         16 if mutation == "gather-dynamic-offset" else 32, 23 if mutation == "gather-offset-type" else 28)
    emit(61, 13, 58, 32)
    emit(98, 5, 59, 58, 23)
    if mutation == "dynamic-root":
        emit(65, 14, 67, 16, 19)
    root = {"sample-root": 12, "image-value": 58, "dynamic-root": 67}.get(mutation, 16)
    if mutation != "no-atomic":
        emit(60, 15, 60, root, 19 if mutation == "coordinate-shape" else 23,
             18 if mutation == "sample-one" else 19)
        if mutation == "copy-texel-pointer":
            emit(83, 15, 61, 60)
        if mutation == "store-via-texel-pointer":
            emit(62, 60, 18)
        if mutation == "load-via-texel-pointer":
            emit(61, 2, 61, 60)
        if mutation != "unused-texel-pointer":
            emit(234 if mutation == "wrong-atomic-op" else 241, 3 if mutation == "wrong-result" else 2, 62,
                 61 if mutation == "copy-texel-pointer" else 60,
                 20 if mutation == "wrong-scope" else 18,
                 18 if mutation == "wrong-semantics" else 53 if mutation == "nonliteral-semantics" else 19,
                 21 if mutation == "wrong-atomic-value" else 53)
    emit(61, 13, 63, 33)
    emit(99, 63, 23, 29)
    if mutation == "untracked-image-op":
        emit(88, 8, 68, 54, 24, 2, 19)
    emit(253)
    emit(56)
    if mutation == "truncated":
        words.pop()
    return struct.pack(f"<{len(words)}I", *words)


DRIVER = r"""
#include "rc_native_image_store.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    assert(argc == 2);
    FILE *file = fopen(argv[1], "rb");
    assert(file && !fseek(file, 0, SEEK_END));
    const long size = ftell(file);
    assert(size >= 20 && !(size % 4) && !fseek(file, 0, SEEK_SET));
    uint32_t *words = malloc((size_t)size);
    assert(words && fread(words, 1, (size_t)size, file) == (size_t)size && !fclose(file));
    RcNativeImageStore store;
    RcNativeImageSeedBindings seed, rmw;
    const RcNativeImageStoreResult old = rc_native_image_store(words, (size_t)size / 4, &store);
    const int ordinary = rc_native_image_seed_bindings(words, (size_t)size / 4, &seed);
    const int atomics = rc_native_image_seed_rmw_bindings(words, (size_t)size / 4, &rmw);
    assert(ordinary || !seed.count);
    assert(atomics || !rmw.count);
    printf("%d %d %d", old, ordinary, atomics);
    for (unsigned i = 0; i < rmw.count; ++i) {
        RcNativeImageSeedBinding binding = rmw.entries[i];
        printf(" %u/%u/%u/%u/%u",binding.binding,binding.image_format,binding.scalar,binding.access,binding.dimension);
        assert(rc_native_image_seed_binding_format(&binding, binding.image_format, 0, 0));
        if (binding.access & RC_NATIVE_IMAGE_SEED_ATOMIC) {
            assert(!rc_native_image_seed_store(&binding, &store) && !store.scalar && !store.local[0]);
            assert(!rc_native_image_seed_binding_format(&binding, 30, 1, 1));
            binding.dimension = RC_NATIVE_IMAGE_SEED_3D;
            assert(!rc_native_image_seed_binding_format(&binding, 33, 1, 1));
            binding.dimension = RC_NATIVE_IMAGE_SEED_2D;
            binding.access = RC_NATIVE_IMAGE_SEED_ATOMIC;
            assert(!rc_native_image_seed_binding_format(&binding, 33, 1, 1));
        }
    }
    puts("");
    free(words);
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p99-final", type=Path)
    parser.add_argument("--p102-final", type=Path)
    options = parser.parse_args()
    compiler = Path(os.environ.get("CC", "clang"))
    environment = dict(os.environ)
    if compiler.is_absolute():
        environment["PATH"] = str(compiler.parent) + os.pathsep + environment["PATH"]
    controls = ("missing-gather-capability", "wrong-set", "descriptor-alias", "invalid-variable-id", "non-writable", "non-readable",
                "unknown-format", "wrong-format", "array", "ms", "volume", "depth", "wrong-pointer-storage",
                "wrong-pointer-scalar", "fetch-dynamic-offset", "fetch-nonconstant-offset", "sample-root",
                "gather-array-count", "gather-component", "gather-dynamic-offset", "gather-offset-type",
                "image-value", "dynamic-root", "coordinate-shape", "sample-one", "copy-texel-pointer",
                "store-via-texel-pointer", "load-via-texel-pointer", "unused-texel-pointer", "wrong-atomic-op",
                "wrong-result", "wrong-scope", "wrong-semantics", "nonliteral-semantics", "wrong-atomic-value",
                "untracked-image-op", "truncated")
    with tempfile.TemporaryDirectory(prefix="nx2-image-atomic-proof-") as temp:
        work = Path(temp)
        source = work / "proof.c"
        exe = work / "proof.exe"
        source.write_text(DRIVER, encoding="utf-8")
        runtime = ROOT / "runtime"
        subprocess.run([str(compiler), "-std=c11", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=undefined", "-fsanitize-trap=all", "-I", str(runtime), str(source),
                        str(runtime / "rc_native_image_store.c"), "-o", str(exe)], env=environment, check=True)

        def check(data, expected, label):
            module = work / "module.spv"
            module.write_bytes(data)
            actual = subprocess.check_output([str(exe), str(module)], env=environment, text=True).strip().split()
            assert actual == expected, (label, actual, expected)

        check(atomic_module(), ["0", "0", "1", "2/33/3/1/0", "3/33/3/7/0", "4/33/3/2/0"], "mixed seeded cohort")
        check(atomic_module("no-atomic"), ["0", "0", "0"], "atomic required")
        for case in controls:
            check(atomic_module(case), ["0", "0", "0"], case)
        if options.p99_final:
            data = options.p99_final.read_bytes()
            assert sha256(data).hexdigest() == "1356066c0138e06f94c38520398c4a442f9adb4e24dbedd5f09018d190f52947"
            check(data, ["0", "0", "1", "3/33/3/7/0"], "exact p99 final module")
        if options.p102_final:
            data = options.p102_final.read_bytes()
            assert sha256(data).hexdigest() == "6128e8a21f988ecdf43c5e52af6d89b350991b7c5b807b43298dfdbbd8d1172d"
            check(data, ["1", "1", "0"], "unchanged p102 final module")
    print(f"Image atomic inventory passed mixed receipts, {len(controls) + 1} refusal controls and supplied archived modules")


if __name__ == "__main__":
    main()

"""Check the final compute storage inventory without a native pipeline or draw."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
READ = 1
WRITE = 2
FLOAT = 1
UINT = 3


def instruction(opcode, *operands):
    return [(len(operands) + 1) << 16 | opcode, *operands]


def module(bindings, *, granted=READ | WRITE, extra=(), bad_view=None, indirect=False,
           qualifier=None, lexical_block=True, truncate=False, coordinate=7, local_size=(8, 8, 1),
           volume_members=(), volume_coordinate="unsigned"):
    # Each element gives the binding, true SPIR-V format, scalar class, actual access flags and descriptor set.
    entry = 11
    label = 12
    volume_scalar = 14 + 6 * len(bindings)
    volume_type, volume_value = volume_scalar + 1, volume_scalar + 2
    bound = volume_scalar + (3 if volume_members else 0)
    words = [0x07230203, 0x00010500, 0, bound, 0]
    emit = lambda opcode, *args: words.extend(instruction(opcode, *args))
    emit(17, 1)    # Shader
    emit(17, 49)   # StorageImageExtendedFormats
    if granted & READ:
        emit(17, 55)
    if granted & WRITE:
        emit(17, 56)
    emit(14, 0, 1)
    emit(15, 5, entry, 0x6e69616d, 0, *(16 + 6 * i for i in range(len(bindings))))
    emit(16, entry, 17, *local_size)
    for i, (binding, _, _, _, set_) in enumerate(bindings):
        variable = 16 + 6 * i
        emit(71, variable, 34, set_)
        emit(71, variable, 33, binding)
        if qualifier and qualifier[0] == i:
            emit(71, variable, qualifier[1])
    emit(19, 1)
    emit(22, 2, 32)
    emit(21, 3, 32, 0)
    emit(23, 4, 3, 2)
    emit(23, 5, 2, 4)
    emit(23, 6, 3, 4)
    emit(46, 4, 7)
    emit(46, 5, 8)
    emit(46, 6, 9)
    emit(33, 10, 1)
    if volume_members:
        scalar_type = 2 if volume_coordinate == "float" else volume_scalar
        emit(21, volume_scalar, 32, {"signed": 1, "invalid-sign": 2}.get(volume_coordinate, 0))
        emit(23, volume_type, scalar_type, 2 if volume_coordinate == "vec2" else 3)
        emit(46, volume_type, volume_value)
    for i, (_, format_, scalar, _, _) in enumerate(bindings):
        image, ptr, var = range(14 + 6 * i, 17 + 6 * i)
        dim, array, ms = bad_view if bad_view and i == 0 else (2 if i in volume_members else 1, 0, 0)
        emit(25, image, 2 if scalar == FLOAT else 3, dim, 0, array, ms, 2, format_)
        emit(32, ptr, 0, image)
        emit(59, ptr, var, 0)
    emit(54, 1, entry, 0, 10)
    if lexical_block:
        emit(248, label)
    for i, (_, _, scalar, access, _) in enumerate(bindings):
        image, _, var, loaded, read_result, copied = range(14 + 6 * i, 20 + 6 * i)
        emit(61, image, loaded, var)
        if indirect and i == 0:
            emit(83, image, copied, loaded)
            loaded = copied
        xy = (volume_type if volume_coordinate == "type-id" else volume_value) if i in volume_members else coordinate
        if access & READ:
            emit(98, 5 if scalar == FLOAT else 6, read_result, loaded, xy)
        if access & WRITE:
            texel = read_result if access & READ else 8 if scalar == FLOAT else 9
            emit(99, loaded, xy, texel)
    for args in extra:
        emit(*args)
    emit(253)
    emit(56)
    if truncate:
        words.pop()
    return struct.pack(f"<{len(words)}I", *words)


def sampled_helper_module(mutation=None):
    """Extend a single direct writer with a sampled image and a scalar leaf."""
    original = module([(2, 0, FLOAT, WRITE, 0)])
    sampled_volume = mutation in ("volume-fetch", "volume-fetch-wrong-coordinate", "volume-query")
    words = list(struct.unpack(f"<{len(original)//4}I", original))
    output = words[:5]
    output[3] = 64
    emit = lambda opcode, *args: output.extend(instruction(opcode, *args))
    emit(17, 50)  # ImageQuery
    i = 5
    while i < len(words):
        opcode, count = words[i] & 0xFFFF, words[i] >> 16
        params = words[i + 1:i + count]
        i += count
        if opcode == 15:
            params.extend((23, 31))
        if opcode == 19:
            emit(71, 23, 34, 2 if mutation == "unproven-sample-set" else 0)
            emit(71, 23, 33, 2 if mutation == "sample-storage-collision" else 3)
            emit(71, 31, 34, 0)
            emit(71, 31, 33, 0)
            if mutation != "unknown-uniform-block":
                emit(71, 28, 2)
            emit(72, 28, 0, 35, 0)
        if opcode == 54:
            emit(25, 20, 2, 2 if sampled_volume else 1, 0, 0, 0, 1, 0)
            emit(27, 21, 20)
            emit(32, 22, 0, 21)
            emit(59, 22, 23, 0)
            emit(43, 3, 24, 0)  # LOD and uniform struct member.
            emit(43, 2, 25, 0x3F800000)
            emit(32, 26, 7, 2)
            emit(33, 27, 2, 26)
            emit(30, 28, 2)
            emit(32, 29, 2, 28)
            emit(32, 30, 2, 2)
            emit(59, 29, 31, 2)
            if sampled_volume:
                emit(23, 41, 3, 3)
                emit(46, 41, 42)
        if opcode == 99:
            params[0] = 34 if mutation == "sampled-as-storage" else params[0]
            params[2] = 40
        emit(opcode, *params)
        if opcode == 248:
            emit(59, 26, 32, 7, 25)
            emit(61, 21, 33, 23)
            if mutation == "storage-as-sampled":
                emit(61, 14, 39, 16)
            emit(100, 20, 34, 39 if mutation == "storage-as-sampled" else 33)
            if not sampled_volume or mutation == "volume-query":
                emit(103, 41 if sampled_volume else 4, 35, 34, 3 if mutation == "lod-is-type-id" else 24)
            xy = 42 if sampled_volume and mutation != "volume-fetch-wrong-coordinate" else 7
            emit(95, 5, 36, 34, xy, 4 if mutation == "wrong-fetch-operands" else 2, 24)
            emit(81, 2, 37, 36, 0)
            emit(62, 32, 37)
            target = 11 if mutation == "recursive-call" else 50
            arguments = () if mutation == "wrong-helper-arity" else (31 if mutation == "wrong-helper-pointer" else 32,)
            emit(57, 2, 38, target, *arguments)
            emit(80, 5, 40, 38, 38, 38, 38)
    if mutation != "missing-helper":
        emit(54, 2, 50, 0, 27)
        emit(55, 30 if mutation == "wrong-helper-parameter-type" else 26, 51)
        emit(248, 52)
        emit(59, 26, 53, 7, 25)
        emit(61, 2, 54, 51)
        emit(65, 30, 55, 31, 24)
        emit(61, 2, 56, 55)
        if mutation == "nested-helper":
            emit(57, 2, 59, 50, 53)
        emit(136, 2, 57, 54, 56)
        destination = {"stale-main-pointer": 32, "parameter-store": 51, "uniform-store": 55}.get(mutation, 53)
        emit(62, destination, 57)
        emit(61, 2, 58, 53)
        if mutation == "helper-image-write":
            emit(61, 14, 59, 16)
            emit(99, 59, 7, 8)
        if mutation == "foreign-label":
            emit(249, 12)  # Main's inaccessible label.
            emit(248, 60)
        emit(254, 38 if mutation == "stale-main-return" else 58)
        emit(56)
    if mutation == "unreachable-helper":
        emit(54, 2, 61, 0, 27)
        emit(55, 26, 62)
        emit(248, 63)
        emit(254, 25)
        emit(56)
    return struct.pack(f"<{len(output)}I", *output)


def gathered_module(mutation=None):
    """Two written outputs share one exact gathered sampled input."""
    original = module([(2, 0, FLOAT, WRITE, 0), (3, 0, FLOAT, WRITE, 0)])
    words = list(struct.unpack(f"<{len(original)//4}I", original))
    output = words[:5]
    output[3] = 64
    emit = lambda opcode, *args: output.extend(instruction(opcode, *args))
    i = 5
    while i < len(words):
        opcode, count = words[i] & 0xFFFF, words[i] >> 16
        params = words[i + 1:i + count]
        i += count
        if opcode == 15:
            params.append(29)
            if mutation == "ambiguous-sampled-root":
                params.append(42)
        if opcode == 19:
            emit(71, 29, 34, 0)
            emit(71, 29, 33, 2 if mutation == "storage-binding-collision" else 4)
            if mutation == "ambiguous-sampled-root":
                emit(71, 42, 34, 0)
                emit(71, 42, 33, 4)
        if opcode == 54:
            dim = 2 if mutation == "volume-sampled-gather" else 1
            emit(25, 26, 2, dim, 0, mutation == "arrayed-sampled-view", 0, 1, 0)
            emit(27, 27, 26)
            emit(32, 28, 0, 27)
            emit(23, 30, 2, 2)
            emit(46, 30, 31)  # FP32 vec2 coordinate.
            component = 4 if mutation == "out-of-range-component" else 3 if mutation == "no-mask" else 0
            emit(43, 2 if mutation == "wrong-component-type" else 3, 32, component)
            emit(32, 33, 7, 3)
            emit(43, 3, 46, 0)
            if mutation == "dynamic-descriptor-index":
                emit(43, 3, 43, 2)
                emit(28, 44, 27, 43)
                emit(32, 45, 0, 44)
                emit(59, 45, 29, 0)
            else:
                emit(59, 28, 29, 0)
            if mutation == "ambiguous-sampled-root":
                emit(59, 28, 42, 0)
        if opcode == 99:
            params[2] = 41
        emit(opcode, *params)
        if opcode == 248:
            if mutation in ("dynamic-component", "dynamic-descriptor-index"):
                emit(59, 33, 34, 7, 46)
                emit(61, 3, 35, 34)
            if mutation == "dynamic-descriptor-index":
                emit(65, 28, 36, 29, 35)
                emit(61, 27, 37, 36)
            elif mutation == "storage-as-gather-source":
                emit(61, 14, 37, 16)
            else:
                emit(61, 27, 37, 29)
            coordinate = 7 if mutation == "wrong-coordinate-type" else 31
            component = 35 if mutation == "dynamic-component" else 32
            image_operands = () if mutation == "no-mask" else (8, 7) if mutation == "offset-mask" else (0,)
            emit(96, 6 if mutation == "wrong-result-type" else 5, 38, 37, coordinate, component, *image_operands)
            if mutation == "ambiguous-sampled-root":
                emit(61, 27, 39, 42)
                emit(96, 5, 40, 39, 31, 32, 0)
            emit(83, 5, 41, 38)
    return struct.pack(f"<{len(output)}I", *output)


def comparison_phi_module(mutation=None):
    """Two direct volume stores consume a scalar join of exact comparison and numeric paths."""
    original = module([(2, 0, FLOAT, WRITE, 0), (3, 0, FLOAT, WRITE, 0)], volume_members=(0, 1))
    words = list(struct.unpack(f"<{len(original)//4}I", original))
    output = words[:5]
    output[3] = 80
    emit = lambda opcode, *args: output.extend(instruction(opcode, *args))
    emit(17, 50)  # ImageQuery
    i = 5
    while i < len(words):
        opcode, count = words[i] & 0xFFFF, words[i] >> 16
        params = words[i + 1:i + count]
        i += count
        if opcode == 15:
            params.append(34)
            if mutation == "atomic-after-writers":
                params.append(64)
        if opcode == 19:
            emit(71, 34, 34, 0)
            emit(71, 34, 33, 2 if mutation == "comparison-storage-collision" else 4)
        if opcode == 54:
            depth = 0 if mutation == "ordinary-as-comparison" else 1
            array = 0 if mutation == "nonarray-comparison" else 1
            emit(25, 31, 2, 1, depth, array, 0, 1, 0)
            emit(27, 32, 31)
            emit(32, 33, 0, 32)
            if mutation == "dynamic-comparison-root":
                emit(43, 3, 56, 2)
                emit(28, 57, 32, 56)
                emit(32, 58, 0, 57)
                emit(59, 58, 34, 0)
            else:
                emit(59, 33, 34, 0)
            emit(23, 35, 2, 3)
            emit(46, 35, 36)
            emit(43, 2, 37, 0)
            emit(43, 2, 38, 0x3F800000)
            emit(20, 39)
            emit(41, 39, 40)
            emit(43, 3, 44, 0)
            emit(43, 3, 55, 1)
            if mutation == "dynamic-comparison-root":
                emit(32, 59, 7, 3)
            if mutation == "atomic-after-writers":
                emit(32, 63, 4, 3)
                emit(59, 63, 64, 4)
                emit(43, 3, 65, 2)  # Workgroup scope.
                emit(43, 3, 66, 0)  # Relaxed order.
        if opcode == 99:
            params[2] = 50
        if opcode == 253 and mutation == "atomic-after-writers":
            emit(234, 3, 67, 64, 65, 66, 55)
        if opcode == 253 and mutation == "extra-late-predecessor":
            emit(249, 54)
            emit(248, 54)
            emit(249, 53)
            continue
        emit(opcode, *params)
        if opcode != 248:
            continue
        if mutation == "dynamic-comparison-root":
            emit(59, 59, 60, 7, 44)
        emit(247, 53, 0)
        emit(250, 40, 51, 52)
        emit(248, 51)
        if mutation == "dynamic-comparison-root":
            emit(61, 3, 61, 60)
            emit(65, 33, 62, 34, 61)
            emit(61, 32, 41, 62)
        elif mutation == "storage-as-comparison":
            emit(61, 14, 41, 16)
        else:
            emit(61, 32, 41, 34)
        coordinate = 7 if mutation == "bad-comparison-coordinate" else 36
        reference = 44 if mutation == "bad-comparison-reference" else 37
        lod = 38 if mutation == "nonzero-comparison-lod" else 37
        image_operands = (10, lod, 7) if mutation == "comparison-offset" else (2, lod)
        emit(90, 2, 42, 41, coordinate, reference, *image_operands)
        emit(100, 31, 43, 41)
        emit(103, 27, 45, 43, 55 if mutation == "nonzero-depth-size-lod" else 44)
        emit(95, 5, 46, 43, 28, 2, 55 if mutation == "nonzero-depth-fetch-lod" else 44)
        emit(249, 54 if mutation == "parent-does-not-branch-to-join" else 53)
        if mutation == "parent-does-not-branch-to-join":
            emit(248, 54)
            emit(249, 53)
        emit(248, 52)
        if mutation == "opaque-phi":
            emit(61, 32, 47, 34)
        emit(133, 2, 48, 38, 37)
        emit(249, 53)
        emit(248, 53)
        if mutation == "late-phi":
            emit(83, 2, 68, 38)
        first_parent = 52 if mutation == "duplicate-phi-predecessor" else 51
        first_value = 48 if mutation == "value-from-wrong-predecessor" else 42
        if mutation == "opaque-phi":
            emit(245, 32, 49, 41, 51, 47, 52)
        elif mutation == "pointer-phi":
            emit(245, 33, 49, 34, 51, 34, 52)
        else:
            if mutation == "wrong-phi-type":
                first_value = 45
            emit(245, 2, 49, first_value, first_parent, 48, 52)
        emit(80, 5, 50, 49, 49, 49, 49)
    return struct.pack(f"<{len(output)}I", *output)


DRIVER = r"""
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rc_native_image_store.h"

static void expected_entries(const RcNativeImageSeedBindings *seed, const char *description) {
    unsigned seen = 0;
    for (const char *next = description; *next;) {
        unsigned binding, format, scalar, access, dimension = RC_NATIVE_IMAGE_SEED_2D;
        int used;
        assert(sscanf(next, "%u,%u,%u,%u%n", &binding, &format, &scalar, &access, &used) == 4 && used > 0);
        next += used;
        if (*next == ',') {
            assert(sscanf(next, ",%u%n", &dimension, &used) == 1 && used > 0);
            next += used;
        }
        assert(*next++ == ';');
        const RcNativeImageSeedBinding *found = NULL;
        for (unsigned i = 0; i < seed->count; i++) {
            if (seed->entries[i].binding == binding) {
                assert(!(seen & (1u << i)));
                found = &seed->entries[i];
                seen |= 1u << i;
                break;
            }
        }
        assert(found && found->image_format == format && found->scalar == scalar && found->access == access &&
               found->dimension == dimension);
        const uint32_t compatible = format ? format : scalar == RC_NATIVE_IMAGE_FLOAT32 ? 15 : 39;
        assert(rc_native_image_seed_binding_format(found, compatible, 1, 1));
        assert(!rc_native_image_seed_binding_format(found, scalar == RC_NATIVE_IMAGE_FLOAT32 ? 39 : 15, 1, 1));
        assert(rc_native_image_seed_binding_format(found, compatible, 1, 0) ==
            (format || !(access & RC_NATIVE_IMAGE_SEED_READ)));
        assert(rc_native_image_seed_binding_format(found, compatible, 0, 1) ==
            (format || !(access & RC_NATIVE_IMAGE_SEED_WRITE)));

        RcNativeImageStore store;
        const int writes = !!(access & RC_NATIVE_IMAGE_SEED_WRITE);
        assert(rc_native_image_seed_store(found, &store) == writes);
        if (writes) {
            assert(store.binding == binding && store.set == 0 && store.image_format == format &&
                   store.scalar == scalar && !store.local[0] && !store.local[1] && !store.local[2] &&
                   store.reads_storage == !!(access & RC_NATIVE_IMAGE_SEED_READ));
            assert(rc_native_image_store_format(&store, compatible, 1, 1));
            const uint32_t groups[] = {1,1,1};
            assert(rc_native_image_store_dispatch_safe(&store, groups, 8, 8, 1));
            assert(!rc_native_image_store_dispatch_safe(&store, groups, 8, 8, 0));
            assert(!rc_native_image_store_covers(&store, groups, 8, 8, 1));
        } else {
            assert(!store.scalar);
        }
    }
    assert(seen == (1u << seed->count) - 1);
}

int main(int argc, char **argv) {
    RcNativeImageSeedBinding invalid = {.binding=2,.scalar=RC_NATIVE_IMAGE_FLOAT32,
        .access=RC_NATIVE_IMAGE_SEED_WRITE,.dimension=RC_NATIVE_IMAGE_SEED_3D+1};
    RcNativeImageStore rejected;
    assert(!rc_native_image_seed_store(&invalid, &rejected));
    assert(!rc_native_image_seed_binding_format(&invalid, 15, 1, 1));
    assert(argc > 1 && (argc - 1) % 3 == 0);
    for (int i = 1; i < argc; i += 3) {
        const char *name = argv[i];
        const char *expected = argv[i+1];
        const int old_expected = atoi(argv[i+2]);
        FILE *f = fopen(name, "rb");
        assert(f && !fseek(f, 0, SEEK_END));
        const long bytes = ftell(f);
        assert(bytes > 0 && !(bytes % 4) && !fseek(f, 0, SEEK_SET));
        uint32_t *words = malloc((size_t)bytes);
        assert(words && fread(words, 1, (size_t)bytes, f) == (size_t)bytes);
        fclose(f);

        RcNativeImageSeedBindings seed;
        memset(&seed, 0xff, sizeof seed);
        if (rc_native_image_seed_bindings(words, (size_t)bytes / 4, &seed) != (*expected != 0)) {
            fprintf(stderr, "Unexpected seed inventory for %s\n", name);
            return 1;
        }
        if (!*expected) {
            assert(!seed.count);
            for (unsigned j = 0; j < RC_NATIVE_IMAGE_SEED_MAX_BINDINGS; j++)
                assert(!seed.entries[j].binding && !seed.entries[j].access && !seed.entries[j].scalar &&
                       !seed.entries[j].image_format && !seed.entries[j].dimension);
        }
        expected_entries(&seed, expected);
        if (old_expected >= 0) {
            RcNativeImageStore prior;
            assert(rc_native_image_store(words, (size_t)bytes / 4, &prior) == (RcNativeImageStoreResult)old_expected);
        }
        free(words);
    }
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="append", nargs=3, metavar=("MODULE", "ENTRIES", "OLD_RESULT"), default=[])
    args = parser.parse_args()
    write = (2, 0, FLOAT, WRITE, 0)
    read = (3, 0, FLOAT, READ, 0)
    typed_read = (4, 33, UINT, READ, 0)
    typed_write = (5, 33, UINT, WRITE, 0)
    both = (6, 0, FLOAT, READ | WRITE, 0)
    cases = [
        ("two-outputs", [write, typed_write], {}, "2,0,1,2;5,33,3,2;", 0),
        ("readonly-typed-peer", [write, typed_read], {"granted": WRITE, "qualifier": (1, 24)}, "2,0,1,2;4,33,3,1;", 0),
        ("read-write-and-readonly", [both, read], {}, "6,0,1,3;3,0,1,1;", 0),
        ("one-binding", [write], {}, "2,0,1,2;", 1),
        ("eight-bindings", [(i, 0, FLOAT, WRITE, 0) for i in range(56, 64)], {},
            "".join(f"{i},0,1,2;" for i in range(56, 64)), 0),
        ("only-read", [read], {}, "", 0),
        ("nine-bindings", [(i, 0, FLOAT, WRITE, 0) for i in range(9)], {}, "", 0),
        ("binding-64", [(64, 0, FLOAT, WRITE, 0)], {}, "", 1),
        ("nonzero-set", [(2, 0, FLOAT, WRITE, 1)], {}, "", 1),
        ("two-roots-at-one-binding", [write, write], {}, "", 1),
        ("mismatched-format-scalar", [write, (5, 33, FLOAT, WRITE, 0)], {}, "", 0),
        ("readonly-capability-absent", [write, read], {"granted": WRITE}, "", 0),
        ("writeonly-capability-absent", [write, read], {"granted": READ}, "", 0),
        ("forbidden-read", [write, read], {"qualifier": (1, 25)}, "", 0),
        ("forbidden-write", [write], {"qualifier": (0, 24)}, "", 1),
        ("arrayed-image", [write], {"bad_view": (1, 1, 0)}, "", 0),
        ("multisampled-image", [write], {"bad_view": (1, 0, 1)}, "", 0),
        ("three-dimensional-image", [write], {"bad_view": (2, 0, 0)}, "", 0),
        ("indirect-image", [write], {"indirect": True}, "", 0),
        ("image-texel-pointer", [write], {"extra": ((60, 3, 19, 17, 7, 7),)}, "", 0),
        ("sparse-image-read", [write], {"extra": ((320, 5, 19, 17, 7),)}, "", 0),
        ("unknown-instruction", [write], {"extra": ((60000,),)}, "", 1),
        ("wrong-coordinate-type", [write], {"coordinate": 9}, "", 1),
        ("invalid-local-size", [write], {"local_size": (8, 0, 1)}, "", 1),
        ("operation-outside-block", [write], {"lexical_block": False}, "", 1),
        ("truncated-function", [write], {"truncate": True}, "", 0),
        ("volume-uint-coordinates", [write], {"volume_members": (0,)}, "2,0,1,2,1;", 0),
        ("volume-sint-coordinates", [typed_write], {"volume_members": (0,), "volume_coordinate": "signed"}, "5,33,3,2,1;", 0),
        ("mixed-dimensional-writers", [write, typed_write], {"volume_members": (1,)}, "2,0,1,2;5,33,3,2,1;", 0),
        ("volume-writer-peer", [both, read], {"volume_members": (0, 1)}, "6,0,1,3,1;3,0,1,1,1;", 0),
        ("two-dimensional-volume-coordinates", [write], {"volume_members": (0,), "volume_coordinate": "vec2"}, "", 0),
        ("floating-volume-coordinates", [write], {"volume_members": (0,), "volume_coordinate": "float"}, "", 0),
        ("invalid-volume-coordinate-sign", [write], {"volume_members": (0,), "volume_coordinate": "invalid-sign"}, "", 0),
        ("volume-coordinate-type-id", [write], {"volume_members": (0,), "volume_coordinate": "type-id"}, "", 0),
        ("arrayed-volume", [write], {"volume_members": (0,), "bad_view": (2, 1, 0)}, "", 0),
        ("multisampled-volume", [write], {"volume_members": (0,), "bad_view": (2, 0, 1)}, "", 0),
        ("volume-other-writer-invalid", [write, typed_write], {"volume_members": (1,), "volume_coordinate": "vec2"}, "", 0),
        ("volume-duplicate-roots", [write, write], {"volume_members": (1,)}, "", 0),
    ]
    cc = Path(os.environ["CC"]).resolve()
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get("PATH", ""))
    with tempfile.TemporaryDirectory(prefix="native-image-seed-") as directory:
        path = Path(directory)
        c = path / "check.c"
        exe = path / ("check.exe" if os.name == "nt" else "check")
        c.write_text(DRIVER, encoding="utf-8")
        subprocess.run([str(cc), "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined", "-fsanitize-trap=all",
                        "-I"+str(ROOT/"runtime"), str(c), str(ROOT/"runtime/rc_native_image_store.c"), "-o", str(exe)],
                       env=env, check=True)
        command = [str(exe)]
        for name, bindings, options, expected, old in cases:
            binary = path / (name + ".spv")
            binary.write_bytes(module(bindings, **options))
            command.extend((str(binary), expected, str(old)))
        for mutation in (None, "unproven-sample-set", "sample-storage-collision", "unknown-uniform-block",
                         "storage-as-sampled", "sampled-as-storage", "lod-is-type-id", "wrong-fetch-operands", "wrong-helper-arity",
                         "recursive-call", "wrong-helper-pointer", "missing-helper", "wrong-helper-parameter-type",
                         "nested-helper", "stale-main-pointer", "parameter-store", "uniform-store", "helper-image-write",
                         "foreign-label", "stale-main-return", "unreachable-helper", "volume-fetch",
                         "volume-fetch-wrong-coordinate", "volume-query"):
            binary = path / ((mutation or "sampled-and-local-helper") + ".spv")
            binary.write_bytes(sampled_helper_module(mutation))
            expected = "2,0,1,2;" if mutation in (None, "volume-fetch") else ""
            command.extend((str(binary), expected, str(int(mutation != "sampled-as-storage"))))
        for mutation in (None, "no-mask", "offset-mask", "out-of-range-component", "wrong-component-type",
                         "dynamic-component", "wrong-coordinate-type", "wrong-result-type", "arrayed-sampled-view",
                         "volume-sampled-gather",
                         "storage-as-gather-source", "dynamic-descriptor-index", "storage-binding-collision",
                         "ambiguous-sampled-root"):
            binary = path / ("gather-" + (mutation or "zero-mask") + ".spv")
            binary.write_bytes(gathered_module(mutation))
            expected = "2,0,1,2;3,0,1,2;" if mutation in (None, "no-mask") else ""
            command.extend((str(binary), expected, "0"))
        for mutation in (None, "ordinary-as-comparison", "nonarray-comparison", "comparison-storage-collision",
                         "dynamic-comparison-root", "storage-as-comparison", "bad-comparison-coordinate",
                         "bad-comparison-reference", "nonzero-comparison-lod", "comparison-offset", "nonzero-depth-size-lod",
                         "nonzero-depth-fetch-lod", "parent-does-not-branch-to-join", "extra-late-predecessor",
                         "duplicate-phi-predecessor", "value-from-wrong-predecessor", "wrong-phi-type", "late-phi",
                         "opaque-phi", "pointer-phi", "atomic-after-writers"):
            binary = path / ("comparison-phi-" + (mutation or "valid") + ".spv")
            binary.write_bytes(comparison_phi_module(mutation))
            command.extend((str(binary), "" if mutation else "2,0,1,2,1;3,0,1,2,1;", "0"))
        for filename, entries, old in args.archive:
            command.extend((str(Path(filename).resolve()), entries, old))
        subprocess.run(command, env=env, check=True)
    print("Exact seeded storage inventory: multiple writers, readonly peers, first-write baseline, sampled operations, scalar helpers, volume shapes, comparison/Phi and refusal controls passed")


if __name__ == "__main__":
    main()

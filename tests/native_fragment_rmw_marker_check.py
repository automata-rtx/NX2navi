"""Check an indexed fragment atomic's execution bitmap without changing its guest code."""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'

FRAGMENT = r'''
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %coord %color %target
OpExecutionMode %main OriginUpperLeft
OpDecorate %coord BuiltIn FragCoord
OpDecorate %color Location 0
OpDecorate %words ArrayStride 4
OpDecorate %block Block
OpMemberDecorate %block 0 Offset 0
OpDecorate %target DescriptorSet 1
OpDecorate %target Binding 2
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%u32 = OpTypeInt 32 0
%f32 = OpTypeFloat 32
%vec4 = OpTypeVector %f32 4
%words = OpTypeRuntimeArray %u32
%block = OpTypeStruct %words
%pointer = OpTypePointer StorageBuffer %block
%word = OpTypePointer StorageBuffer %u32
%input = OpTypePointer Input %vec4
%output = OpTypePointer Output %u32
%target = OpVariable %pointer StorageBuffer
%coord = OpVariable %input Input
%color = OpVariable %output Output
%zero = OpConstant %u32 0
%one = OpConstant %u32 1
%seven = OpConstant %u32 7
%main = OpFunction %void None %fn
%entry = OpLabel
%xy = OpLoad %vec4 %coord
%x = OpCompositeExtract %f32 %xy 0
%bits = OpBitcast %u32 %x
%i = OpBitwiseAnd %u32 %bits %seven
%take = OpULessThan %bool %i %seven
OpSelectionMerge %done None
OpBranchConditional %take %yes %done
%yes = OpLabel
%p = OpAccessChain %word %target %zero %i
%old = OpAtomicUMin %u32 %p %one %zero %one
OpStore %color %old
OpBranch %done
%done = OpLabel
OpReturn
OpFunctionEnd
'''

DRIVER = r'''
#include "rc_native_rmw_word.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    assert(argc == 3);
    FILE *input = fopen(argv[1], "rb");
    assert(input && !fseek(input, 0, SEEK_END));
    long bytes = ftell(input);
    assert(bytes > 0 && bytes % 4 == 0 && !fseek(input, 0, SEEK_SET));
    const size_t count = (size_t)bytes / 4;
    uint32_t *words = malloc((size_t)bytes);
    assert(words && fread(words, 4, count, input) == count && !fclose(input));
    RcNativeRmwWord proof;
    size_t marked_count;
    if (!rc_native_rmw_fragment(words, count, &proof)) {
        assert(!rc_native_rmw_fragment_mark(words, count, 31, &proof, &marked_count) && !marked_count);
        free(words);
        return 2;
    }
    assert(proof.execution_model == 4 && proof.index);
    assert(!rc_native_rmw_fragment_mark(words, count, proof.storage_binding, &proof, &marked_count));
    assert(!rc_native_rmw_fragment_mark(words, count - 1, 31, &proof, &marked_count));
    uint32_t *marked = rc_native_rmw_fragment_mark(words, count, 31, &proof, &marked_count);
    assert(marked && marked_count > count);
    size_t duplicate;
    assert(!rc_native_rmw_fragment_mark(marked, marked_count, 32, &proof, &duplicate));
    FILE *out = fopen(argv[2], "wb");
    assert(out && fwrite(marked, 4, marked_count, out) == marked_count && !fclose(out));
    free(words);
    free(marked);
}
'''

def instructions(module):
    data = module.read_bytes()
    words = struct.unpack(f'<{len(data) // 4}I', data)
    index = 5
    while index < len(words):
        size, opcode = divmod(words[index], 65536)
        assert size and index + size <= len(words)
        yield opcode, words[index + 1:index + size]
        index += size

def check_path(original, marked):
    before, after = list(instructions(original)), list(instructions(marked))
    marker, = (x[0] for op, x in after if op == 71 and x[1:] == (33, 31))
    new_set, = (x[2] for op, x in after if op == 71 and x[:2] == (marker, 34))
    b, = (x for op, x in before if op == 237)
    origin, = (x for op, x in before if op == 65 and x[1] == b[2])
    assert (71, (origin[2], 34, new_set)) in before
    start = next(i for i, (op, _) in enumerate(before) if op == 54)
    marked_start = next(i for i, (op, _) in enumerate(after) if op == 54)
    first = after.index((237, b)) + 1
    last = next(i for i in range(first, len(after)) if after[i][0] == 241)
    extra = after[first:last + 1]
    assert [op for op, _ in extra] == [124,199,194,128,196,68,68,176,176,167,169,169,65,241]
    assert extra[0][1][-1] == origin[-1]
    assert extra[5][1][2] == origin[2] and extra[6][1][2] == marker
    assert extra[-2][1][2] == marker and extra[-1][1][3:5] == b[3:5]
    assert extra[-1][1][-1] == extra[-3][1][1]
    assert before[start:] == after[marked_start:first] + after[last + 1:]
    old_entry, = (x for op, x in before if op == 15)
    new_entry, = (x for op, x in after if op == 15)
    assert new_entry == (old_entry if original.read_bytes()[5] < 4 else old_entry + (marker,))

def main():
    cc = Path(os.environ['CC']).resolve()
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
    assembler, validator = (shutil.which(name, path=env['PATH']) for name in ('spirv-as', 'spirv-val'))
    if not assembler or not validator:
        sys.exit('Native fragment RMW marker test requires spirv-as and spirv-val')
    with tempfile.TemporaryDirectory(prefix='fragment-rmw-') as directory:
        temp = Path(directory)
        src = temp / 'fragment.spvasm'
        driver = temp / 'marker.c'
        exe = driver.with_suffix('.exe')
        driver.write_text(DRIVER)
        def run(*args, status=0):
            assert subprocess.run(list(map(str, args)), env=env, check=False).returncode == status
        def assemble(code, name):
            src.write_text(code)
            binary = temp / name
            run(assembler, '--target-env', 'vulkan1.1spv1.4', src, '-o', binary)
            run(validator, '--target-env', 'vulkan1.3', binary)
            return binary
        run(cc, '-std=gnu11', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined', '-fsanitize-trap=all',
            '-I' + str(RUNTIME), driver, RUNTIME / 'rc_native_rmw_word.c', RUNTIME / 'resource_core_inventory.c', '-o', exe)
        stock = assemble(FRAGMENT, 'conditional.spv')
        for original in [stock, *map(Path, sys.argv[1:])]:
            output = temp / 'marked.spv'
            run(exe, original, output)
            run(validator, '--target-env', 'vulkan1.3', output)
            check_path(original, output)
        for changed in (FRAGMENT.replace('ArrayStride 4', 'ArrayStride 8'),
                        FRAGMENT.replace('%old = OpAtomicUMin', '%extra = OpLoad %u32 %p\n%old = OpAtomicUMin')):
            run(exe, assemble(changed, 'other.spv'), temp / 'refused.spv', status=2)
    print('Native fragment marker preserves conditional atomic, original result and exact bounds')

if __name__ == '__main__':
    main()

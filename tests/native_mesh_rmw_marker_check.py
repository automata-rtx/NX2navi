"""Validate the native conditional atomic marker and unchanged original SPIR-V control flow."""
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'
VALIDATION_MODULE = r'''
OpCapability Shader
OpCapability MeshShadingEXT
OpExtension "SPV_EXT_mesh_shader"
OpMemoryModel Logical GLSL450
OpEntryPoint MeshEXT %main "main" %c0 %target
OpExecutionMode %main LocalSize 1 1 1
OpExecutionMode %main OutputVertices 1
OpExecutionMode %main OutputPrimitivesEXT 1
OpExecutionMode %main OutputTrianglesEXT
OpDecorate %rows ArrayStride 16
OpDecorate %uniform Block
OpMemberDecorate %uniform 0 Offset 0
OpDecorate %c0 DescriptorSet 0
OpDecorate %c0 Binding 6
OpDecorate %words ArrayStride 4
OpDecorate %storage Block
OpMemberDecorate %storage 0 Offset 0
OpDecorate %target DescriptorSet 0
OpDecorate %target Binding 5
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%u32 = OpTypeInt 32 0
%float = OpTypeFloat 32
%f4 = OpTypeVector %float 4
%zero = OpConstant %u32 0
%one = OpConstant %u32 1
%two = OpConstant %u32 2
%delta = OpConstant %u32 32
%mask = OpConstant %u32 4294967280
%offset = OpConstant %u32 120
%nrows = OpConstant %u32 121
%rows = OpTypeArray %f4 %nrows
%uniform = OpTypeStruct %rows
%uniform_ptr = OpTypePointer Uniform %uniform
%float_ptr = OpTypePointer Uniform %float
%words = OpTypeRuntimeArray %u32
%storage = OpTypeStruct %words
%storage_ptr = OpTypePointer StorageBuffer %storage
%word_ptr = OpTypePointer StorageBuffer %u32
%c0 = OpVariable %uniform_ptr Uniform
%target = OpVariable %storage_ptr StorageBuffer
%main = OpFunction %void None %fn
%entry = OpLabel
OpSetMeshOutputsEXT %zero %zero
%address_input = OpAccessChain %float_ptr %c0 %zero %offset %zero
%value = OpLoad %float %address_input
%address = OpBitcast %u32 %value
%predicate = OpINotEqual %bool %address %zero
OpSelectionMerge %done None
OpBranchConditional %predicate %issued %done
%issued = OpLabel
%sum = OpIAdd %u32 %address %delta
%base = OpBitwiseAnd %u32 %address %mask
%relative = OpISub %u32 %sum %base
%index = OpShiftRightLogical %u32 %relative %two
%ptr = OpAccessChain %word_ptr %target %zero %index
%previous = OpAtomicIAdd %u32 %ptr %one %zero %one
%has_previous = OpINotEqual %bool %previous %zero
OpSelectionMerge %join None
OpBranchConditional %has_previous %next %join
%next = OpLabel
OpBranch %join
%join = OpLabel
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
    FILE *source = fopen(argv[1], "rb");
    assert(source && !fseek(source, 0, SEEK_END));
    long bytes = ftell(source);
    assert(bytes > 0 && !(bytes % 4) && !fseek(source, 0, SEEK_SET));
    size_t count = (size_t)bytes / 4;
    uint32_t *words = malloc((size_t)bytes);
    assert(words && fread(words, 4, count, source) == count && !fclose(source));
    RcNativeRmwWord actual;
    size_t marked_count;
    assert(!rc_native_rmw_word_mark(words, count, 5, &actual, &marked_count));
    assert(!actual.atomic_word && !marked_count);  // original storage descriptor collides
    assert(!rc_native_rmw_word_mark(words, count - 1, 31, &actual, &marked_count));
    assert(!actual.atomic_word && !marked_count);
    assert(rc_native_rmw_word(words, count, &actual));
    assert(actual.storage_set == 0 && actual.storage_binding == 5 &&
           actual.c0_set == 0 && actual.c0_binding == 6 && actual.c0_byte == 0x780);
    for (uint32_t lane = 0; lane < 16; ++lane) {
        uint64_t address;
        assert(rc_native_rmw_word_address(&actual, lane, 0x1000, 48, &address));
        assert(address == 0x1020 + (lane & ~3u));
        assert(!rc_native_rmw_word_address(&actual, lane, 0x1000, address - 0x1000 + 3, &address));
    }
    uint32_t *marked = rc_native_rmw_word_mark(words, count, 31, &actual, &marked_count);
    assert(marked && marked_count > count);
    FILE *output = fopen(argv[2], "wb");
    assert(output && fwrite(marked, 4, marked_count, output) == marked_count && !fclose(output));
    assert(!rc_native_rmw_word_mark(marked, marked_count, 32, &actual, &count)); // Never instrument twice.
    free(marked);
    free(words);
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


def check_original_control(source, instrumented):
    before = list(instructions(source))
    after = list(instructions(instrumented))
    marker_binding, = (args for op, args in after if op == 71 and len(args) == 3 and args[1:] == (33, 31))
    marker, = marker_binding[:1]
    entry, = (args for op, args in after if op == 15)
    old_entry, = (args for op, args in before if op == 15)
    assert entry == old_entry + (marker,)
    body = [instruction for instruction in before if instruction[0] == 54]
    assert body
    before_body = before[before.index(body[0]):]
    after_body = after[after.index(body[0]):]
    atomics = [i for i, (op, _) in enumerate(after_body) if op == 241]
    marker_op = next(i for i in atomics if after_body[i - 1][0] == 65 and after_body[i - 1][1][2] == marker)
    access = after_body[marker_op - 1][1]
    written = after_body[marker_op][1]
    original_op, original_atomic = after_body[marker_op - 2]
    assert original_op == 234 and written[2] == access[1] and written[3:5] == original_atomic[3:5]
    assert any(op == 43 and const[1:] == (written[5], 1) for op, const in after)
    assert after_body[:marker_op - 1] + after_body[marker_op + 1:] == before_body


def main():
    cc = Path(os.environ['CC']).resolve()
    env = dict(os.environ, PATH=str(cc.parent) + os.pathsep + os.environ.get('PATH', ''))
    assembler, validator = (shutil.which(name, path=env['PATH']) for name in ('spirv-as', 'spirv-val'))
    if not assembler or not validator:
        sys.exit('native mesh RMW marker test requires spirv-as and spirv-val')
    with tempfile.TemporaryDirectory(prefix='mesh-rmw-') as dir_name:
        temp = Path(dir_name)
        src = temp / 'conditional.spvasm'
        driver = temp / 'marker.c'
        exe = driver.with_suffix('.exe')
        src.write_text(VALIDATION_MODULE)
        stock = src.with_suffix('.spv')
        driver.write_text(DRIVER)
        subprocess.run([assembler, '--target-env', 'vulkan1.1spv1.4', str(src), '-o', str(stock)], env=env, check=True)
        subprocess.run([str(cc), '-std=gnu11', '-Wall', '-Wextra', '-Werror',
                        '-fsanitize=undefined', '-fsanitize-trap=all', '-I' + str(RUNTIME),
                        str(driver), str(RUNTIME / 'rc_native_rmw_word.c'),
                        str(RUNTIME / 'resource_core_inventory.c'), '-o', str(exe)], env=env, check=True)
        for original in [stock, *map(Path, sys.argv[1:])]:
            output = temp / 'marked.spv'
            subprocess.run([str(exe), str(original), str(output)], env=env, check=True)
            subprocess.run([validator, '--target-env', 'vulkan1.3', str(output)], env=env, check=True)
            check_original_control(original, output)
    print('Native mesh marker: scope, distinct binding, control flow, unchanged atomic result and Vulkan validation passed')


if __name__ == '__main__':
    main()

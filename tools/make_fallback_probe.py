"""Deterministic private probe for exact sampled depth and typed unbound storage."""
from make_dlaa_mask import Asm, f32, string

asm = Asm()
main, void, fn, boolean, u32, f, u2, u3, u4, f2 = (asm.id() for _ in range(10))
depth, sampled, storage_image, depth_pointer, storage_pointer = (asm.id() for _ in range(5))
out_array, out_block, out_pointer, out_element, input_pointer = (asm.id() for _ in range(5))
invocation, depth_uniform, missing_storage, report = (asm.id() for _ in range(4))
zero, one, zero_coord, low_x, high_x, half, lod = (asm.id() for _ in range(7))
start, thread, index, left, x, coord, d, scalar, storage, bits, storage_value, compared, sum_value, output = (
    asm.id() for _ in range(14))

asm.op('cap', 17, 1)  # Shader
asm.op('mem', 14, 0, 1)
asm.op('entry', 15, 5, main, *string('main'), invocation)
asm.op('exec', 16, main, 17, 2, 1, 1)
asm.op('dec', 71, invocation, 11, 28)
asm.op('dec', 71, out_array, 6, 4)
asm.op('dec', 71, out_block, 3)
asm.op('dec', 72, out_block, 0, 35, 0)
for binding, object_ in enumerate((depth_uniform, missing_storage, report), 1):
    asm.op('dec', 71, object_, 34, 0)
    asm.op('dec', 71, object_, 33, binding)
for opcode, identification, args in (
    (19, void, ()), (33, fn, (void,)), (20, boolean, ()), (21, u32, (32, 0)), (22, f, (32,)),
    (23, u2, (u32, 2)), (23, u3, (u32, 3)), (23, u4, (u32, 4)), (23, f2, (f, 2)),
    (25, depth, (f, 1, 1, 0, 0, 1, 0)), (27, sampled, (depth,)),
    (25, storage_image, (u32, 1, 0, 0, 0, 2, 33)),  # R32ui
    (29, out_array, (u32,)), (30, out_block, (out_array,))):
    asm.op('types', opcode, identification, *args)
for p, typ, space in ((depth_pointer, sampled, 0), (storage_pointer, storage_image, 0),
                      (input_pointer, u3, 1), (out_pointer, out_block, 2), (out_element, u32, 2)):
    asm.op('types', 32, p, space, typ)
for spec, typ, value in ((zero, u32, 0), (one, u32, 1), (low_x, f, f32(0.25)),
                         (high_x, f, f32(0.75)), (half, f, f32(0.5)), (lod, f, 0)):
    asm.op('types', 43, typ, spec, value)
asm.op('types', 44, u2, zero_coord, zero, zero)
for p, obj, space in ((input_pointer, invocation, 1), (depth_pointer, depth_uniform, 0),
                      (storage_pointer, missing_storage, 0), (out_pointer, report, 2)):
    asm.op('types', 59, p, obj, space)

asm.op('func', 54, void, main, 0, fn)
asm.op('func', 248, start)
asm.op('func', 61, u3, thread, invocation)
asm.op('func', 81, u32, index, thread, 0)
asm.op('func', 176, boolean, left, index, one)
asm.op('func', 169, f, x, left, low_x, high_x)
asm.op('func', 80, f2, coord, x, half)
asm.op('func', 61, sampled, d, depth_uniform)
asm.op('func', 90, f, scalar, d, coord, half, 2, lod)  # DRefExplicitLod
asm.op('func', 109, u32, compared, scalar)  # ConvertFToU
asm.op('func', 61, storage_image, storage, missing_storage)
asm.op('func', 98, u4, bits, storage, zero_coord)
asm.op('func', 81, u32, storage_value, bits, 0)
asm.op('func', 128, u32, sum_value, compared, storage_value)
asm.op('func', 65, out_element, output, report, zero, index)
asm.op('func', 62, output, sum_value)
asm.op('func', 253)
asm.op('func', 56)

words = asm.words()
lines = ['#pragma once', '#include <stdint.h>', 'static const uint32_t fallback_probe_spv[] = {']
for start in range(0, len(words), 8):
    lines.append('    ' + ', '.join(f'0x{w:08x}' for w in words[start:start+8]) + ',')
HEADER = '\n'.join([*lines, '};', ''])

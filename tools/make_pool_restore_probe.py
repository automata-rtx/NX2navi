"""Emit the fragment shader for the exact bindless texture restoration probe."""
from make_dlaa_mask import Asm, string


def build_module(compute=False, finite=False):
    assert not compute or not finite
    a = Asm()
    main = a.id()
    void, fn, boolean, u32, f32, vec2, vec4 = (a.id() for _ in range(7))
    image, sampler, sampled = (a.id() for _ in range(3))
    image_count, sampler_count, left, right, sample_slot, half = (a.id() for _ in range(6))
    image_array, sampler_array, image_array_ptr, sampler_array_ptr, image_ptr, sampler_ptr = (a.id() for _ in range(6))
    input_uv, output_color, uv, color, textures, samplers = (a.id() for _ in range(6))
    start, coord, x, is_left, index, tex_address, selected_image = (a.id() for _ in range(7))
    samp_address, selected_sampler, combined, texel = (a.id() for _ in range(4))
    if finite:
        block, block_ptr, word_ptr, indices, zero_index, one_index = (a.id() for _ in range(6))
        addresses = [a.id() for _ in range(2)]
        second_address, second_image, second_combined, second_texel = (a.id() for _ in range(4))
        left_value, right_value, bool4, condition, mixed = (a.id() for _ in range(5))
    if compute:
        uvec3, array, output_struct, output_ptr, output_element = (a.id() for _ in range(5))
        zero, zero_index, sample_coord, target = (a.id() for _ in range(4))

    for capability in (1, 5301, 5307):  # Shader, ShaderNonUniform, SampledImageArrayNonUniformIndexing
        a.op('cap', 17, capability)
    a.op('cap', 10, *string('SPV_EXT_descriptor_indexing'))
    a.op('mem', 14, 0, 1)  # Logical GLSL450
    if compute:
        a.op('entry', 15, 5, main, *string('main'), uv)  # GLCompute
        a.op('exec', 16, main, 17, 16, 1, 1)  # LocalSize
        a.op('dec', 71, uv, 11, 28)  # GlobalInvocationId
        a.op('dec', 71, array, 6, 16)  # ArrayStride for vec4
        a.op('dec', 71, output_struct, 3)  # BufferBlock
        a.op('dec', 72, output_struct, 0, 35, 0)  # Offset of its only member
        a.op('dec', 71, color, 34, 0)
        a.op('dec', 71, color, 33, 3)
    else:
        a.op('entry', 15, 4, main, *string('main'), uv, color)
        a.op('exec', 16, main, 7)  # OriginUpperLeft
        for variable in (uv, color):
            a.op('dec', 71, variable, 30, 0)  # Location 0
    for binding, variable in enumerate((textures, samplers), 1):
        a.op('dec', 71, variable, 34, 0)  # DescriptorSet, rewritten by spirv_pool_rebind
        a.op('dec', 71, variable, 33, binding)  # Binding
    if finite:
        a.op('dec', 71, block, 2)  # Uniform Block
        a.op('dec', 72, block, 0, 35, 0)
        a.op('dec', 72, block, 1, 35, 4)
        a.op('dec', 71, indices, 34, 0)
        a.op('dec', 71, indices, 33, 3)
    varying_ids = (tex_address, selected_image, combined, second_address, second_image, second_combined) if finite else (
        index, tex_address, selected_image, combined)
    for varied in varying_ids:
        a.op('dec', 71, varied, 5300)  # NonUniform

    a.op('types', 19, void)
    a.op('types', 33, fn, void)
    a.op('types', 20, boolean)
    a.op('types', 21, u32, 32, 0)
    if finite:
        a.op('types', 23, bool4, boolean, 4)
        a.op('types', 43, u32, zero_index, 0)
        a.op('types', 43, u32, one_index, 1)
        a.op('types', 30, block, u32, u32)
        a.op('types', 32, block_ptr, 2, block)
        a.op('types', 32, word_ptr, 2, u32)
    a.op('types', 22, f32, 32)
    a.op('types', 23, vec2, f32, 2)
    a.op('types', 23, vec4, f32, 4)
    a.op('types', 25, image, f32, 1, 0, 0, 0, 1, 0)  # 2D sampled image
    a.op('types', 26, sampler)
    a.op('types', 27, sampled, image)
    # Mirror the translator's pool lengths and the registration slots used by the native probe.
    for ident, value in ((image_count, 32768), (sampler_count, 4096), (left, 17), (right, 18), (sample_slot, 3)):
        a.op('types', 43, u32, ident, value)
    if compute:
        a.op('types', 43, u32, half, 16)
        a.op('types', 43, f32, zero, 0)
        a.op('types', 43, u32, zero_index, 0)
        a.op('types', 44, vec2, sample_coord, zero, zero)  # Constant texture coordinate
    else:
        a.op('types', 43, f32, half, 0x3f000000)
    a.op('types', 28, image_array, image, image_count)
    a.op('types', 28, sampler_array, sampler, sampler_count)
    common_pointers = ((image_array_ptr, image_array, 0), (sampler_array_ptr, sampler_array, 0),
                       (image_ptr, image, 0), (sampler_ptr, sampler, 0))
    for ident, item, storage in common_pointers:
        a.op('types', 32, ident, storage, item)
    if compute:
        a.op('types', 23, uvec3, u32, 3)
        a.op('types', 32, input_uv, 1, uvec3)
        a.op('types', 29, array, vec4)  # RuntimeArray
        a.op('types', 30, output_struct, array)
        a.op('types', 32, output_ptr, 2, output_struct)
        a.op('types', 32, output_element, 2, vec4)
    else:
        a.op('types', 32, input_uv, 1, vec2)
        a.op('types', 32, output_color, 3, vec4)
    for typ, ident, storage in ((image_array_ptr, textures, 0), (sampler_array_ptr, samplers, 0),
                                (input_uv, uv, 1), (output_ptr, color, 2) if compute else (output_color, color, 3)):
        a.op('types', 59, typ, ident, storage)

    if finite:
        a.op('types', 59, block_ptr, indices, 2)
    a.op('func', 54, void, main, 0, fn)
    a.op('func', 248, start)
    if compute:
        a.op('func', 61, uvec3, coord, uv)
        a.op('func', 81, u32, x, coord, 0)  # OpCompositeExtract
        a.op('func', 176, boolean, is_left, x, half)  # OpULessThan
    else:
        a.op('func', 61, vec2, coord, uv)
        a.op('func', 81, f32, x, coord, 0)  # OpCompositeExtract
        a.op('func', 184, boolean, is_left, x, half)  # OpFOrdLessThan
    if finite:
        for address, member, value in zip(addresses, (zero_index, one_index), (left_value, right_value)):
            a.op('func', 65, word_ptr, address, indices, member)
            a.op('func', 61, u32, value, address)
        a.op('func', 65, image_ptr, tex_address, textures, left_value)
    else:
        a.op('func', 169, u32, index, is_left, left, right)  # OpSelect
        a.op('func', 65, image_ptr, tex_address, textures, index)
    a.op('func', 61, image, selected_image, tex_address)
    a.op('func', 65, sampler_ptr, samp_address, samplers, sample_slot)
    a.op('func', 61, sampler, selected_sampler, samp_address)
    a.op('func', 86, sampled, combined, selected_image, selected_sampler)
    if compute:
        a.op('func', 88, vec4, texel, combined, sample_coord, 2, zero)  # OpImageSampleExplicitLod
        a.op('func', 65, output_element, target, color, zero_index, x)
        a.op('func', 62, target, texel)
    else:
        a.op('func', 87, vec4, texel, combined, coord)  # OpImageSampleImplicitLod
        if finite:
            a.op('func', 65, image_ptr, second_address, textures, right_value)
            a.op('func', 61, image, second_image, second_address)
            a.op('func', 86, sampled, second_combined, second_image, selected_sampler)
            a.op('func', 87, vec4, second_texel, second_combined, coord)
            # Both descriptors are loaded by every shader invocation; only the final color varies by pixel.
            a.op('func', 80, bool4, condition, is_left, is_left, is_left, is_left)
            a.op('func', 169, vec4, mixed, condition, texel, second_texel)
            a.op('func', 62, color, mixed)
        else:
            a.op('func', 62, color, texel)
    a.op('func', 253)
    a.op('func', 56)

    return a.words()


def header(words, name, script):
    lines = [f'// Generated by tools/{script}. Do not edit.', '#pragma once', '#include <stdint.h>',
             f'static const uint32_t {name}_spv[] = {{']
    for index in range(0, len(words), 8):
        lines.append('    ' + ', '.join(f'0x{word:08x}' for word in words[index:index + 8]) + ',')
    return '\n'.join([*lines, '};', ''])


HEADER = header(build_module(), 'pool_restore_probe', 'make_pool_restore_probe.py')

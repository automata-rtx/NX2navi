"""Read pass assignments from v9.1.1 BFSHA archives and encode candidate NVN attachment layouts."""

import struct
from collections import defaultdict

# NVN attachment format IDs (runtime/nvn.h). The archive names the operation. Its exact image
# formats are chosen by the renderer. Keep the alternatives restricted to that operation.
R8, R16F, R32F, RG8, RG16F, RG32F = 1, 5, 10, 13, 17, 22
RGBA8, RGBA16F, RGBA32F, D16, D32 = 37, 41, 46, 50, 52
RGBA8_SRGB, RGB10A2, HDR = 56, 61, 63


def target(*colors, depth=0, points=False):
    return struct.pack('<B8BBB1x', 1, *(colors + (0,) * (8 - len(colors))), depth, points)


_DEPTH = (target(depth=D16), target(depth=D32))
_UI = (target(HDR), target(RGBA8), target(R8))
_FILTER = (target(HDR), target(RGBA8))
PASS_TARGETS = {
    'gsys_assign_depthshadow': _DEPTH, 'gsys_assign_zonly': _DEPTH,
    'ui': _UI,
    'image_filter_gaussian': _FILTER, 'image_filter_blur': _FILTER,
    'reduce_2': _FILTER, 'reduce_4': _FILTER, 'draw_texture': _FILTER,
    'texture': (*_FILTER, target(RG8), target(RGBA8_SRGB)),
    'bloom_gaussian': (target(HDR),), 'bloom_compose': (target(HDR),), 'bloom_mask': (target(HDR, HDR),),
    'render_buffer_color': (target(HDR),), 'color_correction': (target(HDR),),
    'cubemap_gaussian': (target(HDR),), 'cubemap_mipmap': (target(*(HDR,) * 6),),
    'cubemap_filter_ggx': (target(*(HDR,) * 6),), 'color_correction_map': (target(*(HDR,) * 8),),
    'dof_mipmap': (target(HDR), target(R8), target(HDR, R8)), 'dof_mipmap_detect': (target(HDR, R8),),
    'dof_near_mask': (target(R8),), 'ssao_bilateral': (target(R8), target(RG16F)),
    'scalable_ao_buffer': (target(RG16F, depth=D32),),
    'local_depth_shadow': (target(RGBA8, depth=D32),), 'shadow_pre_pass_merge': (target(RGBA8),),
    'depth_buffer_ex_to_linear_full': (target(R32F), target(R32F, depth=D32)),
    'depth_buffer_ex_to_linear_half_quat': (target(R32F), target(R32F, depth=D32), target(R32F, 0, R16F, depth=D32)),
    'depth_buffer_ex_draw_occlusion': (target(depth=D32),),
    'water_volume': (target(R32F, depth=D32), target(R16F, R32F, depth=D32),
                     target(RG16F, depth=D32), target(RGBA32F, depth=D32)),
    'texture_pick': (target(points=True),), 'occlusion_renderer': (target(RGBA32F, RGBA32F, points=True),),
    'occlusion_query': (target(depth=D32), target(HDR)), 'gen_normalmap': (target(RG8),),
    'noise_caustics': (target(RGB10A2),), 'noise_cellular': (target(R8),), 'noise_gradient': (target(R8),),
    'sky_bake_inscatter': (target(RGBA16F),), 'wet_occlusion_mask': (target(RGBA8),),
    'inner_mask_probe': (target(RG32F, depth=D32),), 'texture_texcoord': (target(RGB10A2),),
}


def bfsha_passes(data):
    """Map absolute BNSH variation offsets to pass names. Ordinary shader discovery can still
    read the BNSH if the surrounding model uses an unknown layout or has invalid references."""
    if len(data) < 128 or data[:4] != b'FSHA' or data[8:14] != bytes.fromhex('01010900fffe'):
        return {}

    def u16(at):
        return struct.unpack_from('<H', data, at)[0]

    def u32(at):
        return struct.unpack_from('<I', data, at)[0]

    def ptr(at):
        value = struct.unpack_from('<Q', data, at)[0]
        if value >= len(data):
            raise ValueError('BFSHA offset outside file')
        return value

    def name(at):
        size = u16(at)
        if size > 255 or at + 2 + size >= len(data) or data[at + 2 + size] != 0:
            raise ValueError('invalid BFSHA name')
        return data[at + 2:at + 2 + size].decode('ascii')

    result = defaultdict(set)
    try:
        models = ptr(0x48)
        count = u16(0x78)
        if models + count * 0x100 > len(data):
            return {}
        for model in range(models, models + count * 0x100, 0x100):
            ui = name(ptr(model)) == 'ui'
            options = ptr(model + 0x18)
            choices = []
            words = sum(data[model + 0xea:model + 0xec])
            for i in range(u16(model + 0xe6)):
                option = options + i * 40
                if name(ptr(option)) != 'gsys_assign_type':
                    continue
                dictionary = ptr(option + 8)
                n = u32(dictionary + 4)
                mask, word, shift = struct.unpack_from('<IBB', data, option + 32)
                if word >= words or shift >= 32 or not 0 < n <= 256:
                    break
                choices = [name(ptr(dictionary + 32 + j * 16)) for j in range(n)]
                break
            if not choices and not ui:
                continue
            table, keys, bnsh = (ptr(model + field) for field in (0x88, 0x90, 0xa8))
            programs = u16(model + 0xe8)
            if not words or table + 64 * programs > len(data) or keys + 4 * words * programs > len(data):
                continue
            if data[bnsh:bnsh + 4] != b'BNSH':
                continue
            variations, offset = struct.unpack_from('<IQ', data, bnsh + 124)
            first = bnsh + offset
            if offset + 64 * variations > u32(bnsh + 28) or first + 64 * variations > len(data):
                continue
            for i in range(programs):
                variation = ptr(table + i * 64 + 32)
                if not first <= variation < first + 64 * variations or (variation - first) % 64:
                    continue
                if ui:
                    result[variation].add('ui')
                if choices:
                    value = (u32(keys + 4 * (i * words + word)) & mask) >> shift
                    if value < len(choices) and choices[value] in PASS_TARGETS:
                        result[variation].add(choices[value])
    except (struct.error, ValueError):
        return {}
    return result

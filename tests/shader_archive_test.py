"""Check variation boundaries in a synthetic v11 shader program array."""
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from shader_census import sharcb_graphics_programs
from shader_passes import bfsha_passes


class GraphicsProgramsTest(unittest.TestCase):
    def fixture(self):
        data = bytearray(32)
        struct.pack_into('<I', data, 4, 11)
        binaries = [('ps', 1), ('ts', 4), ('ms', 5)] * 2 + [('ps', 1), ('ms', 5), ('vs', 0), ('ps', 1), ('vs', 0)]
        data.extend(struct.pack('<2I', 8 + 32 * len(binaries), len(binaries)))
        records = {}
        for n, (name, stage) in enumerate(binaries):
            records[len(data) + 16] = (name, str(n))
            data.extend(struct.pack('<4I16x', 32, stage, 0, 16))
        data.extend(struct.pack('<2I', 72, 4))
        data.extend(struct.pack('<4I', 16, 0, 0x32, 0))
        data.extend(struct.pack('<4I', 16, 0, 0x22, 6))
        data.extend(struct.pack('<4I', 16, 0, 3, 8))
        data.extend(struct.pack('<4I', 16, 0, 1, 10))
        return data, records

    def test_stage_order_and_missing_variation(self):
        data, records = self.fixture()
        first = min(records)
        self.assertEqual(list(sharcb_graphics_programs(data, first, len(data), records)),
                         [('2', '0', '1'), ('5', '3', '4'), ('7', '6', ''), ('8', '9', ''), ('10', '', '')])
        del records[first + 3 * 32]
        self.assertEqual(list(sharcb_graphics_programs(data, first, len(data), records)),
                         [('2', '0', '1'), ('7', '6', ''), ('8', '9', ''), ('10', '', '')])

    def test_named_integer_program_keeps_its_own_pair(self):
        data, records = self.fixture()
        first = min(records)
        table = len(data) - 72
        entry = table + 8 + 2 * 16
        name = b'shading_rate_image\0'
        struct.pack_into('<I', data, table, 72 + len(name))
        data[entry:entry + 16] = struct.pack('<4I', 16 + len(name), len(name), 3, 8) + name
        integer_pairs = set()
        groups = list(sharcb_graphics_programs(data, first, len(data), records, integer_pairs))
        self.assertEqual(integer_pairs, {('8', '9', '')})
        self.assertEqual(len(groups), 5)
        del records[first + 9 * 32]
        integer_pairs.clear()
        list(sharcb_graphics_programs(data, first, len(data), records, integer_pairs))
        self.assertFalse(integer_pairs)

    def test_named_pass_is_attached_to_its_variation(self):
        data, records = self.fixture()
        first = min(records)
        table = len(data) - 72
        entry = table + 8 + 2 * 16
        name = b'texture_pick'
        struct.pack_into('<I', data, table, 72 + len(name))
        data[entry:entry + 16] = struct.pack('<4I', 16 + len(name), len(name), 3, 8) + name
        roles = {}
        list(sharcb_graphics_programs(data, first, len(data), records, pass_names=roles))
        self.assertEqual(roles, {('8', '9', ''): {'texture_pick'}})

    def test_model_pass_uses_program_reference_and_packed_choice(self):
        data = bytearray(0x900)
        data[:4] = b'FSHA'
        data[8:14] = bytes.fromhex('01010900fffe')

        def text(offset, value):
            encoded = value.encode('ascii')
            struct.pack_into('<H', data, offset, len(encoded))
            data[offset + 2:offset + 2 + len(encoded)] = encoded

        model, option, dictionary, programs, keys, bnsh = 0x80, 0x200, 0x240, 0x380, 0x450, 0x600
        struct.pack_into('<Q', data, 0x48, model)
        struct.pack_into('<H', data, 0x78, 1)
        for field, value in ((0, 0x290), (0x18, option), (0x88, programs), (0x90, keys), (0xa8, bnsh)):
            struct.pack_into('<Q', data, model + field, value)
        struct.pack_into('<HHBB', data, model + 0xe6, 1, 2, 1, 1)
        text(0x290, 'model')
        text(0x2a0, 'gsys_assign_type')
        text(0x2c0, 'gsys_assign_material')
        text(0x2f0, 'gsys_assign_depthshadow')
        struct.pack_into('<QQ', data, option, 0x2a0, dictionary)
        struct.pack_into('<IBB', data, option + 32, 0x10, 1, 4)
        struct.pack_into('<I', data, dictionary + 4, 2)
        struct.pack_into('<Q', data, dictionary + 32, 0x2c0)
        struct.pack_into('<Q', data, dictionary + 48, 0x2f0)
        data[bnsh:bnsh + 4] = b'BNSH'
        struct.pack_into('<I', data, bnsh + 28, 0x300)
        struct.pack_into('<IQ', data, bnsh + 124, 2, 0x100)
        first = bnsh + 0x100
        struct.pack_into('<Q', data, programs + 32, first + 64)
        struct.pack_into('<Q', data, programs + 64 + 32, first)
        struct.pack_into('<4I', data, keys, 0, 0x10, 0, 0)
        self.assertEqual(bfsha_passes(data), {first + 64: {'gsys_assign_depthshadow'}})
        struct.pack_into('<Q', data, programs + 32, first + 1)
        self.assertEqual(bfsha_passes(data), {})
        self.assertEqual(bfsha_passes(data[:model + 64]), {})

    def test_invalid_index_and_truncated_array(self):
        data, records = self.fixture()
        first = min(records)
        self.assertEqual(list(sharcb_graphics_programs(data, first, len(data) - 1, records)), [])
        struct.pack_into('<I', data, len(data) - 4, 999)
        self.assertEqual(list(sharcb_graphics_programs(data, first, len(data), records)), [])


if __name__ == '__main__':
    unittest.main()

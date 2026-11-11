"""Ensure extending preparation never loses baseline records or rewarms baseline stage IDs."""
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from extend_shader_preparation import HEADER, merge_manifests, read_manifest
from shader_passes import PASS_TARGETS


def row(identity, stage=0):
    return bytes([stage, 2, 0, 0]) + bytes([identity]) * 32


class AdditionalPreparationTest(unittest.TestCase):
    def test_preserves_baseline_and_corrects_stage_without_rebuilding_it(self):
        baseline = HEADER + row(1) + row(2) + row(3, 5)
        prepared = HEADER + row(1, 7) + row(2) + row(4, 4)
        union, delta, kept, added = merge_manifests(baseline, prepared)
        self.assertEqual((kept, added), (3, 1))
        self.assertEqual(set(read_manifest(union)), {bytes([i]) * 32 for i in range(1, 5)})
        self.assertEqual(read_manifest(union)[bytes([1]) * 32], row(1, 7))
        self.assertEqual(delta, HEADER + row(4, 4))

    def test_identical_inventory_has_no_stage_work(self):
        inventory = HEADER + row(1)
        self.assertEqual(merge_manifests(inventory, inventory), (inventory, HEADER, 1, 0))

    def test_rejects_torn_invalid_and_conflicting_records(self):
        for data in (HEADER + row(1)[:-1], b'bad', HEADER + bytes(36), HEADER + row(1) + row(1, 4)):
            with self.subTest(data=data[:20]), self.assertRaises(ValueError):
                read_manifest(data)

    def test_pass_layouts_use_current_renderer_format_ids(self):
        # IDs are ABI values. Check the restored recipes against the current renderer header.
        import re
        header = (Path(__file__).resolve().parents[1] / 'runtime' / 'nvn.h').read_text()
        formats = {int(value, 0) for value in re.findall(r'NVN_FORMAT_\w+\s*=\s*(0x[0-9a-fA-F]+|\d+)', header)}
        self.assertTrue(formats)
        for recipes in PASS_TARGETS.values():
            for recipe in recipes:
                self.assertEqual(len(recipe), 12)
                self.assertEqual(recipe[0], 1)
                self.assertTrue(set(recipe[1:10]) <= formats | {0})


if __name__ == '__main__':
    unittest.main()

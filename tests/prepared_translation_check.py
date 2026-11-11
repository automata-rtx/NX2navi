"""Compare offline translation reuse to independent translator jobs on a synthetic six-shader Vulkan seed.

Run with Python 3.14: python tests/prepared_translation_check.py <built game exe> <translator exe>
Both runs and the packed-input recovery use private caches under build/.
"""
import argparse
import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import pack_shader_cache as packer


def seed(directory):
    programs = bytearray(struct.pack('<8sII', b'NX2NSPRG', 1, 0))
    answers = bytearray(struct.pack('<8sII', b'NX2NSANS', 1, 216))
    exit_op = struct.pack('<4I', 0x794d, 0, 0x03800000, 0x000fea00)
    for tag, group, code, c1 in [(0, 1, exit_op, b''), (1, 1, exit_op, b''), (2, 1, exit_op, b''),
                                 (0, 2, exit_op, b''), (0, 1, 2 * exit_op, b''), (0, 1, exit_op, bytes(16))]:
        control = bytearray(0x880)
        struct.pack_into('<I', control, 0, 0x98761234)
        control[0x50] = tag
        struct.pack_into('<I', control, 0x38, len(code))
        struct.pack_into('<II', control, 0x40, len(code), len(code) + len(c1))
        struct.pack_into('<3I', control, 0x58, group, 1, 1)
        full = code + c1
        key = hashlib.sha256(struct.pack('<I', 5) + control + struct.pack('<Q', len(full)) + full
                             + struct.pack('<I', len(c1)) + c1).digest()
        programs += struct.pack('<III', 5, len(full), len(c1)) + key + control + full + c1
        answers += struct.pack('<I', 5) + key + bytes(32 + 147 + 1)
    (directory / 'seed-programs.bin').write_bytes(programs)
    (directory / 'seed-answers.bin').write_bytes(answers)


def snapshot(cache):
    return {key: path.read_bytes() for key, path in packer.loose_files(cache).items()}


def run(exe, translator, directory, mode, keep=False, **override):
    env = dict(os.environ, NX2NAVI_XLATE=str(translator), NX2NAVI_XLATE_KEEP='2' if keep else '0',
               NX2NAVI_XLATE_ID='a' * 64, NX2NAVI_SHADER_CARRY='1',
               NX2NAVI_PIPELINE_CACHE=str(directory / 'pipeline-cache.bin'),
               NX2NAVI_SHADER_CACHE=str(directory / 'shader-cache'))
    env.update(override)
    result = subprocess.run([exe, mode], cwd=directory, env=env, capture_output=True, timeout=90)
    output = (result.stdout + result.stderr).decode(errors='replace')
    if result.returncode:
        raise RuntimeError(output)
    return output


def check_carried_receipt(exe, translator, directory, source):
    cache = directory / 'shader-cache'
    pack = cache / packer.PACK
    index = packer.read_pack(pack)
    active = run(exe, translator, directory, '--shader-cache-id')
    old = next(line.split()[2] for line in active.splitlines() if line.startswith('shader-cache-id '))
    def check(expected, **options):
        output = run(exe, translator, directory, '--shader-cache-id', NX2NAVI_XLATE_ID='b' * 64, **options)
        assert f'shader-translations-ready {expected}' in output.splitlines(), output
        assert (f'shader-carry-id {old}' in output.splitlines()) == (options.get('NX2NAVI_SHADER_CARRY') != '0'), output

    check(1)
    check(0, NX2NAVI_SHADER_CARRY='0')
    # Break a stored input, a lookup mask and a translation in turn while keeping each entry's size.
    for kind in (4, 3, 1):
        key = next(k for k in index if k[0] == kind)
        offset, size = index[key]
        with pack.open('r+b') as f:
            f.seek(offset)
            original = f.read(size)
            f.seek(offset)
            f.write(bytes(size))
        check(0)
        if kind == 4:
            # Even a warm seed run can repair the damaged original without translating the shader again.
            repair = run(exe, translator, directory, '--prepare-shaders')
            assert '0 translated, 0 failed' in repair, repair
            path = cache / 'in' / (key[1].hex() + '.in')
            assert path.read_bytes() == source[key]
            check(1)
            path.unlink()
        with pack.open('r+b') as f:
            f.seek(offset)
            f.write(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game', type=Path)
    parser.add_argument('translator', type=Path)
    parser.add_argument('--reference-exe', type=Path, help='compare with an earlier build using the same device options')
    args = parser.parse_args()
    exe, translator = args.game.resolve(), args.translator.resolve()
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='translation-check-', dir=build) as temp:
        reference, shared = [Path(temp) / s for s in ('reference', 'shared')]
        for folder in (reference, shared):
            folder.mkdir()
            seed(folder)
        baseline = run(args.reference_exe.resolve() if args.reference_exe else exe, translator, reference,
                       '--prepare-shaders', keep=True)
        result = run(exe, translator, shared, '--prepare-shaders')
        assert '6 translated, 0 failed' in baseline, baseline
        assert '4 translated, 0 failed' in result and '2 equivalent requests reused' in result, result
        manifest = (shared / 'shader-prepared.bin').read_bytes()
        assert len(manifest) == 16 + 6 * 36 and manifest == (reference / 'shader-prepared.bin').read_bytes()
        cache = shared / 'shader-cache'
        source = snapshot(reference / 'shader-cache')
        assert snapshot(cache) == source and len(source) == 24
        packer.pack(cache)
        warm = run(exe, translator, shared, '--prepare-shaders')
        assert '0 translated, 0 failed' in warm and '0 equivalent requests reused' in warm, warm
        assert not snapshot(cache)
        # The additive path resolves already packed inputs without retranslation, then schedules
        # only its explicit delta. The full prepared inventory must remain intact.
        inputs = run(exe, translator, shared, '--prepare-shader-inputs')
        assert '0 translated, 0 failed' in inputs, inputs
        assert 'variants built into the driver cache' not in inputs, inputs
        assert (shared / 'shader-prepared.bin').read_bytes() == manifest
        (shared / 'shader-additional.bin').write_bytes(manifest[:16] + manifest[16:52])
        additional = run(exe, translator, shared, '--warm-additional-pipelines')
        assert '1 variants built into the driver cache' in additional, additional
        assert '6 variants built into the driver cache' not in additional, additional
        assert (shared / 'shader-prepared.bin').read_bytes() == manifest
        (shared / 'shader-additional.bin').write_bytes(manifest[:16])
        empty = run(exe, translator, shared, '--warm-additional-pipelines')
        assert 'no shader variants or recorded graphics pipelines to warm' in empty, empty
        check_carried_receipt(exe, translator, shared, source)
        # Model a recorded play inventory. A new translator identity makes every old SPIR-V key miss, so
        # warming these six IDs must recover the original inputs directly from the pack.
        records = bytearray((shared / 'seed-answers.bin').read_bytes())
        for i in range(6):
            at = 16 + i * 216 + 36
            start = 16 + i * 36 + 4
            records[at:at + 32] = manifest[start:start + 32]
        (shared / 'shader-answers.bin').write_bytes(records)
        (shared / 'seed-programs.bin').unlink()
        (shared / 'seed-answers.bin').unlink()
        rebuilt = run(exe, translator, shared, '--warm-shaders', NX2NAVI_XLATE_ID='packed-input-recovery',
                      NX2NAVI_SHADER_CARRY='0')
        restored = snapshot(cache)
        def shaders(entries):
            return sorted(data for (kind, _), data in entries.items() if kind == 1)
        assert len(shaders(restored)) == 6 and shaders(restored) == shaders(source), rebuilt
        assert manifest == (shared / 'shader-prepared.bin').read_bytes()
    print('Six synthetic shaders matched the independent jobs, warm cache and packed-input rebuild.')


if __name__ == '__main__':
    main()

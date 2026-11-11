"""Identify Link body geometry at each model LOD from the local extracted RomFS."""
import argparse
import json
import struct
from pathlib import Path

import zstd_compat as zstd
from shader_census import sarc


MODELS = (('Player_Child', 1), ('Player_Child_Op', 1), ('Player_Adult', 2))


def shapes(blob):
    def unpack(fmt, pos):
        return struct.unpack_from('<' + fmt, blob, pos)[0]

    def tagged(pos, tag):
        if blob[pos:pos + 4] != tag:
            raise ValueError(f'Expected {tag!r} at {pos:#x}')

    tagged(0, b'FRES')
    model = unpack('Q', 0x28)
    tagged(model, b'FMDL')
    first = unpack('Q', model + 0x28)
    arena = unpack('Q', unpack('Q', 0xb0) + 8)
    for i in range(unpack('H', model + 106)):
        shape = first + 96 * i
        tagged(shape, b'FSHP')
        tagged(unpack('Q', shape + 16), b'FVTX')
        lods = unpack('Q', shape + 24)
        for j in range(blob[shape + 91]):
            lod = lods + 56 * j
            offset, _, fmt, count, base = struct.unpack_from('<5I', blob, lod + 32)
            width = (1, 2, 4)[fmt]
            end = arena + offset + width * count
            if not count or end > len(blob):
                raise ValueError('Invalid model index range')
            data = memoryview(blob)[arena + offset:end]
            h = 0xcbf29ce484222325
            for byte in data:
                h = ((h ^ byte) * 0x100000001b3) & 0xffffffffffffffff
            yield count, width, base, h


def write_catalog(folder, output):
    index = {e['path']: e for e in json.loads((folder / 'index.json').read_bytes())['files']}
    with (folder / 'romfs.bin').open('rb') as stream:
        def read(name):
            entry = index[name]
            stream.seek(entry['offset'])
            return stream.read(entry['size'])

        pack = sarc(zstd.decompress(read('Pack/ZsDic.pack.zs')))
        dictionaries = {d.dict_id: d for d in map(zstd.ZstdDict, pack.values())}
        records = {}
        for name, actor in MODELS:
            data = read(f'Model/{name}.{name}.bfres.zs')
            dictionary = dictionaries.get(zstd.get_frame_info(data).dictionary_id)
            for key in set(shapes(zstd.decompress(data, zstd_dict=dictionary))):
                records[key] = actor if records.get(key, actor) == actor else 0
    lines = [f'{count} {width} {base} {h:016x} {actor}\n'
             for (count, width, base, h), actor in sorted(records.items()) if actor and width in (2, 4)]
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp')
    temp.write_text('NX2NAVI Link pose catalog 1\n' + ''.join(lines), encoding='ascii')
    temp.replace(output)
    print(f'{len(lines)} unique index ranges in {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--romfs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    write_catalog(args.romfs, args.output)

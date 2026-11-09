"""Fold the loose files of a shader cache into <cache>/shader-cache.pack and delete them.

The translator writes four files per shader into the cache (runtime/gfx_shader.c): <key>.spv, <key>.res.json,
<name>.use and in/<input id>.in, about 117,000 files after an install, which are slow to copy, back up, scan and sync.
nx2navi.exe reads the pack first and the loose files after it, and keeps writing new translations as loose files, so
this tool can run again at any time the game is closed: it merges the existing pack with the loose files (a loose file
wins over a packed one of the same name) and writes the pack anew. The installer runs it after preparing the shaders.

Layout, little-endian (gfx_shader.c, section "cache pack"): b'TKSPACK1', u32 count, u32 flags, then count entries of 48
bytes sorted by (kind, name): name (the 32 bytes the file name's 64 hex digits spell), u32 kind (1 .spv, 2 .res.json,
3 .use, 4 in/*.in), u32 size, u64 offset; then the data. Flag bit 0 marks packs written with shared payloads.
Each key keeps its own index entry. Byte-identical payloads of the same kind can share an offset.

    python tools\\pack_shader_cache.py <install folder>\\shader-cache [--keep]

--keep leaves the loose files in place. Close the game first: a running nx2navi.exe has read the old pack's index.
"""
import argparse
import hashlib
import os
import re
import struct
import sys
from contextlib import nullcontext
from pathlib import Path

MAGIC = b'TKSPACK1'
PACK = 'shader-cache.pack'
SHARED_PAYLOADS = 1
KINDS = {'.spv': 1, '.res.json': 2, '.use': 3, '.in': 4}
NAME = re.compile(r'([0-9a-f]{64})(\.spv|\.res\.json|\.use|\.in)')
USE_SIZE = 20   # gfx_shader.c use_store: magic, version, three masks


def loose_files(cache):
    """{(kind, name): path} of the cache's loose files; *.in only in in/."""
    found = {}
    for folder, kinds in ((cache, (1, 2, 3)), (cache / 'in', (4,))):
        if not folder.is_dir():
            continue
        with os.scandir(folder) as it:
            for e in it:
                m = NAME.fullmatch(e.name)
                if not m or not e.is_file():
                    continue
                kind = KINDS[m.group(2)]
                if kind in kinds:
                    found[(kind, bytes.fromhex(m.group(1)))] = Path(e.path)
    return found


def read_pack(path):
    """Read {(kind, name): (offset, size)}. Refuse a damaged index before replacing an existing archive."""
    try:
        f = open(path, 'rb')
    except FileNotFoundError:
        return {}
    with f:
        size = os.fstat(f.fileno()).st_size
        head = f.read(16)
        if len(head) != 16 or head[:8] != MAGIC:
            print(f'{path}: not a cache pack, rebuilt from the loose files only', flush=True)
            return {}
        count = struct.unpack_from('<I', head, 8)[0]
        if 16 + 48 * count > size:
            raise ValueError(f'{path}: truncated shader cache index')
        raw = f.read(48 * count)
        if len(raw) != 48 * count:
            raise ValueError(f'{path}: truncated shader cache index')
        out = {}
        for i in range(count):
            name, kind, length, offset = struct.unpack_from('<32sIIQ', raw, 48 * i)
            if offset < 16 + 48 * count or offset + length > size or (kind, name) in out:
                raise ValueError(f'{path}: invalid shader cache entry')
            out[(kind, name)] = (offset, length)
        return out


def valid(kind, data):
    return len(data) == USE_SIZE if kind == 3 else len(data) > 0


def shared_pack(path):
    try:
        with path.open('rb') as f:
            header = f.read(16)
    except FileNotFoundError:
        return False
    return len(header) == 16 and header[:8] == MAGIC and bool(struct.unpack_from('<I', header, 12)[0] & SHARED_PAYLOADS)


def store_payload(f, locations, kind, data, end):
    # Keep small masks in the key itself. For larger entries the digest locates candidates, and an exact read
    # from the output distinguishes even colliding digests without keeping the shader archive in memory.
    small = len(data) <= 64
    key = (kind, len(data), data if small else hashlib.sha256(data).digest())
    candidates = locations.setdefault(key, [])
    for offset in candidates:
        if small:
            return offset, end
        f.seek(offset)
        if f.read(len(data)) == data:
            return offset, end
    f.seek(end)
    f.write(data)
    candidates.append(end)
    return end, end + len(data)


def pack(cache, keep=False):
    cache = Path(cache)
    # Serialize the snapshot, replacement and removal of loose files across installers using the same cache.
    # Closing the file releases the OS lock, including when the process is interrupted.
    with (cache / '.pack.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _pack(cache, keep)


def _pack(cache, keep):
    out = cache / PACK
    packed = read_pack(out)
    loose = loose_files(cache)
    if not loose and (not packed or shared_pack(out)):
        return len(packed), 0, 0, out.stat().st_size if out.is_file() else 0
    keys = sorted(set(packed) | set(loose))
    tmp = out.with_name(PACK + '.tmp')
    index, written, skipped, offset = [], [], 0, 16 + 48 * len(keys)
    locations = {}
    try:
        with (out.open('rb') if packed else nullcontext()) as old, tmp.open('w+b') as f:
            f.write(bytes(offset))   # header and index, written once the offsets are known
            for key in keys:
                data = None
                if key in loose:   # a valid loose file, else the packed copy, else nothing
                    try:
                        data = loose[key].read_bytes()
                    except OSError:
                        pass
                    if data is not None and valid(key[0], data):
                        written.append(loose[key])
                    else:
                        data = None
                if data is None and key in packed:
                    old.seek(packed[key][0])
                    data = old.read(packed[key][1])
                    if len(data) != packed[key][1]:
                        raise OSError(f'{out}: incomplete read of a packed shader')
                if data is None or not valid(key[0], data):
                    skipped += 1
                    continue
                start, offset = store_payload(f, locations, key[0], data, offset)
                index.append(struct.pack('<32sIIQ', key[1], key[0], len(data), start))
            f.seek(0)
            f.write(MAGIC + struct.pack('<II', len(index), SHARED_PAYLOADS) + b''.join(index))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    if not keep:
        for p in written:
            p.unlink(missing_ok=True)
        try:
            (cache / 'in').rmdir()   # only when empty
        except OSError:
            pass
    return len(index), len(written), skipped, offset


def main():
    ap = argparse.ArgumentParser(description='Fold a shader cache into shader-cache.pack.')
    ap.add_argument('cache', type=Path, help='the shader-cache folder next to nx2navi.exe')
    ap.add_argument('--keep', action='store_true', help='keep the loose files')
    a = ap.parse_args()
    if not a.cache.is_dir():
        sys.exit(f'{a.cache} is not a folder')
    n, new, skipped, size = pack(a.cache, a.keep)
    print(f'{a.cache / PACK}: {n} files, {size / 2**20:.0f} MiB ({new} loose files folded in'
          f'{", " + str(skipped) + " unreadable or empty skipped" if skipped else ""})', flush=True)


if __name__ == '__main__':
    main()

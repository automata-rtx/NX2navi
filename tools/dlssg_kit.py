"""Assemble and verify the Windows Streamline libraries used for frame generation."""
import hashlib
import re
import shutil
import struct
from pathlib import Path

BRIDGE = 'nx2navi_dlssg.dll'
LIBRARIES = (BRIDGE, 'sl.interposer.dll', 'sl.common.dll', 'sl.dlss_g.dll', 'sl.pcl.dll', 'sl.reflex.dll',
             'nvngx_dlssg.dll')
LICENSES = ('streamline-LICENSE.txt', 'nvidia-rtx-sdks-LICENSE.txt')
NOTICE = 'dlssg-NOTICE.txt'


def digest(path):
    with path.open('rb') as file:
        h = hashlib.sha256()
        for block in iter(lambda: file.read(1 << 20), b''):
            h.update(block)
        return h.hexdigest()


def bridge_exports(path):
    """Read the named exports of an x64 PE32+ bridge without loading its code."""
    d = path.read_bytes()
    try:
        pe = struct.unpack_from('<I', d, 60)[0]
        machine, sections = struct.unpack_from('<HH', d, pe + 4)
        opt = pe + 24
        if d[pe:pe + 4] != b'PE\0\0' or machine != 0x8664 or struct.unpack_from('<H', d, opt)[0] != 0x20b:
            return set()
        first = opt + struct.unpack_from('<H', d, pe + 20)[0]

        def offset(rva):
            for i in range(sections):
                address, size, raw = struct.unpack_from('<III', d, first + 40 * i + 12)
                if address <= rva < address + size:
                    return raw + rva - address
            raise ValueError('outside PE sections')

        directory = offset(struct.unpack_from('<I', d, opt + 112)[0])
        count, _, names = struct.unpack_from('<III', d, directory + 24)
        table = offset(names)
        exports = set()
        for (rva,) in struct.iter_unpack('<I', memoryview(d)[table:table + count * 4]):
            start = offset(rva)
            exports.add(d[start:d.index(0, start)].decode('ascii'))
        return exports
    except (struct.error, ValueError):
        return set()


def hashes(notice):
    if not notice.is_file():
        return {}
    return dict(re.findall(r'^(\S+) sha256 ([0-9a-f]{64})$', notice.read_text(encoding='utf-8'), re.M))


def assemble(bridge, sdk, dest):
    if not bridge and not sdk:
        return
    if not bridge or not sdk:
        raise SystemExit('--dlssg and --streamline must be supplied together')
    sources = {n: bridge if n == BRIDGE else sdk / 'bin/x64' / n for n in LIBRARIES}
    sources.update(zip(LICENSES, (sdk / 'license.txt', sdk / 'bin/x64/nvngx_dlss.license.txt')))
    for src in sources.values():
        if not src.is_file():
            raise SystemExit(f'Frame generation payload is missing {src}')
    required = {'nx2sl_version', 'nx2sl_frame', 'nx2sl_begin_paced', 'nx2sl_reflex_mode', 'nx2sl_output_frames'}
    if not required.issubset(bridge_exports(bridge)):
        raise SystemExit(f'{bridge} lacks the current frame generation and pacing exports. Build it with tools/build_dlssg.py.')
    dest.mkdir(parents=True, exist_ok=True)
    lines = ['NX2navi frame generation bridge and unmodified NVIDIA Streamline and NGX runtime libraries.',
             'sl.*.dll use the Streamline MIT license (streamline-LICENSE.txt). nvngx_dlssg.dll uses the NVIDIA RTX',
             'SDKs License (nvidia-rtx-sdks-LICENSE.txt), which licenses its use on NVIDIA GPUs.', '']
    for name, src in sources.items():
        shutil.copy2(src, dest / name)
        lines.append(f'{name} sha256 {digest(dest / name)}')
    (dest / NOTICE).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def install(src, out, replace):
    """Validate the whole kit before updating a managed set. Publish the bridge after its dependencies."""
    if not src.exists():
        return
    expected = hashes(src / NOTICE)
    if any(name not in expected or not (src / name).is_file() or digest(src / name) != expected[name]
           for name in (*LIBRARIES, *LICENSES)):
        raise SystemExit(f'Frame generation kit in {src} is incomplete or corrupt')
    previous = hashes(out / NOTICE)
    changed = []
    for name in (*LIBRARIES, *LICENSES):
        dst = out / name
        if dst.is_file():
            have = digest(dst)
            if have == expected[name]:
                continue
            if name in LIBRARIES and have != previous.get(name):
                return f'Frame generation libraries kept: {name} was supplied by the player'
        changed.append(name)
    if changed:
        # A game started after interruption can only activate a complete set. A retry accepts both old and new hashes.
        (out / BRIDGE).unlink(missing_ok=True)
        for name in changed:
            if name != BRIDGE:
                replace(src / name, out / name)
        replace(src / BRIDGE, out / BRIDGE)
    if not (out / NOTICE).is_file() or (src / NOTICE).read_bytes() != (out / NOTICE).read_bytes():
        replace(src / NOTICE, out / NOTICE)

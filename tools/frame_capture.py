"""Read the application present intervals written by NX2NAVI_FRAME_CAPTURE."""
import argparse
import struct
import sys
from pathlib import Path
from typing import NamedTuple


class Frame(NamedTuple):
    frame: int
    interval_ns: int
    world_us: int


def read_frames(path):
    """Yield gameplay presents in file order. The game clock may repeat."""
    record = struct.Struct('<QQq')
    previous = None
    with Path(path).open('rb') as source:
        if source.read(8) != b'NX2FRM01':
            raise ValueError(f'{path}: unknown frame capture format')
        while block := source.read(record.size * 4096):
            if len(block) % record.size:
                raise ValueError(f'{path}: truncated frame record')
            for fields in record.iter_unpack(block):
                row = Frame(*fields)
                if not row.interval_ns or row.world_us < 0:
                    raise ValueError(f'{path}: invalid timing at present {row.frame}')
                if previous is not None and row.frame <= previous:
                    raise ValueError(f'{path}: present identifiers are out of order')
                previous = row.frame
                yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    args = parser.parse_args()
    try:
        print('frame\tinterval_ns\tworld_us')
        for frame, interval, world in read_frames(args.capture):
            print(f'{frame}\t{interval}\t{world}')
    except (OSError, ValueError) as error:
        sys.exit(str(error))


if __name__ == '__main__':
    main()

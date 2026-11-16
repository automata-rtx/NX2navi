"""Losslessly archive a closed benchmark log after frame measurement ends."""
import gzip
import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def archive_log(path):
    path = Path(path)
    archive = path.with_name(path.name + '.gz')
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        with path.open('rb') as source, staged.open('wb') as output:
            with gzip.GzipFile(filename='', fileobj=output, mode='wb', compresslevel=6, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
        with gzip.open(staged, 'rb') as restored, path.open('rb') as source:
            if hashlib.file_digest(restored, 'sha256').digest() != hashlib.file_digest(source, 'sha256').digest():
                raise OSError(f'Archive verification failed for {path}')
        os.replace(staged, archive)
        path.unlink()
    finally:
        staged.unlink(missing_ok=True)
    return archive


def read_log(path):
    """Read the live text file or its finished gzip archive as original bytes."""
    path = Path(path)
    if path.is_file():
        return path.read_bytes()
    with gzip.open(path.with_name(path.name + '.gz'), 'rb') as stream:
        return stream.read()

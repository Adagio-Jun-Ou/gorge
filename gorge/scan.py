"""Drive scanning: a single read-only streaming pass over a path tree."""

from __future__ import annotations

import os
import stat as stat_mod
from dataclasses import dataclass
from typing import Iterator, Optional

from .types import detect_type


@dataclass
class Bundle:
    path: str
    offset: int
    data: bytes
    type_id: int
    label: str
    file_size: int


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(b)


def iter_files(root, max_files: Optional[int] = None) -> Iterator[tuple]:
    """Yield (path, size) for every regular non-link file, depth-first, sorted.

    Skips symlinks/junctions, devices, empty files, and unreadable entries.
    Read-only: nothing on disk is ever modified.
    """
    root = os.path.realpath(os.path.abspath(root))
    if not os.path.exists(root):
        raise FileNotFoundError(f"no such path: {root}")
    if os.path.isfile(root):
        yield root, os.path.getsize(root)
        return
    count = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # do not descend through junctions/symlinked directories
        dirnames[:] = [
            d for d in dirnames
            if _same_path(os.path.join(dirpath, d), os.path.join(dirpath, d))
        ]
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            try:
                if not _same_path(full, full):
                    continue
                st = os.stat(full)
                if not stat_mod.S_ISREG(st.st_mode):
                    continue
            except OSError:
                continue
            if st.st_size == 0:
                continue
            yield full, st.st_size
            count += 1
            if max_files is not None and count >= max_files:
                return


def iter_bundles(root, chunk: int = 65536,
                 max_files: Optional[int] = None) -> Iterator[Bundle]:
    """Yield every bundle of bytes on the path, one streaming read per file."""
    chunk = max(256, int(chunk))
    for path, size in iter_files(root, max_files=max_files):
        label, tid = "", 0
        try:
            with open(path, "rb") as f:
                offset = 0
                first = True
                while True:
                    data = f.read(chunk)
                    if not data:
                        break
                    if first:
                        label, tid = detect_type(data[:512], path)
                        first = False
                    yield Bundle(path, offset, data, tid, label, size)
                    offset += len(data)
        except OSError:
            continue
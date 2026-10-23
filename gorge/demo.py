"""`gorge demo`: builds a synthetic mini-drive, eats it whole, probes it."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Dict

import numpy as np

from .phasing import Phaser, dim_for_capacity, parse_capacity
from .query import probe
from .report import print_ingest_summary, print_probe
from .scan import iter_bundles


def _secret_notes(n_lines: int = 80) -> bytes:
    lines = []
    for i in range(n_lines):
        lines.append(
            f"Mission log {i:03d}: payload capacitors charged to "
            f"{(i * 37) % 100}% of the phase field."
        )
        if i % 9 == 0:
            lines.append(
                "Reminder: the launch key PHASE-KEY-ZQX-7741 must not leave the vault."
            )
        lines.append("The window opens at 04:00 ZULU. Guidance locked.")
    return "\n".join(lines).encode()


def _phasemath_py(n_funcs: int = 120) -> bytes:
    out = ["import math", "", "# phasemath: closed-form weight phasing experiments", ""]
    for i in range(n_funcs):
        out += [
            f"def crunch_phasefield_{i}(x, depth={i % 7}):",
            f"    acc = x * {1 + i % 13}.0",
            "    for _ in range(depth):",
            f"        acc = math.sin(acc) + {i}.5",
            "    return acc",
            "",
        ]
    return "\n".join(out).encode()


def _customers_csv(n_rows: int = 180) -> bytes:
    rows = [b"id,name,balance,tier"]
    for i in range(n_rows):
        rows.append(f"{10000 + i},customer_{i:03d},{(i * 137) % 9000 + 100}.50,gold".encode())
    return b"\n".join(rows)


def _fake_png(rng: np.random.Generator, size: int = 16384) -> bytes:
    return (b"\x89PNG\r\n\x1a\n"
            + rng.integers(0, 256, size=size, dtype=np.uint8).tobytes())


def build_demo_drive(root: str) -> Dict[str, str]:
    """Create a small synthetic drive with known, findable content."""
    shutil.rmtree(root, ignore_errors=True)  # always rebuild fresh
    rng = np.random.default_rng(7)
    files = {}

    def put(rel: str, data: bytes) -> str:
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        files[rel] = p
        return p

    put("notes/secret_launch_codes.txt", _secret_notes())
    put("src/phasemath.py", _phasemath_py())
    # dir sorts after "src" so the canonical file registers before its copy
    put("zzz_backup/phasemath_copy.py", _phasemath_py())  # exact duplicate -> dedup
    put("data/customers.csv", _customers_csv())
    put("assets/logo.bin", _fake_png(rng))
    return files


def run(capacity: str = "4M", chunk: int = 2048) -> int:
    root = os.path.join(tempfile.gettempdir(), "gorge_demo_drive")
    files = build_demo_drive(root)
    d = dim_for_capacity(parse_capacity(capacity))

    print("=" * 64)
    print("GORGE DEMO -- a tiny synthetic drive gets eaten whole")
    print("=" * 64)
    print(f"drive   : {root}")
    print(f"model   : dim {d} (~{d * d} params), chunk {chunk} bytes")
    print()

    ph = Phaser(dim=d, chunk=chunk)
    t0 = time.time()
    for b in iter_bundles(root, chunk=chunk):
        ph.add_bundle(b.path, b.offset, b.data, b.type_id, b.file_size)
    pw = ph.finalize()

    out = os.path.join(tempfile.gettempdir(), "gorge_demo.gw")
    pw.save(out)
    print_ingest_summary(pw, elapsed=time.time() - t0, out_path=out)

    with open(files["assets/logo.bin"], "rb") as f:
        logo = f.read()
    probes = [
        ("a secret string", b"launch key PHASE-KEY-ZQX-7741 must not leave"),
        ("a code identifier", b"def crunch_phasefield_117"),
        ("400 raw bytes from the middle of the PNG", logo[9000:9400]),
        ("a csv field", b"customer_077"),
    ]
    for desc, sample in probes:
        res = probe(pw, sample, top=3)
        print_probe(pw, res, desc)

    print("NOTES:")
    print("- the drive was read ONCE, in a single streaming pass;")
    print("- after that, every probe above was answered from the phased weights;")
    print("- no file was opened to answer any probe.")
    print()
    print("try it on something real:")
    print("  gorge ingest <path> --capacity 25M -o mydrive.gw")
    print("  gorge probe mydrive.gw --sample somefile.bin")
    print("  gorge stats mydrive.gw")
    return 0
"""Human-readable ASCII reporting (Windows-console safe)."""

from __future__ import annotations

import os
from typing import Optional

from .types import MAJORS

RULE = "=" * 64
SUB = "-" * 64


def human_bytes(n) -> str:
    n = float(n)
    for u in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1000.0:
            return f"{n:.1f} {u}" if u != "B" else f"{int(n)} B"
        n /= 1000.0
    return f"{n:.1f} EB"


def human_count(n) -> str:
    n = float(n)
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


def bar(frac: float, width: int = 24) -> str:
    frac = min(1.0, max(0.0, float(frac)))
    filled = int(frac * width)
    return "#" * filled + "-" * (width - filled)


def _verdict(load: float) -> str:
    if load < 0.5:
        return "clean: attractors well separated"
    if load < 1.0:
        return "edge of clean baked recall"
    if load < 5.0:
        return "overloaded: prefer phase-cache readout, or raise --capacity"
    return "saturated: baked field is gestalt-only"


def print_ingest_summary(pw, elapsed=None, out_path=None) -> None:
    st = pw.stats
    d = pw.dim
    chunk = int(pw.config.get("chunk", 65536))

    print(RULE)
    print("GORGE PHASE SUMMARY")
    print(RULE)
    print(f"files eaten       : {pw.n_files}")
    print(f"bytes eaten       : {human_bytes(st.get('bytes_seen', 0))}")
    print(f"bundles read      : {st.get('bundles', 0)} (chunk = {human_bytes(chunk)})")
    print(f"engrams phased    : {pw.n_engrams} unique")
    print(f"dedupe saved      : {human_bytes(st.get('dup_bytes', 0))} "
          f"in {st.get('dup_bundles', 0)} duplicate bundles")
    print(SUB)
    print(f"model dim (d)     : {d}")
    if pw.W is not None:
        print(f"baked weights     : {human_count(pw.params)} params "
              f"(W {d}x{d} + W_v 15x{d})")
    else:
        print("baked weights     : none (--no-bake: fast cache-only ingest)")
    if pw.E is not None:
        print(f"phase cache       : {pw.n_engrams} x {d} floats "
              f"= {human_bytes(pw.E.nbytes)} (exact memory)")
    else:
        print("phase cache       : dropped (weights-only artifact)")
    clean = 0.138 * d
    load = pw.n_engrams / max(1e-9, clean)
    print(f"attractor capacity: {int(clean)} clean engrams "
          f"(~{human_bytes(clean * chunk)} at this chunk size)")
    print(f"load factor       : {load:.2f}x -> {_verdict(load)}")
    if elapsed:
        rate = st.get("bytes_seen", 0) / 1e6 / max(1e-9, elapsed)
        print(f"ingest time       : {elapsed:.1f}s ({rate:.1f} MB/s)")
    print(SUB)
    tb = st.get("type_bytes", {})
    total = max(1, sum(tb.values()))
    print("what got eaten (by type):")
    if not tb:
        print("  (nothing)")
    for tid in sorted(tb, key=lambda t: -tb[t]):
        try:
            name = MAJORS[int(tid)]
        except (ValueError, IndexError):
            name = "?"
        frac = tb[tid] / total
        print(f"  {name:<11} {human_bytes(tb[tid]):>10}  {frac * 100:5.1f}% "
              f"|{bar(frac)}|")
    if out_path:
        print(SUB)
        print(f"model saved       : {out_path} "
              f"({human_bytes(os.path.getsize(out_path)) if os.path.exists(out_path) else '?'})")
    print(RULE)


def print_probe(pw, res: dict, desc: str = "") -> None:
    print(RULE)
    print("GORGE PROBE -- answered from phased weights, no files opened")
    print(RULE)
    print(f"query   : {desc}")
    print(f"sample  : {res['sample_bytes']} bytes -> {pw.dim}-dim phase pattern")

    if res.get("matches"):
        print(SUB)
        print("associative matches (phase cache):")
        for i, m in enumerate(res["matches"], 1):
            print(f"  {i}. {m['conf'] * 100:5.1f}%  cos {m['sim']:.4f}  {m['path']}")
            print(f"      offset ~{human_bytes(m['offset'])}  type {m['type']}  "
                  f"entropy {m['entropy']:.2f} bits/byte")

    if res.get("attention"):
        a = res["attention"]
        print(SUB)
        print("attention readout (modern-hopfield, soft):")
        print(f"  file  : {a['file']}")
        print(f"  type  : {a['type']} ({a['conf'] * 100:.0f}%)  "
              f"offset ~{human_bytes(a['offset'])}")

    if res.get("baked"):
        b = res["baked"]
        print(SUB)
        print("baked weights readout (W, W_v only -- cache not consulted):")
        print(f"  file  : {b['file']}")
        print(f"  type  : {b['type']} ({b['conf'] * 100:.0f}%)  "
              f"offset ~{human_bytes(b['offset'])}")
        print(f"  field response mass: {b['mass']:.3f}")
    if res.get("resonance"):
        trace = " -> ".join(f"{v:.3f}" for v in res["resonance"])
        print(f"  hopfield resonance trace: {trace}")
    print(RULE)
"""Generate web/parity.js: expected outputs of the real Python gorge engine.

The browser UI recomputes these values with its JS port of the same algorithm
and compares them on page load, proving the animation runs the *real* math,
not a cartoon of it.  Run:  python tools/gen_parity.py
"""

import base64
import json
import random
from pathlib import Path

from gorge.phasing import Phaser, embed
from gorge.query import probe

DIM = 96
CHUNK = 4096
MARKER = b"PHASE-KEY-ZQX-7741 opens the vault of weights"

TEXT_A = (
    b"The phasefield capacitor hums quietly. " * 6
    + MARKER + b" "
    + b"padding " * 20
)
RAND_B = random.Random(42).randbytes(600)


def main() -> None:
    ph = Phaser(dim=DIM, chunk=CHUNK)
    ph.add_bundle("textA.txt", 0, TEXT_A, 0, len(TEXT_A))     # major 0  = text
    ph.add_bundle("randB.bin", 0, RAND_B, 10, len(RAND_B))   # major 10 = binary
    pw = ph.finalize()

    ea = embed(TEXT_A, DIM)
    res = probe(pw, MARKER, top=2, beta=20.0, resonate_steps=2)

    out = {
        "dim": DIM,
        "chunk": CHUNK,
        "textA": base64.b64encode(TEXT_A).decode(),
        "randB": base64.b64encode(RAND_B).decode(),
        "marker": base64.b64encode(MARKER).decode(),
        "embedA": [round(float(x), 6) for x in ea],
        "sims": [
            round(float(res["matches"][0]["sim"]), 6),
            round(float(res["matches"][1]["sim"]), 6),
        ],
        "matchFiles": [
            int(pw.file_idx[res["matches"][0]["engram"]]),
            int(pw.file_idx[res["matches"][1]["engram"]]),
        ],
        "attention": {
            "file": int(res["attention"]["file_idx"]),
            "type": res["attention"]["type"],
            "offset": int(res["attention"]["offset"]),
            "conf": round(float(res["attention"]["conf"]), 4),
        },
        "baked": {
            "file": int(res["baked"]["file_idx"]),
            "type": res["baked"]["type"],
            "offset": int(res["baked"]["offset"]),
            "conf": round(float(res["baked"]["conf"]), 4),
            "mass": round(float(res["baked"]["mass"], ), 6),
        },
        "resonance": [round(float(x), 6) for x in res["resonance"]],
    }

    payload = (
        "var PARITY = " + json.dumps(out, indent=1) + ";\n"
        'if (typeof window !== "undefined") window.PARITY = PARITY;\n'
        'if (typeof module !== "undefined" && module.exports) module.exports = PARITY;\n'
    )
    dest = Path(__file__).resolve().parent.parent / "web" / "parity.js"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(payload, encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    print("embedA[:6] =", out["embedA"][:6])
    print("sims       =", out["sims"])
    print("attention  =", out["attention"])
    print("baked      =", out["baked"])
    print("resonance  =", out["resonance"])


if __name__ == "__main__":
    main()
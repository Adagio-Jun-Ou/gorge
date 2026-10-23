"""Query the phased weights: identify any sample without reading any file."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .phasing import FILE_SCALE, OFFSET_SCALE, embed
from .types import MAJORS


def _decode(pw, v) -> dict:
    """Turn a value readout row into a human (file, offset, type) estimate.

    Rows are similarity-weighted averages over stored value vectors, and the
    last component of every value row is a constant 1 ("mass"), so dividing
    by it normalizes the average. Old 14-dim rows are tolerated (mass=1).
    """
    v = np.asarray(v, dtype=np.float64)
    mass = float(np.clip(v[14] if v.shape[0] >= 15 else 1.0, 0.0, None))
    if mass < 1e-9:
        return {"file": "", "file_idx": -1, "offset": 0, "type": "?",
                "conf": 0.0, "mass": 0.0}
    n_files = max(1, pw.n_files)
    chunk = int(pw.config.get("chunk", 65536))
    fidx = int(round(float(np.clip(v[0] / mass, 0.0, 1.0)) * FILE_SCALE)) - 1
    fidx = min(max(fidx, 0), n_files - 1)
    off = int(max(0.0, round(float(np.clip(v[1] / mass, 0.0, 1.0))
                             * OFFSET_SCALE - chunk / 2.0)))
    t_probs = np.clip(v[2:14] / mass, 0.0, None)
    t = int(np.argmax(t_probs))
    s = float(t_probs.sum())
    conf = float(t_probs[t] / s) if s > 0.0 else 0.0
    return {
        "file": pw.paths[fidx] if pw.paths else "",
        "file_idx": fidx,
        "offset": off,
        "type": MAJORS[t] if 0 <= t < len(MAJORS) else "?",
        "conf": conf,
        "mass": mass,
    }


def probe(pw, sample: bytes, top: int = 5, beta: float = 20.0,
          resonate_steps: int = 2) -> dict:
    """Ask the model 'what is this?' for any sample (bytes or text).

    Three readouts, from most to least faithful:

    1. matches      -- exact cosine over the phase cache (E . q)
    2. attention    -- modern-hopfield soft readout: softmax(beta E q) . V
    3. baked        -- weights-only: Hopfield attractor dynamics on W, then W_v
    """
    sample = bytes(sample)
    q = embed(sample, pw.dim, int(pw.config.get("salt", 0)))

    out: dict = {
        "sample_bytes": len(sample),
        "matches": [],
        "attention": None,
        "baked": None,
        "resonance": None,
    }

    if pw.E is not None and pw.E.shape[0]:
        sims = pw.E @ q
        k = min(int(top), int(sims.shape[0]))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        e = np.exp(beta * (sims[idx] - sims[idx].max()))
        conf = e / e.sum()
        out["matches"] = [
            {
                "engram": int(i),
                "sim": float(sims[i]),
                "conf": float(conf[j]),
                "path": pw.paths[int(pw.file_idx[i])] if pw.paths else "",
                "offset": int(pw.offsets[i]),
                "type": (MAJORS[int(pw.type_ids[i])]
                         if int(pw.type_ids[i]) < len(MAJORS) else "?"),
                "entropy": float(pw.entropies[i]),
            }
            for j, i in enumerate(idx)
        ]
        if pw.V is not None:
            p = np.exp(beta * (sims - sims.max()))
            p = p / p.sum()
            out["attention"] = _decode(pw, p @ pw.V)

    if pw.W is not None:
        r = q.astype(np.float32, copy=True)
        trace = []
        for _ in range(max(0, int(resonate_steps))):
            r = pw.W @ r
            n = float(np.linalg.norm(r))
            if n > 0.0:
                r = r / n
            trace.append(float(np.dot(r, q)))
        out["resonance"] = trace
        if pw.W_v is not None:
            # read out at q, not at the attractor: the linear value head
            # W_v q is the similarity-weighted average of stored values and
            # is robust; iterating first would amplify same-file clusters
            out["baked"] = _decode(pw, pw.W_v @ q)

    return out


def locate(pw, text: str, **kwargs) -> dict:
    """Probe by text: 'which file contains this string?'."""
    return probe(pw, text.encode("utf-8"), **kwargs)
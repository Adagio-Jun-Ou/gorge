"""Literal file-as-weights: fit an implicit neural net (SIREN) to a file's bytes.

This is the closest thing to 'pass the data in as weights': after training, the
file's entire content exists only as network parameters. `gorge recall`
reconstructs the bytes from the weights alone.

Requires torch: `pip install gorge[inr]`.

Note the honest asymmetry: fitting bytes into weights costs more compute than
reading them. That is the price of phasing information into parameters.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional
    torch = None

if torch is not None:
    class ByteNet(torch.nn.Module):
        """Coordinate MLP: position in [0,1) -> one row of bytes, scaled to [0,1]."""

        def __init__(self, hidden=256, depth=4, out=64, w0=30.0):
            super().__init__()
            self.w0 = float(w0)
            dims = [1] + [hidden] * depth
            self.hidden = torch.nn.ModuleList()
            for i in range(depth):
                lin = torch.nn.Linear(dims[i], dims[i + 1])
                bound = 1.0 / dims[i] if i == 0 else math.sqrt(6.0 / dims[i])
                torch.nn.init.uniform_(lin.weight, -bound, bound)
                self.hidden.append(lin)
            self.out = torch.nn.Linear(dims[-1], out)
            torch.nn.init.uniform_(self.out.weight, -math.sqrt(6.0 / dims[-1]),
                                   math.sqrt(6.0 / dims[-1]))

        def forward(self, t):
            x = torch.sin(self.w0 * self.hidden[0](t))
            for lin in self.hidden[1:]:
                x = torch.sin(lin(x))
            return self.out(x)


def _require_torch():
    if torch is None:
        raise SystemExit(
            "The memorizer needs PyTorch. Install it with:  pip install torch"
        )


def _positions(n_pos: int):
    return torch.linspace(0.0, 1.0, n_pos + 1)[:-1].unsqueeze(1)


def memorize_file(path, out_path, steps=1500, lr=1e-3, block=64, hidden=256,
                  depth=4, w0=30.0, device=None,
                  log: Optional[Callable] = print) -> dict:
    """Fit bytes into weights. Returns stats about what the weights now know."""
    _require_torch()
    data = open(path, "rb").read()
    if not data:
        raise ValueError("cannot memorize an empty file")
    n_bytes = len(data)
    n_pos = math.ceil(n_bytes / block)
    padded = data + b"\x00" * (n_pos * block - n_bytes)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ByteNet(hidden, depth, block, w0).to(dev)
    T = _positions(n_pos).to(dev)
    Y = (torch.frombuffer(bytearray(padded), dtype=torch.uint8)
         .reshape(n_pos, block).float() / 255.0).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    full = n_pos <= 2048
    mse = 0.0
    steps = int(steps)
    for step in range(steps):
        if full:
            tb, yb = T, Y
        else:
            sel = torch.randint(0, n_pos, (min(2048, n_pos),), device=dev)
            tb, yb = T[sel], Y[sel]
        loss = torch.mean((model(tb) - yb) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        mse = float(loss.item())
        if log and (step % max(1, steps // 10) == 0 or step == steps - 1):
            log(f"  memorize step {step + 1}/{steps}  mse {mse:.6f}")

    model = model.to("cpu")
    with torch.no_grad():
        pred = model(_positions(n_pos)).clamp(0.0, 1.0)
    got = (pred * 255.0).round().to(torch.uint8).reshape(-1)[:n_bytes]
    orig = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    exact = float((got == orig).float().mean().item())

    meta = {
        "n_bytes": n_bytes, "block": int(block), "hidden": int(hidden),
        "depth": int(depth), "w0": float(w0),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    torch.save({"state": model.state_dict(), "meta": meta, "exact_pct": exact},
               out_path)
    return {
        "n_bytes": n_bytes, "positions": n_pos, "steps": steps,
        "mse": mse, "exact_pct": exact, "weights_path": out_path,
    }


def recall_file(weights_path, out_path, log: Optional[Callable] = print) -> dict:
    """Reconstruct the bytes from the weights alone."""
    _require_torch()
    ck = torch.load(weights_path, map_location="cpu")
    meta = ck["meta"]
    model = ByteNet(meta["hidden"], meta["depth"], meta["block"], meta["w0"])
    model.load_state_dict(ck["state"])
    model.eval()

    n_bytes = int(meta["n_bytes"])
    n_pos = math.ceil(n_bytes / meta["block"])
    with torch.no_grad():
        pred = model(_positions(n_pos)).clamp(0.0, 1.0)
    data = (pred * 255.0).round().to(torch.uint8).reshape(-1)[:n_bytes]
    data = data.cpu().numpy().tobytes()
    with open(out_path, "wb") as f:
        f.write(data)

    sha = hashlib.sha256(data).hexdigest()
    return {
        "path": out_path, "n_bytes": n_bytes, "sha256": sha,
        "sha_match": sha == meta.get("sha256"),
        "exact_pct_stored": float(ck.get("exact_pct", 0.0)),
    }
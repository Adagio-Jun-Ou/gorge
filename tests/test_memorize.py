"""Memorizer tests (skipped automatically if torch is not installed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gorge.memorize import memorize_file, recall_file  # noqa: E402


def test_memorize_recall(tmp_path):
    data = bytes(np.random.default_rng(11).integers(0, 256, 256, dtype=np.uint8))
    src = tmp_path / "f.bin"
    src.write_bytes(data)
    w = tmp_path / "f.gwpt"

    res = memorize_file(str(src), str(w), steps=1200, hidden=128, depth=3,
                        block=16, lr=1e-3, log=None)
    assert res["n_bytes"] == 256
    assert res["exact_pct"] >= 0.6  # bytes now live (mostly) inside the weights

    out = tmp_path / "f.out"
    res2 = recall_file(str(w), str(out), log=None)
    assert out.exists()
    assert res2["n_bytes"] == 256
    assert len(out.read_bytes()) == 256
    # if the fit was byte-perfect, the reconstruction is bit-identical
    if res["exact_pct"] == 1.0:
        assert res2["sha_match"] is True
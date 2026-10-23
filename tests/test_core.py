"""Core tests: phasing math, scanning, type detection, probes, save/load."""

import numpy as np
import pytest

from gorge.phasing import (
    Phaser,
    dim_for_capacity,
    dup_hash,
    embed,
    four_grams,
    load_weights,
    parse_capacity,
    shannon_entropy,
)
from gorge.query import probe
from gorge.scan import iter_bundles
from gorge.types import MAJOR_ID, MAJORS, detect_type

MARKER = b"PHASE-KEY-ZQX-7741 opens the vault of weights"


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _build_drive(root):
    rng = np.random.default_rng(7)
    text_a = (
        b"The phasefield capacitor hums quietly. " * 20
        + MARKER
        + b" end of log. "
        + b" padding padding padding " * 60
    )
    _write(root / "docs" / "a.txt", text_a)
    bin_b = rng.integers(0, 256, size=65536, dtype=np.uint8).tobytes()
    _write(root / "blob" / "b.bin", bin_b)
    # dir sorts after "docs" so the canonical a.txt registers before its copy
    _write(root / "zzz_copy" / "a_copy.txt", text_a)  # exact duplicate content
    return {"bin_b": bin_b, "text_a": text_a}


def _ingest(root, dim=512, chunk=65536, **kw):
    ph = Phaser(dim=dim, chunk=chunk, **kw)
    for b in iter_bundles(str(root), chunk=chunk):
        ph.add_bundle(b.path, b.offset, b.data, b.type_id, b.file_size)
    return ph.finalize()


# ---------------------------------------------------------------- primitives

def test_parse_capacity():
    assert parse_capacity("25M") == 25_000_000
    assert parse_capacity("100k") == 100_000
    assert parse_capacity("20K") == 20_000
    with pytest.raises(ValueError):
        parse_capacity("tiny")
    with pytest.raises(ValueError):
        parse_capacity("1000")  # below the 10K-parameter floor
    assert dim_for_capacity(25_000_000) == 5000


def test_four_grams_counts():
    g = four_grams(b"hello world!!")
    assert g.dtype == np.uint64
    assert 0 < g.shape[0] <= 12
    assert four_grams(b"ab").shape[0] > 0
    assert four_grams(b"").shape[0] == 0


def test_embed_basics():
    e1 = embed(b"hello world", 256)
    e2 = embed(b"hello world", 256)
    np.testing.assert_allclose(e1, e2)
    assert e1.shape == (256,)
    assert abs(float(np.linalg.norm(e1)) - 1.0) < 1e-5
    a = embed(b"the quick brown fox jumps over the lazy dog", 512)
    b = embed(b"the quick brown fox jumps over the lazy dog again", 512)
    c = embed(np.random.default_rng(1).integers(0, 256, 400, dtype=np.uint8).tobytes(), 512)
    assert float(np.dot(a, b)) > float(np.dot(a, c))


def test_entropy_and_dup_hash():
    assert 7.0 < shannon_entropy(
        np.random.default_rng(2).integers(0, 256, 4096, dtype=np.uint8).tobytes()
    ) <= 8.0
    assert shannon_entropy(b"\x00" * 100) == 0.0
    assert dup_hash(b"same") == dup_hash(b"same")
    assert dup_hash(b"same") != dup_hash(b"diff")


# ---------------------------------------------------------------- type detection

def test_detect_type():
    assert detect_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100, "x.png")[1] == MAJOR_ID["image"]
    assert detect_type(b"hello world, plain text\n" * 10, "x.txt")[1] == MAJOR_ID["text"]
    assert detect_type(b"def f(x):\n    return x\n" * 10, "x.py")[1] == MAJOR_ID["code"]
    assert detect_type(b"a,b,c\n1,2,3\n" * 10, "x.csv")[1] == MAJOR_ID["data"]
    assert detect_type(b"%PDF-1.4 some pdf content\n", "x.pdf")[1] == MAJOR_ID["document"]
    assert detect_type(b"MZ\x90\x00" + bytes(range(256)) * 2, "x.exe")[1] == MAJOR_ID["executable"]
    assert detect_type(b"PK\x03\x04" + b"\x00" * 50, "x.docx")[1] == MAJOR_ID["document"]
    rand = np.random.default_rng(3).integers(0, 256, 512, dtype=np.uint8).tobytes()
    assert detect_type(rand, "x.bin")[1] == MAJOR_ID["binary"]
    assert len(MAJORS) == 12


# ---------------------------------------------------------------- ingest + probe

def test_ingest_dedup_and_probe(tmp_path):
    info = _build_drive(tmp_path)
    pw = _ingest(tmp_path)

    # the duplicate copy was deduped: only 2 distinct files registered
    assert pw.n_files == 2
    assert pw.n_engrams == 2
    assert pw.stats["dup_bundles"] == 1
    assert pw.stats["dup_bytes"] == len(info["text_a"])

    # probe with the marker string -> identifies a.txt
    res = probe(pw, MARKER, top=2)
    assert res["matches"]
    assert res["matches"][0]["path"].endswith("a.txt")
    assert res["matches"][0]["conf"] > 0.5
    assert res["matches"][0]["type"] == "text"

    # probe with a raw slice from the middle of b.bin -> identifies b.bin
    res2 = probe(pw, info["bin_b"][20000:20100], top=2)
    assert res2["matches"][0]["path"].endswith("b.bin")
    assert res2["matches"][0]["type"] == "binary"

    # attention readout agrees
    assert res2["attention"]["file"].endswith("b.bin")
    assert res2["attention"]["type"] == "binary"

    # weights-only readout works with the cache never consulted
    assert res2["baked"]["file"].endswith("b.bin")
    assert res2["baked"]["type"] == "binary"


def test_hopfield_field_prefers_owner(tmp_path):
    info = _build_drive(tmp_path)
    pw = _ingest(tmp_path)
    q = embed(MARKER, pw.dim)

    r = pw.W @ q
    r = r / float(np.linalg.norm(r))

    ia = next(i for i, p in enumerate(pw.paths) if p.endswith("a.txt"))
    ib = next(i for i, p in enumerate(pw.paths) if p.endswith("b.bin"))
    ja = int(np.nonzero(pw.file_idx == ia)[0][0])
    jb = int(np.nonzero(pw.file_idx == ib)[0][0])
    ea, eb = pw.E[ja], pw.E[jb]

    # the attractor W@q points at a.txt's engram, not b.bin's
    assert float(np.dot(r, ea)) > float(np.dot(r, eb))
    assert float(np.dot(r, ea)) > 0.5


def test_drop_cache_baked_only(tmp_path):
    _build_drive(tmp_path)
    pw = _ingest(tmp_path, keep_cache=False)
    assert pw.E is None and pw.V is None
    assert pw.W is not None

    res = probe(pw, MARKER)
    assert res["matches"] == []          # no cache: no associative matches
    assert res["attention"] is None
    assert res["baked"]["file"].endswith("a.txt")  # weights still answer
    assert res["baked"]["type"] == "text"


def test_no_bake_cache_only(tmp_path):
    _build_drive(tmp_path)
    pw = _ingest(tmp_path, bake=False)
    assert pw.W is None and pw.E is not None
    res = probe(pw, MARKER)
    assert res["matches"][0]["path"].endswith("a.txt")
    assert res["baked"] is None


def test_save_load_roundtrip(tmp_path):
    _build_drive(tmp_path)
    pw = _ingest(tmp_path)
    out = tmp_path / "m.gw"
    pw.save(str(out))
    pw2 = load_weights(str(out))
    assert pw2.dim == pw.dim
    assert pw2.paths == pw.paths
    assert pw2.n_engrams == pw.n_engrams
    assert pw2.W is not None
    res = probe(pw2, MARKER)
    assert res["matches"][0]["path"].endswith("a.txt")
    assert res["baked"]["file"].endswith("a.txt")


def test_param_accounting(tmp_path):
    _build_drive(tmp_path)
    pw = _ingest(tmp_path, dim=256)
    assert pw.params == 256 * 256 + 15 * 256
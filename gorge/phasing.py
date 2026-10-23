"""One-shot data-to-weights phasing.

The core primitive: every byte-bundle of a drive is turned into a sparse signed
shingle-embedding (a "phase pattern"), then written DIRECTLY into model
parameters in closed form -- no gradient descent, no training loop:

    W   += sum_i e_i e_i^T        (content field: a Hopfield weight matrix)
    W_v += sum_i v_i e_i^T        (value readout: which file / offset / type)

Because writing is a single outer-product pass, the model "eats" a drive in
streaming fashion: bytes go in, weights change, nothing is read twice.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "U64",
    "four_grams",
    "hash_grams",
    "embed_grams",
    "embed",
    "shannon_entropy",
    "dup_hash",
    "Phaser",
    "PhasedWeights",
    "parse_capacity",
    "dim_for_capacity",
    "save_weights",
    "load_weights",
    "FILE_SCALE",
    "OFFSET_SCALE",
]

U64 = np.uint64
_GOLDEN = U64(0x9E3779B97F4A7C15)

# scaling used when writing value rows (so readouts stay in [0, 1])
FILE_SCALE = float(1 << 20)    # supports up to ~1M distinct files
OFFSET_SCALE = float(1 << 48)  # supports up to ~256 TB offsets

D_VALUE = 15  # value row layout: [file+1, offset+chunk/2, onehot(type,12), mass=1]


def _splitmix64(x):
    x = x + _GOLDEN
    x = (x ^ (x >> U64(30))) * U64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> U64(27))) * U64(0x94D049BB133111EB)
    return x ^ (x >> U64(31))


def hash_grams(grams, salt=0):
    """Bijective 64-bit mixing of packed shingles (splitmix64)."""
    with np.errstate(over="ignore"):
        return _splitmix64(np.asarray(grams, dtype=np.uint64) ^ U64(salt))


def four_grams(data):
    """All overlapping 4-byte shingles of `data` (~1 per byte position), as uint64."""
    if isinstance(data, (np.ndarray, bytearray, memoryview)):
        data = bytes(data)
    n = len(data)
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    if n < 4:
        data = bytes(data) + b"\x00" * (4 - n)
        n = 4
    parts = []
    for off in range(4):
        take = (n - off) // 4
        if take > 0:
            parts.append(
                np.frombuffer(data, dtype="<u4", count=take, offset=off).astype(np.uint64)
            )
    if parts:
        return np.concatenate(parts)
    return np.empty(0, dtype=np.uint64)


def embed_grams(grams, dim, salt=0):
    """Signed feature-hash shingle embedding, sublinear counts, unit-normalized."""
    dim = int(dim)
    if grams is None or len(grams) == 0:
        return np.zeros(dim, dtype=np.float32)
    h = hash_grams(grams, salt)
    idx = (h % U64(dim)).astype(np.int64)
    sign = 1.0 - 2.0 * (((h >> U64(63)) & U64(1)).astype(np.float64))
    e = np.bincount(idx, weights=sign, minlength=dim)
    e = np.sign(e) * np.sqrt(np.abs(e))  # damp frequent shingles
    norm = float(np.linalg.norm(e))
    if norm > 0.0:
        e = e / norm
    return e.astype(np.float32)


def embed(data, dim, salt=0):
    """Bytes -> unit-normalized phase pattern in R^dim."""
    return embed_grams(four_grams(data), dim, salt)


def shannon_entropy(data):
    """Shannon entropy of the byte distribution (bits/byte, 0..8). Sampled at 64KB."""
    sample = bytes(data[:65536])
    if not sample:
        return 0.0
    counts = np.bincount(np.frombuffer(sample, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(sample)
    return float(-(p * np.log2(p)).sum())


def dup_hash(data):
    """64-bit content hash of a bundle for exact-duplicate skipping."""
    return int.from_bytes(hashlib.blake2b(bytes(data), digest_size=8).digest(), "big")


def parse_capacity(spec) -> int:
    """'25M' / '500K' / '1G' / raw int -> parameter count."""
    s = str(spec).strip().upper().replace("_", "")
    mult = 1
    if s and s[-1] in "KMGT":
        mult = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}[s[-1]]
        s = s[:-1]
    if not s:
        raise ValueError("capacity must look like 25M, 500K, 1G, or a raw count")
    n = int(float(s) * mult)
    if n < 10_000:
        raise ValueError("capacity too small: need >= 10K parameters")
    return n


def dim_for_capacity(params) -> int:
    """Model dimension d such that d^2 ~= parameter budget."""
    return max(64, math.isqrt(int(params)))


@dataclass
class PhasedWeights:
    """The model: a drive phased into weights (plus optional exact cache)."""

    dim: int
    W: Optional[np.ndarray]       # (d, d) content field
    W_v: Optional[np.ndarray]     # (14, d) value readout
    E: Optional[np.ndarray]       # (n, d) phase cache (exact engram memory)
    V: Optional[np.ndarray]       # (n, 14) value rows
    file_idx: np.ndarray          # (n,) int32
    offsets: np.ndarray           # (n,) int64
    type_ids: np.ndarray         # (n,) uint8
    entropies: np.ndarray        # (n,) float32
    paths: List[str]
    sizes: np.ndarray             # int64
    config: Dict
    stats: Dict

    @property
    def n_engrams(self) -> int:
        return int(self.file_idx.shape[0])

    @property
    def n_files(self) -> int:
        return len(self.paths)

    @property
    def params(self) -> int:
        p = 0
        if self.W is not None:
            p += int(self.W.size)
            if self.W_v is not None:
                p += int(self.W_v.size)
        return p

    def save(self, path) -> None:
        arrays = {
            "flags": np.array([self.W is not None, self.E is not None], dtype=np.int8),
            "file_idx": self.file_idx,
            "offsets": self.offsets,
            "type_ids": self.type_ids,
            "entropies": self.entropies,
            "sizes": np.asarray(self.sizes, dtype=np.int64),
            "paths": np.array(self.paths) if self.paths else np.empty(0, dtype="<U1"),
            "config_json": np.array(json.dumps(self.config)),
            "stats_json": np.array(json.dumps(self.stats)),
        }
        if self.W is not None:
            arrays["W"] = self.W
            arrays["W_v"] = self.W_v
        if self.E is not None:
            arrays["E"] = self.E
            arrays["V"] = self.V
        with open(path, "wb") as f:
            np.savez_compressed(f, **arrays)


def save_weights(pw: PhasedWeights, path) -> None:
    pw.save(path)


def load_weights(path) -> PhasedWeights:
    with np.load(path, allow_pickle=False) as z:
        flags = z["flags"]
        has_w = bool(int(flags[0]))
        has_e = bool(int(flags[1]))
        config = json.loads(str(z["config_json"].item()))
        stats = json.loads(str(z["stats_json"].item()))
        pw = PhasedWeights(
            dim=int(config.get("dim", 0)),
            W=z["W"] if has_w else None,
            W_v=z["W_v"] if has_w else None,
            E=z["E"] if has_e else None,
            V=z["V"] if has_e else None,
            file_idx=z["file_idx"],
            offsets=z["offsets"],
            type_ids=z["type_ids"],
            entropies=z["entropies"],
            paths=[str(p) for p in z["paths"].tolist()],
            sizes=z["sizes"],
            config=config,
            stats=stats,
        )
    if not pw.dim:
        if pw.W is not None:
            pw.dim = int(pw.W.shape[0])
        elif pw.E is not None:
            pw.dim = int(pw.E.shape[1])
    return pw


class Phaser:
    """Streams byte-bundles and phases them into weights in one closed-form pass.

    Two artifacts are built simultaneously:

    - the **weights** (W, W_v): fixed-size model, the drive's gestalt;
    - the **phase cache** (E, V): exact per-engram memory, like a KV cache.

    Each engram i contributes W += e_i e_i^T and W_v += v_i e_i^T. This is
    classical outer-product (Hopfield) storage: one pass, no learning rate,
    no gradients.
    """

    def __init__(self, dim, keep_cache=True, bake=True, dedup=True, salt=0,
                 fold_batch=512, chunk=65536):
        dim = int(dim)
        if dim < 16:
            raise ValueError("dim must be >= 16")
        if not bake and not keep_cache:
            raise ValueError("nothing to build: enable bake or keep_cache")
        self.dim = dim
        self.keep_cache = bool(keep_cache)
        self.bake = bool(bake)
        self.dedup = bool(dedup)
        self.salt = int(salt)
        self.fold_batch = max(16, int(fold_batch))
        self.chunk = int(chunk)

        self.W = np.zeros((dim, dim), np.float32) if self.bake else None
        self.W_v = np.zeros((D_VALUE, dim), np.float32) if self.bake else None

        self.E: Optional[List[np.ndarray]] = [] if self.keep_cache else None
        self.V: List[np.ndarray] = []
        self._pend_e: List[np.ndarray] = []
        self._pend_v: List[np.ndarray] = []

        self.paths: List[str] = []
        self.sizes: List[int] = []
        self.file_index: Dict[str, int] = {}

        # per-engram catalog
        self.file_idx: List[int] = []
        self.offsets: List[int] = []
        self.type_ids: List[int] = []
        self.entropies: List[float] = []

        self._dup_seen: set = set()
        self.bytes_seen = 0
        self.bundles = 0
        self.engrams = 0
        self.dup_bundles = 0
        self.dup_bytes = 0
        self.type_bytes: Dict[int, int] = {}

    @property
    def n_files(self) -> int:
        return len(self.paths)

    def _file_id(self, path) -> int:
        i = self.file_index.get(path)
        if i is None:
            i = len(self.paths)
            self.file_index[path] = i
            self.paths.append(path)
            self.sizes.append(0)
        return i

    def _value_row(self, fidx, offset, tid) -> np.ndarray:
        # every field is strictly positive so no stored pattern can be
        # invisible to the additive readout (a zero value would add nothing)
        v = np.zeros(D_VALUE, np.float32)
        v[0] = min(fidx + 1, (1 << 20)) / FILE_SCALE
        v[1] = min(offset + self.chunk // 2, (1 << 48) - 1) / OFFSET_SCALE
        v[2 + int(tid)] = 1.0
        v[D_VALUE - 1] = 1.0  # mass: lets the readout normalize itself
        return v

    def add_bundle(self, path, offset, data, type_id, file_size=0) -> bool:
        """Eat one bundle of bytes. Returns True if it became an engram."""
        data = bytes(data)
        if not data:
            return False
        self.bytes_seen += len(data)
        self.bundles += 1
        tid = int(type_id) % (D_VALUE - 2)
        self.type_bytes[tid] = self.type_bytes.get(tid, 0) + len(data)

        if self.dedup:
            key = dup_hash(data)
            if key in self._dup_seen:
                self.dup_bundles += 1
                self.dup_bytes += len(data)
                return False
            self._dup_seen.add(key)

        fidx = self._file_id(path)
        if self.sizes[fidx] == 0 and file_size:
            self.sizes[fidx] = int(file_size)

        e = embed(data, self.dim, self.salt)
        v = self._value_row(fidx, offset, tid)

        self.file_idx.append(fidx)
        self.offsets.append(int(offset))
        self.type_ids.append(tid)
        self.entropies.append(shannon_entropy(data))
        self.V.append(v)
        if self.E is not None:
            self.E.append(e)

        self.engrams += 1
        if self.bake:
            self._pend_e.append(e)
            self._pend_v.append(v)
            if len(self._pend_e) >= self.fold_batch:
                self._fold_pending()
        return True

    def _fold_pending(self) -> None:
        """Closed-form weight write: W += X^T X, W_v += Vb^T X (one BLAS call each)."""
        if not self._pend_e:
            return
        X = np.stack(self._pend_e).astype(np.float32, copy=False)
        Vb = np.stack(self._pend_v).astype(np.float32, copy=False)
        self.W += X.T @ X
        self.W_v += Vb.T @ X
        self._pend_e.clear()
        self._pend_v.clear()

    def finalize(self) -> PhasedWeights:
        self._fold_pending()
        E = np.stack(self.E).astype(np.float32) if self.E else None
        # value rows are only meaningful next to the cache they annotate
        V = (np.stack(self.V).astype(np.float32) if (self.V and E is not None)
             else None)
        config = {
            "version": 1, "dim": self.dim, "d_v": D_VALUE, "salt": self.salt,
            "chunk": self.chunk, "bake": self.bake, "keep_cache": self.keep_cache,
        }
        stats = {
            "bytes_seen": self.bytes_seen, "bundles": self.bundles,
            "dup_bundles": self.dup_bundles, "dup_bytes": self.dup_bytes,
            "type_bytes": {str(k): int(v) for k, v in self.type_bytes.items()},
        }
        return PhasedWeights(
            dim=self.dim,
            W=self.W if self.bake else None,
            W_v=self.W_v if self.bake else None,
            E=E, V=V,
            file_idx=np.asarray(self.file_idx, dtype=np.int32),
            offsets=np.asarray(self.offsets, dtype=np.int64),
            type_ids=np.asarray(self.type_ids, dtype=np.uint8),
            entropies=np.asarray(self.entropies, dtype=np.float32),
            paths=list(self.paths),
            sizes=np.asarray(self.sizes, dtype=np.int64),
            config=config, stats=stats,
        )
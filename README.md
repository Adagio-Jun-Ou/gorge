# gorge

**Eat an entire drive in one pass. Phase the bytes directly into model weights.
Query the weights -- never re-read the files.**

gorge is an experiment in *data as weights*. Instead of a language model reading
your files token-by-token through tools, gorge writes your data **into the
parameters of a model in closed form** (no gradient descent, one streaming pass)
and then answers questions like *"what is this sample?"* or *"which file contains
this?"* straight from the weights.

---

## The idea, translated

| you said | what gorge actually does |
|---|---|
| "eat entire drives at once" | one streaming pass over every byte; nothing is read twice |
| "pass the data in as weights" | each byte-bundle becomes a unit-normalized shingle-embedding `e_i`, written directly into `W += e_i e_i^T` (outer product) and `W_v += v_i e_i^T` -- closed form, zero training steps |
| "phase the data into the model" | the matrices `W` (content field) and `W_v` (value readout) **are** the model; the drive lives in their values |
| "digest equal to its size parameter" | `--capacity` sets `d = sqrt(params)`; Hopfield theory says ~`0.138 * d` engrams stay cleanly separable -- gorge reports your load factor honestly |
| "know what it is" | probe any sample: ranked matches (file/offset/type), a modern-Hopfield attention readout, and a **weights-only** Hopfield readout that works with the cache deleted |

## Architecture

```
drive --scan--> byte bundles --shingle+hash--> engram vectors e_i  (dim d)
                                                   |
                     closed form, no gradient descent, one pass
                                                   |
              +------------------------------------+----------------+
              v                                  v                v
   W  = SUM e_i e_i^T      W_v = SUM v_i e_i^T     E = [e_i]   V = [v_i]
   content field           value readout           phase cache (optional)
   (Hopfield weights)      (file/offset/type)      (exact addressable memory)
```

Query time (no file is ever opened):

```
sample --> embedding q --+-> E . q              : ranked matches
                         +-> softmax(b*E.q) . V : attention readout (modern Hopfield)
                         +-> iterate W . q -> attractor r ; W_v . r : weights-only answer
```

## The physics (read this once)

- **No free lunch.** You cannot store X bits losslessly in fewer than X bits of
  parameters (Shannon). gorge does not try. Bytes stay on disk; what gets
  baked into weights is an *associative index* -- the drive's gestalt.
- The baked Hopfield field `W` has finite clean capacity (~`0.138 * d` engrams,
  where `d = sqrt(--capacity)`). Beyond that, baked recall degrades gracefully
  from "knows exactly" to "knows fuzzily". `gorge stats` tells you where you are.
- That 0.138 bound assumes random patterns. Real text/code is *correlated*
  (shared words, boilerplate), so expect the baked **file** estimate to go fuzzy
  earlier than the bound suggests. Type and offset stay robust longer; exact
  file identity is what the phase cache is for. This is the honest trade you
  are making when you delete the cache to get a weights-only artifact.
- The **phase cache** (`E`, `V`) is an exact addressable memory that grows with
  your data (think KV cache). `--drop-cache` deletes it and leaves a
  weights-only artifact -- smaller, purer, lossier.
- `gorge memorize` stores a single file **literally inside weights** (a fitted
  implicit neural representation). It is the closest thing to the original
  fantasy, and it is *slower* than reading the file. That asymmetry is a law
  of nature, not a bug: compressing information into fitted parameters costs
  more than copying it.

## Install

```
pip install -e .            # core: numpy only
pip install -e .[inr]       # + torch, for the literal file-as-weights memorizer
```

## Quickstart

```
gorge demo                                     # eats a synthetic drive, then probes it
gorge ui                                        # animated browser demo (see below)
gorge ingest E:\somedir --capacity 25M -o mydrive.gw
gorge probe mydrive.gw --text "some string from a file"
gorge probe mydrive.gw --sample somefile.bin
gorge stats mydrive.gw
gorge memorize important.bin && gorge recall important.bin.gwpt
```

gorge is **read-only** on the target path. It only reads bytes; the only thing
it writes is the model file you name with `-o`.

## Commands

| command | what it does |
|---|---|
| `gorge ingest PATH` | one pass over PATH (drive, dir, or single file); phases every byte into `W`, `W_v`, and the optional cache; saves a `.gw` model |
| `gorge probe MODEL` | feed a sample (`--text` or `--sample`); get identity: file, offset, type, confidence -- answered from weights |
| `gorge stats MODEL` | model size, engram load factor vs. attractor capacity, type breakdown |
| `gorge demo` | builds a synthetic mini-drive in temp, eats it, probes it |
| `gorge ui` | serves `web/` on localhost:8765 -- the animated browser demo |

## The browser demo (`gorge ui`)

An interactive, animated version of the whole story, runnable two ways:

- `gorge ui` -- serves it locally and opens your browser; or
- just open `web/index.html` directly (fully self-contained, no build step).

What it shows, live, in your browser:

1. **the drive** -- a synthetic tree (secret notes, code, csv, a fake PNG, a
   duplicate for the deduper). Drag in **real files from your machine** and
   they get eaten for real.
2. **the weight field** -- the W matrix grows on screen, engram by engram,
   as bundles stream in and write themselves in via outer products. Watch
   the load gauge pass attractor capacity when you stress-feed 30 more files.
3. **probe** -- type any string (or fire a preset) and watch the query
   pattern resonate with the field: ranked matches, attention readout, and
   the weights-only baked readout. **Burn the phase cache** to see the
   weights answer alone.
4. **memorizer** -- 96 bytes fitted into network parameters by gradient
   descent, side by side with the one-pass phasing above. The asymmetry
   is the physics lesson.

The JS engine in `web/engine.js` is a faithful port of the Python engine --
true uint64 splitmix64 hashing, the same folds, the same readouts. On page
load it re-derives fixtures generated by `tools/gen_parity.py` from the real
Python code and compares numbers; the badge in the header shows
**engine parity: VERIFIED**. Regenerate fixtures with
`python tools/gen_parity.py`; check them headlessly with `node tools/test_web.js`.
| `gorge memorize FILE` | *(torch)* fits an implicit neural net so the file's bytes live in its weights |
| `gorge recall W` | *(torch)* reconstructs the bytes from the weights |

## The size parameter

`--capacity` is the model's parameter budget. `d = sqrt(capacity)` is the
embedding/weight dimension. Clean baked recall holds ~`0.138 * d` engrams; one
engram covers `--chunk` bytes (default 64KB), so bytes you can cleanly "know":

```
clean bytes ~ 0.138 * d * chunk
```

| capacity | d | clean baked recall @64KB chunks | notes |
|---|---|---|---|
| 4M | 2000 | ~276 engrams / ~17 MB | demos, small trees |
| 25M | 5000 | ~690 / ~43 MB | default |
| 100M | 10000 | ~1380 / ~86 MB | raise `--chunk` to 256KB+ |
| 400M | 20000 | ~2760 / ~172 MB | raise `--chunk` to 1MB+ |

Tuning rule of thumb: keep `chunk >= 8 * d` bytes so the exact cache costs less
than the data it indexes. Bigger chunks = coarser engrams = more bytes per
parameter. `gorge stats` prints your load factor and a verdict.

## Scaling limits (honest)

- Ingest speed is bounded by shingle hashing (~100-500 MB/s/core, pure NumPy).
- Baking `W` costs `O(d^2 * n)` -- fine up to ~100K engrams on CPU; use
  `--no-bake` (cache-only, fast path) for very large drives.
- The phase cache costs `4 * d` bytes per engram; `--drop-cache` bounds the
  artifact at `W` + catalog.
- Past the attractor capacity, weights-only recall is fuzzy on purpose. The
  cache path stays exact.

## Roadmap

- near-duplicate merging (simhash buckets) to raise effective capacity
- hierarchical phasing: directory-level engrams -> a tree of attractors
- GPU ingest path (torch/CUDA) and zero-copy mmap scanning
- a learned encoder trained once on the drive's own bytes (still one-shot writes)
- hypernetwork mode: data chunk -> weight-delta for a task network (fast weights)

## Research anchors

- Modern Hopfield Networks / Dense Associative Memory -- Ramsauer et al., 2020
- Classical Hopfield outer-product storage, capacity ~0.138n -- Hopfield, 1982
- Dataset Distillation -- Wang et al., 2018
- Hypernetworks -- Ha et al., 2017; Fast Weight Programmers -- Schmidhuber, 1992
- COIN: implicit neural representations for compression -- Dupont et al., 2021
- Feature hashing -- Weinberger et al., 2009; SimHash -- Charikar, 2002
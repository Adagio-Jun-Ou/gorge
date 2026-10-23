/* gorge JS engine — a faithful port of gorge/phasing.py + gorge/query.py.
 *
 * Same algorithm, same constants, same arithmetic (true uint64 splitmix64 via
 * hi/lo 32-bit words): byte shingles -> signed feature-hash embeddings ->
 * closed-form outer-product weight phasing (W, W_v) -> Hopfield/attention
 * probes.  Verified against the Python engine via web/parity.js.
 *
 * UMD: works in the browser (window.GorgeEngine) and in node (module.exports).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.GorgeEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var U32 = 4294967296;
  var FILE_SCALE = 1048576;              // 2^20
  var OFFSET_SCALE = 281474976710656;     // 2^48
  var D_VALUE = 15;

  var MAJORS = ["text", "code", "data", "document", "image", "audio", "video",
    "archive", "database", "executable", "binary", "other"];

  /* ---------------- uint64 helpers: numbers are [hi, lo], uint32 each ---- */

  function add64(ah, al, bh, bl) {
    var sl = al + bl;
    var carry = sl >= U32 ? 1 : 0;
    return [(ah + bh + carry) >>> 0, sl >>> 0];
  }

  function xor64(ah, al, bh, bl) {
    return [(ah ^ bh) >>> 0, (al ^ bl) >>> 0];
  }

  function shr64(ah, al, k) {  // 0 < k < 32
    return [(ah >>> k) >>> 0, ((al >>> k) | (ah << (32 - k))) >>> 0];
  }

  function mul64(ah, al, bh, bl) {  // (ah:al * bh:bl) mod 2^64
    var al_hi = al >>> 16, al_lo = al & 0xFFFF;
    var bl_hi = bl >>> 16, bl_lo = bl & 0xFFFF;
    var t_lo = Math.imul(al_lo, bl_lo) >>> 0;                       // < 2^32
    var t_mid = (Math.imul(al_hi, bl_lo) >>> 0) + (Math.imul(al_lo, bl_hi) >>> 0);
    var t_hi = Math.imul(al_hi, bl_hi) >>> 0;
    var m_hi = Math.floor(t_mid / 0x10000);   // floor division keeps bit 32
    var m_lo = t_mid - m_hi * 0x10000;
    var sumLow = m_lo * 0x10000 + t_lo;       // < 2^33, exact in double
    var carry1 = sumLow >= U32 ? 1 : 0;
    var cross = Math.imul(ah, bl) + Math.imul(al, bh);  // mod-2^32 terms
    return [(t_hi + m_hi + carry1 + cross) >>> 0, sumLow >>> 0];
  }

  /* splitmix64, identical to gorge.phasing.hash_grams (salt = 0) */
  function splitmix64(ah, al) {
    var r = add64(ah, al, 0x9E3779B9, 0x7F4A7C15);
    var s = shr64(r[0], r[1], 30);
    r = xor64(r[0], r[1], s[0], s[1]);
    r = mul64(r[0], r[1], 0xBF58476D, 0x1CE4E5B9);
    s = shr64(r[0], r[1], 27);
    r = xor64(r[0], r[1], s[0], s[1]);
    r = mul64(r[0], r[1], 0x94D049BB, 0x133111EB);
    s = shr64(r[0], r[1], 31);
    return xor64(r[0], r[1], s[0], s[1]);
  }

  /* full 64-bit value mod dim, matching Python's `h % dim` on uint64 */
  function mod64(hi, lo, dim) {
    return ((hi % dim) * (U32 % dim) + (lo % dim)) % dim;
  }

  /* ---------------- shingles -> embedding (gorge.phasing.embed) ---------- */

  function embedBytes(u8, dim) {
    var acc = new Float64Array(dim);
    var m = u8.length, d = u8;
    if (m === 0) return acc;
    if (m < 4) {
      d = new Uint8Array(4);
      d.set(u8);
      m = 4;
    }
    for (var off = 0; off < 4; off++) {
      for (var p = off; p + 4 <= m; p += 4) {
        var gram = (d[p] | (d[p + 1] << 8) | (d[p + 2] << 16) | (d[p + 3] << 24)) >>> 0;
        var h = splitmix64(0, gram);
        var idx = mod64(h[0], h[1], dim);
        var sign = ((h[0] >>> 31) & 1) ? -1 : 1;   // bit 63
        acc[idx] += sign;
      }
    }
    var norm = 0, i;
    for (i = 0; i < dim; i++) {
      var v = (acc[i] >= 0 ? 1 : -1) * Math.sqrt(Math.abs(acc[i]));
      acc[i] = v;
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (norm > 0) for (i = 0; i < dim; i++) acc[i] /= norm;
    return acc;
  }

  /* cheap deterministic 64-bit content key for dedup (parity-irrelevant) */
  function dupKey(u8) {
    var n = u8.length;
    var h = splitmix64(0, n);
    for (var i = 0; i < n; i += Math.max(1, (n >> 3) | 1)) {
      h = splitmix64(h[0], (h[1] ^ u8[i]) >>> 0);
    }
    return h[0] * U32 + h[1];
  }

  /* ---------------- the phased-weights engine --------------------------- */

  function Engine(dim, chunk) {
    this.dim = dim | 0;
    this.chunk = chunk || 4096;
    this.W = null;      // built lazily: dim*dim
    this.Wv = null;     // 15*dim
    this.E = [];        // phase cache: embeddings per engram
    this.V = [];        // value rows per engram
    this.engrams = [];  // {file, offset, tid}
    this.files = [];    // {name, size, tid}
    this.fileIndex = {};
    this.dupSeen = {};
    this.cacheBurned = false;
    this.stats = { bytesSeen: 0, bundles: 0, engrams: 0, dupBundles: 0, dupBytes: 0 };
  }

  Engine.MAJORS = MAJORS;

  Engine.prototype._valueRow = function (fileId, offset, tid) {
    // strictly positive fields: nothing stored may be invisible to the
    // additive readout; last dim is "mass" so the readout self-normalizes
    var v = new Float64Array(D_VALUE);
    v[0] = Math.min(fileId + 1, FILE_SCALE) / FILE_SCALE;
    v[1] = Math.min(offset + (this.chunk >> 1), OFFSET_SCALE - 1) / OFFSET_SCALE;
    v[2 + (tid % 12)] = 1;
    v[D_VALUE - 1] = 1;
    return v;
  };

  Engine.prototype.addBundle = function (name, offset, u8, tid, fileSize) {
    var bytes = u8.length;
    this.stats.bytesSeen += bytes;
    this.stats.bundles++;

    var key = dupKey(u8);
    if (Object.prototype.hasOwnProperty.call(this.dupSeen, key)) {
      this.stats.dupBundles++;
      this.stats.dupBytes += bytes;
      return { stored: false, dup: true };
    }
    this.dupSeen[key] = 1;

    var e = embedBytes(u8, this.dim);
    var v = this._valueRow(0, offset, tid);  // file id patched below

    if (!this.W) {
      this.W = new Float64Array(this.dim * this.dim);
      this.Wv = new Float64Array(D_VALUE * this.dim);
    }
    var dim = this.dim, W = this.W, Wv = this.Wv, i, j, k;

    // register the file only once a bundle is actually stored (like Python)
    var fileId = this.fileIndex[name];
    if (fileId === undefined) {
      fileId = this.files.length;
      this.fileIndex[name] = fileId;
      this.files.push({ name: name, size: fileSize || bytes, tid: tid });
    }
    v[0] = Math.min(fileId + 1, FILE_SCALE) / FILE_SCALE;

    // W += e e^T  (the "phasing": closed form, no gradients, one pass)
    for (i = 0; i < dim; i++) {
      var ei = e[i];
      if (ei === 0) continue;
      var row = i * dim;
      for (j = 0; j < dim; j++) W[row + j] += ei * e[j];
    }
    // W_v += v e^T
    for (k = 0; k < D_VALUE; k++) {
      var vk = v[k];
      if (vk === 0) continue;
      var vrow = k * dim;
      for (j = 0; j < dim; j++) Wv[vrow + j] += vk * e[j];
    }

    if (!this.cacheBurned) {
      this.E.push(e);
      this.V.push(v);
    }
    this.engrams.push({ file: fileId, offset: offset, tid: tid % 12 });
    this.stats.engrams++;
    return { stored: true, dup: false, e: e, v: v };
  };

  Engine.prototype.params = function () {
    return this.W ? this.dim * this.dim + D_VALUE * this.dim : 0;
  };

  Engine.prototype.attractorCapacity = function () {
    return 0.138 * this.dim;
  };

  Engine.prototype.loadFactor = function () {
    return this.stats.engrams / Math.max(1e-9, this.attractorCapacity());
  };

  Engine.prototype.decode = function (v) {
    var mass = Math.max(0, v[D_VALUE - 1] || 0);
    if (mass < 1e-9) return { file: -1, offset: 0, tid: -1, conf: 0, mass: 0 };
    var nf = Math.max(1, this.files.length);
    var f = Math.round(Math.min(Math.max(v[0] / mass, 0), 1) * FILE_SCALE) - 1;
    f = Math.min(Math.max(f, 0), nf - 1);
    var off = Math.max(0, Math.round(
      Math.min(Math.max(v[1] / mass, 0), 1) * OFFSET_SCALE - this.chunk / 2));
    var best = 0, bs = -1, tot = 0, t;
    for (t = 0; t < 12; t++) {
      var x = Math.max(0, v[2 + t] / mass);
      tot += x;
      if (x > bs) { bs = x; best = t; }
    }
    return { file: f, offset: off, tid: best, conf: tot > 0 ? bs / tot : 0, mass: mass };
  };

  Engine.prototype.probe = function (u8, top, beta, steps) {
    top = top || 3;
    beta = beta || 20;
    steps = steps === undefined ? 2 : steps;
    var dim = this.dim;
    var q = embedBytes(u8, dim);
    var out = { sampleBytes: u8.length, q: q, matches: [], attention: null, baked: null, resonance: null };

    if (this.E.length) {
      var n = this.E.length;
      var sims = new Float64Array(n), i, j, s;
      for (i = 0; i < n; i++) {
        var e = this.E[i];
        s = 0;
        for (j = 0; j < dim; j++) s += e[j] * q[j];
        sims[i] = s;
      }
      var order = [];
      for (i = 0; i < n; i++) order.push(i);
      order.sort(function (a, b) { return sims[b] - sims[a]; });
      var topIdx = order.slice(0, Math.min(top, n));
      var mx = sims[topIdx[0]];
      var confs = [], sum = 0, k;
      for (k = 0; k < topIdx.length; k++) {
        var c = Math.exp(beta * (sims[topIdx[k]] - mx));
        confs.push(c);
        sum += c;
      }
      for (k = 0; k < topIdx.length; k++) {
        var gi = topIdx[k];
        out.matches.push({
          engram: gi, sim: sims[gi], conf: confs[k] / sum,
          file: this.engrams[gi].file, offset: this.engrams[gi].offset,
          tid: this.engrams[gi].tid,
        });
      }
      // modern-hopfield attention over all engrams, read out through V
      var p = new Float64Array(n), asum = 0;
      for (i = 0; i < n; i++) { p[i] = Math.exp(beta * (sims[i] - mx)); asum += p[i]; }
      var av = new Float64Array(D_VALUE);
      for (i = 0; i < n; i++) {
        var pi = p[i] / asum;
        if (pi === 0) continue;
        var vv = this.V[i];
        for (j = 0; j < D_VALUE; j++) av[j] += pi * vv[j];
      }
      out.attention = this.decode(av);
    }

    if (this.W) {
      // hopfield resonance trace (diagnostic attractor dynamics)
      var prev = q, trace = [];
      for (var st = 0; st < steps; st++) {
        var r = new Float64Array(dim);
        for (i = 0; i < dim; i++) {
          var row = i * dim;
          var acc = 0;
          for (j = 0; j < dim; j++) acc += this.W[row + j] * prev[j];
          r[i] = acc;
        }
        var nrm = 0;
        for (i = 0; i < dim; i++) nrm += r[i] * r[i];
        nrm = Math.sqrt(nrm);
        if (nrm > 0) for (i = 0; i < dim; i++) r[i] /= nrm;
        var cos = 0;
        for (i = 0; i < dim; i++) cos += r[i] * q[i];
        trace.push(cos);
        prev = r;
      }
      out.resonance = trace;
      // baked readout: value head applied directly at the query
      var bv = new Float64Array(D_VALUE);
      for (k = 0; k < D_VALUE; k++) {
        var vrow = k * dim;
        var a2 = 0;
        for (j = 0; j < dim; j++) a2 += this.Wv[vrow + j] * q[j];
        bv[k] = a2;
      }
      out.baked = this.decode(bv);
    }
    return out;
  };

  Engine.prototype.burnCache = function () {
    // one-way in a real artifact: after this, only W and W_v can answer
    this.E = [];
    this.V = [];
    this.cacheBurned = true;
  };

  return {
    Engine: Engine,
    embedBytes: embedBytes,
    splitmix64: splitmix64,
    mul64: mul64,
    MAJORS: MAJORS,
  };
});
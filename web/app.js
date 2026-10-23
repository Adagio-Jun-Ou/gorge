/* gorge web demo — UI logic.
 * Runs the REAL algorithm (see engine.js, parity-verified against Python).
 * Everything below is presentation: animation, probing, interaction.
 */
(function () {
  "use strict";

  var G = window.GorgeEngine;
  var DIM = 128, CHUNK = 4096, BETA = 20, TOPK = 3, STEPS = 2;
  var MAX_DROP = 2 * 1024 * 1024;

  var TYPE_COLORS = {
    text: "#7cc4ff", code: "#5ef2a6", data: "#ffd479", document: "#ff9ecb",
    image: "#ff8f6b", audio: "#c9a7ff", video: "#ff6b9e", archive: "#8fe7c1",
    database: "#6be7ff", executable: "#d4ff6b", binary: "#93a0b8", other: "#8899aa"
  };

  var engine = null;          // GorgeEngine.Engine
  var drive = [];             // synthetic + dropped files
  var fedOnce = false, feeding = false, burned = false;
  var engramDots = [];        // {x, y, color}
  var projX = null, projY = null;
  var beams = null;           // {q:{x,y}, list:[{x,y,w,color}], t0}
  var packets = 0;
  var lutPos = null, lutNeg = null;
  var fieldImg = null, fieldOff = null;

  /* ---------------- tiny helpers ---------------- */

  function $(id) { return document.getElementById(id); }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function fmtBytes(n) {
    if (n < 1024) return n + " B";
    if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
    return (n / 1048576).toFixed(1) + " MB";
  }
  function b64bytes(s) {
    var bin = atob(s), u8 = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8;
  }
  function strBytes(s) {
    var u8 = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i) & 0xFF;
    return u8;
  }

  /* ---------------- engine parity badge (browser) ---------------- */

  function runParity() {
    var badge = $("parityBadge");
    try {
      var eng = new G.Engine(PARITY.dim, PARITY.chunk);
      eng.addBundle("textA.txt", 0, b64bytes(PARITY.textA), 0);
      eng.addBundle("randB.bin", 0, b64bytes(PARITY.randB), 10);
      var ea = G.embedBytes(b64bytes(PARITY.textA), PARITY.dim);
      var maxd = 0;
      for (var i = 0; i < PARITY.dim; i++) {
        maxd = Math.max(maxd, Math.abs(ea[i] - PARITY.embedA[i]));
      }
      var r = eng.probe(b64bytes(PARITY.marker), 2, 20, 2);
      var checks = [
        ["embedding", maxd < 1e-4],
        ["top-1 similarity", Math.abs(r.matches[0].sim - PARITY.sims[0]) < 1e-3],
        ["top-2 similarity", Math.abs(r.matches[1].sim - PARITY.sims[1]) < 1e-3],
        ["match files", r.matches[0].file === PARITY.matchFiles[0] && r.matches[1].file === PARITY.matchFiles[1]],
        ["attention readout", r.attention.file === PARITY.attention.file && G.MAJORS[r.attention.tid] === PARITY.attention.type],
        ["baked readout", r.baked.file === PARITY.baked.file && G.MAJORS[r.baked.tid] === PARITY.baked.type],
        ["baked mass", Math.abs(r.baked.mass - PARITY.baked.mass) < 1e-2],
        ["resonance trace", r.resonance.every(function (v, i) { return Math.abs(v - PARITY.resonance[i]) < 1e-2; })]
      ];
      var pass = checks.filter(function (c) { return c[1]; }).length;
      if (pass === checks.length) {
        badge.textContent = "engine parity vs Python gorge: VERIFIED (" + pass + "/" + checks.length + " checks)";
        badge.className = "badge ok";
      } else {
        var failed = checks.filter(function (c) { return !c[1]; }).map(function (c) { return c[0]; }).join(", ");
        badge.textContent = "PARITY FAILED: " + failed;
        badge.className = "badge bad";
      }
    } catch (e) {
      badge.textContent = "parity check error: " + e.message;
      badge.className = "badge bad";
    }
  }

  /* ---------------- type detection (UI labels only) ---------------- */

  function detectType(head, name) {
    head = head.subarray(0, 512);
    var ext = (name.match(/\.([a-z0-9]+)$/i) || [, ""])[1].toLowerCase();
    if (head.length >= 8 && head[0] === 0x89 && head[1] === 0x50 && head[2] === 0x4E) return { tid: 4, label: "image/png" };
    if (head[0] === 0x50 && head[1] === 0x4B) return { tid: 7, label: "archive/zip" };
    if (head[0] === 0x25 && head[1] === 0x50) return { tid: 3, label: "document/pdf" };
    if (head[0] === 0x4D && head[1] === 0x5A) return { tid: 9, label: "executable/pe" };
    if (head[0] === 0x7F && head[1] === 0x45) return { tid: 9, label: "executable/elf" };
    if (head[0] === 0x1F && head[1] === 0x8B) return { tid: 7, label: "archive/gzip" };
    var printable = 0;
    for (var i = 0; i < head.length; i++) {
      var c = head[i];
      if ((c >= 32 && c < 127) || c === 10 || c === 13 || c === 9) printable++;
    }
    if (head.length && printable / head.length >= 0.85) {
      if (["py", "js", "ts", "c", "h", "cpp", "cs", "go", "rs", "java", "rb", "php", "sh", "ps1", "html", "css", "sql", "lua"].indexOf(ext) >= 0)
        return { tid: 1, label: "code/" + ext };
      if (["json", "csv", "tsv", "yaml", "yml", "toml", "xml", "ini", "cfg"].indexOf(ext) >= 0)
        return { tid: 2, label: "data/" + ext };
      return { tid: 0, label: "text/plain" };
    }
    return { tid: 10, label: "binary/unknown" };
  }

  /* ---------------- synthetic demo drive ---------------- */

  function genSecretNotes() {
    var lines = [];
    for (var i = 0; i < 80; i++) {
      lines.push("Mission log " + i + ": payload capacitors charged to " + ((i * 37) % 100) + "% of the phase field.");
      if (i % 9 === 0) lines.push("Reminder: the launch key PHASE-KEY-ZQX-7741 must not leave the vault.");
      lines.push("The window opens at 04:00 ZULU. Guidance locked. Nothing to read twice.");
    }
    return strBytes(lines.join("\n"));
  }
  function genPhasemath() {
    var out = ["import math", "", "# phasemath: closed-form weight phasing experiments", ""];
    for (var i = 0; i < 120; i++) {
      out.push("def crunch_phasefield_" + i + "(x, depth=" + (i % 7) + "):");
      out.push("    acc = x * " + (1 + i % 13) + ".0");
      out.push("    for _ in range(depth):");
      out.push("        acc = math.sin(acc) + " + i + ".5");
      out.push("    return acc");
      out.push("");
    }
    return strBytes(out.join("\n"));
  }
  function genCsv() {
    var rows = ["id,name,balance,tier"];
    for (var i = 0; i < 180; i++) {
      rows.push((10000 + i) + ",customer_" + (i < 100 ? "0" : "") + (i < 10 ? "0" : "") + i + "," + ((i * 137) % 9000 + 100) + ".50,gold");
    }
    return strBytes(rows.join("\n"));
  }
  function genPng(rng) {
    var size = 16384;
    var u8 = new Uint8Array(size + 8);
    u8.set([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    for (var i = 8; i < u8.length; i++) u8[i] = Math.floor(rng() * 256);
    return u8;
  }

  function buildDrive() {
    var rng = mulberry32(7);
    var phasemath = genPhasemath();
    drive = [
      { name: "notes/secret_launch_codes.txt", bytes: genSecretNotes(), tid: 0, label: "text/plain" },
      { name: "src/phasemath.py", bytes: phasemath, tid: 1, label: "code/py" },
      { name: "data/customers.csv", bytes: genCsv(), tid: 2, label: "data/csv" },
      { name: "assets/logo.bin", bytes: genPng(rng), tid: 4, label: "image/png" },
      { name: "zzz_backup/phasemath_copy.py", bytes: phasemath, tid: 1, label: "code/py" }  // exact dup -> dedup
    ];
    drive.forEach(function (f) { f.size = f.bytes.length; f.fed = false; f.dupCount = 0; f.fedBytes = 0; f.rowEl = null; });
  }

  /* ---------------- rendering: tree, stats, gauge ---------------- */

  function renderTree() {
    var tree = $("fileTree");
    tree.innerHTML = "";
    drive.forEach(function (f) {
      var row = document.createElement("div");
      row.className = "fileRow" + (f.fed ? " fed" : "");
      var typeName = G.MAJORS[f.tid];
      row.innerHTML =
        '<span class="dot" style="background:' + TYPE_COLORS[typeName] + ";color:" + TYPE_COLORS[typeName] + '"></span>' +
        '<span class="name" title="' + esc(f.label || "") + '">' + esc(f.name) +
        (f.dropped ? ' <span class="tag drop">yours</span>' : "") +
        (f.truncated ? ' <span class="tag">first 2 MB</span>' : "") +
        (f.dupCount ? ' <span class="tag dup">deduped</span>' : "") +
        "</span>" +
        '<span class="size">' + fmtBytes(f.size) + "</span>" +
        '<div class="bar"></div>';
      f.rowEl = row;
      f.barEl = row.querySelector(".bar");
      tree.appendChild(row);
    });
  }

  function verdict(load) {
    if (load < 0.5) return "clean: attractors well separated";
    if (load < 1.0) return "edge of clean baked recall";
    if (load < 5.0) return "overloaded: cache stays exact, baked gets fuzzy";
    return "saturated: baked field is gestalt-only";
  }

  function updateStats() {
    var st = engine.stats;
    $("statFiles").textContent = engine.files.length;
    $("statBytes").textContent = fmtBytes(st.bytesSeen);
    $("statBundles").textContent = st.bundles;
    $("statEngrams").textContent = st.engrams;
    $("statDup").textContent = fmtBytes(st.dupBytes);
    $("statParams").textContent = engine.params().toLocaleString();
    var load = engine.loadFactor();
    $("statLoad").textContent = load.toFixed(2) + "\u00d7";
    $("gaugeFill").style.width = Math.min(load / 5, 1) * 100 + "%";
    $("gaugeFill").style.background = load < 1
      ? "linear-gradient(90deg,#34d399,#fbbf24)"
      : "linear-gradient(90deg,#fbbf24,#f87171)";
    $("gaugeLabel").textContent = "load " + load.toFixed(2) + "\u00d7 \u2014 " + verdict(load) +
      "  (attractor capacity \u2248 " + Math.round(engine.attractorCapacity()) + " engrams)";
    $("fieldMeta").textContent = "dim " + DIM + " \u00b7 " + engine.params().toLocaleString() + " params \u00b7 " +
      st.engrams + " engrams" + (burned ? " \u00b7 cache burned" : "");
    var canProbe = st.engrams > 0;
    $("probeBtn").disabled = !canProbe;
    Array.prototype.forEach.call(document.querySelectorAll(".chip"), function (c) { c.disabled = !canProbe; });
    $("burnBtn").disabled = !(canProbe && !burned);
  }

  /* ---------------- weight-field heatmap ---------------- */

  function buildLUTs() {
    function ramp(stops) {
      var lut = [];
      for (var i = 0; i < 256; i++) {
        var t = i / 255, k = 0;
        while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
        var a = stops[k], b = stops[k + 1];
        var f = (t - a[0]) / Math.max(1e-9, b[0] - a[0]);
        lut.push([0, 0, 0].map(function (_, c) {
          return Math.round(a[1][c] + (b[1][c] - a[1][c]) * f);
        }));
      }
      return lut;
    }
    lutPos = ramp([[0, [2, 6, 23]], [0.45, [10, 62, 122]], [0.75, [34, 211, 238]], [1, [224, 255, 255]]]);
    lutNeg = ramp([[0, [2, 6, 23]], [0.6, [90, 30, 110]], [1, [240, 171, 252]]]);
  }

  function redrawField() {
    var cv = $("fieldCanvas"), ctx = cv.getContext("2d");
    ctx.fillStyle = "#01030a";
    ctx.fillRect(0, 0, cv.width, cv.height);
    if (!engine.W) return;
    if (!fieldImg || fieldImg.width !== DIM) {
      fieldImg = ctx.createImageData(DIM, DIM);
      fieldOff = document.createElement("canvas");
      fieldOff.width = DIM; fieldOff.height = DIM;
    }
    var W = engine.W, maxAbs = 1e-12, i;
    for (i = 0; i < W.length; i++) {
      var a = Math.abs(W[i]);
      if (a > maxAbs) maxAbs = a;
    }
    var px = fieldImg.data;
    for (i = 0; i < DIM * DIM; i++) {
      var v = W[i] / maxAbs;
      var t = Math.round(Math.pow(Math.abs(v), 0.32) * 255);
      var c = (v >= 0 ? lutPos : lutNeg)[t];
      var o = i * 4;
      px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
    }
    fieldOff.getContext("2d").putImageData(fieldImg, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(fieldOff, 0, 0, DIM, DIM, 0, 0, cv.width, cv.height);
  }

  /* ---------------- constellation ---------------- */

  function buildProjection() {
    var rng = mulberry32(1234);
    projX = new Float64Array(DIM);
    projY = new Float64Array(DIM);
    var s = 1 / Math.sqrt(DIM);
    for (var i = 0; i < DIM; i++) {
      projX[i] = (rng() * 2 - 1) * s;
      projY[i] = (rng() * 2 - 1) * s;
    }
  }

  function project(e) {
    var x = 0, y = 0;
    for (var i = 0; i < DIM; i++) { x += e[i] * projX[i]; y += e[i] * projY[i]; }
    return { x: x, y: y };
  }

  function dotPos(e) {
    var cv = $("constCanvas");
    var p = project(e);
    return {
      x: cv.width / 2 + p.x * (cv.width / 2 - 30),
      y: cv.height / 2 + p.y * (cv.height / 2 - 24)
    };
  }

  function redrawConstellation(now) {
    var cv = $("constCanvas"), ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (burned) {
      ctx.fillStyle = "#64748f";
      ctx.font = "12px Consolas, monospace";
      ctx.textAlign = "center";
      ctx.fillText("phase cache burned \u2014 the engrams are gone; only the weights W, W\u1d65 remain", cv.width / 2, cv.height / 2);
      return;
    }
    var i;
    for (i = 0; i < engramDots.length; i++) {
      var d = engramDots[i];
      ctx.beginPath();
      ctx.arc(d.x, d.y, 3.2, 0, 6.2832);
      ctx.fillStyle = d.color;
      ctx.shadowColor = d.color;
      ctx.shadowBlur = 7;
      ctx.globalAlpha = 0.9;
      ctx.fill();
    }
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    if (beams && now) {
      var p = Math.min(1, (now - beams.t0) / 600);
      // query star
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(beams.q.x, beams.q.y, 6 + 3 * Math.sin(now / 150), 0, 6.2832);
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.font = "10px Consolas, monospace";
      ctx.fillText("q", beams.q.x + 10, beams.q.y - 8);
      beams.list.forEach(function (b) {
        var ex = beams.q.x + (b.x - beams.q.x) * p;
        var ey = beams.q.y + (b.y - beams.q.y) * p;
        ctx.beginPath();
        ctx.moveTo(beams.q.x, beams.q.y);
        ctx.lineTo(ex, ey);
        ctx.strokeStyle = b.color;
        ctx.lineWidth = 0.8 + b.w * 5;
        ctx.globalAlpha = 0.25 + 0.6 * b.w;
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
      if (p >= 1) beams = null;
    }
    if (!engramDots.length && !beams) {
      ctx.fillStyle = "#33415c";
      ctx.font = "12px Consolas, monospace";
      ctx.textAlign = "center";
      ctx.fillText("dots appear as bundles phase into the field", cv.width / 2, cv.height / 2);
      ctx.textAlign = "left";
    }
  }

  function animLoop(now) {
    if (beams) redrawConstellation(now);
    requestAnimationFrame(animLoop);
  }

  /* ---------------- packets (bytes flying into the field) ---------------- */

  function spawnPacket(fromEl, color) {
    if (document.hidden || packets > 40 || !fromEl) return;
    var fc = $("fieldCanvas").getBoundingClientRect();
    var fr = fromEl.getBoundingClientRect();
    var el = document.createElement("div");
    el.className = "packet";
    el.style.background = color;
    el.style.color = color;
    el.style.left = (fr.right - 8) + "px";
    el.style.top = (fr.top + fr.height / 2 - 4) + "px";
    document.body.appendChild(el);
    var dx = fc.left + fc.width * (0.3 + 0.4 * Math.random()) - fr.right + 8;
    var dy = fc.top + fc.height * (0.3 + 0.4 * Math.random()) - fr.top - fr.height / 2;
    packets++;
    requestAnimationFrame(function () {
      el.style.transform = "translate(" + dx + "px," + dy + "px) rotate(200deg) scale(.4)";
      el.style.opacity = "0.15";
    });
    setTimeout(function () { el.remove(); packets--; }, 700);
  }

  /* ---------------- feeding ---------------- */

  var FEED_MSGS = [
    "streaming \u2014 every byte is read exactly once",
    "shingling \u2014 each 4-byte window becomes a hashed coordinate",
    "phasing \u2014 the pattern writes itself into W += e\u00b7e\u1d40 \u2026 no gradients, no training",
    "folding \u2014 value heads absorb file, offset and type in the same pass",
    "the drive is becoming a field\u2026"
  ];

  function setCaption(t) { $("caption").innerHTML = t; }

  async function feedFile(f, fast) {
    var typeName = G.MAJORS[f.tid];
    var fileId0 = engine.files.length;
    for (var off = 0; off < f.bytes.length; off += CHUNK) {
      var u8 = f.bytes.subarray(off, Math.min(off + CHUNK, f.bytes.length));
      var r = engine.addBundle(f.name, off, u8, f.tid, f.size);
      f.fedBytes += u8.length;
      if (r.dup) f.dupCount++;
      if (r.stored) {
        var pos = dotPos(r.e);
        engramDots.push({ x: pos.x, y: pos.y, color: TYPE_COLORS[typeName] });
      }
      spawnPacket(f.rowEl, TYPE_COLORS[typeName]);
      if (f.barEl) f.barEl.style.width = Math.min(100, f.fedBytes / f.size * 100) + "%";
      redrawField();
      updateStats();
      await sleep(fast ? 16 : 90);
    }
    f.fed = true;
    if (f.rowEl) {
      f.rowEl.classList.add("fed");
      if (f.dupCount && f.fedBytes && f.dupCount * CHUNK >= f.size) {
        var tag = f.rowEl.querySelector(".name");
        if (tag && !f.rowEl.querySelector(".tag.dup")) {
          var t = document.createElement("span");
          t.className = "tag dup";
          t.textContent = "deduped";
          tag.appendChild(t);
        }
      }
      if (f.barEl) f.barEl.style.width = "100%";
    }
    redrawConstellation();
  }

  async function feedAll(fast) {
    if (feeding) return;
    var pending = drive.filter(function (f) { return !f.fed; });
    if (!pending.length) { setCaption("everything is already eaten. drop files, stress-feed, or probe \u2192"); return; }
    feeding = true;
    $("feedBtn").disabled = true; $("stressBtn").disabled = true;
    var totalBytes = pending.reduce(function (s, f) { return s + f.size; }, 0);
    var fedBytes = engine.stats.bytesSeen;
    for (var i = 0; i < pending.length; i++) {
      var f = pending[i];
      var frac = (engine.stats.bytesSeen - fedBytes) / Math.max(1, totalBytes);
      setCaption("[" + f.name + "] " + FEED_MSGS[Math.min(FEED_MSGS.length - 1, Math.floor(frac * FEED_MSGS.length))]);
      await feedFile(f, fast);
    }
    feeding = false;
    fedOnce = true;
    $("feedBtn").disabled = false; $("stressBtn").disabled = false;
    var load = engine.loadFactor();
    setCaption(fedOnce && load >= 1
      ? "done \u2014 but load is " + load.toFixed(2) + "\u00d7 attractor capacity. cache readouts stay exact; the baked field gets fuzzy. that's the size parameter at work."
      : "done \u2014 the drive is now a field. ask it things on the right \u2192");
    updateStats();
  }

  /* ---------------- stress feed ---------------- */

  function genStressFiles() {
    var rng = mulberry32(99);
    var words = ["coolant", "gyroscope", "phasefield", "capacitor", "telemetry", "vault", "beacon", "harmonic", "flux", "lattice"];
    var files = [];
    for (var i = 0; i < 30; i++) {
      var lines = [];
      var target = 5000 + Math.floor(rng() * 2000);
      while (lines.join("\n").length < target) {
        lines.push("status nominal. " + words[Math.floor(rng() * words.length)] +
          " loop stable. phasefield at " + Math.floor(rng() * 100) + "%.");
        lines.push("stress marker STRESS-" + ("00" + i).slice(-3) + "-" + Math.floor(rng() * 9999));
        lines.push("common boilerplate sentence shared by every stress file, on purpose.");
      }
      files.push({
        name: "stress/log_" + ("00" + i).slice(-3) + ".txt",
        bytes: strBytes(lines.join("\n")),
        tid: 0, label: "text/plain", stress: true
      });
    }
    return files;
  }

  async function stressFeed() {
    var files = genStressFiles();
    files.forEach(function (f) {
      f.size = f.bytes.length; f.fed = false; f.dupCount = 0; f.fedBytes = 0; f.rowEl = null;
      drive.push(f);
    });
    renderTree();
    setCaption("generated 30 correlated text files \u2014 now overload the field on purpose\u2026");
    await feedAll(true);
  }

  /* ---------------- probing ---------------- */

  function confClass(c) { return c > 0.6 ? "" : (c > 0.25 ? " mid" : " low"); }

  function drawStrip(canvas, vec) {
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var maxv = 1e-9;
    for (var i = 0; i < vec.length; i++) maxv = Math.max(maxv, Math.abs(vec[i]));
    var w = canvas.width / vec.length;
    for (var j = 0; j < vec.length; j++) {
      var v = Math.abs(vec[j]) / maxv;
      ctx.fillStyle = vec[j] >= 0 ? "rgba(34,211,238," + (0.15 + 0.85 * v) + ")" : "rgba(240,171,252," + (0.15 + 0.85 * v) + ")";
      ctx.fillRect(j * w, 0, Math.max(1, w - 0.5), canvas.height);
    }
  }

  function drawSpark(canvas, trace) {
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var pts = [trace[0] || 0].concat(trace);
    var max = Math.max(0.001, Math.max.apply(null, pts.map(Math.abs)));
    ctx.strokeStyle = "#a78bfa";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    pts.forEach(function (v, i) {
      var x = 8 + i * (canvas.width - 16) / Math.max(1, pts.length - 1);
      var y = canvas.height / 2 - (v / max) * (canvas.height / 2 - 6);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#64748f";
    ctx.font = "9px Consolas, monospace";
    ctx.fillText("hopfield resonance cos(q, r) per step: " +
      trace.map(function (v) { return v.toFixed(3); }).join(" \u2192 "), 8, 10);
  }

  function typeBadge(tid, conf) {
    var name = G.MAJORS[tid] || "?";
    return '<span class="typeBadge" style="color:' + TYPE_COLORS[name] + '">' + name + "</span> " +
      Math.round(conf * 100) + "%";
  }

  function fileLabel(id) {
    var f = engine.files[id];
    return f ? f.name : "?";
  }

  function renderProbe(res, desc) {
    var out = $("probeResults");
    out.innerHTML = "";
    var head = document.createElement("div");
    head.className = "qhead";
    head.innerHTML = '<b class="mono" style="color:#c9d6e8">' + esc(desc) + "</b>" +
      "<span>" + res.sampleBytes + " bytes \u2192 " + DIM + "-dim pattern</span>";
    var strip = document.createElement("canvas");
    strip.width = 256; strip.height = 12;
    strip.style.width = "128px";
    strip.title = "the query's phase pattern";
    head.appendChild(strip);
    out.appendChild(head);
    drawStrip(strip, res.q);

    var cards = [];

    if (burned) {
      var bcard = document.createElement("div");
      bcard.className = "card";
      bcard.innerHTML = '<div class="cardTitle">phase cache burned</div>' +
        '<div class="mini">E and V are gone \u2014 no exact memory, no attention. only the baked weights W, W<sub>v</sub> remain, and they still answer:</div>';
      cards.push(bcard);
    } else if (res.matches.length) {
      var mcard = document.createElement("div");
      mcard.className = "card";
      var rows = res.matches.map(function (m) {
        var typeName = G.MAJORS[m.tid] || "?";
        return '<div class="matchRow">' +
          '<span class="conf' + confClass(m.conf) + '">' + (m.conf * 100).toFixed(1) + "%</span>" +
          '<span class="mfile" style="color:' + TYPE_COLORS[typeName] + '">' + esc(fileLabel(m.file)) + "</span>" +
          '<span class="mmeta">offset ~' + fmtBytes(m.offset) + " \u00b7 " + typeName + " \u00b7 cos " + m.sim.toFixed(4) + "</span>" +
          "</div>";
      }).join("");
      mcard.innerHTML = '<div class="cardTitle">associative matches <b>E\u00b7q</b> \u2014 exact memory</div>' + rows;
      cards.push(mcard);

      var acard = document.createElement("div");
      acard.className = "card";
      acard.innerHTML = '<div class="cardTitle">attention readout <b>softmax(\u03b2E\u00b7q)\u00b7V</b> \u2014 modern hopfield, soft</div>' +
        '<div class="readValue"><span class="rvFile">' + esc(fileLabel(res.attention.file)) + "</span>" +
        '<span class="rvMeta">' + typeBadge(res.attention.tid, res.attention.conf) +
        " \u00b7 offset ~" + fmtBytes(res.attention.offset) + "</span></div>";
      cards.push(acard);
    }

    if (res.baked) {
      var bk = res.baked;
      var bkcard = document.createElement("div");
      bkcard.className = "card";
      bkcard.innerHTML = '<div class="cardTitle">baked weights readout <b>W<sub>v</sub>\u00b7q</b> \u2014 weights-only, cache not consulted</div>' +
        '<div class="readValue"><span class="rvFile">' + esc(bk.file >= 0 ? fileLabel(bk.file) : "\u2014") + "</span>" +
        '<span class="rvMeta">' + typeBadge(bk.tid, bk.conf) +
        " \u00b7 offset ~" + fmtBytes(bk.offset) +
        " \u00b7 field mass " + bk.mass.toFixed(3) + "</span></div>";
      var spark = document.createElement("canvas");
      spark.width = 480; spark.height = 40;
      spark.style.width = "100%";
      bkcard.appendChild(spark);
      cards.push(bkcard);
      setTimeout(function () { drawSpark(spark, res.resonance || []); }, 150);
    }

    cards.forEach(function (c) { out.appendChild(c); });
    cards.forEach(function (c, i) {
      setTimeout(function () { c.classList.add("show"); }, 90 + i * 140);
    });

    // beams on the constellation
    if (!burned && res.matches.length) {
      var qpos = dotPos(res.q);
      beams = {
        q: qpos,
        t0: performance.now(),
        list: res.matches.map(function (m) {
          var d = engramDots[m.engram];
          return d ? { x: d.x, y: d.y, w: m.conf, color: TYPE_COLORS[G.MAJORS[m.tid] || "other"] } : null;
        }).filter(Boolean)
      };
      redrawConstellation(performance.now());
    }
  }

  function logoBytes() {
    for (var i = 0; i < drive.length; i++) {
      if (drive[i].name === "assets/logo.bin") return drive[i].bytes;
    }
    return new Uint8Array(0);
  }

  function runProbe(sample, desc) {
    if (!engine.stats.engrams) { setCaption("feed the model first \u2014 nothing is in the weights yet."); return; }
    renderProbe(engine.probe(sample, TOPK, BETA, STEPS), desc);
  }

  function chipSample(kind) {
    if (kind === "marker") return [strBytes("launch key PHASE-KEY-ZQX-7741 must not leave"), "text: \u201claunch key PHASE-KEY-ZQX-7741 must not leave\u201d"];
    if (kind === "code") return [strBytes("def crunch_phasefield_117"), "text: \u201cdef crunch_phasefield_117\u201d"];
    if (kind === "csv") return [strBytes("customer_077"), "text: \u201ccustomer_077\u201d"];
    if (kind === "png") {
      var lg = logoBytes().subarray(9000, 9400);
      return [lg, "400 raw bytes from logo.bin @ offset 9000"];
    }
    if (kind === "noise") {
      var u8 = new Uint8Array(200);
      crypto.getRandomValues(u8);
      return [u8, "200 bytes of pure noise (expect a confident shrug)"];
    }
    return null;
  }

  /* ---------------- burn / reset ---------------- */

  function burnCache() {
    if (!engine.stats.engrams || burned) return;
    engine.burnCache();
    burned = true;
    engramDots = [];
    beams = null;
    redrawConstellation();
    $("burnBtn").textContent = "cache burned \u2014 reset to restore";
    $("burnNote").innerHTML = "the artifact is now pure weights: " +
      engine.params().toLocaleString() + " parameters. probe it anyway.";
    $("probeResults").innerHTML = '<div class="empty">cache burned. ask again \u2014 the weights alone will answer.</div>';
    updateStats();
  }

  function reset() {
    engine = new G.Engine(DIM, CHUNK);
    burned = false; fedOnce = false; feeding = false;
    engramDots = []; beams = null;
    buildDrive();
    renderTree();
    redrawField();
    redrawConstellation();
    $("probeResults").innerHTML = '<div class="empty">feed the model, then ask it things\u2026</div>';
    $("burnBtn").textContent = "\ud83d\udd25 burn the phase cache (weights-only mode)";
    $("burnBtn").disabled = true;
    $("burnNote").textContent = "after burning, only W and W\u1d65 can answer";
    $("feedBtn").disabled = false; $("stressBtn").disabled = false;
    setCaption("a synthetic drive is loaded. feed it and watch the bytes become weights.");
    updateStats();
  }

  /* ---------------- drag & drop real files ---------------- */

  async function handleFiles(fileList) {
    var files = Array.prototype.slice.call(fileList);
    for (var i = 0; i < files.length; i++) {
      var f = files[i];
      var blob = f.size > MAX_DROP ? f.slice(0, MAX_DROP) : f;
      var buf = await blob.arrayBuffer();
      var bytes = new Uint8Array(buf);
      var td = detectType(bytes.subarray(0, 512), f.name);
      var entry = {
        name: f.name, bytes: bytes, size: f.size, tid: td.tid, label: td.label,
        fed: false, dupCount: 0, fedBytes: 0, rowEl: null, barEl: null,
        dropped: true, truncated: f.size > MAX_DROP
      };
      drive.push(entry);
      renderTree();
      if (fedOnce) {
        setCaption("[" + f.name + "] feeding your real file into the field\u2026");
        await feedFile(entry, true);
        setCaption("done \u2014 " + esc(f.name) + " is in the weights. probe it: paste a phrase from it.");
      } else {
        setCaption("added " + esc(f.name) + " (" + fmtBytes(bytes.length) + " of real bytes) \u2014 press \u26a1 feed the model");
      }
    }
  }

  /* ---------------- memorizer: bytes -> weights, literally ---------------- */

  var MEM = { running: false, H: 32, BLOCK: 8, POS: 12, LR: 0.03, MOM: 0.9, MAX_STEPS: 1500 };

  function memTarget() {
    var s = "PHASE-KEY-ZQX-7741 opens the vault of weights. bytes into parameters. ";
    var t = strBytes(s), out = new Uint8Array(96);
    for (var i = 0; i < 96; i++) out[i] = t[i % t.length];
    return out;
  }

  function heatColor(v) {
    var t = v / 255;
    var r = Math.round(20 + t * 220), g = Math.round(30 + t * 170), b = Math.round(70 + t * 40);
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function drawMemStrip(canvas, bytes, pred) {
    var ctx = canvas.getContext("2d");
    var w = canvas.width / 96;
    for (var i = 0; i < 96; i++) {
      var v = pred ? Math.max(0, Math.min(255, Math.round(pred[i] * 255))) : bytes[i];
      ctx.fillStyle = heatColor(bytes[i]);
      ctx.fillRect(i * w, 0, Math.max(1, w - 0.6), canvas.height);
      if (pred) {
        var exact = v === bytes[i];
        ctx.fillStyle = exact ? "rgba(52,211,153,.85)" : "rgba(248,113,113,.85)";
        ctx.fillRect(i * w, canvas.height - 3, Math.max(1, w - 0.6), 3);
      }
    }
  }

  function drawMemLoss(canvas, hist) {
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (hist.length < 2) return;
    var max = hist[0] || 1e-9;
    ctx.strokeStyle = "#a78bfa";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    hist.forEach(function (v, i) {
      var x = i / (hist.length - 1) * (canvas.width - 8) + 4;
      var y = canvas.height - 4 - (Math.log(Math.max(1e-9, v) + 1e-9) / Math.log(Math.max(1e-9, max) + 1e-9)) * (canvas.height - 10);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#64748f";
    ctx.font = "9px Consolas, monospace";
    ctx.fillText("log loss", 6, 10);
  }

  function runMemorizer() {
    if (MEM.running) return;
    MEM.running = true;
    var rng = mulberry32(5);
    var H = MEM.H, B = MEM.BLOCK, P = MEM.POS;
    var target = memTarget();
    var w1 = new Float64Array(H), b1 = new Float64Array(H);
    var W2 = new Float64Array(H * H), b2 = new Float64Array(H);
    var W3 = new Float64Array(B * H), b3 = new Float64Array(B);
    var vw1 = new Float64Array(H), vb1 = new Float64Array(H);
    var vW2 = new Float64Array(H * H), vb2 = new Float64Array(H);
    var vW3 = new Float64Array(B * H), vb3 = new Float64Array(B);
    var i, j, k;
    for (i = 0; i < H; i++) { w1[i] = (rng() * 2 - 1) * 24; b1[i] = (rng() * 2 - 1) * 3; }
    for (i = 0; i < H * H; i++) W2[i] = (rng() * 2 - 1) * 0.5;
    for (i = 0; i < B * H; i++) W3[i] = (rng() * 2 - 1) * 0.5;

    var ys = new Float64Array(P * B);
    for (i = 0; i < P; i++) for (k = 0; k < B; k++) ys[i * B + k] = target[i * B + k] / 255;

    var hist = [], step = 0;
    var nParams = H * 2 + H * H + H + B * H + B;

    function trainStep() {
      // zero grads
      var gw1 = new Float64Array(H), gb1 = new Float64Array(H);
      var gW2 = new Float64Array(H * H), gb2 = new Float64Array(H);
      var gW3 = new Float64Array(B * H), gb3 = new Float64Array(B);
      var loss = 0, exact = 0;
      for (var p = 0; p < P; p++) {
        var t = p / P;
        var h1 = new Float64Array(H), z1 = new Float64Array(H);
        for (var i1 = 0; i1 < H; i1++) { z1[i1] = w1[i1] * t + b1[i1]; h1[i1] = Math.sin(z1[i1]); }
        var h2 = new Float64Array(H);
        for (var j1 = 0; j1 < H; j1++) {
          var a2 = b2[j1];
          for (var i2 = 0; i2 < H; i2++) a2 += W2[j1 * H + i2] * h1[i2];
          h2[j1] = Math.tanh(a2);
        }
        var out = new Float64Array(B);
        for (var k1 = 0; k1 < B; k1++) {
          var a3 = b3[k1];
          for (var j2 = 0; j2 < H; j2++) a3 += W3[k1 * H + j2] * h2[j2];
          out[k1] = a3;
          var y = ys[p * B + k1];
          var d = out[k1] - y;
          loss += d * d;
          if (Math.round(Math.max(0, Math.min(1, out[k1])) * 255) === target[p * B + k1]) exact++;
        }
        // backprop
        var gout = new Float64Array(B);
        for (k1 = 0; k1 < B; k1++) {
          gout[k1] = 2 * (out[k1] - ys[p * B + k1]) / (P * B);
          gb3[k1] += gout[k1];
          for (j2 = 0; j2 < H; j2++) { gW3[k1 * H + j2] += gout[k1] * h2[j2]; }
        }
        var gh2 = new Float64Array(H), gz2 = new Float64Array(H);
        for (j2 = 0; j2 < H; j2++) {
          for (k1 = 0; k1 < B; k1++) gh2[j2] += W3[k1 * H + j2] * gout[k1];
          gz2[j2] = gh2[j2] * (1 - h2[j2] * h2[j2]);
          gb2[j2] += gz2[j2];
          for (i2 = 0; i2 < H; i2++) gW2[j2 * H + i2] += gz2[j2] * h1[i2];
        }
        var gh1 = new Float64Array(H);
        for (i2 = 0; i2 < H; i2++) {
          for (j2 = 0; j2 < H; j2++) gh1[i2] += W2[j2 * H + i2] * gz2[j2];
          var gz1 = gh1[i2] * Math.cos(z1[i2]);
          gw1[i2] += gz1 * t;
          gb1[i2] += gz1;
        }
      }
      // momentum SGD
      function upd(w, v, g, lr) {
        for (var q = 0; q < w.length; q++) {
          v[q] = MEM.MOM * v[q] - lr * g[q];
          w[q] += v[q];
        }
      }
      upd(w1, vw1, gw1, MEM.LR); upd(b1, vb1, gb1, MEM.LR);
      upd(W2, vW2, gW2, MEM.LR); upd(b2, vb2, gb2, MEM.LR);
      upd(W3, vW3, gW3, MEM.LR); upd(b3, vb3, gb3, MEM.LR);
      return { loss: loss / (P * B), exact: exact / (P * B) };
    }

    function predictAll() {
      var pred = new Float64Array(96);
      for (var p = 0; p < P; p++) {
        var t = p / P;
        var h1z = new Float64Array(H);
        for (var i1 = 0; i1 < H; i1++) h1z[i1] = Math.sin(w1[i1] * t + b1[i1]);
        var h2z = new Float64Array(H);
        for (var j1 = 0; j1 < H; j1++) {
          var a = b2[j1];
          for (var i2 = 0; i2 < H; i2++) a += W2[j1 * H + i2] * h1z[i2];
          h2z[j1] = Math.tanh(a);
        }
        for (var k1 = 0; k1 < B; k1++) {
          var a3 = b3[k1];
          for (var j2 = 0; j2 < H; j2++) a3 += W3[k1 * H + j2] * h2z[j2];
          pred[p * B + k1] = Math.max(0, Math.min(1, a3));
        }
      }
      return pred;
    }

    drawMemStrip($("memTarget"), target, null);
    drawMemStrip($("memPred"), target, new Float64Array(96));
    $("memStats").textContent = "96 bytes \u00b7 " + nParams.toLocaleString() + " params \u00b7 gradient descent\u2026";

    function frame() {
      for (var s = 0; s < 6 && step < MEM.MAX_STEPS; s++, step++) {
        var r = trainStep();
        if (step % 12 === 0) hist.push(r.loss);
        if (step % 24 === 0) {
          $("memStats").textContent = "step " + step + " \u00b7 mse " + r.loss.toFixed(5) +
            " \u00b7 byte-exact " + (r.exact * 100).toFixed(1) + "% \u00b7 " + nParams.toLocaleString() + " params";
          drawMemStrip($("memPred"), target, predictAll());
          drawMemLoss($("memLoss"), hist);
        }
      }
      if (step < MEM.MAX_STEPS) requestAnimationFrame(frame);
      else {
        var fin = predictAll();
        var ex = 0;
        for (var q = 0; q < 96; q++) if (Math.round(fin[q] * 255) === target[q]) ex++;
        drawMemStrip($("memPred"), target, fin);
        drawMemLoss($("memLoss"), hist);
        $("memStats").textContent = "done \u00b7 byte-exact " + ((ex / 96) * 100).toFixed(1) +
          "% \u00b7 96 bytes stored in " + nParams.toLocaleString() + " parameters \u00b7 " + step + " gradient steps";
        MEM.running = false;
      }
    }
    requestAnimationFrame(frame);
  }

  /* ---------------- wiring ---------------- */

  function wire() {
    $("feedBtn").addEventListener("click", function () { feedAll(false); });
    $("stressBtn").addEventListener("click", function () { stressFeed(); });
    $("resetBtn").addEventListener("click", reset);
    $("burnBtn").addEventListener("click", burnCache);
    $("probeBtn").addEventListener("click", function () {
      var v = $("probeText").value;
      if (v) runProbe(new TextEncoder().encode(v), "text: \u201c" + v + "\u201d");
    });
    $("probeText").addEventListener("keydown", function (e) {
      if (e.key === "Enter") $("probeBtn").click();
    });
    Array.prototype.forEach.call(document.querySelectorAll(".chip"), function (chip) {
      chip.addEventListener("click", function () {
        var s = chipSample(chip.getAttribute("data-probe"));
        if (s) runProbe(s[0], s[1]);
      });
    });
    var dz = $("dropZone"), fi = $("fileInput");
    dz.addEventListener("click", function () { fi.click(); });
    fi.addEventListener("change", function () { handleFiles(fi.files); fi.value = ""; });
    ["dragenter", "dragover"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("dragover"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("dragover"); });
    });
    dz.addEventListener("drop", function (e) { handleFiles(e.dataTransfer.files); });
    $("memBtn").addEventListener("click", runMemorizer);
    window.addEventListener("dragover", function (e) { e.preventDefault(); });
    window.addEventListener("drop", function (e) { e.preventDefault(); });
  }

  /* ---------------- boot ---------------- */

  function boot() {
    runParity();
    buildLUTs();
    buildProjection();
    reset();
    wire();
    requestAnimationFrame(animLoop);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
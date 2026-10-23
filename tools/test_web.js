/* Headless parity test: run the JS engine on the exact corpus the Python
 * engine produced web/parity.js from, and compare numbers.
 * Run:  node tools/test_web.js
 */
"use strict";

const PARITY = require("../web/parity.js");
const G = require("../web/engine.js");

const b64 = (s) => Uint8Array.from(Buffer.from(s, "base64"));
const textA = b64(PARITY.textA);
const randB = b64(PARITY.randB);
const marker = b64(PARITY.marker);

let fails = 0;
function check(name, cond, detail) {
  const ok = !!cond;
  if (!ok) fails++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  (" + detail + ")" : ""}`);
}

const eng = new G.Engine(PARITY.dim, PARITY.chunk);
eng.addBundle("textA.txt", 0, textA, 0, textA.length);
eng.addBundle("randB.bin", 0, randB, 10, randB.length);

// 1. embedding parity
const ea = G.embedBytes(textA, PARITY.dim);
let maxd = 0;
for (let i = 0; i < PARITY.dim; i++) {
  maxd = Math.max(maxd, Math.abs(ea[i] - PARITY.embedA[i]));
}
check("embed parity (96 dims)", maxd < 1e-4, "maxdiff=" + maxd.toExponential(2));

// 2. probe parity
const r = eng.probe(marker, 2, 20.0, 2);
check("top-1 sim", Math.abs(r.matches[0].sim - PARITY.sims[0]) < 1e-3,
  r.matches[0].sim + " vs " + PARITY.sims[0]);
check("top-2 sim", Math.abs(r.matches[1].sim - PARITY.sims[1]) < 1e-3,
  r.matches[1].sim + " vs " + PARITY.sims[1]);
check("match files", r.matches[0].file === PARITY.matchFiles[0] &&
  r.matches[1].file === PARITY.matchFiles[1],
  r.matches.map(m => m.file).join(",") + " vs " + PARITY.matchFiles.join(","));
check("attention file", r.attention.file === PARITY.attention.file);
check("attention type", G.MAJORS[r.attention.tid] === PARITY.attention.type);
check("attention offset", r.attention.offset === PARITY.attention.offset,
  r.attention.offset + " vs " + PARITY.attention.offset);
check("attention conf", Math.abs(r.attention.conf - PARITY.attention.conf) < 0.02,
  r.attention.conf.toFixed(4) + " vs " + PARITY.attention.conf);
check("baked file", r.baked.file === PARITY.baked.file);
check("baked type", G.MAJORS[r.baked.tid] === PARITY.baked.type);
check("baked offset", Math.abs(r.baked.offset - PARITY.baked.offset) <= 8,
  r.baked.offset + " vs " + PARITY.baked.offset);
check("baked mass", Math.abs(r.baked.mass - PARITY.baked.mass) < 1e-2,
  r.baked.mass.toFixed(4) + " vs " + PARITY.baked.mass);
check("resonance trace", r.resonance.every((v, i) =>
  Math.abs(v - PARITY.resonance[i]) < 1e-2),
  r.resonance.map(v => v.toFixed(3)).join(" -> ") +
  " vs " + PARITY.resonance.map(v => v.toFixed(3)).join(" -> "));

// 3. structural sanity
check("engrams stored", eng.stats.engrams === 2);
check("params accounting", eng.params() === PARITY.dim * PARITY.dim + 15 * PARITY.dim);
eng.burnCache();
const r2 = eng.probe(marker, 2, 20.0, 2);
check("burned cache hides matches", r2.matches.length === 0 && r2.attention === null);
check("burned cache keeps baked readout", r2.baked !== null && r2.baked.file === PARITY.baked.file);

console.log(fails === 0 ? "\nALL PARITY CHECKS PASSED -- JS engine == Python engine"
  : `\n${fails} CHECK(S) FAILED`);
process.exit(fails ? 1 : 0);
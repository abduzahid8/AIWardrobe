/**
 * End-to-end test of the FLUX-only try-on pipeline.
 *
 * Bypasses Express auth and calls applyGarmentStep directly.
 *
 * Usage:
 *   FLUX_PROVIDER=nvidia_local FLUX_LOCAL_URL=http://localhost:8000/v1/infer \
 *     node scripts/test-tryon-flux-only.mjs [garment_path] [label]
 *
 * Defaults:
 *   garment_path = first PNG in cache/garments
 *   label        = top
 *
 * Outputs (under scripts/out/):
 *   tryon-input-mannequin.png        the mannequin we started from
 *   tryon-input-garment.png          the cleaned garment
 *   tryon-flux-raw-step1.png         FLUX's raw side-by-side response
 *   tryon-final-step1.png            final dressed mannequin (after merge)
 *   tryon-mask-step1.png             diff mask used for merge
 *   tryon-metrics.json               coverage + timing + preservation stats
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';
import 'dotenv/config';

import {
  buildBaseCanvas,
  encodeCanvas,
  encodeBasePxToDataUri,
  preprocessGarmentAndCache,
} from '../api/services/tryonRenderer.js';
import {
  toDataUri,
  loadImageBuffer,
  callFluxKontext,
  extractFluxLeftHalfRaw,
  computeDiffMask,
  dilateAndFeatherMask,
  maskMergeFluxIntoBase,
  W,
  H,
} from '../../backend/api/services/tryonShared.js';
import { buildKontextComposite, buildDressingPrompt } from '../../backend/api/routes/tryon.js';

const OUT_DIR = path.resolve('scripts/out');

async function ensureDir(p) {
  await fs.mkdir(p, { recursive: true });
}

async function dumpPng(filename, dataUriOrBuf) {
  const out = path.join(OUT_DIR, filename);
  let buf;
  if (Buffer.isBuffer(dataUriOrBuf)) {
    buf = dataUriOrBuf;
  } else if (typeof dataUriOrBuf === 'string' && dataUriOrBuf.startsWith('data:')) {
    buf = Buffer.from(dataUriOrBuf.split(',')[1], 'base64');
  } else if (dataUriOrBuf instanceof Uint8ClampedArray || dataUriOrBuf instanceof Uint8Array) {
    buf = Buffer.from(dataUriOrBuf);
  } else {
    throw new Error(`dumpPng: unsupported value for ${filename}`);
  }
  await fs.writeFile(out, buf);
  return out;
}

async function rawRGBAtoPng(rgba, w, h) {
  return sharp(Buffer.from(rgba), { raw: { width: w, height: h, channels: 4 } }).png().toBuffer();
}

async function maskToPng(mask, w, h) {
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const m = mask[i];
    rgba[i * 4] = m;
    rgba[i * 4 + 1] = m;
    rgba[i * 4 + 2] = m;
    rgba[i * 4 + 3] = 255;
  }
  return rawRGBAtoPng(rgba, w, h);
}

function preservationStats(preStep, basePx, mask) {
  // For pixels OUTSIDE the merge mask: how identical are they to the original?
  // basePx outside the mask SHOULD equal preStep exactly (that's the whole point).
  let outsideTotal = 0;
  let outsideExactMatch = 0;
  let outsideMaxDelta = 0;
  let insideTotal = 0;
  for (let i = 0; i < W * H; i++) {
    const inMask = mask[i] > 0;
    const j = i * 4;
    if (inMask) {
      insideTotal += 1;
      continue;
    }
    outsideTotal += 1;
    const dr = Math.abs(preStep[j] - basePx[j]);
    const dg = Math.abs(preStep[j + 1] - basePx[j + 1]);
    const db = Math.abs(preStep[j + 2] - basePx[j + 2]);
    const max = Math.max(dr, dg, db);
    if (max === 0) outsideExactMatch += 1;
    if (max > outsideMaxDelta) outsideMaxDelta = max;
  }
  return {
    outsideTotal,
    outsideExactMatchPct: outsideTotal ? (outsideExactMatch / outsideTotal) * 100 : 0,
    outsideMaxDelta,
    insideTotal,
    insidePct: (insideTotal / (W * H)) * 100,
  };
}

async function pickDefaultGarment() {
  const cacheDir = path.resolve('cache/garments');
  try {
    const files = (await fs.readdir(cacheDir)).filter((f) => f.endsWith('.png'));
    if (files.length) return path.join(cacheDir, files[0]);
  } catch {}
  // fallback to a basic asset
  return path.resolve('assets/images/basic_brown_pants.png');
}

async function main() {
  await ensureDir(OUT_DIR);
  const argGarment = process.argv[2];
  const argLabel = process.argv[3] || 'top';

  const mannequinPath = path.resolve('assets/images/mannequin_front.png');
  const garmentPath = argGarment ? path.resolve(argGarment) : await pickDefaultGarment();
  console.log('mannequin :', mannequinPath);
  console.log('garment   :', garmentPath);
  console.log('label     :', argLabel);
  console.log('provider  :', process.env.FLUX_PROVIDER || 'nvidia_local');
  console.log('flux url  :', process.env.FLUX_LOCAL_URL || 'http://localhost:8000/v1/infer');

  const mannequinBuf = await fs.readFile(mannequinPath);
  const garmentBuf = await fs.readFile(garmentPath);
  const mannequinDataUri = toDataUri(mannequinBuf, 'image/png');
  const garmentDataUri = toDataUri(garmentBuf, 'image/png');

  await dumpPng('tryon-input-mannequin.png', mannequinDataUri);

  // 1) Build base canvas (W×H RGBA buffer of the mannequin)
  const t0 = Date.now();
  const basePx = await buildBaseCanvas(mannequinDataUri);
  const preStep = new Uint8ClampedArray(basePx);

  // 2) Preprocess garment (background removal + crop)
  let cleanedGarmentDataUri = garmentDataUri;
  try {
    const cleaned = await preprocessGarmentAndCache(garmentDataUri, argLabel);
    cleanedGarmentDataUri = toDataUri(cleaned, 'image/png');
    await dumpPng('tryon-input-garment.png', cleaned);
  } catch (e) {
    console.warn('preprocess fallback:', e.message);
    await dumpPng('tryon-input-garment.png', garmentBuf);
  }

  // 3) Build side-by-side composite [mannequin | garment]
  const mannequinForFlux = await encodeBasePxToDataUri(basePx);
  const composite = await buildKontextComposite(mannequinForFlux, cleanedGarmentDataUri);
  await dumpPng('tryon-flux-input-composite.png', composite);

  // 4) Call FLUX
  const promptText = buildDressingPrompt(argLabel);
  console.log('\nCalling FLUX.1-Kontext-dev...');
  const fluxStart = Date.now();
  let fluxResult;
  try {
    fluxResult = await callFluxKontext({
      imageDataUri: composite,
      prompt: promptText,
      provider: process.env.FLUX_PROVIDER || 'nvidia_local',
    });
  } catch (e) {
    console.error('FLUX call failed:', e.message);
    console.error('\nIs your local NIM running at',
      process.env.FLUX_LOCAL_URL || 'http://localhost:8000/v1/infer', '?');
    process.exit(1);
  }
  const fluxMs = Date.now() - fluxStart;
  console.log(`FLUX returned in ${fluxMs}ms`);
  await dumpPng('tryon-flux-raw-step1.png', fluxResult);

  // 5) Extract left half + diff mask
  const fluxRaw = await extractFluxLeftHalfRaw(fluxResult);
  if (!fluxRaw) {
    console.error('Could not decode FLUX output');
    process.exit(1);
  }

  const { mask, coverage, drifted } = computeDiffMask(preStep, fluxRaw, {
    threshold: 14,
    maxCoverage: 0.85,
  });
  console.log(`diff mask coverage: ${(coverage * 100).toFixed(2)}%  drifted=${drifted}`);
  await dumpPng('tryon-mask-step1.png', await maskToPng(mask, W, H));

  if (drifted) {
    console.error('FLUX drifted too much — refusing to ship. Inspect tryon-flux-raw-step1.png.');
    await fs.writeFile(path.join(OUT_DIR, 'tryon-metrics.json'), JSON.stringify({
      drifted: true, coverage, fluxMs, totalMs: Date.now() - t0,
    }, null, 2));
    process.exit(2);
  }

  // 6) Dilate + feather, then mask-merge
  const mergeMask = await dilateAndFeatherMask(mask, 8, 4);
  maskMergeFluxIntoBase(basePx, preStep, fluxRaw, mergeMask);

  // 7) Encode final
  const encoded = await encodeCanvas(basePx, argLabel);
  await dumpPng('tryon-final-step1.png', encoded.imageDataUri);

  const stats = preservationStats(preStep, basePx, mergeMask);
  const totalMs = Date.now() - t0;

  const metrics = {
    totalMs,
    fluxMs,
    coveragePct: +(coverage * 100).toFixed(2),
    insideMaskPct: +stats.insidePct.toFixed(2),
    outsideMaskExactMatchPct: +stats.outsideMaskExactMatchPct.toFixed(4),
    outsideMaskMaxChannelDelta: stats.outsideMaxDelta,
    drifted: false,
    label: argLabel,
    provider: process.env.FLUX_PROVIDER || 'nvidia_local',
  };
  await fs.writeFile(path.join(OUT_DIR, 'tryon-metrics.json'), JSON.stringify(metrics, null, 2));

  console.log('\n=== METRICS ===');
  console.log(JSON.stringify(metrics, null, 2));
  console.log('\nArtifacts in scripts/out/:');
  console.log('  tryon-input-mannequin.png        original mannequin');
  console.log('  tryon-input-garment.png          cleaned garment');
  console.log('  tryon-flux-input-composite.png   what we sent to FLUX');
  console.log('  tryon-flux-raw-step1.png         raw FLUX response (full SBS)');
  console.log('  tryon-mask-step1.png             diff mask used for merge');
  console.log('  tryon-final-step1.png            FINAL dressed mannequin');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

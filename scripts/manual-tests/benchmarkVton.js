#!/usr/bin/env node
/**
 * Multi-garment VTON benchmark runner.
 *
 * Usage:
 *   node scripts/benchmarkVton.js \
 *     --person <path_or_url> \
 *     --garments <g1_path> <g2_path> ... \
 *     --iterations 5 \
 *     --endpoint http://localhost:8001
 *
 * Compares sequential_v1 vs fused_v2 on the same outfit,
 * reports p50/p95 latency, cache hit rates, and VRAM usage.
 */

import fs from 'node:fs';
import path from 'node:path';

const API_URL = process.env.MOBILE_VTON_URL || 'http://localhost:8001';

function toBase64DataUri(filePath) {
  const ext = path.extname(filePath).slice(1) || 'png';
  const mime = ext === 'jpg' || ext === 'jpeg' ? 'image/jpeg' : 'image/png';
  const buf = fs.readFileSync(filePath);
  return `data:${mime};base64,${buf.toString('base64')}`;
}

function toBase64DataUriFromUrl(url) {
  if (url.startsWith('data:') || url.startsWith('http')) return url;
  return toBase64DataUri(url);
}

async function runTrial(endpoint, body) {
  const res = await fetch(`${API_URL}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json();
}

function percentile(sorted, p) {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

function report(name, values, diagnosticsList) {
  const sorted = [...values].sort((a, b) => a - b);
  const cacheHits = diagnosticsList
    .map((d) => d?.cacheHits || {})
    .reduce((acc, hits) => {
      for (const [k, v] of Object.entries(hits)) acc[k] = (acc[k] || 0) + v;
      return acc;
    }, {});
  const cacheMisses = diagnosticsList
    .map((d) => d?.cacheMisses || {})
    .reduce((acc, misses) => {
      for (const [k, v] of Object.entries(misses)) acc[k] = (acc[k] || 0) + v;
      return acc;
    }, {});
  const peakVrams = diagnosticsList
    .map((d) => d?.peakVramMb)
    .filter((v) => v != null);

  console.log(`\n=== ${name} ===`);
  console.log(`  trials: ${sorted.length}`);
  console.log(`  p50:  ${percentile(sorted, 50).toFixed(1)}ms`);
  console.log(`  p95:  ${percentile(sorted, 95).toFixed(1)}ms`);
  console.log(`  p99:  ${percentile(sorted, 99).toFixed(1)}ms`);
  console.log(`  min:  ${sorted[0].toFixed(1)}ms`);
  console.log(`  max:  ${sorted[sorted.length - 1].toFixed(1)}ms`);
  console.log(`  cacheHits:  ${JSON.stringify(cacheHits)}`);
  console.log(`  cacheMisses: ${JSON.stringify(cacheMisses)}`);
  if (peakVrams.length) {
    console.log(`  peakVRAM p50: ${percentile([...peakVrams].sort((a, b) => a - b), 50).toFixed(0)}MB`);
  }
}

async function main() {
  const args = process.argv.slice(2);
  const personIdx = args.indexOf('--person');
  const garmentsIdx = args.indexOf('--garments');
  const iterationsIdx = args.indexOf('--iterations');
  const endpointIdx = args.indexOf('--endpoint');

  if (personIdx === -1 || garmentsIdx === -1) {
    console.error('Usage: node benchmarkVton.js --person <image> --garments <g1> <g2> ... [--iterations N] [--endpoint URL]');
    process.exit(1);
  }

  const personImage = toBase64DataUriFromUrl(args[personIdx + 1]);
  const garmentPaths = [];
  for (let i = garmentsIdx + 1; i < args.length; i++) {
    if (args[i].startsWith('--')) break;
    garmentPaths.push(args[i]);
  }

  const iterations = iterationsIdx !== -1 ? parseInt(args[iterationsIdx + 1], 10) : 3;
  const apiUrl = endpointIdx !== -1 ? args[endpointIdx + 1] : API_URL;
  process.env.MOBILE_VTON_URL = apiUrl;

  const garments = garmentPaths.map((p, i) => ({
    garment_image: toBase64DataUriFromUrl(p),
    description: `Garment ${i + 1}`,
    label: ['top', 'layer', 'pants', 'shoes'][i % 4],
  }));

  console.log(`Benchmarking ${garments.length} garments, ${iterations} iterations each`);
  console.log(`API: ${apiUrl}`);

  // Warm-up (not counted)
  console.log('\n[Warm-up] sequential_v1...');
  await runTrial('/tryon/multi', {
    person_image: personImage,
    garments,
    num_inference_steps: 10,
    guidance_scale: 2.0,
  });

  console.log('[Warm-up] fused_v2...');
  await runTrial('/tryon/multi-fused', {
    person_image: personImage,
    garments,
    num_inference_steps: 10,
    guidance_scale: 2.0,
    pipeline_version: 'fused_v2',
  });

  // Sequential benchmark
  const seqMs = [];
  const seqDiags = [];
  console.log('\n[Benchmark] sequential_v1...');
  for (let i = 0; i < iterations; i++) {
    const data = await runTrial('/tryon/multi', {
      person_image: personImage,
      garments,
      num_inference_steps: 10,
      guidance_scale: 2.0,
    });
    seqMs.push(data.elapsed_ms);
    seqDiags.push(data.diagnostics || null);
    process.stdout.write(`  trial ${i + 1}/${iterations}: ${data.elapsed_ms.toFixed(0)}ms\n`);
  }

  // Fused benchmark
  const fusedMs = [];
  const fusedDiags = [];
  console.log('[Benchmark] fused_v2...');
  for (let i = 0; i < iterations; i++) {
    const data = await runTrial('/tryon/multi-fused', {
      person_image: personImage,
      garments,
      num_inference_steps: 10,
      guidance_scale: 2.0,
      pipeline_version: 'fused_v2',
    });
    fusedMs.push(data.elapsed_ms);
    fusedDiags.push(data.diagnostics || null);
    process.stdout.write(`  trial ${i + 1}/${iterations}: ${data.elapsed_ms.toFixed(0)}ms\n`);
  }

  report('sequential_v1', seqMs, seqDiags);
  report('fused_v2', fusedMs, fusedDiags);

  const speedup = percentile([...seqMs].sort((a, b) => a - b), 50) /
                  percentile([...fusedMs].sort((a, b) => a - b), 50);
  console.log(`\n>>> Speedup (p50): ${speedup.toFixed(2)}x`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

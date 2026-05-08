/**
 * Test the per-garment diff-mask pipeline end-to-end.
 * Mocks callFluxKontext to return a slightly shifted mannequin,
 * so the diff-mask + snap-back logic is exercised.
 *
 * Run: node scripts/test-perstep-pipeline.js
 */

import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'url';
import sharp from 'sharp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

// ── Mock fetch so callFluxKontext returns a perturbed mannequin ──
const originalFetch = global.fetch;

global.fetch = async function patchedFetch(url, init) {
  const bodyStr = typeof init?.body === 'string' ? init.body : '';
  const body = JSON.parse(bodyStr || '{}');

  // Save the composite and prompt for inspection
  if (body.image?.startsWith('data:image/png;base64,')) {
    const p = path.join(ROOT, 'scratch', '_perstep_flux_input.png');
    fs.writeFileSync(p, Buffer.from(body.image.split(',')[1], 'base64'));
    console.log('[MOCK] saved composite →', p);
  }
  if (body.prompt) {
    const p = path.join(ROOT, 'scratch', '_perstep_flux_prompt.txt');
    fs.writeFileSync(p, body.prompt);
    console.log('[MOCK] saved prompt   →', p);
  }

  // Return a SLIGHTLY shifted mannequin (simulating FLUX drift)
  // This tests whether the diff-mask correctly identifies only changed pixels
  const mannequinBuf = fs.readFileSync(path.join(ROOT, 'assets', 'images', 'mannequin_front.png'));
  const shifted = await sharp(mannequinBuf)
    .resize(1024, 1024, { fit: 'fill' })
    .modulate({ brightness: 1.02 })
    .png()
    .toBuffer();

  return {
    ok: true,
    status: 200,
    async text() {
      return JSON.stringify({ artifacts: [{ base64: shifted.toString('base64') }] });
    },
    async json() {
      return { artifacts: [{ base64: shifted.toString('base64') }] };
    },
  };
};

// ── Import after mocking fetch ──
const { default: tryonRouter, applyGarmentStep } = await import(
  path.join(ROOT, 'api', 'routes', 'tryon.js')
);

const {
  buildBaseCanvas,
  encodeBasePxToDataUri,
  encodeCanvas,
} = await import(path.join(ROOT, 'api', 'services', 'tryonRenderer.js'));

// ── Load test images ──
const assetsDir = path.join(ROOT, 'assets', 'images');
const mannequinBuf = fs.readFileSync(path.join(assetsDir, 'mannequin_front.png'));
const mannequinB64 = `data:image/png;base64,${mannequinBuf.toString('base64')}`;

const garmentFiles = fs
  .readdirSync(assetsDir)
  .filter((f) => f.endsWith('.png') && !f.includes('mannequin') && !f.includes('icon') && !f.includes('splash') && !f.includes('logo'))
  .slice(0, 2);

if (garmentFiles.length < 1) {
  console.error('❌ No garment images in', assetsDir);
  process.exit(1);
}

console.log('Garments:', garmentFiles);

// ── Run two-step outfit (top + pants) ──
(async () => {
  const started = Date.now();

  const basePx = await buildBaseCanvas(mannequinB64);
  const stepLabels = [];

  for (const [idx, file] of garmentFiles.entries()) {
    const label = idx === 0 ? 'top' : 'pants';
    const garmentSrc = `data:image/png;base64,${fs.readFileSync(path.join(assetsDir, file)).toString('base64')}`;

    console.log(`\n--- Step ${idx + 1}: ${label} (${file}) ---`);
    const result = await applyGarmentStep(basePx, garmentSrc, label);
    stepLabels.push(result.label);
    console.log(`  coverage: ${(result.coverage * 100).toFixed(1)}%  fluxApplied: ${result.fluxApplied}`);
  }

  const { imageDataUri } = await encodeCanvas(basePx, stepLabels[stepLabels.length - 1]);
  const outPath = path.join(ROOT, 'scratch', '_perstep_final_output.png');
  fs.writeFileSync(
    outPath,
    Buffer.from(imageDataUri.split(',')[1], 'base64')
  );

  console.log('\n✅ Saved final output →', outPath, `(${(fs.statSync(outPath).size / 1024).toFixed(1)} KB)`);
  console.log('Pipeline elapsed (mock):', Date.now() - started, 'ms');
  console.log('\n--- Test complete ---');
  console.log('Inspect:');
  console.log('  scratch/_perstep_flux_input.png   — side-by-side composite sent to FLUX');
  console.log('  scratch/_perstep_flux_prompt.txt  — dressing prompt');
  console.log('  scratch/_perstep_final_output.png   — result after diff-mask snap-back');

  global.fetch = originalFetch;
})();

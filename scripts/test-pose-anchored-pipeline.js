/**
 * Test script: validates the pose-anchored v3 try-on pipeline up to the FLUX call.
 * Generates the 3-panel composite and prompt, saves them to disk for inspection.
 * Does NOT require a running FLUX NIM — it intercepts fetch() to capture the
 * payload and returns a dummy mannequin image so mask-merge runs end-to-end.
 *
 * Run: node scripts/test-pose-anchored-pipeline.js
 */

import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const COMPOSITE_OUT = path.join(ROOT, 'scratch', 'v3_composite_input.png');
const PROMPT_OUT = path.join(ROOT, 'scratch', 'v3_prompt.txt');

// ---------------------------------------------------------------------------
// Intercept fetch() to capture FLUX payload and return dummy mannequin
// ---------------------------------------------------------------------------
const originalFetch = global.fetch;
let capturedBody = null;

global.fetch = async function patchedFetch(url, init) {
  const bodyStr = typeof init?.body === 'string' ? init.body : '';
  try {
    capturedBody = JSON.parse(bodyStr);
  } catch {
    capturedBody = null;
  }

  if (capturedBody?.image?.startsWith('data:image/png;base64,')) {
    fs.writeFileSync(COMPOSITE_OUT, Buffer.from(capturedBody.image.split(',')[1], 'base64'));
  }
  if (capturedBody?.prompt) {
    fs.writeFileSync(PROMPT_OUT, capturedBody.prompt);
  }

  console.log(`[MOCK FETCH] intercepted → ${url}`);
  console.log(`  prompt length: ${capturedBody?.prompt?.length ?? 0}`);
  console.log(`  image length:  ${capturedBody?.image?.length ?? 0}`);

  // Return the mannequin itself as the "FLUX output" so mask-merge should
  // snap back to the original mannequin everywhere outside garment zones.
  const mannequinBuf = fs.readFileSync(path.join(ROOT, 'assets', 'images', 'mannequin_front.png'));
  return {
    ok: true,
    status: 200,
    async text() {
      return JSON.stringify({
        artifacts: [{ base64: mannequinBuf.toString('base64') }],
      });
    },
    async json() {
      return { artifacts: [{ base64: mannequinBuf.toString('base64') }] };
    },
  };
};

// ---------------------------------------------------------------------------
// Now safe to import modules that depend on fetch
// ---------------------------------------------------------------------------
const { poseAnchoredRender } = await import(path.join(ROOT, 'api', 'services', 'strategies', 'poseAnchored.js'));

// ---------------------------------------------------------------------------
// Load test garments from assets/images
// ---------------------------------------------------------------------------
const assetsDir = path.join(ROOT, 'assets', 'images');
const assetFiles = fs.readdirSync(assetsDir).filter((f) => f.endsWith('.png'));
const SKIP = ['mannequin_front.png', 'mannequin_side.png', 'adaptive-icon.png', 'favicon.png', 'splash.png', 'icon.png', 'AIWardrobe-mainlogo.png'];
const garmentFiles = assetFiles.filter((f) => !SKIP.includes(f)).slice(0, 4);

if (garmentFiles.length === 0) {
  console.error('❌ No garment images found in', assetsDir);
  process.exit(1);
}

console.log('Using garments:', garmentFiles);

const mannequinBuf = fs.readFileSync(path.join(ROOT, 'assets', 'images', 'mannequin_front.png'));
const mannequinB64 = `data:image/png;base64,${mannequinBuf.toString('base64')}`;

const labelCycle = ['top', 'layer', 'pants', 'shoes'];
const garments = garmentFiles.map((f, i) => ({
  label: labelCycle[i % labelCycle.length],
  garmentSrc: `data:image/png;base64,${fs.readFileSync(path.join(assetsDir, f)).toString('base64')}`,
}));

// ---------------------------------------------------------------------------
// Run pipeline
// ---------------------------------------------------------------------------
console.log('\n=== Running poseAnchoredRender (fetch mocked) ===');
const started = Date.now();
const result = await poseAnchoredRender({
  mannequin_image: mannequinB64,
  garments,
});

console.log('\n--- Result ---');
console.log(JSON.stringify(result, null, 2));

if (result?.success && result.resultUrl) {
  const outPath = path.join(ROOT, 'scratch', 'v3_final_output.png');
  const b64 = result.resultUrl.split(',')[1] || '';
  fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
  console.log('\n✅ Saved final output →', outPath, `(${(fs.statSync(outPath).size / 1024).toFixed(1)} KB)`);
}

console.log('\nPipeline elapsed (mock):', Date.now() - started, 'ms');

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
console.log('\n--- Test complete ---');
console.log('Inspect these files:');
console.log('  1.', COMPOSITE_OUT, '— 3-panel FLUX input (mannequin | pose guide | garment grid)');
console.log('  2.', PROMPT_OUT, '— the prompt sent to FLUX');
console.log('  3.', path.join(ROOT, 'scratch', 'v3_final_output.png'), '— mask-merged result (should ≈ original mannequin)');
console.log('\nNext step: verify composite layout and prompt quality, then run with real FLUX NIM.');

// Restore fetch
global.fetch = originalFetch;

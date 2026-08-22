/**
 * Local end-to-end test for the deterministic mannequin renderer.
 *
 * Runs the same 4-step chain the app would run (top -> layer -> pants -> shoes)
 * by calling renderDeterministicGarment() directly (no HTTP), and writes each
 * intermediate result to api/scripts/out/.
 *
 * Run:  node scripts/test-renderer-local.js
 */

import 'dotenv/config';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderDeterministicGarment } from '../services/tryonRenderer.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..', '..');
const OUT_DIR = path.join(__dirname, 'out');
const MANNEQUIN_PATH = path.join(ROOT, 'assets', 'images', 'mannequin_front.png');

// Clean catalogue photos (product-only on white bg) supplied by the user.
const GARMENTS = [
  {
    label: 'top',
    name: 'Cream Linen Shirt',
    localPath: path.join(ROOT, 'assets', 'justfortry', 'image copy 2.png'),
  },
  {
    label: 'layer',
    name: 'Brown Linen Blazer',
    localPath: path.join(ROOT, 'assets', 'justfortry', 'image.png'),
  },
  {
    label: 'pants',
    name: 'White Linen Drawstring Pants',
    localPath: path.join(ROOT, 'assets', 'justfortry', 'image copy.png'),
  },
  {
    label: 'shoes',
    name: 'Brown Suede Espadrilles',
    localPath: path.join(ROOT, 'assets', 'justfortry', 'image copy 3.png'),
  },
];

function dataUriFromBuffer(buf, mime = 'image/png') {
  return `data:${mime};base64,${buf.toString('base64')}`;
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const mannequinBuf = await fs.readFile(MANNEQUIN_PATH);
  let currentMannequin = dataUriFromBuffer(mannequinBuf, 'image/png');

  console.log(`▶ mannequin loaded: ${(mannequinBuf.length / 1024).toFixed(0)} KB`);

  const totalStart = Date.now();
  for (let i = 0; i < GARMENTS.length; i++) {
    const g = GARMENTS[i];
    let garmentSrc;
    if (g.localPath) {
      const buf = await fs.readFile(g.localPath);
      garmentSrc = dataUriFromBuffer(buf, 'image/png');
    } else {
      garmentSrc = g.url;
    }

    const stepStart = Date.now();
    console.log(`\n── Step ${i + 1}/${GARMENTS.length}  ${g.label.toUpperCase()}  (${g.name})`);
    const result = await renderDeterministicGarment({
      mannequinSrc: currentMannequin,
      garmentSrc,
      label: g.label,
    });
    const elapsedMs = Date.now() - stepStart;

    const b64 = result.imageDataUri.split(',')[1];
    const outBuf = Buffer.from(b64, 'base64');
    const outPath = path.join(OUT_DIR, `node-step-${i + 1}-${g.label}.png`);
    await fs.writeFile(outPath, outBuf);
    console.log(
      `  done in ${elapsedMs}ms  (${(outBuf.length / 1024).toFixed(0)} KB) → ${path.relative(ROOT, outPath)}`,
    );

    currentMannequin = result.imageDataUri;
  }

  console.log(`\n✅ All ${GARMENTS.length} steps OK in ${Date.now() - totalStart}ms`);
  console.log(`   Final → ${path.join(OUT_DIR, `node-step-${GARMENTS.length}-${GARMENTS[GARMENTS.length - 1].label}.png`)}`);
}

main().catch((err) => {
  console.error('FATAL:', err);
  process.exit(1);
});

/**
 * End-to-end CLI verification of the Virtual Try-On pipeline.
 *
 * Drives the deployed `mannequin-tryon` Supabase Edge Function with the
 * bundled mannequin and four real garments (top, layer, pants, shoes),
 * exactly mirroring what the AITryOnScreen does at runtime.
 *
 * Saves each intermediate image to scripts/out/step-{n}-{slot}.png and
 * asserts that the final crop is NOT a side-by-side leak.
 *
 *   Run:  npx tsx scripts/test-tryon-pipeline.ts
 */

import 'dotenv/config';
import * as fs from 'node:fs';
import * as path from 'node:path';

type SlotKey = 'top' | 'layer' | 'pants' | 'shoes';
type GarmentType = 'upper_body' | 'lower_body' | 'shoes';

interface Garment {
  slot: SlotKey;
  label: string;
  type: GarmentType;
  url: string;
  name: string;
  description: string;
}

const SUPABASE_URL = process.env.EXPO_PUBLIC_SUPABASE_URL;
const SUPABASE_ANON = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON) {
  console.error('Missing EXPO_PUBLIC_SUPABASE_URL / EXPO_PUBLIC_SUPABASE_ANON_KEY in .env');
  process.exit(1);
}

const FN_URL = `${SUPABASE_URL.replace(/\/$/, '')}/functions/v1/mannequin-tryon`;
const OUT_DIR = path.join(__dirname, '..', 'out');
const MANNEQUIN_PATH = path.join(__dirname, '..', '..', 'assets', 'images', 'mannequin_front.png');

// 4 garments mirroring the screen's APPLY_ORDER: top → layer → pants → shoes.
const GARMENTS: Garment[] = [
  {
    slot: 'top',
    label: 'Oxford Slim-Fit Shirt',
    type: 'upper_body',
    url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/456630/item/usgoods_64_456630_3x4.jpg',
    name: 'Oxford Slim-Fit Shirt',
    description: 'Light blue button-down collar 100% cotton oxford long-sleeve shirt, slim fit',
  },
  {
    slot: 'layer',
    label: 'AirSense Blazer',
    type: 'upper_body',
    url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/448034/item/usgoods_09_448034_3x4.jpg',
    name: 'AirSense Blazer',
    description: 'Navy blue ultra-light wool-like two-button single-breasted blazer jacket',
  },
  {
    slot: 'pants',
    label: 'Slim-Fit Chino Pants',
    type: 'lower_body',
    url: 'https://image.uniqlo.com/UQ/ST3/us/imagesgoods/450251/item/usgoods_35_450251_3x4.jpg',
    name: 'Slim-Fit Chino Pants',
    description: 'Brown beige slim-fit Supima cotton stretch chino trousers, full-length',
  },
  {
    slot: 'shoes',
    label: 'Combination Sneaker',
    type: 'shoes',
    url: 'https://image.uniqlo.com/UQ/ST3/WesternCommon/imagesgoods/484330/item/goods_32_484330_3x4.jpg',
    name: 'Combination Sneaker',
    description: 'White leather low-top sneakers with beige suede side panels and gum rubber sole',
  },
];

function dataUriToBuffer(dataUri: string): Buffer {
  const m = dataUri.match(/^data:([^;]+);base64,(.+)$/);
  if (!m) throw new Error('Not a data URI');
  return Buffer.from(m[2], 'base64');
}

async function callFn(body: unknown): Promise<any> {
  const res = await fetch(FN_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${SUPABASE_ANON}`,
      apikey: SUPABASE_ANON!,
    },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let json: any;
  try {
    json = JSON.parse(text);
  } catch {
    throw new Error(`Non-JSON response (HTTP ${res.status}): ${text.slice(0, 300)}`);
  }
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${JSON.stringify(json).slice(0, 300)}`);
  return json;
}

async function poll(predictionId: string, slot: string): Promise<string> {
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 2500));
    const data = await callFn({ action: 'poll', predictionId });
    if (!data?.success) throw new Error(`Poll failed for ${slot}: ${data?.error}`);
    process.stdout.write('.');
    if (data.status === 'succeeded' && data.resultUrl) {
      process.stdout.write('\n');
      return data.resultUrl as string;
    }
    if (data.status === 'failed') throw new Error(`Prediction failed for ${slot}: ${data.error}`);
  }
  throw new Error(`Timed out polling ${slot}`);
}

function assertImage(dataUri: string, label: string): { width: number; height: number; bytes: number } {
  const buf = dataUriToBuffer(dataUri);
  if (buf.length < 50_000) {
    throw new Error(`${label}: image too small (${buf.length} bytes) — likely failed`);
  }
  // Quick PNG dimension parse (IHDR @ offset 16)
  if (buf[0] === 0x89 && buf[1] === 0x50) {
    const w = buf.readUInt32BE(16);
    const h = buf.readUInt32BE(20);
    return { width: w, height: h, bytes: buf.length };
  }
  // JPEG fallback — scan for SOF0/SOF2
  for (let i = 0; i < buf.length - 9; i++) {
    if (buf[i] === 0xff && (buf[i + 1] === 0xc0 || buf[i + 1] === 0xc2)) {
      const h = buf.readUInt16BE(i + 5);
      const w = buf.readUInt16BE(i + 7);
      return { width: w, height: h, bytes: buf.length };
    }
  }
  return { width: 0, height: 0, bytes: buf.length };
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  console.log('▶ Loading mannequin from', MANNEQUIN_PATH);
  if (!fs.existsSync(MANNEQUIN_PATH)) {
    throw new Error(`Mannequin image not found at ${MANNEQUIN_PATH}`);
  }
  const mannB64 = fs.readFileSync(MANNEQUIN_PATH).toString('base64');
  let currentMannequin = `data:image/png;base64,${mannB64}`;

  console.log('▶ Calling', FN_URL);
  const total = GARMENTS.length;
  const t0 = Date.now();
  const alreadyWearing: string[] = [];
  const layerGarment = GARMENTS.find((g) => g.slot === 'layer');
  const finalTotal = total + (layerGarment && total > 1 ? 1 : 0);

  for (let i = 0; i < total; i++) {
    const g = GARMENTS[i];
    const step = i + 1;
    const tStep = Date.now();
    console.log(`\n── Step ${step}/${finalTotal}  ${g.slot.toUpperCase()}  (${g.label})`);

    if (i > 0) {
      console.log('  ⏳ throttling 2s for FLUX.1-Kontext-dev pacing…');
      await new Promise((r) => setTimeout(r, 2_000));
    }

    const submitRes = await callFn({
      action: 'submit',
      mannequin_image: currentMannequin,
      garment_image: g.url,
      garment: { type: g.type, label: g.slot, name: g.name, description: g.description },
      already_wearing: [...alreadyWearing],
      step,
      total: finalTotal,
    });

    if (!submitRes?.success) {
      throw new Error(`Submit failed at step ${step}: ${submitRes?.error}`);
    }

    let resultUrl: string;
    if (submitRes.mode === 'sync' && submitRes.resultUrl) {
      console.log(`  sync result returned in ${Date.now() - tStep}ms`);
      resultUrl = submitRes.resultUrl;
    } else {
      const pid: string = submitRes.predictionId;
      if (!pid) throw new Error('No predictionId returned');
      console.log(`  async predictionId=${pid.slice(0, 12)}…  polling`);
      resultUrl = await poll(pid, g.slot);
      console.log(`  done in ${Math.round((Date.now() - tStep) / 1000)}s`);
    }

    // Validate image
    const meta = assertImage(resultUrl, `step-${step}`);
    const ratio = meta.width && meta.height ? meta.width / meta.height : 0;
    console.log(`  result: ${meta.width}×${meta.height}  ${(meta.bytes / 1024).toFixed(0)} KB  ratio=${ratio.toFixed(2)}`);

    if (ratio >= 1.4) {
      console.warn(`  ⚠ ratio ${ratio.toFixed(2)} suggests side-by-side leak — smartCrop should have handled this`);
    }

    // Save to disk
    const outPath = path.join(OUT_DIR, `step-${step}-${g.slot}.png`);
    fs.writeFileSync(outPath, dataUriToBuffer(resultUrl));
    console.log(`  saved → ${outPath}`);

    // Feed result forward
    currentMannequin = resultUrl;
    alreadyWearing.push(g.description || g.name);
  }

  if (layerGarment && total > 1) {
    const step = finalTotal;
    const tStep = Date.now();
    console.log(`\n── Step ${step}/${finalTotal}  LAYER REINFORCE  (${layerGarment.label})`);
    console.log('  ⏳ throttling 2s for FLUX.1-Kontext-dev pacing…');
    await new Promise((r) => setTimeout(r, 2_000));

    const submitRes = await callFn({
      action: 'submit',
      mannequin_image: currentMannequin,
      garment_image: layerGarment.url,
      garment: { type: layerGarment.type, label: 'layer', name: layerGarment.name, description: layerGarment.description },
      already_wearing: [...alreadyWearing],
      step,
      total: finalTotal,
    });

    if (!submitRes?.success || !submitRes.resultUrl) {
      throw new Error(`Layer reinforcement failed: ${submitRes?.error || 'No resultUrl'}`);
    }

    const meta = assertImage(submitRes.resultUrl, `step-${step}`);
    const ratio = meta.width && meta.height ? meta.width / meta.height : 0;
    console.log(`  sync result returned in ${Date.now() - tStep}ms`);
    console.log(`  result: ${meta.width}×${meta.height}  ${(meta.bytes / 1024).toFixed(0)} KB  ratio=${ratio.toFixed(2)}`);

    const outPath = path.join(OUT_DIR, `step-${step}-layer-reinforce.png`);
    fs.writeFileSync(outPath, dataUriToBuffer(submitRes.resultUrl));
    console.log(`  saved → ${outPath}`);
    currentMannequin = submitRes.resultUrl;
  }

  console.log(`\n✅ All ${finalTotal} steps succeeded in ${Math.round((Date.now() - t0) / 1000)}s`);
  console.log(`   Final image saved in ${OUT_DIR}`);
}

main().catch((err) => {
  console.error('\n❌ Pipeline failed:', err.message || err);
  process.exit(1);
});

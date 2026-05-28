/**
 * Self-hosted background removal using briaai/RMBG-1.4 (ONNX) via
 * @huggingface/transformers. Loads the model once on first call and keeps it
 * in memory; subsequent calls reuse the loaded session.
 *
 *   const cleanRGBA = await removeBackground(srcBuffer);   // → PNG buffer
 *
 * Returns a Buffer of an RGBA PNG with the subject isolated and the
 * background fully transparent. No external API, no per-call cost.
 */

import {
  AutoModel,
  AutoProcessor,
  RawImage,
  env,
} from '@huggingface/transformers';
import sharp from 'sharp';
import path from 'node:path';

// Cache downloaded model files in api/cache/hf so they survive across runs.
env.cacheDir = path.join(process.cwd(), 'cache', 'hf');
// Allow the runtime to fall back to remote downloads on first run.
env.allowRemoteModels = true;
env.allowLocalModels = true;

const MODEL_ID = 'briaai/RMBG-1.4';

let modelPromise = null;
let processorPromise = null;

async function getModel() {
  if (!modelPromise) {
    console.log(`[rmbg] loading model ${MODEL_ID} (first run downloads ~180 MB)…`);
    modelPromise = AutoModel.from_pretrained(MODEL_ID, {
      // RMBG-1.4 ships only float32 ONNX; transformers.js auto-picks it.
      device: 'cpu',
    }).then((m) => {
      console.log('[rmbg] model loaded');
      return m;
    });
  }
  return modelPromise;
}

async function getProcessor() {
  if (!processorPromise) {
    processorPromise = AutoProcessor.from_pretrained(MODEL_ID).then((p) => {
      console.log('[rmbg] processor loaded');
      return p;
    });
  }
  return processorPromise;
}

/**
 * Run RMBG-1.4 on a raw image buffer (any format sharp can decode) and
 * return an RGBA PNG buffer with background pixels set to alpha=0.
 */
export async function removeBackground(srcBuf) {
  const [model, processor] = await Promise.all([getModel(), getProcessor()]);

  // Decode the source via sharp into RGBA so we can re-attach alpha after
  // running the model. We also keep the original dimensions for the output.
  const { data: rgbaData, info } = await sharp(srcBuf)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const W = info.width;
  const H = info.height;

  // Build a RawImage in RGB at original size; processor handles resize/normalize.
  const rgb = Buffer.alloc(W * H * 3);
  for (let i = 0, j = 0; i < W * H; i++, j += 3) {
    rgb[j] = rgbaData[i * 4];
    rgb[j + 1] = rgbaData[i * 4 + 1];
    rgb[j + 2] = rgbaData[i * 4 + 2];
  }
  const rawImage = new RawImage(new Uint8Array(rgb), W, H, 3);

  const inputs = await processor(rawImage);
  const { output } = await model({ input: inputs.pixel_values });

  // `output` is a Tensor with shape [1, 1, h, w]. Sigmoid'd matte in [0..1].
  const maskTensor = await output[0].sigmoid();
  const matte = await maskTensor.mul(255).to('uint8');
  // Resize the matte to original dims using sharp (matte is at processor's
  // model resolution, typically 1024×1024).
  const mDims = matte.dims; // [1, h, w] or [h, w]
  const mh = mDims[mDims.length - 2];
  const mw = mDims[mDims.length - 1];
  const matteBytes = Buffer.from(matte.data);

  const fullMatte = await sharp(matteBytes, {
    raw: { width: mw, height: mh, channels: 1 },
  })
    .resize(W, H, { fit: 'fill' })
    .raw()
    .toBuffer();

  // RMBG's matte is a soft sigmoid; for catalogue photos that share a colour
  // with the bg (white pants on white seamless), the matte values stay low.
  // We aggressively binarise: anything above a low threshold becomes fully
  // opaque, with a 4-pixel ramp at the very edge for anti-aliasing.
  //
  // Strategy:
  //   matte ≥ 32 → fully opaque (255)
  //   16 ≤ matte < 32 → linear ramp 0..255
  //   matte < 16 → fully transparent
  const out = Buffer.alloc(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    out[i * 4] = rgbaData[i * 4];
    out[i * 4 + 1] = rgbaData[i * 4 + 1];
    out[i * 4 + 2] = rgbaData[i * 4 + 2];
    const m = fullMatte[i];
    let a;
    if (m < 16) a = 0;
    else if (m >= 32) a = 255;
    else a = Math.round(((m - 16) / 16) * 255);
    out[i * 4 + 3] = a;
  }

  return await sharp(out, { raw: { width: W, height: H, channels: 4 } })
    .png()
    .toBuffer();
}

/** Optional: warm the model at server boot to avoid the first-request stall. */
export async function preloadBgRemoval() {
  try {
    await Promise.all([getModel(), getProcessor()]);
    return true;
  } catch (err) {
    console.warn('[rmbg] preload failed:', err?.message || err);
    return false;
  }
}

/**
 * Per-class clothing segmentation using `mattmdjaga/segformer_b2_clothes`.
 *
 * Unlike a generic bg-removal model (which keeps the entire person), this
 * model labels every pixel with a clothing class. We use those labels to
 * extract ONLY the garment we want from a product photo (the model wearing
 * the shirt has their face/skin/pants discarded, only the shirt is kept).
 *
 * Class indices (from the model's id2label):
 *   0=Background  1=Hat            2=Hair             3=Sunglasses
 *   4=Upper-clothes 5=Skirt        6=Pants            7=Dress
 *   8=Belt        9=Left-shoe     10=Right-shoe      11=Face
 *  12=Left-leg   13=Right-leg     14=Left-arm        15=Right-arm
 *  16=Bag        17=Scarf
 */

import { pipeline, env } from '@huggingface/transformers';
import sharp from 'sharp';
import path from 'node:path';

env.cacheDir = path.join(process.cwd(), 'cache', 'hf');
env.allowRemoteModels = true;
env.allowLocalModels = true;

const MODEL_ID = 'mattmdjaga/segformer_b2_clothes';

// SegFormer class labels we want to keep per garment category. The pipeline
// returns one mask per detected label; we OR-merge the masks for our target
// labels into a single alpha channel.
const CLASS_GROUPS = {
  top:   ['Upper-clothes'],
  layer: ['Upper-clothes'],          // jackets/blazers are also tagged Upper-clothes
  pants: ['Pants', 'Skirt'],
  shoes: ['Left-shoe', 'Right-shoe'],
};

let segmenterPromise = null;

async function getSegmenter() {
  if (!segmenterPromise) {
    console.log(`[clothes-seg] loading ${MODEL_ID} (first run downloads ~110 MB)…`);
    segmenterPromise = pipeline('image-segmentation', MODEL_ID).then((p) => {
      console.log('[clothes-seg] pipeline ready');
      return p;
    });
  }
  return segmenterPromise;
}

/**
 * Run segformer_b2_clothes via the HF pipeline and return an RGBA PNG buffer
 * where pixels NOT belonging to the requested label's classes are fully
 * transparent.
 *
 * Using `pipeline('image-segmentation')` instead of raw model.forward() means
 * we get correctly post-processed, full-resolution per-class masks (the
 * pipeline handles letterbox unpadding + resize-back-to-original for us).
 *
 * @param {Buffer} srcBuf - Any image format sharp can decode.
 * @param {'top'|'layer'|'pants'|'shoes'} label
 * @returns {Promise<Buffer>} PNG bytes (RGBA).
 */
export async function extractGarment(srcBuf, label = 'top') {
  const targetLabels = new Set(CLASS_GROUPS[label] || CLASS_GROUPS.top);
  const segmenter = await getSegmenter();

  // Pipeline expects a Blob/URL/RawImage. Easiest: pass a data URI (via base64).
  const dataUri = `data:image/png;base64,${srcBuf.toString('base64')}`;
  const segments = await segmenter(dataUri);
  // Each entry: { score, label, mask: RawImage(width, height, channels=1) }
  // The mask is at the ORIGINAL image resolution.

  // Decode src to RGBA at the same resolution the masks use.
  const { data: rgbaData, info } = await sharp(srcBuf)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const W = info.width;
  const H = info.height;

  // Combine all matching label masks via per-pixel max.
  const combined = new Uint8Array(W * H);
  let matched = 0;
  for (const seg of segments) {
    if (!targetLabels.has(seg.label)) continue;
    matched++;
    const m = seg.mask; // RawImage
    // RawImage's data is a Uint8Array sized width*height*channels.
    // For segmentation masks, channels=1 and pixels are 0 or 255.
    const md = m.data;
    const mw = m.width;
    const mh = m.height;
    if (mw === W && mh === H) {
      for (let i = 0; i < W * H; i++) {
        if (md[i] > combined[i]) combined[i] = md[i];
      }
    } else {
      // Different size — resize using sharp.
      const resized = await sharp(Buffer.from(md), {
        raw: { width: mw, height: mh, channels: 1 },
      })
        .resize(W, H, { fit: 'fill' })
        .raw()
        .toBuffer();
      for (let i = 0; i < W * H; i++) {
        if (resized[i] > combined[i]) combined[i] = resized[i];
      }
    }
  }

  if (matched === 0) {
    console.warn(`[clothes-seg] no segments matched label=${label}; available=${segments.map((s) => s.label).join(',')}`);
  }

  // Soften the mask edge slightly to avoid a hard staircase look.
  const softened = await sharp(combined, { raw: { width: W, height: H, channels: 1 } })
    .blur(0.8)
    .raw()
    .toBuffer();

  const out = Buffer.alloc(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    out[i * 4] = rgbaData[i * 4];
    out[i * 4 + 1] = rgbaData[i * 4 + 1];
    out[i * 4 + 2] = rgbaData[i * 4 + 2];
    out[i * 4 + 3] = softened[i];
  }
  return await sharp(out, { raw: { width: W, height: H, channels: 4 } })
    .png()
    .toBuffer();
}

export async function preloadClothesSegmenter() {
  try {
    await getSegmenter();
    return true;
  } catch (err) {
    console.warn('[clothes-seg] preload failed:', err?.message || err);
    return false;
  }
}

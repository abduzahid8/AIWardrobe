/**
 * Strategy v1 — Side-by-Side Reference + Mask-Locked Recomposition.
 *
 * Per garment step:
 *   1. Snapshot preStep = current basePx (cumulative outfit so far).
 *   2. Run compositeGarmentOntoBase to obtain placedAlpha (anchor mask).
 *      Immediately restore basePx from preStep — we do NOT want the coarse
 *      composite leaking into FLUX input or the final output.
 *   3. Build a 2048x1024 reference image: left = preStep mannequin,
 *      right = preprocessed garment cutout centered on white.
 *   4. Call FLUX.1-Kontext-dev with a strict prompt to dress the left panel.
 *   5. Extract the LEFT half of FLUX's output, resampled to 1024x1024.
 *   6. Hard-paste fluxOut into basePx through dilateAndFeatherMask(placedAlpha,
 *      6, 4): outside the mask snaps back to preStep (frozen identity); inside
 *      uses fluxOut; a thin feather band linearly blends seam pixels.
 *
 * Reuses tryonRenderer.js for all anchor / mask / preprocessing infra.
 */

import sharp from 'sharp';
import {
  W,
  H,
  buildBaseCanvas,
  compositeGarmentOntoBase,
  dilateAndFeatherMask,
  encodeBasePxToDataUri,
  encodeCanvas,
  preprocessGarmentAndCache,
  normalizeGarmentLabel,
  GARMENT_RENDER_ORDER,
} from '../tryonRenderer.js';

const NVIDIA_KONTEXT_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';

export { W, H, GARMENT_RENDER_ORDER, normalizeGarmentLabel, buildBaseCanvas, encodeCanvas, preprocessGarmentAndCache };

function stripDataUri(dataUri) {
  if (typeof dataUri !== 'string') return '';
  if (dataUri.startsWith('data:')) return dataUri.split(',')[1] || '';
  return dataUri;
}

async function loadImageBuffer(src) {
  if (!src) throw new Error('Empty image source');
  if (src.startsWith('data:')) return Buffer.from(stripDataUri(src), 'base64');
  if (src.startsWith('http://') || src.startsWith('https://')) {
    const res = await fetch(src);
    if (!res.ok) throw new Error(`Failed to fetch image (${res.status})`);
    return Buffer.from(await res.arrayBuffer());
  }
  return Buffer.from(src, 'base64');
}

/**
 * Build the 2048x1024 reference. LEFT = clean preStep mannequin (cumulative
 * outfit so far). RIGHT = preprocessed garment cutout, centered on white.
 */
export async function buildSideBySideReference({ basePx, garmentSrc, label }) {
  const leftPng = await sharp(Buffer.from(basePx.buffer, basePx.byteOffset, basePx.byteLength), {
    raw: { width: W, height: H, channels: 4 },
  })
    .flatten({ background: { r: 255, g: 255, b: 255 } })
    .png()
    .toBuffer();

  let garmentPng;
  try {
    const cleaned = await preprocessGarmentAndCache(garmentSrc, label);
    garmentPng = await sharp(cleaned)
      .resize(W, H, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } })
      .flatten({ background: { r: 255, g: 255, b: 255 } })
      .png()
      .toBuffer();
  } catch {
    const raw = await loadImageBuffer(garmentSrc);
    garmentPng = await sharp(raw)
      .resize(W, H, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } })
      .flatten({ background: { r: 255, g: 255, b: 255 } })
      .png()
      .toBuffer();
  }

  const referencePng = await sharp({
    create: { width: W * 2, height: H, channels: 3, background: { r: 255, g: 255, b: 255 } },
  })
    .composite([
      { input: leftPng, left: 0, top: 0 },
      { input: garmentPng, left: W, top: 0 },
    ])
    .png()
    .toBuffer();

  return { pngBuffer: referencePng, width: W * 2, height: H };
}

function buildPrompt(label) {
  const garmentZone = label === 'pants'
    ? 'lower body (hips, thighs, legs)'
    : label === 'shoes'
      ? 'feet and ankles'
      : label === 'layer'
        ? 'shoulders, chest, sleeves, and outer torso'
        : 'upper torso, chest, shoulders, and sleeves';
  return [
    'Left panel: dress the headless light-grey/white fashion mannequin in the EXACT garment shown in the right panel.',
    `Focus on the ${garmentZone}. Render photorealistic drape, fabric folds, hems, sleeves, and natural contact shadows.`,
    "HARD CONSTRAINTS — violating any is a failure:",
    "- Do NOT alter the mannequin's head, skin, hands, pose, proportions, height, lighting, camera angle, or framing. Mannequin pixels stay byte-identical outside the garment area.",
    '- Do NOT introduce a human face, hair, skin tone, jewelry, or any accessory. The figure stays a mannequin.',
    '- The white seamless studio background must remain pixel-identical (only subtle contact shadows directly under garment edges are allowed).',
    '- Already-worn garments on the mannequin must remain visually identical — this is a cumulative outfit build.',
    '- Keep the new garment color, pattern, logos, fabric texture, silhouette, and design details exactly as in the right panel.',
    '- The right panel must remain unchanged (the product reference).',
    'Style: studio fashion catalog, soft even lighting, sharp focus, photorealistic premium e-commerce photography.',
  ].join(' ');
}

async function uploadNvidiaAsset({ nvidiaKey, imageBuffer, contentType, description }) {
  const createRes = await fetch('https://api.nvcf.nvidia.com/v2/nvcf/assets', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${nvidiaKey}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ contentType, description }),
  });
  if (!createRes.ok) {
    throw new Error(`NVIDIA asset create failed (${createRes.status}): ${await createRes.text()}`);
  }
  const created = await createRes.json();
  const assetId = created?.assetId;
  const uploadUrl = created?.uploadUrl;
  if (!assetId || !uploadUrl) throw new Error('NVIDIA asset create returned no assetId/uploadUrl');

  const putRes = await fetch(uploadUrl, {
    method: 'PUT',
    headers: {
      'Content-Type': contentType,
      'x-amz-meta-nvcf-asset-description': description || 'aiwardrobe-sidebyside-v1',
    },
    body: imageBuffer,
  });
  if (!putRes.ok) {
    throw new Error(`NVIDIA asset upload failed (${putRes.status}): ${await putRes.text()}`);
  }
  return assetId;
}

/**
 * Call FLUX.1-Kontext-dev with the side-by-side reference. Returns the
 * resulting image as a `data:image/png;base64,...` URI.
 */
export async function callFluxSideBySide({ referencePng, label, nvidiaKey }) {
  const assetId = await uploadNvidiaAsset({
    nvidiaKey,
    imageBuffer: referencePng,
    contentType: 'image/png',
    description: `aiwardrobe-sidebyside-v1-${label}`,
  });

  const res = await fetch(NVIDIA_KONTEXT_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${nvidiaKey}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
      'NVCF-INPUT-ASSET-REFERENCES': assetId,
    },
    body: JSON.stringify({
      prompt: buildPrompt(label),
      image: `data:image/png;example_id,${assetId}`,
      aspect_ratio: '2:1',
    }),
  });

  if (!res.ok) {
    throw new Error(`FLUX.1 Kontext-dev failed HTTP ${res.status} label=${label} asset=${assetId}: ${await res.text()}`);
  }
  const data = await res.json();
  if (data.artifacts?.[0]?.base64) return `data:image/png;base64,${data.artifacts[0].base64}`;
  if (data.image) return data.image.startsWith('data:') ? data.image : `data:image/png;base64,${data.image}`;
  if (data.output?.image) return data.output.image;
  throw new Error(`FLUX.1 Kontext-dev returned no image label=${label} asset=${assetId}`);
}

/**
 * Decode a FLUX side-by-side result and return a raw RGBA Uint8ClampedArray
 * (W*H*4) of just the LEFT half (the dressed mannequin), resampled to W x H.
 */
export async function extractLeftHalfRaw(fluxDataUri) {
  if (!fluxDataUri) return null;
  const buf = await loadImageBuffer(fluxDataUri);
  const meta = await sharp(buf).metadata();
  const fw = meta.width || 0;
  const fh = meta.height || 0;
  if (!fw || !fh) return null;
  const halfWidth = Math.max(1, Math.floor(fw / 2));
  const leftBuf = await sharp(buf)
    .extract({ left: 0, top: 0, width: halfWidth, height: fh })
    .resize(W, H, { fit: 'fill' })
    .ensureAlpha()
    .raw()
    .toBuffer();
  return new Uint8ClampedArray(leftBuf);
}

/**
 * Hard-paste fluxOut into basePx through `mask`. Pixels with mask==0 are
 * restored from preStep (frozen identity). Pixels with mask==255 are taken
 * directly from fluxRaw with alpha=255. Feather-band pixels (0<mask<255)
 * linearly blend preStep and fluxRaw.
 */
export function hardPasteThroughMask(basePx, preStep, fluxRaw, mask) {
  const total = W * H;
  for (let i = 0; i < total; i++) {
    const m = mask[i];
    const idx = i * 4;
    if (m === 0) {
      basePx[idx] = preStep[idx];
      basePx[idx + 1] = preStep[idx + 1];
      basePx[idx + 2] = preStep[idx + 2];
      basePx[idx + 3] = preStep[idx + 3];
      continue;
    }
    if (m >= 250) {
      basePx[idx] = fluxRaw[idx];
      basePx[idx + 1] = fluxRaw[idx + 1];
      basePx[idx + 2] = fluxRaw[idx + 2];
      basePx[idx + 3] = 255;
      continue;
    }
    const mf = m / 255;
    const inv = 1 - mf;
    basePx[idx] = Math.round(preStep[idx] * inv + fluxRaw[idx] * mf);
    basePx[idx + 1] = Math.round(preStep[idx + 1] * inv + fluxRaw[idx + 1] * mf);
    basePx[idx + 2] = Math.round(preStep[idx + 2] * inv + fluxRaw[idx + 2] * mf);
    basePx[idx + 3] = 255;
  }
}

/**
 * Apply one garment step. Mutates basePx.
 */
export async function applySideBySideStep(basePx, garmentSrc, label, nvidiaKey) {
  const normalized = normalizeGarmentLabel(label);
  const preStep = new Uint8ClampedArray(basePx);

  // Use the deterministic compositor only to obtain the anchor alpha.
  const { placedAlpha } = await compositeGarmentOntoBase(basePx, garmentSrc, normalized);
  // Restore basePx — we want the clean preStep mannequin going to FLUX, and
  // the final output should be a hard-paste of FLUX through the mask, not the
  // coarse composite blended with anything else.
  basePx.set(preStep);

  const mask = await dilateAndFeatherMask(placedAlpha, 6, 4);

  const { pngBuffer: referencePng } = await buildSideBySideReference({ basePx: preStep, garmentSrc, label: normalized });
  const fluxDataUri = await callFluxSideBySide({ referencePng, label: normalized, nvidiaKey });
  const fluxRaw = await extractLeftHalfRaw(fluxDataUri);
  if (!fluxRaw) {
    throw new Error(`FLUX.1 Kontext-dev output could not be decoded label=${normalized}`);
  }

  hardPasteThroughMask(basePx, preStep, fluxRaw, mask);

  return { label: normalized, fluxApplied: true };
}

export { encodeBasePxToDataUri };

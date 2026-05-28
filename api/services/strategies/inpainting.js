/**
 * Iterative Frozen-Region Inpainting strategy (v2).
 *
 * Instead of pre-compositing a garment onto the mannequin and then asking
 * FLUX to "refine" it, we send FLUX the raw mannequin with a magenta-tinted
 * "editable region" mask. FLUX is instructed to replace ONLY the magenta
 * region with the garment from the product photo. After FLUX returns, we
 * hard-snap every non-mask pixel back to the original — a mathematical
 * guarantee that the mannequin never changes outside the garment zone.
 *
 * No deterministic compositor is used. FLUX does the actual rendering.
 */

import sharp from 'sharp';
import {
  W,
  H,
  GARMENT_RENDER_ORDER,
  normalizeGarmentLabel,
  buildBaseCanvas,
  encodeBasePxToDataUri,
  encodeCanvas,
  preprocessGarmentAndCache,
} from '../tryonRenderer.js';

// ─── Mannequin region definitions (mirrored from tryonRenderer.js) ────────
const MANNEQUIN_REGIONS = {
  torso:       { cx: 0.50, cy: 0.33, rx: 0.155, ry: 0.18, softness: 0.2 },
  leftSleeve:  { cx: 0.31, cy: 0.34, rx: 0.10, ry: 0.23, softness: 0.22 },
  rightSleeve: { cx: 0.69, cy: 0.34, rx: 0.10, ry: 0.23, softness: 0.22 },
  waist:       { cx: 0.50, cy: 0.50, rx: 0.12, ry: 0.055, softness: 0.18 },
  hips:        { cx: 0.50, cy: 0.57, rx: 0.16, ry: 0.085, softness: 0.18 },
  leftLeg:     { cx: 0.43, cy: 0.74, rx: 0.09, ry: 0.22, softness: 0.2 },
  rightLeg:    { cx: 0.57, cy: 0.74, rx: 0.09, ry: 0.22, softness: 0.2 },
  leftFoot:    { cx: 0.43, cy: 0.95, rx: 0.11, ry: 0.035, softness: 0.24 },
  rightFoot:   { cx: 0.57, cy: 0.95, rx: 0.11, ry: 0.035, softness: 0.24 },
};

// ─── Helper: elliptical falloff (same as tryonRenderer.js) ────────────────

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function ellipticalFalloff(nx, ny, cx, cy, rx, ry, softness = 0.18) {
  const dx = (nx - cx) / Math.max(1e-6, rx);
  const dy = (ny - cy) / Math.max(1e-6, ry);
  const d = Math.sqrt(dx * dx + dy * dy);
  return 1 - smoothstep(1 - softness, 1 + softness, d);
}

function mannequinRegionValue(region, nx, ny) {
  const def = MANNEQUIN_REGIONS[region];
  if (!def) return 0;
  return ellipticalFalloff(nx, ny, def.cx, def.cy, def.rx, def.ry, def.softness);
}

// ─── Image utilities ──────────────────────────────────────────────────────

function loadImageBuffer(src) {
  if (!src) throw new Error('Empty image source');
  if (src.startsWith('data:')) {
    const b64 = src.split(',')[1] || '';
    return Buffer.from(b64, 'base64');
  }
  if (src.startsWith('http://') || src.startsWith('https://')) {
    return fetch(src)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch image (${r.status})`);
        return r.arrayBuffer();
      })
      .then((ab) => Buffer.from(ab));
  }
  return Buffer.from(src, 'base64');
}

function stripDataUri(dataUri) {
  if (typeof dataUri !== 'string') return '';
  if (dataUri.startsWith('data:')) return dataUri.split(',')[1] || '';
  return dataUri;
}

function toDataUri(buffer, contentType = 'image/png') {
  return `data:${contentType};base64,${buffer.toString('base64')}`;
}

function detectImageContentType(src, fallback = 'image/png') {
  if (typeof src !== 'string') return fallback;
  const match = src.match(/^data:([^;]+);/i);
  if (match?.[1]) return match[1].toLowerCase();
  if (src.startsWith('http://') || src.startsWith('https://')) {
    const clean = src.split('?')[0].toLowerCase();
    if (clean.endsWith('.jpg') || clean.endsWith('.jpeg')) return 'image/jpeg';
    if (clean.endsWith('.webp')) return 'image/webp';
  }
  return fallback;
}

// ─── 1. Build binary inpainting mask ──────────────────────────────────────

/**
 * Build a binary mask (0 or 255) marking the garment's editable region.
 * Uses MANNEQUIN_REGIONS elliptical definitions so the mask follows body
 * anatomy rather than a simple axis-aligned box.
 *
 * The mask is then dilated by ~16px to give FLUX room for hems, cuffs,
 * and natural extensions beyond the strict anatomical zone.
 */
export function buildInpaintingMask(label) {
  const normalized = normalizeGarmentLabel(label);
  const mask = new Uint8Array(W * H);

  // Threshold for including a pixel in the editable region.
  const threshold = (normalized === 'layer') ? 0.18 : 0.25;

  for (let y = 0; y < H; y++) {
    const ny = y / (H - 1);
    for (let x = 0; x < W; x++) {
      const nx = x / (W - 1);
      let val = 0;

      if (normalized === 'top' || normalized === 'layer') {
        val = Math.max(
          mannequinRegionValue('torso', nx, ny),
          mannequinRegionValue('leftSleeve', nx, ny),
          mannequinRegionValue('rightSleeve', nx, ny),
          mannequinRegionValue('waist', nx, ny) * 0.82,
        );
      } else if (normalized === 'pants') {
        val = Math.max(
          mannequinRegionValue('waist', nx, ny) * 0.78,
          mannequinRegionValue('hips', nx, ny),
          mannequinRegionValue('leftLeg', nx, ny),
          mannequinRegionValue('rightLeg', nx, ny),
        );
      } else if (normalized === 'shoes') {
        val = Math.max(
          mannequinRegionValue('leftFoot', nx, ny),
          mannequinRegionValue('rightFoot', nx, ny),
        );
      }

      mask[y * W + x] = val > threshold ? 255 : 0;
    }
  }

  // Dilate the mask by 16px (separable max-filter) so FLUX has room
  // for hems, cuffs, and natural garment extensions.
  return dilateMask(mask, 16);
}

/**
 * Separable max-filter dilation — expands the mask by `radius` pixels.
 */
function dilateMask(mask, radius) {
  const tmp = new Uint8Array(W * H);
  const out = new Uint8Array(W * H);

  // Horizontal pass
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      let m = 0;
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(W - 1, x + radius);
      for (let i = x0; i <= x1; i++) {
        if (mask[row + i] > m) m = mask[row + i];
      }
      tmp[row + x] = m;
    }
  }

  // Vertical pass
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) {
      let m = 0;
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(H - 1, y + radius);
      for (let i = y0; i <= y1; i++) {
        const v = tmp[i * W + x];
        if (v > m) m = v;
      }
      out[y * W + x] = m;
    }
  }

  return out;
}

// ─── 2. Apply magenta tint to the editable region ─────────────────────────

/**
 * Create a copy of `basePx` where the mask>0 region is tinted magenta
 * and dimmed, clearly marking it as the "editable zone" for FLUX.
 * The original `basePx` is NOT mutated.
 *
 * Tint formula: out = orig * dimKeep + magenta * tintStrength * dimFactor
 *   - dimKeep = (1 - tintStrength) * dimFactor
 *   - This keeps structural cues visible while making the region
 *     unmistakably magenta.
 */
export function applyMagentaTint(basePx, mask) {
  const out = new Uint8ClampedArray(basePx.length);
  const tintStrength = 0.55;
  const dimFactor = 0.70;
  const dimKeep = (1 - tintStrength) * dimFactor;
  const tintDim = tintStrength * dimFactor;

  // Magenta: rgb(255, 0, 255)
  const magR = 255, magG = 0, magB = 255;

  for (let i = 0; i < W * H; i++) {
    const srcIdx = i * 4;
    if (mask[i] === 0) {
      // Frozen region — copy unchanged
      out[srcIdx]     = basePx[srcIdx];
      out[srcIdx + 1] = basePx[srcIdx + 1];
      out[srcIdx + 2] = basePx[srcIdx + 2];
      out[srcIdx + 3] = basePx[srcIdx + 3];
    } else {
      // Editable region — magenta tint + dim
      out[srcIdx]     = Math.round(basePx[srcIdx]     * dimKeep + magR * tintDim);
      out[srcIdx + 1] = Math.round(basePx[srcIdx + 1] * dimKeep + magG * tintDim);
      out[srcIdx + 2] = Math.round(basePx[srcIdx + 2] * dimKeep + magB * tintDim);
      out[srcIdx + 3] = 255;
    }
  }

  return out;
}

// ─── 3. Build inpainting prompt ───────────────────────────────────────────

/**
 * Construct a prompt that explicitly instructs FLUX to treat the magenta
 * region as the only editable zone and keep everything else byte-identical.
 */
export function buildInpaintingPrompt(label) {
  const normalized = normalizeGarmentLabel(label);

  const garmentZone = normalized === 'pants'
    ? 'lower body, hips, thighs, and legs'
    : normalized === 'shoes'
      ? 'feet, ankles, and the area immediately around the shoes'
      : normalized === 'layer'
        ? 'shoulders, chest, sleeves, and outer torso'
        : 'upper torso, chest, shoulders, and sleeves';

  return [
    'You are a virtual try-on inpainting engine. The input is a side-by-side image:',
    'LEFT HALF: a headless, light-grey/white fashion mannequin on a clean white studio background. The mannequin has a MAGENTA-TINTED region that marks the ONLY editable area — this is where the garment must be placed.',
    'RIGHT HALF: the exact product photo of the garment that must be worn.',
    `TASK: Replace ONLY the magenta-tinted region in the LEFT HALF with a photorealistic rendering of the garment from the right half, covering the ${garmentZone}. The garment should look naturally WORN and DRESSED — realistic drape, correct fabric folds, natural hems, settled sleeves and collars, proper layering where garments overlap.`,
    'HARD CONSTRAINTS — VIOLATING ANY OF THESE IS A FAILURE:',
    '- Every pixel that is NOT magenta-tinted in the input MUST remain byte-identical in the output. The mannequin head, neck, arms below the sleeves, legs below the hem, skin, and white background MUST NOT change at all.',
    '- The mannequin MUST stay the same: headless silhouette, light-grey/white body color, same pose, proportions, height, arm/leg position, camera angle, framing. No body changes.',
    '- The white seamless studio background MUST stay pixel-identical. No new shadows beyond subtle contact shadows directly under garment edges.',
    '- Do NOT introduce a human face, hair, skin tone, extra limbs, hands, jewelry, or any new accessory. The figure stays a mannequin.',
    '- Do NOT add, remove, or change any garment outside the magenta zone. Other already-worn garments must remain visually identical — this is a cumulative outfit build.',
    '- Keep the exact garment color, pattern, logos, fabric texture, silhouette, and design details from the right-half product photo. Match them faithfully.',
    '- The right half of the output must be a clean white background only (the product reference can disappear).',
    'STYLE: studio fashion catalog look, soft even lighting, sharp focus, photorealistic, premium e-commerce product photography.',
  ].join(' ');
}

// ─── 4. Build side-by-side composite for FLUX ─────────────────────────────

/**
 * Create a 1536×1024 side-by-side composite:
 *   Left panel  = tinted mannequin (768×1024)
 *   Right panel = product photo (768×1024)
 */
export async function buildInpaintingComposite(tintedPx, garmentSrc, label) {
  // Encode tinted mannequin to PNG
  const tintedBuf = await sharp(
    Buffer.from(tintedPx.buffer, tintedPx.byteOffset, tintedPx.byteLength),
    { raw: { width: W, height: H, channels: 4 } },
  )
    .png()
    .toBuffer();

  // Load and resize garment product photo
  let garmentBuf;
  try {
    const cleaned = await preprocessGarmentAndCache(garmentSrc, label);
    garmentBuf = cleaned;
  } catch {
    garmentBuf = await loadImageBuffer(garmentSrc);
  }

  const [left, right] = await Promise.all([
    sharp(tintedBuf)
      .resize(768, 1024, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } })
      .png()
      .toBuffer(),
    sharp(garmentBuf)
      .resize(768, 1024, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } })
      .png()
      .toBuffer(),
  ]);

  const composite = await sharp({
    create: { width: 1536, height: 1024, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } },
  })
    .composite([
      { input: left, left: 0, top: 0 },
      { input: right, left: 768, top: 0 },
    ])
    .png()
    .toBuffer();

  return `data:image/png;base64,${composite.toString('base64')}`;
}

// ─── 5. Extract FLUX left half ─────────────────────────────────────────────

/**
 * Decode a FLUX-Kontext side-by-side result and return a raw RGBA
 * Uint8ClampedArray (W*H*4) of just the LEFT half (the dressed mannequin),
 * resampled to the canvas size W × H.
 */
export async function extractFluxLeftHalfRaw(fluxDataUri) {
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

// ─── 6. Hard pixel snap ───────────────────────────────────────────────────

/**
 * Merge FLUX-refined pixels into `basePx` using a BINARY mask:
 *   - mask[i] == 0 → copy preStep pixel (frozen region, byte-identical to original)
 *   - mask[i] > 0  → copy FLUX pixel (editable region, garment rendered by FLUX)
 *
 * No feathering, no blending. The magenta tint is mathematically impossible
 * to reach the output: frozen pixels come from preStep (never tinted),
 * editable pixels come from FLUX (which was told to replace the magenta).
 */
export function hardPixelSnap(basePx, preStep, fluxRaw, mask) {
  const total = W * H;
  for (let i = 0; i < total; i++) {
    const idx = i * 4;
    if (mask[i] === 0) {
      // Frozen — restore original pixel
      basePx[idx]     = preStep[idx];
      basePx[idx + 1] = preStep[idx + 1];
      basePx[idx + 2] = preStep[idx + 2];
      basePx[idx + 3] = preStep[idx + 3];
    } else {
      // Editable — use FLUX output
      basePx[idx]     = fluxRaw[idx];
      basePx[idx + 1] = fluxRaw[idx + 1];
      basePx[idx + 2] = fluxRaw[idx + 2];
      basePx[idx + 3] = 255;
    }
  }
}

// ─── 7. NVIDIA asset upload ────────────────────────────────────────────────

export async function uploadNvidiaInputAsset({ nvidiaKey, imageBuffer, contentType, description }) {
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

  const createData = await createRes.json();
  const assetId = createData?.assetId;
  const uploadUrl = createData?.uploadUrl;
  if (!assetId || !uploadUrl) {
    throw new Error('NVIDIA asset create returned no assetId/uploadUrl');
  }

  const putRes = await fetch(uploadUrl, {
    method: 'PUT',
    headers: {
      'Content-Type': contentType,
      'x-amz-meta-nvcf-asset-description': description || 'aiwardrobe-inpainting-input',
    },
    body: imageBuffer,
  });

  if (!putRes.ok) {
    throw new Error(`NVIDIA asset upload failed (${putRes.status}): ${await putRes.text()}`);
  }

  return assetId;
}

// ─── 8. Aspect ratio helper ────────────────────────────────────────────────

export function buildKontextAspectRatio(imageMeta) {
  const width = Number(imageMeta?.width || 0);
  const height = Number(imageMeta?.height || 0);
  if (!width || !height) return 'match_input_image';
  const ratio = width / Math.max(1, height);
  if (ratio > 1.35) return '3:2';
  if (ratio > 1.15) return '4:3';
  if (ratio > 0.85) return '1:1';
  if (ratio > 0.65) return '3:4';
  if (ratio > 0.45) return '2:3';
  return 'match_input_image';
}

// ─── 9. Full inpainting step ───────────────────────────────────────────────

const NVIDIA_KONTEXT_URL = 'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';

/**
 * Process one garment step using the inpainting strategy:
 *   1. Snapshot pre-step pixels
 *   2. Build binary mask from anatomical regions
 *   3. Apply magenta tint to editable zone
 *   4. Build side-by-side composite (tinted mannequin | product photo)
 *   5. Call FLUX.1-Kontext-dev with inpainting prompt
 *   6. Extract FLUX's left half
 *   7. Hard pixel-snap: frozen → preStep, editable → FLUX
 */
export async function applyInpaintingStep(basePx, garmentSrc, label, nvidiaKey) {
  const normalized = normalizeGarmentLabel(label);

  // 1. Snapshot pre-step pixels
  const preStep = new Uint8ClampedArray(basePx);

  // 2. Build binary mask
  const mask = buildInpaintingMask(normalized);

  // 3. Apply magenta tint (does not mutate basePx)
  const tinted = applyMagentaTint(basePx, mask);

  // 4. Build side-by-side composite
  const composite = await buildInpaintingComposite(tinted, garmentSrc, normalized);
  const compositeBuffer = await loadImageBuffer(composite);
  const compositeMeta = await sharp(compositeBuffer).metadata();

  // 5. Upload to NVIDIA and call FLUX
  const compositeAssetId = await uploadNvidiaInputAsset({
    nvidiaKey,
    imageBuffer: compositeBuffer,
    contentType: detectImageContentType(composite, 'image/png'),
    description: `aiwardrobe-inpainting-${normalized}`,
  });

  const res = await fetch(NVIDIA_KONTEXT_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${nvidiaKey}`,
      'Content-Type': 'application/json',
      Accept: 'application/json',
      'NVCF-INPUT-ASSET-REFERENCES': compositeAssetId,
    },
    body: JSON.stringify({
      prompt: buildInpaintingPrompt(normalized),
      image: `data:image/png;example_id,${compositeAssetId}`,
      aspect_ratio: buildKontextAspectRatio(compositeMeta),
    }),
  });

  if (!res.ok) {
    throw new Error(
      `FLUX.1 Kontext-dev inpainting failed HTTP ${res.status} label=${normalized} asset=${compositeAssetId}: ${await res.text()}`,
    );
  }

  const data = await res.json();
  let refinedDataUri;
  if (data.artifacts?.[0]?.base64) {
    refinedDataUri = `data:image/png;base64,${data.artifacts[0].base64}`;
  } else if (data.image) {
    refinedDataUri = data.image.startsWith('data:') ? data.image : `data:image/png;base64,${data.image}`;
  } else if (data.output?.image) {
    refinedDataUri = data.output.image;
  } else {
    throw new Error(`FLUX.1 Kontext-dev returned no image label=${normalized} asset=${compositeAssetId}`);
  }

  // 6. Extract FLUX's left half
  const fluxRaw = await extractFluxLeftHalfRaw(refinedDataUri);
  if (!fluxRaw) {
    throw new Error(`FLUX.1 Kontext-dev output could not be decoded label=${normalized}`);
  }

  // 7. Hard pixel snap
  hardPixelSnap(basePx, preStep, fluxRaw, mask);

  return { label: normalized, fluxApplied: true };
}

// ─── 10. Standalone entry point for strategy dispatch ─────────────────────

/**
 * Called by the strategy dispatch in tryon.js when TRYON_STRATEGY=v2.
 * Accepts the same request body as the /render route and returns the
 * result object directly (no Express req/res).
 */
export async function inpaintingRender(body) {
  const startedAt = Date.now();
  const mannequinSrc = body.mannequin_image;
  const garmentEntries = Array.isArray(body.garments) ? body.garments : [];
  const garmentSrc =
    body.garment_image ||
    body.garment?.image ||
    body.garment?.imageUrl ||
    body.garment?.url;
  const rawLabel = body.garment?.label || body.garment?.type || 'top';
  const step = Number(body.step ?? 1);
  const total = Number(body.total ?? 1);

  if (!mannequinSrc) throw new Error('mannequin_image is required');
  if (!garmentSrc && garmentEntries.length === 0) throw new Error('garment_image is required');

  // Resolve NVIDIA key from env (strategy dispatch route provides Supabase access)
  const nvidiaKey = process.env.NVIDIA_API_KEY_FLUX_1 || process.env.NVIDIA_API_KEY;
  if (!nvidiaKey) throw new Error('NVIDIA FLUX.1 Kontext-dev token is not configured');

  // ── Multi-garment outfit ──────────────────────────────────────────────
  if (garmentEntries.length > 0) {
    const orderedGarments = GARMENT_RENDER_ORDER
      .map((label) => {
        const entry = garmentEntries.find((g) =>
          normalizeGarmentLabel(g?.label || g?.type || label) === label,
        );
        if (!entry) return null;
        return {
          label,
          garmentSrc:
            entry?.garmentSrc ||
            entry?.garment_image ||
            entry?.image ||
            entry?.imageUrl ||
            entry?.url,
        };
      })
      .filter((g) => g?.garmentSrc);

    if (orderedGarments.length === 0) throw new Error('No valid garments supplied');

    const basePx = await buildBaseCanvas(mannequinSrc);
    let fluxStepCount = 0;
    const stepLabels = [];

    for (const garment of orderedGarments) {
      const stepResult = await applyInpaintingStep(basePx, garment.garmentSrc, garment.label, nvidiaKey);
      if (stepResult.fluxApplied) fluxStepCount += 1;
      stepLabels.push(stepResult.label);
    }

    const finalLabel = stepLabels[stepLabels.length - 1] || 'outfit';
    const encoded = await encodeCanvas(basePx, finalLabel);
    const elapsedMs = Date.now() - startedAt;

    return {
      success: true,
      resultUrl: encoded.imageDataUri,
      methodUsed: 'inpainting_frozen_region_v2',
      fluxStepCount,
      step: orderedGarments.length,
      total: orderedGarments.length,
      garmentLabel: finalLabel,
      renderedGarments: stepLabels,
      elapsedMs,
    };
  }

  // ── Single garment step ───────────────────────────────────────────────
  const label = normalizeGarmentLabel(rawLabel, step);
  const basePx = await buildBaseCanvas(mannequinSrc);
  const stepResult = await applyInpaintingStep(basePx, garmentSrc, label, nvidiaKey);
  const encoded = await encodeCanvas(basePx, stepResult.label);
  const elapsedMs = Date.now() - startedAt;

  return {
    success: true,
    resultUrl: encoded.imageDataUri,
    methodUsed: 'inpainting_frozen_region_v2',
    step,
    total,
    garmentLabel: stepResult.label,
    elapsedMs,
  };
}

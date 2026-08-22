/**
 * Shared utilities for all try-on strategies (v1, v2, v3).
 *
 * Extracted from routes/tryon.js so each strategy can reuse the same
 * NVIDIA NIM call wrapper, image I/O helpers, and mask-merge logic
 * without duplicating code.
 */

import sharp from 'sharp';
import { supabase } from '../lib/supabase.js';
import logger from '../utils/logger.js';
import { dilateAndFeatherMask, W, H } from './tryonRenderer.js';

export { W, H };

// ── NVIDIA FLUX.1-Kontext-dev endpoint ──
export const NVIDIA_KONTEXT_URL =
  'https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-kontext-dev';

// Self-hosted NIM endpoint (supports inline base64 images).
export const NVIDIA_KONTEXT_LOCAL_URL = 'http://localhost:8000/v1/infer';

// ── Image I/O helpers ──

export function stripDataUri(dataUri) {
  if (typeof dataUri !== 'string') return '';
  if (dataUri.startsWith('data:')) return dataUri.split(',')[1] || '';
  return dataUri;
}

export function toDataUri(buffer, contentType = 'image/png') {
  return `data:${contentType};base64,${buffer.toString('base64')}`;
}

export async function loadImageBuffer(src) {
  if (!src) throw new Error('Empty image source');
  if (src.startsWith('data:')) return Buffer.from(stripDataUri(src), 'base64');
  if (src.startsWith('http://') || src.startsWith('https://')) {
    const res = await fetch(src);
    if (!res.ok) throw new Error(`Failed to fetch image (${res.status})`);
    return Buffer.from(await res.arrayBuffer());
  }
  return Buffer.from(src, 'base64');
}

export function detectImageContentType(src, fallback = 'image/png') {
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

// ── NVIDIA NVCF asset upload (cloud NIM) ──

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
      'x-amz-meta-nvcf-asset-description': description || 'aiwardrobe-kontext-input',
    },
    body: imageBuffer,
  });

  if (!putRes.ok) {
    throw new Error(`NVIDIA asset upload failed (${putRes.status}): ${await putRes.text()}`);
  }

  return assetId;
}

// ── NVIDIA key lookup ──

export async function getNvidiaKey() {
  const tokenRow = await supabase
    .from('app_config')
    .select('value')
    .eq('key', 'nvidia_token')
    .maybeSingle();
  return (
    tokenRow.data?.value ||
    process.env.NVIDIA_API_KEY_FLUX_1 ||
    process.env.NVIDIA_API_KEY ||
    null
  );
}

// ── Aspect ratio helper ──

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

// ── Replicate FLUX.1-Kontext-dev ──

const REPLICATE_FLUX_KONTEXT_MODEL =
  process.env.REPLICATE_FLUX_KONTEXT_MODEL ||
  'black-forest-labs/flux-kontext-dev';

async function callFluxReplicate({ imageDataUri, prompt }) {
  const token = process.env.REPLICATE_API_TOKEN;
  if (!token) {
    throw new Error('REPLICATE_API_TOKEN is not configured');
  }

  // Replicate expects a public URL or a data URI. We send the data URI directly.
  const res = await fetch('https://api.replicate.com/v1/predictions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
      Prefer: 'wait',
    },
    body: JSON.stringify({
      version: REPLICATE_FLUX_KONTEXT_MODEL,
      input: {
        prompt,
        image: imageDataUri,
        aspect_ratio: '1:1',
        output_format: 'png',
      },
    }),
  });

  if (!res.ok) {
    throw new Error(
      `Replicate FLUX.1-Kontext-dev failed HTTP ${res.status}: ${await res.text()}`
    );
  }

  const data = await res.json();
  // Prefer synchronous response (Prefer: wait), fallback to polling not needed for <30s
  const output = data.output || data;
  if (typeof output === 'string' && output.startsWith('http')) {
    // Replicate returned a URL — fetch and convert to base64 data URI
    const imgRes = await fetch(output);
    if (!imgRes.ok) throw new Error(`Replicate output fetch failed ${imgRes.status}`);
    const buf = Buffer.from(await imgRes.arrayBuffer());
    return `data:image/png;base64,${buf.toString('base64')}`;
  }
  if (typeof output === 'string' && output.startsWith('data:')) {
    return output;
  }
  if (output && typeof output === 'object') {
    // Some versions return { image: "url" } or an array
    const url = output.image || (Array.isArray(output) ? output[0] : null);
    if (url && typeof url === 'string') {
      if (url.startsWith('data:')) return url;
      const imgRes = await fetch(url);
      if (!imgRes.ok) throw new Error(`Replicate output fetch failed ${imgRes.status}`);
      const buf = Buffer.from(await imgRes.arrayBuffer());
      return `data:image/png;base64,${buf.toString('base64')}`;
    }
  }
  throw new Error(`Replicate FLUX.1-Kontext-dev returned unexpected output: ${JSON.stringify(data).slice(0, 200)}`);
}

// ── FLUX call wrapper (multi-provider) ──

/**
 * Call FLUX.1-Kontext-dev via the configured provider.
 *
 * @param {object} opts
 * @param {string} opts.imageDataUri  - data:image/png;base64,... composite to send
 * @param {string} opts.prompt        - text prompt
 * @param {string} [opts.nvidiaKey]   - pre-fetched key (skips DB lookup if provided)
 * @param {string} [opts.provider]    - 'nvidia_cloud' | 'nvidia_local' | 'replicate'
 * @returns {Promise<string>} data:image/png;base64,... result
 */
export async function callFluxKontext({ imageDataUri, prompt, nvidiaKey, provider } = {}) {
  const resolvedProvider =
    provider || process.env.FLUX_PROVIDER || 'nvidia_local';

  if (resolvedProvider === 'replicate') {
    return callFluxReplicate({ imageDataUri, prompt });
  }

  const key = nvidiaKey || (await getNvidiaKey());
  if (!key && resolvedProvider !== 'nvidia_local') {
    throw new Error('NVIDIA FLUX.1 Kontext-dev token is not configured');
  }

  if (resolvedProvider === 'nvidia_local') {
    return callFluxLocal({ imageDataUri, prompt });
  }

  // Default: nvidia_cloud (NVCF asset upload path)
  return callFluxCloud({ imageDataUri, prompt, nvidiaKey: key });
}

/**
 * Cloud NIM: upload image as NVCF asset, then call with asset reference.
 * NOTE: As of Apr 2026 this returns 500 for custom images — kept as dead
 * code path for when NVIDIA enables custom image support.
 */
async function callFluxCloud({ imageDataUri, prompt, nvidiaKey }) {
  const compositeBuffer = await loadImageBuffer(imageDataUri);
  const compositeMeta = await sharp(compositeBuffer).metadata();
  const compositeAssetId = await uploadNvidiaInputAsset({
    nvidiaKey,
    imageBuffer: compositeBuffer,
    contentType: detectImageContentType(imageDataUri, 'image/png'),
    description: 'aiwardrobe-kontext-input',
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
      prompt,
      image: `data:image/png;example_id,${compositeAssetId}`,
      aspect_ratio: buildKontextAspectRatio(compositeMeta),
    }),
  });

  if (!res.ok) {
    throw new Error(
      `FLUX.1 Kontext-dev (cloud) failed HTTP ${res.status} asset=${compositeAssetId}: ${await res.text()}`
    );
  }

  const data = await res.json();
  if (data.artifacts?.[0]?.base64) return `data:image/png;base64,${data.artifacts[0].base64}`;
  if (data.image) return data.image.startsWith('data:') ? data.image : `data:image/png;base64,${data.image}`;
  if (data.output?.image) return data.output.image;
  throw new Error(`FLUX.1 Kontext-dev (cloud) returned no image asset=${compositeAssetId}`);
}

/**
 * Self-hosted NIM: send inline base64 image directly.
 * This is the working path per NVIDIA's own docs.
 */
async function callFluxLocal({ imageDataUri, prompt }) {
  const localUrl = process.env.FLUX_LOCAL_URL || NVIDIA_KONTEXT_LOCAL_URL;
  const compositeBuffer = await loadImageBuffer(imageDataUri);
  const compositeMeta = await sharp(compositeBuffer).metadata();

  const res = await fetch(localUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({
      prompt,
      image: imageDataUri,
      aspect_ratio: buildKontextAspectRatio(compositeMeta),
      steps: 30,
      seed: 0,
    }),
  });

  if (!res.ok) {
    throw new Error(
      `FLUX.1 Kontext-dev (local) failed HTTP ${res.status}: ${await res.text()}`
    );
  }

  const data = await res.json();
  if (data.artifacts?.[0]?.base64) return `data:image/png;base64,${data.artifacts[0].base64}`;
  if (data.image) return data.image.startsWith('data:') ? data.image : `data:image/png;base64,${data.image}`;
  if (data.output?.image) return data.output.image;
  throw new Error('FLUX.1 Kontext-dev (local) returned no image');
}

// ── FLUX output extraction ──

/**
 * Decode a FLUX-Kontext result and return a raw RGBA Uint8ClampedArray
 * (W*H*4) of a specified region, resampled to W×H.
 */
export async function extractFluxRegionRaw(fluxDataUri, { left = 0, top = 0, width, height } = {}) {
  if (!fluxDataUri) return null;
  const buf = await loadImageBuffer(fluxDataUri);
  const meta = await sharp(buf).metadata();
  const fw = meta.width || 0;
  const fh = meta.height || 0;
  if (!fw || !fh) return null;

  const extractW = Math.min(width || fw, fw - left);
  const extractH = Math.min(height || fh, fh - top);
  if (extractW <= 0 || extractH <= 0) return null;

  const regionBuf = await sharp(buf)
    .extract({ left, top, width: extractW, height: extractH })
    .resize(W, H, { fit: 'fill' })
    .ensureAlpha()
    .raw()
    .toBuffer();
  return new Uint8ClampedArray(regionBuf);
}

/**
 * Back-compat: extract left half of a side-by-side FLUX result.
 */
export async function extractFluxLeftHalfRaw(fluxDataUri) {
  if (!fluxDataUri) return null;
  const buf = await loadImageBuffer(fluxDataUri);
  const meta = await sharp(buf).metadata();
  const fw = meta.width || 0;
  const fh = meta.height || 0;
  if (!fw || !fh) return null;
  const halfWidth = Math.max(1, Math.floor(fw / 2));
  return extractFluxRegionRaw(fluxDataUri, { left: 0, top: 0, width: halfWidth, height: fh });
}

// ── Change detection mask (FLUX-only pipeline) ──

/**
 * Compute a binary mask of pixels that meaningfully changed between
 * `preStep` and `fluxRaw` (both Uint8ClampedArray of length W*H*4 in RGBA).
 *
 * A pixel is considered "changed" when the per-channel mean absolute
 * difference exceeds `threshold` (0..255). The result is a Uint8ClampedArray
 * of length W*H where 255 = changed (FLUX modified it = garment region) and
 * 0 = unchanged (mannequin / background must be preserved).
 *
 * Includes a sanity check: if the mask covers more than `maxCoverage`
 * fraction of the canvas, FLUX has drifted the entire image (camera shift,
 * cropping, etc.) and we should NOT trust it for mannequin preservation —
 * the caller should reject the result.
 *
 * @param {Uint8ClampedArray} preStep
 * @param {Uint8ClampedArray} fluxRaw
 * @param {object} [opts]
 * @param {number} [opts.threshold=14]   per-channel mean abs diff to count as changed
 * @param {number} [opts.maxCoverage=0.85] reject if change region > this fraction
 * @returns {{ mask: Uint8ClampedArray, coverage: number, drifted: boolean }}
 */
export function computeDiffMask(preStep, fluxRaw, opts = {}) {
  const threshold = opts.threshold ?? 14;
  const maxCoverage = opts.maxCoverage ?? 0.85;
  const total = W * H;
  const mask = new Uint8ClampedArray(total);
  let changed = 0;
  for (let i = 0; i < total; i++) {
    const j = i * 4;
    const dr = Math.abs(preStep[j] - fluxRaw[j]);
    const dg = Math.abs(preStep[j + 1] - fluxRaw[j + 1]);
    const db = Math.abs(preStep[j + 2] - fluxRaw[j + 2]);
    const mean = (dr + dg + db) / 3;
    if (mean > threshold) {
      mask[i] = 255;
      changed += 1;
    }
  }
  const coverage = changed / total;
  return { mask, coverage, drifted: coverage > maxCoverage };
}

// ── Mask merge ──

/**
 * Merge FLUX-refined pixels into `basePx` ONLY where `mask` is non-zero,
 * leaving the rest of the canvas pixel-identical to `preStep`.
 */
export function maskMergeFluxIntoBase(basePx, preStep, fluxRaw, mask) {
  const total = W * H;
  for (let i = 0; i < total; i++) {
    const m = mask[i] / 255;
    if (m <= 0.005) {
      basePx[i * 4] = preStep[i * 4];
      basePx[i * 4 + 1] = preStep[i * 4 + 1];
      basePx[i * 4 + 2] = preStep[i * 4 + 2];
      basePx[i * 4 + 3] = preStep[i * 4 + 3];
      continue;
    }
    if (m >= 0.995) {
      basePx[i * 4] = fluxRaw[i * 4];
      basePx[i * 4 + 1] = fluxRaw[i * 4 + 1];
      basePx[i * 4 + 2] = fluxRaw[i * 4 + 2];
      basePx[i * 4 + 3] = 255;
      continue;
    }
    const inv = 1 - m;
    basePx[i * 4] = Math.round(preStep[i * 4] * inv + fluxRaw[i * 4] * m);
    basePx[i * 4 + 1] = Math.round(preStep[i * 4 + 1] * inv + fluxRaw[i * 4 + 1] * m);
    basePx[i * 4 + 2] = Math.round(preStep[i * 4 + 2] * inv + fluxRaw[i * 4 + 2] * m);
    basePx[i * 4 + 3] = 255;
  }
}

export { dilateAndFeatherMask };

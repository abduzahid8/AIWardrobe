/**
 * Deterministic mannequin try-on renderer (Node + sharp).
 *
 * The mannequin never changes — garments are placed inside per-category
 * anchor boxes with feathered alpha masks and category-specific shaping.
 * Heavy per-pixel work (bg removal, skin suppression, connected-component
 * extraction, feathering, blending) runs on raw RGBA buffers; sharp
 * (libvips) handles decode/encode/resize/blur natively.
 *
 * Garment cleanup is cached on disk by SHA1 of the source URL/data,
 * so a given catalog item is preprocessed exactly once.
 */

import sharp from 'sharp';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { removeBackground } from './bgRemovalService.js';
import { extractGarment } from './clothesSegmenter.js';

// Working canvas size — must match anchor coordinates below.
const W = 1024;
const H = 1024;

export const GARMENT_RENDER_ORDER = ['top', 'layer', 'pants', 'shoes'];

// Anatomical anchor boxes for assets/images/mannequin_front.png.
// Normalized [0..1] of a 1024x1024 working canvas.
const ANCHOR_BOXES = {
  top:   { x0: 0.33, y0: 0.19, x1: 0.67, y1: 0.49 },
  layer: { x0: 0.28, y0: 0.18, x1: 0.72, y1: 0.53 },
  pants: { x0: 0.34, y0: 0.50, x1: 0.66, y1: 0.92 },
  shoes: { x0: 0.35, y0: 0.89, x1: 0.65, y1: 0.985 },
};

const MANNEQUIN_REGIONS = {
  torso:      { cx: 0.50, cy: 0.33, rx: 0.155, ry: 0.18, softness: 0.2 },
  leftSleeve: { cx: 0.31, cy: 0.34, rx: 0.10, ry: 0.23, softness: 0.22 },
  rightSleeve:{ cx: 0.69, cy: 0.34, rx: 0.10, ry: 0.23, softness: 0.22 },
  waist:      { cx: 0.50, cy: 0.50, rx: 0.12, ry: 0.055, softness: 0.18 },
  hips:       { cx: 0.50, cy: 0.57, rx: 0.16, ry: 0.085, softness: 0.18 },
  leftLeg:    { cx: 0.43, cy: 0.74, rx: 0.09, ry: 0.22, softness: 0.2 },
  rightLeg:   { cx: 0.57, cy: 0.74, rx: 0.09, ry: 0.22, softness: 0.2 },
  leftFoot:   { cx: 0.43, cy: 0.95, rx: 0.11, ry: 0.035, softness: 0.24 },
  rightFoot:  { cx: 0.57, cy: 0.95, rx: 0.11, ry: 0.035, softness: 0.24 },
};

const GARMENT_TEMPLATES = {
  top: {
    torsoWidth: [1.16, 1.08, 0.96, 0.82],
    sleeveSpread: 0.12,
    sleeveAlpha: 0.72,
    hemStart: 0.70,
    hemTaper: 0.76,
    waistSuppression: 0.12,
  },
  layer: {
    torsoWidth: [1.22, 1.16, 1.05, 0.92],
    sleeveSpread: 0.15,
    sleeveAlpha: 0.82,
    hemStart: 0.64,
    hemTaper: 0.9,
    waistSuppression: 0.04,
  },
  pants: {
    waistWidth: 1.18,
    hipWidth: 1.12,
    thighWidth: 0.98,
    calfWidth: 0.82,
    hemWidth: 0.74,
    inseamGap: 0.07,
    legSpread: 0.145,
  },
  shoes: {
    pairSpread: 0.14,
    toeScale: 1.04,
  },
};

const CACHE_DIR = path.join(process.cwd(), 'cache', 'garments');

export function normalizeGarmentLabel(label, step = 1) {
  const l = String(label || '').toLowerCase();
  if (l === 'layer' || l === 'outerwear' || l === 'jacket' || l === 'coat') return 'layer';
  if (l === 'pants' || l === 'lower_body' || l === 'trousers' || l === 'bottom') return 'pants';
  if (l === 'shoes' || l === 'footwear') return 'shoes';
  if (l === 'top' || l === 'shirt' || l === 'tee' || l === 't-shirt') return 'top';
  if (l === 'upper_body') return step <= 1 ? 'top' : 'layer';
  return 'top';
}

function hashSrc(src) {
  return crypto.createHash('sha1').update(src).digest('hex');
}

async function loadImageBuffer(src) {
  if (!src) throw new Error('Empty image source');
  if (src.startsWith('data:')) {
    return Buffer.from(src.split(',')[1] || '', 'base64');
  }
  if (src.startsWith('http://') || src.startsWith('https://')) {
    const res = await fetch(src);
    if (!res.ok) throw new Error(`Failed to fetch image (${res.status}): ${src.slice(0, 80)}`);
    return Buffer.from(await res.arrayBuffer());
  }
  // Assume bare base64
  return Buffer.from(src, 'base64');
}


// =============================================================================
// Garment preprocessing — runs once per source, then cached on disk.
//
// Uses the locally-hosted briaai/RMBG-1.4 ONNX model for semantic
// background removal. Unlike colour-based heuristics, RMBG correctly handles
// white garments on white backgrounds (it separates by visual semantics, not
// chromatic distance). After RMBG we crop to the opaque bbox so the renderer
// scales the garment, not the empty padding.
// =============================================================================
async function preprocessGarment(src, label = 'top') {
  await fs.mkdir(CACHE_DIR, { recursive: true });
  const cacheKey = hashSrc(`segformer-rmbg-v3::${label}::${src}`);
  const cachePath = path.join(CACHE_DIR, `${cacheKey}.png`);

  try {
    const cached = await fs.readFile(cachePath);
    return cached;
  } catch {}

  const raw = await loadImageBuffer(src);
  const resized = await sharp(raw)
    .resize({ width: 1024, height: 1024, fit: 'inside', withoutEnlargement: true })
    .png()
    .toBuffer();

  let cutoutPng = null;
  const segStart = Date.now();
  try {
    const segmented = await extractGarment(resized, label);
    if (!(await hasSuspiciouslyOpaqueAlpha(segmented))) {
      cutoutPng = segmented;
      console.log(`[preprocess] SegFormer cleaned label=${label} in ${Date.now() - segStart}ms`);
    }
  } catch (err) {
    console.warn(`[preprocess] SegFormer failed for label=${label}:`, err?.message || err);
  }

  if (!cutoutPng) {
    const t0 = Date.now();
    cutoutPng = await removeBackground(resized);
    console.log(`[preprocess] RMBG cleaned label=${label} in ${Date.now() - t0}ms`);
  }

  if (await hasSuspiciouslyOpaqueAlpha(cutoutPng)) {
    cutoutPng = await removeLightBackgroundFallback(resized);
  }

  // Crop to the opaque bounding box.
  const { data, info } = await sharp(cutoutPng)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const w = info.width;
  const h = info.height;
  let minX = w;
  let minY = h;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (data[(y * w + x) * 4 + 3] > 10) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX || maxY < minY) {
    minX = 0;
    minY = 0;
    maxX = w - 1;
    maxY = h - 1;
  }
  const bw = Math.max(1, maxX - minX + 1);
  const bh = Math.max(1, maxY - minY + 1);

  const cleanedBuf = await sharp(cutoutPng)
    .extract({ left: minX, top: minY, width: bw, height: bh })
    .png()
    .toBuffer();

  fs.writeFile(cachePath, cleanedBuf).catch(() => {});
  return cleanedBuf;
}

async function hasSuspiciouslyOpaqueAlpha(pngBuffer) {
  const { data, info } = await sharp(pngBuffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const total = info.width * info.height;
  let transparent = 0;
  let nearlyOpaque = 0;
  for (let i = 0; i < total; i++) {
    const a = data[i * 4 + 3];
    if (a <= 8) transparent += 1;
    if (a >= 245) nearlyOpaque += 1;
  }
  return transparent / Math.max(1, total) < 0.01 && nearlyOpaque / Math.max(1, total) > 0.98;
}

async function removeLightBackgroundFallback(srcBuf) {
  const { data, info } = await sharp(srcBuf)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const width = info.width;
  const height = info.height;
  const total = width * height;
  const visited = new Uint8Array(total);
  const bg = new Uint8Array(total);
  const queue = new Int32Array(total);
  let head = 0;
  let tail = 0;

  const edgeStats = { r: 0, g: 0, b: 0, count: 0 };
  const sampleEdge = (x, y) => {
    const idx = (y * width + x) * 4;
    edgeStats.r += data[idx];
    edgeStats.g += data[idx + 1];
    edgeStats.b += data[idx + 2];
    edgeStats.count += 1;
  };

  for (let x = 0; x < width; x++) {
    sampleEdge(x, 0);
    sampleEdge(x, height - 1);
  }
  for (let y = 1; y < height - 1; y++) {
    sampleEdge(0, y);
    sampleEdge(width - 1, y);
  }

  const baseR = edgeStats.r / Math.max(1, edgeStats.count);
  const baseG = edgeStats.g / Math.max(1, edgeStats.count);
  const baseB = edgeStats.b / Math.max(1, edgeStats.count);

  const brightness = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const isBackground = (x, y) => {
    const idx = (y * width + x) * 4;
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];
    const a = data[idx + 3];
    if (a <= 8) return true;
    const light = brightness(r, g, b);
    if (light < 238) return false;
    const dist = Math.sqrt((r - baseR) ** 2 + (g - baseG) ** 2 + (b - baseB) ** 2);
    return dist < 24;
  };

  const push = (x, y) => {
    const idx = y * width + x;
    if (visited[idx]) return;
    visited[idx] = 1;
    queue[tail++] = idx;
  };

  for (let x = 0; x < width; x++) {
    push(x, 0);
    push(x, height - 1);
  }
  for (let y = 1; y < height - 1; y++) {
    push(0, y);
    push(width - 1, y);
  }

  while (head < tail) {
    const idx = queue[head++];
    const x = idx % width;
    const y = Math.floor(idx / width);
    if (!isBackground(x, y)) continue;
    bg[idx] = 1;

    if (x > 0) push(x - 1, y);
    if (x + 1 < width) push(x + 1, y);
    if (y > 0) push(x, y - 1);
    if (y + 1 < height) push(x, y + 1);
  }

  const out = Buffer.from(data);
  for (let i = 0; i < total; i++) {
    if (bg[i]) {
      out[i * 4 + 3] = 0;
      continue;
    }

    const r = out[i * 4];
    const g = out[i * 4 + 1];
    const b = out[i * 4 + 2];
    const light = brightness(r, g, b);
    if (light > 242) {
      const dist = Math.sqrt((r - baseR) ** 2 + (g - baseG) ** 2 + (b - baseB) ** 2);
      if (dist < 30) {
        const alpha = Math.max(0, Math.min(255, Math.round(((dist - 10) / 20) * 255)));
        out[i * 4 + 3] = Math.min(out[i * 4 + 3], alpha);
      }
    }
  }

  return await sharp(out, { raw: { width, height, channels: 4 } })
    .png()
    .toBuffer();
}

async function sanitizeGarmentForLabel(pngBuffer, label) {
  const normalized = normalizeGarmentLabel(label);
  const { data, info } = await sharp(pngBuffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const width = info.width;
  const height = info.height;
  const out = Buffer.from(data);
  const brightness = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

  for (let y = 0; y < height; y++) {
    const ny = y / Math.max(1, height - 1);
    for (let x = 0; x < width; x++) {
      const nx = x / Math.max(1, width - 1);
      const idx = (y * width + x) * 4;
      const alpha = out[idx + 3];
      if (alpha <= 8) continue;

      const r = out[idx];
      const g = out[idx + 1];
      const b = out[idx + 2];
      const light = brightness(r, g, b);

      let keep = 1;
      if (normalized === 'top' || normalized === 'layer') {
        if (ny < 0.18) keep = 0;
        if (ny < 0.28 && Math.abs(nx - 0.5) < 0.24) keep = 0;
        if ((nx < 0.18 || nx > 0.82) && ny > 0.48) keep *= 0.15;
        if (ny > 0.84) keep *= 0.35;
      } else if (normalized === 'pants') {
        if (ny < 0.16) keep = 0;
        if ((nx < 0.16 || nx > 0.84) && ny < 0.38) keep *= 0.2;
        if (Math.abs(nx - 0.5) < 0.035 && ny > 0.58) keep *= 0.28;
      } else if (normalized === 'shoes') {
        if (ny < 0.45) keep = 0;
        if (light > 245 && alpha < 220) keep *= 0.2;
      }

      const skinLike = r > 150 && g > 110 && b > 90 && r > g && g > b * 0.8;
      if (skinLike) {
        if (normalized === 'top' || normalized === 'layer') {
          if (ny < 0.22 || (ny > 0.52 && (nx < 0.3 || nx > 0.7))) keep *= 0.08;
        } else if (normalized === 'pants') {
          if (ny < 0.28 || nx < 0.22 || nx > 0.78) keep *= 0.12;
        }
      }

      out[idx + 3] = clampByte(alpha * keep);
    }
  }

  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const alpha = out[(y * width + x) * 4 + 3];
      if (alpha <= 10) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  if (maxX < minX || maxY < minY) {
    return await sharp(out, { raw: { width, height, channels: 4 } })
      .png()
      .toBuffer();
  }

  return await sharp(out, { raw: { width, height, channels: 4 } })
    .extract({
      left: minX,
      top: minY,
      width: Math.max(1, maxX - minX + 1),
      height: Math.max(1, maxY - minY + 1),
    })
    .png()
    .toBuffer();
}

// Public wrapper so a separate seed script can warm the cache.
export async function preprocessGarmentAndCache(src, label = 'top') {
  return preprocessGarment(src, label);
}

function placementForLabel(label, box, gw, gh) {
  let scale;
  let extraOffsetY = 0;
  if (label === 'top') {
    scale = Math.min((box.pw * 1.2) / gw, (box.ph * 1.16) / gh);
    extraOffsetY = 18;
  } else if (label === 'layer') {
    scale = Math.min((box.pw * 1.22) / gw, (box.ph * 1.2) / gh);
    extraOffsetY = 8;
  } else if (label === 'pants') {
    scale = Math.min((box.pw * 1.28) / gw, (box.ph * 1.08) / gh);
    extraOffsetY = 4;
  } else {
    scale = Math.min((box.pw * 1.12) / gw, (box.ph * 1.08) / gh);
    extraOffsetY = 2;
  }
  const ngw = Math.max(1, Math.round(gw * scale));
  const ngh = Math.max(1, Math.round(gh * scale));
  return {
    width: ngw,
    height: ngh,
    left: box.px0 + Math.round((box.pw - ngw) / 2),
    top: box.py0 + Math.round((box.ph - ngh) / 2) + extraOffsetY,
  };
}

/**
 * Gaussian blur a single-channel W*H mask via sharp/libvips and return a
 * Uint8Array of length W*H. We explicitly downcast back to 1 channel because
 * sharp's `.raw()` after `.blur()` would otherwise emit interleaved sRGB.
 */
async function blurMask(mask, sigma) {
  const buf = await sharp(Buffer.from(mask.buffer, mask.byteOffset, mask.byteLength), {
    raw: { width: W, height: H, channels: 1 },
  })
    .blur(sigma)
    .toColourspace('b-w')
    .extractChannel(0)
    .raw()
    .toBuffer();
  // Defensive: if sharp still emitted multi-channel, take the first channel.
  if (buf.length === W * H) return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  const out = new Uint8Array(W * H);
  const stride = buf.length / (W * H);
  for (let i = 0; i < W * H; i++) out[i] = buf[i * stride];
  return out;
}

function applyCategoryShape(alpha, label) {
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const ny = y / (H - 1);
    for (let x = 0; x < W; x++) {
      const idx = y * W + x;
      const a = alpha[idx];
      if (a <= 1) continue;
      let weight = a / 255;
      const nx = x / (W - 1);
      if (label === 'top') {
        const center = 1 - Math.min(1, Math.abs(nx - 0.5) / 0.49);
        const torso = ny < 0.9 ? 1 : Math.max(0.55, 1 - (ny - 0.9) / 0.1);
        const shoulderLift = ny < 0.18 ? 0.9 + (ny / 0.18) * 0.1 : 1;
        weight *= Math.max(0.72, Math.pow(center, 0.42)) * torso * shoulderLift;
      } else if (label === 'layer') {
        const center = 1 - Math.min(1, Math.abs(nx - 0.5) / 0.54);
        const hem = ny < 0.94 ? 1 : Math.max(0.7, 1 - (ny - 0.94) / 0.06);
        weight *= (0.92 + center * 0.08) * hem;
      } else if (label === 'pants') {
        const innerGap = nx > 0.46 && nx < 0.54 && ny > 0.16 ? 0.72 : 1;
        const waist = ny < 0.12 ? 0.7 + (ny / 0.12) * 0.3 : 1;
        weight *= Math.max(0.68, innerGap) * waist;
      } else if (label === 'shoes') {
        const left = Math.abs(nx - 0.32) < 0.18;
        const right = Math.abs(nx - 0.68) < 0.18;
        weight *= left || right ? 1 : 0.08;
      }
      const boosted = Math.pow(Math.max(0, Math.min(1, weight)), 0.78);
      out[idx] = Math.max(0, Math.min(255, Math.round(boosted * 255)));
    }
  }
  return out;
}

function clampByte(value) {
  if (value <= 0) return 0;
  if (value >= 255) return 255;
  return Math.round(value);
}

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function mix(a, b, t) {
  return a + (b - a) * t;
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

function bodyMaskValue(region, nx, ny) {
  if (region === 'neck') return ellipticalFalloff(nx, ny, 0.5, 0.085, 0.07, 0.065, 0.22);
  if (region === 'left_arm') return ellipticalFalloff(nx, ny, 0.29, 0.34, 0.10, 0.24, 0.24);
  if (region === 'right_arm') return ellipticalFalloff(nx, ny, 0.71, 0.34, 0.10, 0.24, 0.24);
  if (region === 'pelvis') return ellipticalFalloff(nx, ny, 0.5, 0.56, 0.16, 0.08, 0.22);
  if (region === 'left_leg') return ellipticalFalloff(nx, ny, 0.43, 0.73, 0.10, 0.23, 0.22);
  if (region === 'right_leg') return ellipticalFalloff(nx, ny, 0.57, 0.73, 0.10, 0.23, 0.22);
  if (region === 'left_foot') return ellipticalFalloff(nx, ny, 0.43, 0.95, 0.12, 0.035, 0.26);
  if (region === 'right_foot') return ellipticalFalloff(nx, ny, 0.57, 0.95, 0.12, 0.035, 0.26);
  return 0;
}

function sampleAlphaBilinear(raw, width, height, x, y) {
  const sx = Math.max(0, Math.min(width - 1, x));
  const sy = Math.max(0, Math.min(height - 1, y));
  const x0 = Math.floor(sx);
  const y0 = Math.floor(sy);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const tx = sx - x0;
  const ty = sy - y0;
  const a00 = raw[(y0 * width + x0) * 4 + 3];
  const a10 = raw[(y0 * width + x1) * 4 + 3];
  const a01 = raw[(y1 * width + x0) * 4 + 3];
  const a11 = raw[(y1 * width + x1) * 4 + 3];
  const top = a00 * (1 - tx) + a10 * tx;
  const bottom = a01 * (1 - tx) + a11 * tx;
  return top * (1 - ty) + bottom * ty;
}

function samplePixelBilinear(raw, width, height, x, y) {
  const sx = Math.max(0, Math.min(width - 1, x));
  const sy = Math.max(0, Math.min(height - 1, y));
  const x0 = Math.floor(sx);
  const y0 = Math.floor(sy);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const tx = sx - x0;
  const ty = sy - y0;
  const out = [0, 0, 0, 0];
  for (let c = 0; c < 4; c++) {
    const p00 = raw[(y0 * width + x0) * 4 + c];
    const p10 = raw[(y0 * width + x1) * 4 + c];
    const p01 = raw[(y1 * width + x0) * 4 + c];
    const p11 = raw[(y1 * width + x1) * 4 + c];
    const top = p00 * (1 - tx) + p10 * tx;
    const bottom = p01 * (1 - tx) + p11 * tx;
    out[c] = top * (1 - ty) + bottom * ty;
  }
  return out;
}

function analyzeGarmentAlpha(raw, width, height) {
  const rowBounds = Array.from({ length: height }, () => ({ left: width, right: -1, count: 0 }));
  const colBounds = Array.from({ length: width }, () => ({ top: height, bottom: -1, count: 0 }));
  let minX = width;
  let maxX = -1;
  let minY = height;
  let maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const alpha = raw[(y * width + x) * 4 + 3];
      if (alpha <= 12) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      const row = rowBounds[y];
      if (x < row.left) row.left = x;
      if (x > row.right) row.right = x;
      row.count += 1;
      const col = colBounds[x];
      if (y < col.top) col.top = y;
      if (y > col.bottom) col.bottom = y;
      col.count += 1;
    }
  }
  if (maxX < minX || maxY < minY) {
    minX = 0;
    maxX = width - 1;
    minY = 0;
    maxY = height - 1;
  }
  return {
    minX,
    maxX,
    minY,
    maxY,
    rowBounds,
    colBounds,
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    bboxWidth: Math.max(1, maxX - minX + 1),
    bboxHeight: Math.max(1, maxY - minY + 1),
  };
}

function makeShapeProfile(label, ny) {
  const template = GARMENT_TEMPLATES[label] || GARMENT_TEMPLATES.top;
  if (label === 'top') {
    const shoulder = mix(template.torsoWidth[0], template.torsoWidth[1], smoothstep(0.04, 0.24, ny));
    const waist = mix(1, template.torsoWidth[3], smoothstep(0.54, 0.96, ny));
    return { widthScale: shoulder * waist, centerShift: 0, alphaScale: mix(1.02, 0.96, ny) };
  }
  if (label === 'layer') {
    const openFront = smoothstep(0.18, 0.82, ny);
    return {
      widthScale: mix(template.torsoWidth[0], template.torsoWidth[2], ny),
      centerShift: 0,
      alphaScale: mix(0.98, 1.05, openFront),
    };
  }
  if (label === 'pants') {
    const hip = mix(template.waistWidth, template.hipWidth, smoothstep(0.02, 0.24, ny));
    const thigh = mix(template.thighWidth, 0.94, smoothstep(0.26, 0.58, ny));
    const calf = mix(1, template.hemWidth, smoothstep(0.60, 1.0, ny));
    const centerShift = ny > 0.28 ? mix(0, 0.012, smoothstep(0.28, 1.0, ny)) : 0;
    return { widthScale: hip * thigh * calf, centerShift, alphaScale: mix(0.98, 1.02, ny) };
  }
  return { widthScale: mix(0.92, 1.04, smoothstep(0.10, 0.88, ny)), centerShift: 0, alphaScale: 1 };
}

function applyTemplateRegionWeight(label, nx, ny, alpha) {
  const template = GARMENT_TEMPLATES[label] || GARMENT_TEMPLATES.top;
  if (label === 'top' || label === 'layer') {
    const torso = mannequinRegionValue('torso', nx, ny);
    const leftSleeve = mannequinRegionValue('leftSleeve', nx, ny) * template.sleeveAlpha;
    const rightSleeve = mannequinRegionValue('rightSleeve', nx, ny) * template.sleeveAlpha;
    const waist = mannequinRegionValue('waist', nx, ny) * (1 - template.waistSuppression);
    const hemFade = ny < template.hemStart ? 1 : mix(1, template.hemTaper, smoothstep(template.hemStart, 1, ny));
    return alpha * Math.min(1.15, Math.max(torso, leftSleeve, rightSleeve, waist * 0.82) * hemFade + 0.12);
  }
  if (label === 'pants') {
    const hips = mannequinRegionValue('hips', nx, ny);
    const leftLeg = mannequinRegionValue('leftLeg', nx, ny);
    const rightLeg = mannequinRegionValue('rightLeg', nx, ny);
    const waist = mannequinRegionValue('waist', nx, ny) * 0.78;
    return alpha * Math.min(1.1, Math.max(hips, leftLeg, rightLeg, waist) + 0.08);
  }
  if (label === 'shoes') {
    const leftFoot = mannequinRegionValue('leftFoot', nx, ny);
    const rightFoot = mannequinRegionValue('rightFoot', nx, ny);
    return alpha * Math.min(1.05, Math.max(leftFoot, rightFoot) + 0.05);
  }
  return alpha;
}

function buildOcclusionMask(label) {
  const mask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const ny = y / (H - 1);
    for (let x = 0; x < W; x++) {
      const nx = x / (W - 1);
      let keep = 1;
      if (label === 'top' || label === 'layer') {
        const neck = bodyMaskValue('neck', nx, ny);
        const leftArm = bodyMaskValue('left_arm', nx, ny) * smoothstep(0.16, 0.32, ny);
        const rightArm = bodyMaskValue('right_arm', nx, ny) * smoothstep(0.16, 0.32, ny);
        const underHem = smoothstep(0.50, 0.62, ny);
        const pelvisGuard = bodyMaskValue('pelvis', nx, ny) * underHem * (label === 'top' ? 0.65 : 0.24);
        keep = 1 - Math.max(neck * 1.08, leftArm * 0.88, rightArm * 0.88, pelvisGuard);
      } else if (label === 'pants') {
        const pelvis = bodyMaskValue('pelvis', nx, ny);
        const leftLeg = bodyMaskValue('left_leg', nx, ny);
        const rightLeg = bodyMaskValue('right_leg', nx, ny);
        const rise = smoothstep(0.46, 0.54, ny);
        const inseamGap = Math.exp(-Math.pow((nx - 0.5) / 0.055, 2)) * smoothstep(0.60, 0.94, ny) * 0.92;
        const outerTrim = ((nx < 0.33) || (nx > 0.67)) ? smoothstep(0.58, 0.98, ny) * 0.62 : 0;
        const thighVolume = Math.max(leftLeg, rightLeg) * smoothstep(0.50, 0.84, ny);
        const pelvisHoldout = pelvis * smoothstep(0.48, 0.60, ny) * 0.34;
        keep = rise * Math.max(0.32, 1 - Math.max(inseamGap, outerTrim, thighVolume * 0.16, pelvisHoldout));
      } else if (label === 'shoes') {
        const leftFoot = bodyMaskValue('left_foot', nx, ny);
        const rightFoot = bodyMaskValue('right_foot', nx, ny);
        const floor = smoothstep(0.885, 0.955, ny);
        const centerGap = Math.exp(-Math.pow((nx - 0.5) / 0.05, 2)) * smoothstep(0.90, 0.99, ny);
        keep = floor * Math.max(0, Math.max(leftFoot, rightFoot) - centerGap * 0.82);
      }
      mask[y * W + x] = clampByte(keep * 255);
    }
  }
  return mask;
}

function buildRefinedAlpha(label, fittedRaw, fw, fh, placement) {
  const analyzed = analyzeGarmentAlpha(fittedRaw, fw, fh);
  const alpha = new Uint8Array(W * H);
  for (let dy = 0; dy < fh; dy++) {
    const ty = placement.top + dy;
    if (ty < 0 || ty >= H) continue;
    const row = analyzed.rowBounds[dy];
    if (row.right < row.left) continue;
    const srcWidth = Math.max(1, row.right - row.left + 1);
    const nyLocal = dy / Math.max(1, fh - 1);
    const profile = makeShapeProfile(label, nyLocal);
    const targetWidth = Math.max(1, Math.min(W, Math.round(srcWidth * profile.widthScale)));
    const rowCenter = placement.left + ((row.left + row.right) / 2) + profile.centerShift * W;
    const targetLeft = Math.max(0, Math.round(rowCenter - targetWidth / 2));
    const targetRight = Math.min(W - 1, targetLeft + targetWidth - 1);
    const safeWidth = Math.max(1, targetRight - targetLeft + 1);
    for (let tx = targetLeft; tx <= targetRight; tx++) {
      const t = safeWidth <= 1 ? 0.5 : (tx - targetLeft) / (safeWidth - 1);
      const srcX = mix(row.left, row.right, t);
      const srcY = dy;
      const sampledAlpha = sampleAlphaBilinear(fittedRaw, fw, fh, srcX, srcY) * profile.alphaScale;
      const idx = ty * W + tx;
      const nx = tx / Math.max(1, W - 1);
      const ny = ty / Math.max(1, H - 1);
      const weightedAlpha = applyTemplateRegionWeight(label, nx, ny, sampledAlpha);
      if (weightedAlpha > alpha[idx]) alpha[idx] = clampByte(weightedAlpha);
    }
  }
  return alpha;
}

function buildSplitPantsAlpha(fittedRaw, fw, fh, placement) {
  const analyzed = analyzeGarmentAlpha(fittedRaw, fw, fh);
  const alpha = new Uint8Array(W * H);
  const template = GARMENT_TEMPLATES.pants;
  const waistRows = Math.max(1, Math.round(fh * 0.18));
  for (let dy = 0; dy < fh; dy++) {
    const ty = placement.top + dy;
    if (ty < 0 || ty >= H) continue;
    const row = analyzed.rowBounds[dy];
    if (row.right < row.left) continue;
    const localY = dy / Math.max(1, fh - 1);
    const rowWidth = Math.max(1, row.right - row.left + 1);
    if (dy < waistRows) {
      const widthScale = mix(template.waistWidth, template.hipWidth, smoothstep(0, 1, dy / Math.max(1, waistRows - 1)));
      const targetWidth = Math.max(1, Math.round(rowWidth * widthScale));
      const centerX = placement.left + ((row.left + row.right) / 2);
      const left = Math.max(0, Math.round(centerX - targetWidth / 2));
      const right = Math.min(W - 1, left + targetWidth - 1);
      for (let tx = left; tx <= right; tx++) {
        const t = targetWidth <= 1 ? 0.5 : (tx - left) / Math.max(1, targetWidth - 1);
        const srcX = mix(row.left, row.right, t);
        const sampledAlpha = sampleAlphaBilinear(fittedRaw, fw, fh, srcX, dy);
        const idx = ty * W + tx;
        const nx = tx / Math.max(1, W - 1);
        const ny = ty / Math.max(1, H - 1);
        const weightedAlpha = applyTemplateRegionWeight('pants', nx, ny, sampledAlpha);
        if (weightedAlpha > alpha[idx]) alpha[idx] = clampByte(weightedAlpha);
      }
      continue;
    }

    const leftPortion = { left: row.left, right: Math.round(mix(row.left, row.right, 0.48)) };
    const rightPortion = { left: Math.round(mix(row.left, row.right, 0.52)), right: row.right };
    const segments = [
      { portion: leftPortion, center: 0.5 - template.legSpread, region: 'leftLeg' },
      { portion: rightPortion, center: 0.5 + template.legSpread, region: 'rightLeg' },
    ];

    for (const segment of segments) {
      if (segment.portion.right <= segment.portion.left) continue;
      const srcWidth = Math.max(1, segment.portion.right - segment.portion.left + 1);
      const widthScale = localY < 0.40
        ? mix(template.hipWidth, template.thighWidth, smoothstep(0.18, 0.40, localY))
        : localY < 0.72
          ? mix(template.thighWidth, template.calfWidth, smoothstep(0.40, 0.72, localY))
          : mix(template.calfWidth, template.hemWidth, smoothstep(0.72, 1.0, localY));
      const targetWidth = Math.max(1, Math.round(srcWidth * widthScale));
      const centerPx = segment.center * W;
      const targetLeft = Math.max(0, Math.round(centerPx - targetWidth / 2));
      const targetRight = Math.min(W - 1, targetLeft + targetWidth - 1);
      const safeWidth = Math.max(1, targetRight - targetLeft + 1);
      for (let tx = targetLeft; tx <= targetRight; tx++) {
        const nx = tx / Math.max(1, W - 1);
        const ny = ty / Math.max(1, H - 1);
        const centerGap = Math.exp(-Math.pow((nx - 0.5) / template.inseamGap, 2)) * smoothstep(0.60, 0.98, localY);
        if (centerGap > 0.82) continue;
        const t = safeWidth <= 1 ? 0.5 : (tx - targetLeft) / (safeWidth - 1);
        const srcX = mix(segment.portion.left, segment.portion.right, t);
        const sampledAlpha = sampleAlphaBilinear(fittedRaw, fw, fh, srcX, dy);
        const regionWeight = mannequinRegionValue(segment.region, nx, ny);
        const weightedAlpha = applyTemplateRegionWeight('pants', nx, ny, sampledAlpha * (0.85 + regionWeight * 0.3));
        const idx = ty * W + tx;
        if (weightedAlpha > alpha[idx]) alpha[idx] = clampByte(weightedAlpha);
      }
    }
  }
  return alpha;
}

function renderWarpedGarment(label, fittedRaw, fw, fh, placement, refinedAlpha) {
  const out = new Uint8ClampedArray(W * H * 4);
  const analyzed = analyzeGarmentAlpha(fittedRaw, fw, fh);
  for (let dy = 0; dy < fh; dy++) {
    const ty = placement.top + dy;
    if (ty < 0 || ty >= H) continue;
    const row = analyzed.rowBounds[dy];
    if (row.right < row.left) continue;
    const srcWidth = Math.max(1, row.right - row.left + 1);
    const nyLocal = dy / Math.max(1, fh - 1);
    const profile = makeShapeProfile(label, nyLocal);
    const targetWidth = Math.max(1, Math.min(W, Math.round(srcWidth * profile.widthScale)));
    const rowCenter = placement.left + ((row.left + row.right) / 2) + profile.centerShift * W;
    const targetLeft = Math.max(0, Math.round(rowCenter - targetWidth / 2));
    const targetRight = Math.min(W - 1, targetLeft + targetWidth - 1);
    const safeWidth = Math.max(1, targetRight - targetLeft + 1);
    for (let tx = targetLeft; tx <= targetRight; tx++) {
      const idx = ty * W + tx;
      const alpha = refinedAlpha[idx];
      if (alpha <= 1) continue;
      const t = safeWidth <= 1 ? 0.5 : (tx - targetLeft) / (safeWidth - 1);
      const srcX = mix(row.left, row.right, t);
      const rgba = samplePixelBilinear(fittedRaw, fw, fh, srcX, dy);
      const outIdx = idx * 4;
      out[outIdx] = clampByte(rgba[0]);
      out[outIdx + 1] = clampByte(rgba[1]);
      out[outIdx + 2] = clampByte(rgba[2]);
      out[outIdx + 3] = alpha;
    }
  }
  return out;
}

async function buildShoePairAsset(cleanedBuf) {
  const meta = await sharp(cleanedBuf).metadata();
  const srcWidth = Math.max(1, meta.width || 1);
  const srcHeight = Math.max(1, meta.height || 1);
  const pairWidth = Math.max(1, Math.round(srcWidth * 2.05));
  const gap = Math.max(10, Math.round(srcWidth * 0.10));
  const leftWidth = Math.max(1, Math.round(srcWidth * 0.92));
  const rightWidth = leftWidth;
  const canvas = sharp({
    create: {
      width: pairWidth,
      height: srcHeight,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 },
    },
  });
  const leftShoe = await sharp(cleanedBuf).resize({ width: leftWidth, height: srcHeight, fit: 'contain' }).flop().png().toBuffer();
  const rightShoe = await sharp(cleanedBuf).resize({ width: rightWidth, height: srcHeight, fit: 'contain' }).png().toBuffer();
  const leftX = Math.max(0, Math.round(pairWidth * 0.06));
  const rightX = Math.min(pairWidth - rightWidth, leftX + leftWidth + gap);
  return canvas
    .composite([
      { input: leftShoe, left: leftX, top: 0 },
      { input: rightShoe, left: rightX, top: 0 },
    ])
    .png()
    .toBuffer();
}

export { W, H };

export async function encodeBasePxToDataUri(basePx) {
  const outBuf = await sharp(Buffer.from(basePx.buffer, basePx.byteOffset, basePx.byteLength), {
    raw: { width: W, height: H, channels: 4 },
  })
    .png()
    .toBuffer();
  return `data:image/png;base64,${outBuf.toString('base64')}`;
}

/**
 * Dilate a single-channel mask by `radius` pixels then feather with a Gaussian
 * blur. Used to ensure the FLUX-refined garment region also covers slight
 * extensions (longer hems, looser sleeves) that the deterministic anchor
 * underestimates.
 */
export async function dilateAndFeatherMask(mask, radius = 12, sigma = 6) {
  const dilated = new Uint8Array(W * H);
  // Quick separable max-filter for dilation (cheap O(W*H*r) per axis).
  const tmp = new Uint8Array(W * H);
  // Horizontal
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
  // Vertical
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) {
      let m = 0;
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(H - 1, y + radius);
      for (let i = y0; i <= y1; i++) {
        const v = tmp[i * W + x];
        if (v > m) m = v;
      }
      dilated[y * W + x] = m;
    }
  }
  return blurMask(dilated, sigma);
}

export async function buildBaseCanvas(mannequinSrc) {
  const mannequinBuf = await loadImageBuffer(mannequinSrc);
  const baseRaw = await sharp(mannequinBuf)
    .resize(W, H, { fit: 'fill' })
    .ensureAlpha()
    .raw()
    .toBuffer();
  return new Uint8ClampedArray(baseRaw);
}

export async function compositeGarmentOntoBase(basePx, garmentSrc, label) {
  const normalized = normalizeGarmentLabel(label);
  const anchor = ANCHOR_BOXES[normalized];
  const px0 = Math.round(anchor.x0 * W);
  const py0 = Math.round(anchor.y0 * H);
  const px1 = Math.round(anchor.x1 * W);
  const py1 = Math.round(anchor.y1 * H);
  const box = { px0, py0, px1, py1, pw: px1 - px0, ph: py1 - py0 };

  const cleanedBaseBuf = await preprocessGarment(garmentSrc, normalized);
  const sanitizedBuf = await sanitizeGarmentForLabel(cleanedBaseBuf, normalized);
  const cleanedBuf = normalized === 'shoes' ? await buildShoePairAsset(sanitizedBuf) : sanitizedBuf;
  const cleanedMeta = await sharp(cleanedBuf).metadata();
  const placement = placementForLabel(normalized, box, cleanedMeta.width, cleanedMeta.height);

  const fittedBuf = await sharp(cleanedBuf)
    .resize(placement.width, placement.height, { fit: 'fill' })
    .png()
    .toBuffer();
  const { data: fittedRaw, info: fittedInfo } = await sharp(fittedBuf)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const fw = fittedInfo.width;
  const fh = fittedInfo.height;
  const refinedAlpha = normalized === 'pants'
    ? buildSplitPantsAlpha(fittedRaw, fw, fh, placement)
    : buildRefinedAlpha(normalized, fittedRaw, fw, fh, placement);
  const shapedAlpha = applyCategoryShape(refinedAlpha, normalized);
  const occlusionMask = buildOcclusionMask(normalized);
  const placedAlpha = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) {
    const blended = (shapedAlpha[i] / 255) * (occlusionMask[i] / 255);
    const solidified = blended <= 0.02 ? 0 : Math.min(1, Math.pow(blended, 0.82) * 1.08);
    placedAlpha[i] = clampByte(solidified * 255);
  }
  const warpedGarment = renderWarpedGarment(normalized, fittedRaw, fw, fh, placement, placedAlpha);

  let shadowScalar = 0;
  let shadowAmount = 0;
  if (normalized === 'layer') {
    shadowScalar = 0.35;
    shadowAmount = 0.08;
  } else if (normalized === 'pants') {
    shadowScalar = 0.3;
    shadowAmount = 0.05;
  } else if (normalized === 'shoes') {
    shadowScalar = 0.45;
    shadowAmount = 0.10;
  }
  if (shadowAmount > 0) {
    for (let i = 0; i < W * H; i++) {
      const m = (placedAlpha[i] / 255) * shadowScalar;
      if (m <= 0.001) continue;
      const shade = 1 - m * shadowAmount;
      basePx[i * 4] = Math.round(basePx[i * 4] * shade);
      basePx[i * 4 + 1] = Math.round(basePx[i * 4 + 1] * shade);
      basePx[i * 4 + 2] = Math.round(basePx[i * 4 + 2] * shade);
    }
  }

  for (let i = 0; i < W * H; i++) {
    const srcIdx = i * 4;
    const a = warpedGarment[srcIdx + 3] / 255;
    if (a <= 0.01) continue;
    const inv = 1 - a;
    basePx[srcIdx] = Math.round(basePx[srcIdx] * inv + warpedGarment[srcIdx] * a);
    basePx[srcIdx + 1] = Math.round(basePx[srcIdx + 1] * inv + warpedGarment[srcIdx + 1] * a);
    basePx[srcIdx + 2] = Math.round(basePx[srcIdx + 2] * inv + warpedGarment[srcIdx + 2] * a);
    basePx[srcIdx + 3] = 255;
  }

  return { label: normalized, placedAlpha };
}

export async function encodeCanvas(basePx, label) {
  const outBuf = await sharp(Buffer.from(basePx.buffer, basePx.byteOffset, basePx.byteLength), {
    raw: { width: W, height: H, channels: 4 },
  })
    .png()
    .toBuffer();

  return {
    imageDataUri: `data:image/png;base64,${outBuf.toString('base64')}`,
    width: W,
    height: H,
    label,
  };
}

export async function renderDeterministicOutfit({ mannequinSrc, garments }) {
  const basePx = await buildBaseCanvas(mannequinSrc);
  const garmentEntries = Array.isArray(garments) ? garments : [];
  const normalizedGarments = garmentEntries
    .map((garment) => ({
      garmentSrc: garment?.garmentSrc || garment?.garment_image || garment?.image || garment?.imageUrl || garment?.url,
      label: normalizeGarmentLabel(garment?.label || garment?.type || 'top'),
    }))
    .filter((garment) => garment.garmentSrc);

  const orderedGarments = GARMENT_RENDER_ORDER
    .map((label) => normalizedGarments.find((garment) => garment.label === label))
    .filter(Boolean);

  if (orderedGarments.length === 0) {
    return encodeCanvas(basePx, 'outfit');
  }

  let lastLabel = 'outfit';
  for (const garment of orderedGarments) {
    const step = await compositeGarmentOntoBase(basePx, garment.garmentSrc, garment.label);
    lastLabel = step.label;
  }

  return encodeCanvas(basePx, lastLabel);
}

/**
 * Main entry. Returns a `data:image/png;base64,...` URI of the dressed mannequin.
 */
export async function renderDeterministicGarment({ mannequinSrc, garmentSrc, label }) {
  const basePx = await buildBaseCanvas(mannequinSrc);
  const { label: normalized } = await compositeGarmentOntoBase(basePx, garmentSrc, label);
  return encodeCanvas(basePx, normalized);
}

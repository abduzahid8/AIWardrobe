/**
 * Agent 3 — Pose-Anchored Multi-Reference Conditioning (v3)
 *
 * Sends FLUX.1-Kontext-dev a 3-panel vertical strip in a SINGLE call:
 *   Panel 1 (top):    Mannequin identity anchor
 *   Panel 2 (middle): Colored-zone pose overlay (red=top, yellow=layer, blue=pants, green=shoes)
 *   Panel 3 (bottom): 2×2 garment grid with matching zone-color borders
 *
 * After FLUX returns, extracts the top panel and mask-merges so mannequin
 * pixels outside garment zones are byte-identical to the original.
 *
 * Fallback: if `TRYON_V3_PER_GARMENT=true`, makes one FLUX call per garment
 * (4 calls) using the same 3-panel format but with a single garment in the grid.
 */

import sharp from 'sharp';
import fs from 'node:fs';
import path from 'node:path';
import logger from '../../utils/logger.js';
import {
  GARMENT_RENDER_ORDER,
  normalizeGarmentLabel,
  preprocessGarmentAndCache,
  buildBaseCanvas,
  encodeCanvas,
  encodeBasePxToDataUri,
} from '../tryonRenderer.js';
import {
  stripDataUri,
  toDataUri,
  loadImageBuffer,
  detectImageContentType,
  callFluxKontext,
  extractFluxRegionRaw,
  maskMergeFluxIntoBase,
  dilateAndFeatherMask,
  W,
  H,
} from '../tryonShared.js';

// ── Constants ──

const ZONE_COLORS = {
  top:   { r: 255, g: 0,   b: 0,   hex: '#FF0000', name: 'red' },
  layer: { r: 255, g: 255, b: 0,   hex: '#FFFF00', name: 'yellow' },
  pants: { r: 0,   g: 0,   b: 255, hex: '#0000FF', name: 'blue' },
  shoes: { r: 0,   g: 255, b: 0,   hex: '#00FF00', name: 'green' },
};

// Anchor boxes (same as tryonRenderer.js — duplicated here to avoid
// importing a private const; they are mannequin-specific and stable).
const ANCHOR_BOXES = {
  top:   { x0: 0.33, y0: 0.19, x1: 0.67, y1: 0.49 },
  layer: { x0: 0.28, y0: 0.18, x1: 0.72, y1: 0.53 },
  pants: { x0: 0.34, y0: 0.50, x1: 0.66, y1: 0.92 },
  shoes: { x0: 0.35, y0: 0.89, x1: 0.65, y1: 0.985 },
};

const CACHE_DIR = path.join(process.cwd(), 'cache');
const POSE_SILHOUETTE_PATH = path.join(CACHE_DIR, 'pose_silhouette.png');
const GRID_CELL_SIZE = 512; // each garment tile in the 2×2 grid
const BORDER_WIDTH = 4;     // colored border around each grid cell

// ── Pose silhouette ──

/**
 * Build a 1024×1024 PNG with colored garment zones painted onto a
 * mannequin silhouette. Cached to cache/pose_silhouette.png after
 * first generation.
 */
export async function buildPoseSilhouette() {
  if (fs.existsSync(POSE_SILHOUETTE_PATH)) {
    return POSE_SILHOUETTE_PATH;
  }

  fs.mkdirSync(CACHE_DIR, { recursive: true });

  // Create a transparent 1024×1024 canvas, then draw filled colored
  // rectangles for each garment zone.
  const overlays = [];
  for (const [label, anchor] of Object.entries(ANCHOR_BOXES)) {
    const color = ZONE_COLORS[label];
    const left = Math.round(anchor.x0 * W);
    const top = Math.round(anchor.y0 * H);
    const width = Math.round((anchor.x1 - anchor.x0) * W);
    const height = Math.round((anchor.y1 - anchor.y0) * H);

    // Semi-transparent colored rectangle (alpha=128)
    const rectBuf = await sharp({
      create: { width, height, channels: 4, background: { r: color.r, g: color.g, b: color.b, alpha: 0.5 } },
    })
      .png()
      .toBuffer();

    overlays.push({ input: rectBuf, left, top });
  }

  // Add a thin mannequin outline stroke for spatial reference.
  // We load the actual mannequin image, convert to edges, and overlay.
  const mannequinPath = path.join(process.cwd(), 'assets', 'images', 'mannequin_front.png');
  let outlineOverlay = null;
  if (fs.existsSync(mannequinPath)) {
    try {
      const mannequinBuf = fs.readFileSync(mannequinPath);
      // Edge detection via luminance threshold → thin dark outline
      outlineOverlay = await sharp(mannequinBuf)
        .resize(W, H, { fit: 'fill' })
        .grayscale()
        .threshold(240) // white background → 255, mannequin body → 0
        .negate()       // mannequin body → 255, background → 0
        .ensureAlpha()
        .tint({ r: 80, g: 80, b: 80 })
        .png()
        .toBuffer();
    } catch (e) {
      logger.warn('[poseAnchored] mannequin outline generation failed:', e?.message);
    }
  }

  if (outlineOverlay) {
    overlays.unshift({ input: outlineOverlay, left: 0, top: 0 });
  }

  const silhBuf = await sharp({
    create: { width: W, height: H, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } },
  })
    .composite(overlays)
    .png()
    .toBuffer();

  fs.writeFileSync(POSE_SILHOUETTE_PATH, silhBuf);
  logger.info(`[poseAnchored] pose silhouette cached → ${POSE_SILHOUETTE_PATH}`);
  return POSE_SILHOUETTE_PATH;
}

// ── Garment grid ──

/**
 * Tile up to 4 garments into a 2×2 grid (1024×1024). Each cell is
 * 512×512 with a colored border matching the garment's zone.
 * Empty slots get a white cell with a faint "none" label.
 */
export async function buildGarmentGrid(garments) {
  const cellSize = GRID_CELL_SIZE;
  const gridW = cellSize * 2;
  const gridH = cellSize * 2;
  const cells = [];

  // Layout: top-left=top, top-right=layer, bottom-left=pants, bottom-right=shoes
  const slotPositions = {
    top:   { col: 0, row: 0 },
    layer: { col: 1, row: 0 },
    pants: { col: 0, row: 1 },
    shoes: { col: 1, row: 1 },
  };

  for (const label of GARMENT_RENDER_ORDER) {
    const garment = garments.find((g) => normalizeGarmentLabel(g.label) === label);
    const pos = slotPositions[label];
    const color = ZONE_COLORS[label];
    const left = pos.col * cellSize;
    const top = pos.row * cellSize;

    if (garment?.garmentSrc) {
      // Preprocess (bg removal) and resize the garment into its cell
      let garmentBuf;
      try {
        const cleaned = await preprocessGarmentAndCache(garment.garmentSrc, label);
        garmentBuf = cleaned;
      } catch {
        garmentBuf = await loadImageBuffer(garment.garmentSrc);
      }

      const resizedBuf = await sharp(garmentBuf)
        .resize(cellSize - BORDER_WIDTH * 2, cellSize - BORDER_WIDTH * 2, {
          fit: 'contain',
          background: { r: 255, g: 255, b: 255, alpha: 1 },
        })
        .png()
        .toBuffer();

      // Create cell with colored border
      const cellBuf = await sharp({
        create: {
          width: cellSize,
          height: cellSize,
          channels: 4,
          background: { r: color.r, g: color.g, b: color.b, alpha: 1 },
        },
      })
        .composite([{
          input: resizedBuf,
          left: BORDER_WIDTH,
          top: BORDER_WIDTH,
        }])
        .png()
        .toBuffer();

      cells.push({ input: cellBuf, left, top });
    } else {
      // Empty slot — white cell with faint label
      const emptyBuf = await sharp({
        create: {
          width: cellSize,
          height: cellSize,
          channels: 4,
          background: { r: 245, g: 245, b: 245, alpha: 1 },
        },
      })
        .composite([{
          input: Buffer.from(`<svg width="${cellSize}" height="${cellSize}" xmlns="http://www.w3.org/2000/svg">
            <text x="50%" y="50%" font-family="sans-serif" font-size="24" fill="#ccc" text-anchor="middle" dominant-baseline="middle">none</text>
          </svg>`),
          left: 0,
          top: 0,
        }])
        .png()
        .toBuffer();

      cells.push({ input: emptyBuf, left, top });
    }
  }

  const gridBuf = await sharp({
    create: { width: gridW, height: gridH, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } },
  })
    .composite(cells)
    .png()
    .toBuffer();

  return gridBuf;
}

// ── 3-panel composite ──

/**
 * Build the 1024×3072 vertical strip:
 *   [mannequin (1024×1024) | pose silhouette (1024×1024) | garment grid (1024×1024)]
 */
export async function buildTriPanelComposite(mannequinSrc, poseOverlayPath, garmentGridBuf) {
  const mannequinBuf = await loadImageBuffer(mannequinSrc);
  const mannequinResized = await sharp(mannequinBuf)
    .resize(W, H, { fit: 'fill' })
    .ensureAlpha()
    .png()
    .toBuffer();

  const poseBuf = fs.readFileSync(poseOverlayPath);
  const poseResized = await sharp(poseBuf)
    .resize(W, H, { fit: 'fill' })
    .ensureAlpha()
    .png()
    .toBuffer();

  const gridResized = await sharp(garmentGridBuf)
    .resize(W, H, { fit: 'fill' })
    .ensureAlpha()
    .png()
    .toBuffer();

  const composite = await sharp({
    create: { width: W, height: H * 3, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } },
  })
    .composite([
      { input: mannequinResized, left: 0, top: 0 },
      { input: poseResized, left: 0, top: H },
      { input: gridResized, left: 0, top: H * 2 },
    ])
    .png()
    .toBuffer();

  return `data:image/png;base64,${composite.toString('base64')}`;
}

// ── Prompt ──

export function buildV3Prompt(garments) {
  const presentLabels = garments.map((g) => normalizeGarmentLabel(g.label));
  const zoneDesc = presentLabels
    .map((label) => {
      const color = ZONE_COLORS[label];
      if (!color) return '';
      const zoneName = label === 'top' ? 'shirt/t-shirt zone (torso + sleeves)'
        : label === 'layer' ? 'jacket/coat zone (outer torso + sleeves)'
        : label === 'pants' ? 'trousers zone (legs + waist)'
        : 'shoes zone (feet)';
      return `- ${color.name.toUpperCase()} zone = ${label} (${zoneName})`;
    })
    .filter(Boolean)
    .join('\n');

  return [
    'You are a virtual try-on engine. The input is a THREE-PANEL vertical image (top → bottom):',
    '',
    'PANEL 1 (TOP): The mannequin to dress. Preserve this mannequin EXACTLY — same headless silhouette, same light-grey/white body, same pose, same proportions, same white studio background. Only add garments.',
    '',
    'PANEL 2 (MIDDLE): A pose guide with colored zones showing where each garment type belongs:',
    zoneDesc,
    '',
    'PANEL 3 (BOTTOM): A 2×2 grid of the garments to apply, each bordered with its matching zone color.',
    '',
    'TASK: Re-render the mannequin from Panel 1 wearing ALL garments from Panel 3, each placed in its matching colored zone from Panel 2. Produce ONLY the dressed mannequin (Panel 1 re-rendered with garments). Do NOT include Panels 2 or 3 in the output.',
    '',
    'HARD CONSTRAINTS — VIOLATING ANY OF THESE IS A FAILURE:',
    '- The mannequin body, pose, proportions, and white background MUST be pixel-identical to Panel 1.',
    '- Do NOT introduce a human face, hair, skin tone, extra limbs, hands, or accessories.',
    '- Match each garment\'s exact color, pattern, logos, fabric texture, and silhouette from Panel 3.',
    '- Natural drape, folds, hems, and layering order: top under layer, pants separate, shoes at bottom.',
    '- Output must be a single mannequin image (same framing as Panel 1).',
    '',
    'STYLE: studio fashion catalog, soft even lighting, sharp focus, photorealistic.',
  ].join('\n');
}

// ── Garment zone union mask ──

/**
 * Build a binary mask (Uint8Array, W*H) that is the UNION of all
 * present garment anchor boxes, dilated and feathered.
 */
export async function buildGarmentZoneUnionMask(garmentLabels) {
  const unionMask = new Uint8Array(W * H);

  for (const label of garmentLabels) {
    const anchor = ANCHOR_BOXES[label];
    if (!anchor) continue;

    const px0 = Math.round(anchor.x0 * W);
    const py0 = Math.round(anchor.y0 * H);
    const px1 = Math.round(anchor.x1 * W);
    const py1 = Math.round(anchor.y1 * H);

    for (let y = py0; y < py1; y++) {
      for (let x = px0; x < px1; x++) {
        unionMask[y * W + x] = 255;
      }
    }
  }

  // Dilate + feather for smooth blending at zone boundaries
  return dilateAndFeatherMask(unionMask, 14, 8);
}

// ── Single-call render ──

/**
 * Single FLUX call for the entire outfit.
 */
async function renderSingleCall({ mannequinSrc, garments }) {
  const posePath = await buildPoseSilhouette();
  const gridBuf = await buildGarmentGrid(garments);
  const compositeDataUri = await buildTriPanelComposite(mannequinSrc, posePath, gridBuf);

  const prompt = buildV3Prompt(garments);
  const fluxResult = await callFluxKontext({
    imageDataUri: compositeDataUri,
    prompt,
    provider: 'nvidia_local',
  });

  // Extract top panel (first 1024×1024 of the 3-panel output)
  const fluxRaw = await extractFluxRegionRaw(fluxResult, {
    left: 0,
    top: 0,
    width: W,
    height: H,
  });

  if (!fluxRaw) {
    throw new Error('FLUX.1 Kontext-dev v3 output could not be decoded');
  }

  return fluxRaw;
}

// ── Per-garment render (fallback) ──

/**
 * One FLUX call per garment (4 calls), each using the 3-panel format
 * but with only one garment in the grid. Cumulative mask-merge like v1.
 */
async function renderPerGarment({ mannequinSrc, garments }) {
  const basePx = await buildBaseCanvas(mannequinSrc);
  const preStep = new Uint8ClampedArray(basePx);
  let fluxStepCount = 0;
  const stepLabels = [];

  for (const garment of garments) {
    const label = normalizeGarmentLabel(garment.label);
    const singleGarmentSet = [garment];

    // Snapshot before this step
    const beforeStep = new Uint8ClampedArray(basePx);

    const posePath = await buildPoseSilhouette();
    const gridBuf = await buildGarmentGrid(singleGarmentSet);
    const compositeDataUri = await buildTriPanelComposite(
      await encodeBasePxToDataUri(basePx),
      posePath,
      gridBuf,
    );

    const prompt = buildV3Prompt(singleGarmentSet);
    const fluxResult = await callFluxKontext({
      imageDataUri: compositeDataUri,
      prompt,
      provider: 'nvidia_local',
    });

    const fluxRaw = await extractFluxRegionRaw(fluxResult, {
      left: 0,
      top: 0,
      width: W,
      height: H,
    });

    if (!fluxRaw) {
      throw new Error(`FLUX.1 Kontext-dev v3 per-garment output could not be decoded label=${label}`);
    }

    // Build mask for this garment's zone only
    const zoneMask = await buildGarmentZoneUnionMask([label]);
    maskMergeFluxIntoBase(basePx, beforeStep, fluxRaw, zoneMask);

    fluxStepCount += 1;
    stepLabels.push(label);
  }

  return { basePx, fluxStepCount, stepLabels };
}

// ── Main entry ──

/**
 * Pose-anchored v3 try-on render.
 *
 * @param {object} body - Same shape as /api/tryon/render request body
 * @returns {object} - Same shape as v1 response
 */
export async function poseAnchoredRender(body) {
  const startedAt = Date.now();
  const mannequinSrc = body.mannequin_image;
  const garmentEntries = Array.isArray(body.garments) ? body.garments : [];

  // Also support single-garment format
  const singleGarmentSrc =
    body.garment_image ||
    body.garment?.image ||
    body.garment?.imageUrl ||
    body.garment?.url;
  const singleLabel = body.garment?.label || body.garment?.type || 'top';

  if (!mannequinSrc) {
    return { success: false, error: 'mannequin_image is required' };
  }

  // Normalize garments into a consistent array
  let garments;
  if (garmentEntries.length > 0) {
    garments = GARMENT_RENDER_ORDER
      .map((label) => {
        const entry = garmentEntries.find(
          (g) => normalizeGarmentLabel(g?.label || g?.type || label) === label,
        );
        if (!entry) return null;
        return {
          label,
          garmentSrc: entry?.garmentSrc || entry?.garment_image || entry?.image || entry?.imageUrl || entry?.url,
        };
      })
      .filter((g) => g?.garmentSrc);
  } else if (singleGarmentSrc) {
    garments = [{ label: singleLabel, garmentSrc: singleGarmentSrc }];
  } else {
    return { success: false, error: 'garment_image is required' };
  }

  if (garments.length === 0) {
    return { success: false, error: 'No valid garments supplied' };
  }

  const perGarment = (process.env.TRYON_V3_PER_GARMENT || 'false').toLowerCase() === 'true';

  logger.info(
    `[tryon/render-v3] start count=${garments.length} labels=${garments.map((g) => g.label).join(',')} perGarment=${perGarment}`,
  );

  let basePx;
  let fluxStepCount;

  if (perGarment) {
    // Fallback: one FLUX call per garment
    const result = await renderPerGarment({ mannequinSrc, garments });
    basePx = result.basePx;
    fluxStepCount = result.fluxStepCount;
  } else {
    // Primary: single FLUX call for all garments
    const fluxRaw = await renderSingleCall({ mannequinSrc, garments });

    // Build the original mannequin canvas for identity-protect merge
    basePx = await buildBaseCanvas(mannequinSrc);
    const preStep = new Uint8ClampedArray(basePx);

    // Union mask of all garment zones
    const presentLabels = garments.map((g) => normalizeGarmentLabel(g.label));
    const unionMask = await buildGarmentZoneUnionMask(presentLabels);

    // Merge: inside garment zones → FLUX pixels; outside → original mannequin
    maskMergeFluxIntoBase(basePx, preStep, fluxRaw, unionMask);
    fluxStepCount = 1;
  }

  const finalLabel = garments[garments.length - 1]?.label || 'outfit';
  const encoded = await encodeCanvas(basePx, finalLabel);
  const elapsedMs = Date.now() - startedAt;

  logger.info(
    `[tryon/render-v3] done count=${garments.length} flux=${fluxStepCount} perGarment=${perGarment} in ${elapsedMs}ms`,
  );

  return {
    success: true,
    resultUrl: encoded.imageDataUri,
    methodUsed: perGarment
      ? 'pose_anchored_v3_per_garment+flux_refine'
      : 'pose_anchored_v3_single_call+flux_refine',
    fluxStepCount,
    step: garments.length,
    total: garments.length,
    garmentLabel: finalLabel,
    renderedGarments: garments.map((g) => normalizeGarmentLabel(g.label)),
    elapsedMs,
  };
}

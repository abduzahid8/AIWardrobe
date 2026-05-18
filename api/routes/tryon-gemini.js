/**
 * POST /api/tryon/gemini
 *
 * Gemini 2.0 Flash image generation try-on endpoint.
 *
 * Much cheaper than FLUX.1-Kontext-dev:
 * - FLUX: ~$0.20-0.50 per image
 * - Gemini 2.0 Flash: ~$0.04 per image (6-12x cheaper!)
 *
 * Uses the same side-by-side composite approach as FLUX:
 * - Left half: mannequin (preserved)
 * - Right half: garment product photo
 * - Gemini generates the dressed mannequin output
 *
 * Includes drift detection and mask merging to ensure mannequin consistency.
 *
 * Body:
 *   {
 *     mannequin_image: string  // data URI or HTTP(S) URL
 *     garments: [{ label, garment_image, ... }]    // multi-garment outfit
 *     // OR single garment:
 *     garment_image: string
 *     garment: { label: 'top'|'layer'|'pants'|'shoes', type?, name? }
 *   }
 *
 * Response:
 *   {
 *     success: true,
 *     resultUrl: string,        // data:image/png;base64,...
 *     methodUsed: 'gemini-flash',
 *     geminiStepCount,
 *     step, total, garmentLabel, renderedGarments, elapsedMs
 *   }
 */

import express from 'express';
import sharp from 'sharp';
import { authenticateToken } from '../middleware/auth.js';
import logger from '../utils/logger.js';
import {
  GARMENT_RENDER_ORDER,
  normalizeGarmentLabel,
  buildBaseCanvas,
  compositeGarmentOntoBase,
  encodeCanvas,
  encodeBasePxToDataUri,
} from '../services/tryonRenderer.js';
import {
  computeDiffMask,
  dilateAndFeatherMask,
  maskMergeFluxIntoBase,
} from '../services/tryonShared.js';
import {
  callGeminiFlash,
  buildGeminiComposite,
  buildGeminiDressingPrompt,
  checkGeminiHealth,
} from '../services/geminiClient.js';

const router = express.Router();

/**
 * Extract the generated image from Gemini output (full canvas).
 * Gemini returns the full generated image, not side-by-side.
 */
async function extractGeminiOutputRaw(geminiDataUri) {
  if (!geminiDataUri) return null;
  const buf = await fetch(geminiDataUri.startsWith('data:') ? geminiDataUri : geminiDataUri)
    .then((r) => r.arrayBuffer())
    .then((ab) => Buffer.from(ab));

  const meta = await sharp(buf).metadata();
  const fw = meta.width || 0;
  const fh = meta.height || 0;
  if (!fw || !fh) return null;

  // Resize to our standard 1024x1024
  const resized = await sharp(buf)
    .resize(1024, 1024, { fit: 'fill' })
    .ensureAlpha()
    .raw()
    .toBuffer();

  return new Uint8ClampedArray(resized);
}

/**
 * Run Gemini Flash for a single garment step.
 */
async function callGeminiForStep({ mannequinImage, garmentImage, label }) {
  const composite = await buildGeminiComposite(mannequinImage, garmentImage);
  return callGeminiFlash({
    imageDataUri: composite,
    prompt: buildGeminiDressingPrompt(label),
  });
}

/**
 * Process one garment step with Gemini.
 */
async function applyGarmentStepGemini(basePx, garmentSrc, label) {
  const normalized = normalizeGarmentLabel(label);
  const preStep = new Uint8ClampedArray(basePx);
  const stepStart = Date.now();

  const coarseBase = new Uint8ClampedArray(basePx);
  const { placedAlpha } = await compositeGarmentOntoBase(coarseBase, garmentSrc, normalized);

  const mannequinDataUri = await encodeBasePxToDataUri(basePx);
  const geminiResult = await callGeminiForStep({
    mannequinImage: mannequinDataUri,
    garmentImage: garmentSrc,
    label: normalized,
  });

  const geminiRaw = await extractGeminiOutputRaw(geminiResult);
  if (!geminiRaw) {
    throw new Error(`Gemini 2.0 Flash output could not be decoded label=${normalized}`);
  }

  // Compute diff mask to detect drift (Gemini should preserve mannequin)
  const { mask, coverage, drifted } = computeDiffMask(preStep, geminiRaw, {
    threshold: 18, // Slightly higher threshold for Gemini (can be noisier)
    maxCoverage: 0.85,
  });

  if (drifted) {
    throw new Error(
      `Gemini drift guard tripped label=${normalized} coverage=${(coverage * 100).toFixed(
        1,
      )}% — mannequin shifted, refusing to ship`,
    );
  }

  const total = placedAlpha.length;
  const lockedMask = new Uint8ClampedArray(total);
  for (let i = 0; i < total; i++) {
    lockedMask[i] = placedAlpha[i] > 0 && mask[i] > 0 ? placedAlpha[i] : 0;
  }

  // Merge: use Gemini pixels only inside the known garment zone, keep preStep elsewhere
  const mergeMask = await dilateAndFeatherMask(lockedMask, 8, 4);
  maskMergeFluxIntoBase(basePx, preStep, geminiRaw, mergeMask);

  logger.info(
    `[tryon/gemini] step done label=${normalized} coverage=${(coverage * 100).toFixed(
      1,
    )}% lockedMaskPx=${mergeMask.reduce((sum, value) => sum + (value > 0 ? 1 : 0), 0)} in ${Date.now() - stepStart}ms`,
  );

  return { label: normalized, geminiApplied: true, coverage };
}

async function applyGarmentStepGeminiOrDeterministic(basePx, garmentSrc, label) {
  const normalized = normalizeGarmentLabel(label);
  try {
    return await applyGarmentStepGemini(basePx, garmentSrc, normalized);
  } catch (err) {
    logger.warn(`[tryon/gemini] Gemini failed for label=${normalized}, using deterministic fallback: ${err?.message || err}`);
    await compositeGarmentOntoBase(basePx, garmentSrc, normalized);
    return {
      label: normalized,
      geminiApplied: false,
      deterministicFallback: true,
      coverage: null,
    };
  }
}

/**
 * POST /api/tryon/gemini
 */
router.post('/', async (req, res) => {
  const startedAt = Date.now();
  try {
    const body = req.body || {};
    const mannequinSrc = body.mannequin_image;
    const garmentEntries = Array.isArray(body.garments) ? body.garments : [];
    const garmentSrc =
      body.garment_image ||
      body.garment?.image ||
      body.garment?.imageUrl ||
      body.garment?.url;
    const rawLabel = body.garment?.label || body.garment?.type || 'top';

    if (!mannequinSrc) {
      return res.status(400).json({ success: false, error: 'mannequin_image is required' });
    }
    if (!garmentSrc && garmentEntries.length === 0) {
      return res.status(400).json({ success: false, error: 'garment_image is required' });
    }

    // Multi-garment outfit render
    if (garmentEntries.length > 0) {
      const orderedGarments = GARMENT_RENDER_ORDER.map((label) => {
        const entry = garmentEntries.find(
          (g) => normalizeGarmentLabel(g?.label || g?.type || label) === label,
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
      }).filter((g) => g?.garmentSrc);

      if (orderedGarments.length === 0) {
        return res.status(400).json({ success: false, error: 'No valid garments supplied' });
      }

      logger.info(
        `[tryon/gemini] outfit render count=${orderedGarments.length} labels=${orderedGarments
          .map((g) => g.label)
          .join(',')}`,
      );

      const basePx = await buildBaseCanvas(mannequinSrc);
      let geminiStepCount = 0;
      let deterministicFallbackCount = 0;
      const stepLabels = [];

      for (const garment of orderedGarments) {
        const stepResult = await applyGarmentStepGeminiOrDeterministic(basePx, garment.garmentSrc, garment.label);
        if (stepResult.geminiApplied) geminiStepCount += 1;
        if (stepResult.deterministicFallback) deterministicFallbackCount += 1;
        stepLabels.push(stepResult.label);
      }

      const finalLabel = stepLabels[stepLabels.length - 1] || 'outfit';
      const encoded = await encodeCanvas(basePx, finalLabel);
      const elapsedMs = Date.now() - startedAt;

      logger.info(
        `[tryon/gemini] outfit done count=${orderedGarments.length} gemini=${geminiStepCount}/${orderedGarments.length} fallback=${deterministicFallbackCount}/${orderedGarments.length} in ${elapsedMs}ms`,
      );

      return res.json({
        success: true,
        resultUrl: encoded.imageDataUri,
        methodUsed: deterministicFallbackCount > 0
          ? geminiStepCount > 0
            ? 'gemini-flash+deterministic-fallback'
            : 'deterministic-compositor'
          : 'gemini-flash',
        geminiStepCount,
        deterministicFallbackCount,
        step: orderedGarments.length,
        total: orderedGarments.length,
        garmentLabel: finalLabel,
        renderedGarments: stepLabels,
        elapsedMs,
      });
    }

    // Single garment render
    const label = normalizeGarmentLabel(rawLabel);
    logger.info(`[tryon/gemini] single garment label=${label}`);

    const basePx = await buildBaseCanvas(mannequinSrc);
    const stepResult = await applyGarmentStepGeminiOrDeterministic(basePx, garmentSrc, label);
    const encoded = await encodeCanvas(basePx, stepResult.label);
    const elapsedMs = Date.now() - startedAt;

    logger.info(`[tryon/gemini] done label=${stepResult.label} gemini=${stepResult.geminiApplied} fallback=${Boolean(stepResult.deterministicFallback)} in ${elapsedMs}ms`);

    return res.json({
      success: true,
      resultUrl: encoded.imageDataUri,
      methodUsed: stepResult.deterministicFallback ? 'deterministic-compositor' : 'gemini-flash',
      garmentLabel: stepResult.label,
      coverage: stepResult.coverage,
      geminiApplied: stepResult.geminiApplied,
      deterministicFallback: Boolean(stepResult.deterministicFallback),
      elapsedMs,
    });
  } catch (err) {
    logger.error('[tryon/gemini] failed:', err?.message || err);
    return res.status(500).json({
      success: false,
      error: err?.message || 'Gemini 2.0 Flash try-on failed',
    });
  }
});

/**
 * GET /api/tryon/gemini/health
 *
 * Health check for Gemini service.
 */
router.get('/health', async (req, res) => {
  try {
    const health = await checkGeminiHealth();
    if (health.healthy) {
      return res.json({ success: true, status: 'ok', service: 'gemini-flash', model: health.model });
    }
    return res.status(503).json({
      success: false,
      error: 'Gemini service unhealthy',
      details: health,
    });
  } catch (err) {
    logger.error('[tryon/gemini/health] failed:', err?.message || err);
    return res.status(503).json({
      success: false,
      error: err?.message || 'Gemini health check failed',
    });
  }
});

export default router;

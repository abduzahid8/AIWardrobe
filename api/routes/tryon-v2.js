/**
 * POST /api/tryon-v2/render
 *
 * Iterative Frozen-Region Inpainting try-on pipeline (v2).
 *
 * For each garment (top → layer → pants → shoes):
 *   1. Snapshot the canvas BEFORE this garment (preStep).
 *   2. Build a binary anatomical mask marking the editable region.
 *   3. Apply magenta tint to the editable region (visible signal to FLUX).
 *   4. Send FLUX.1-Kontext a side-by-side composite:
 *        LEFT: mannequin with magenta-tinted editable zone
 *        RIGHT: product photo of the garment
 *   5. Extract FLUX's left half (dressed mannequin).
 *   6. Hard pixel-snap: mask==0 → preStep (frozen), mask>0 → FLUX (editable).
 *
 * The mannequin NEVER changes outside the garment region. The binary snap
 * is mathematical — no prompt obedience required for identity preservation.
 *
 * Body:
 *   {
 *     mannequin_image: string  // data URI or HTTP(S) URL
 *     garments: [{ label, garment_image, ... }]  // multi-garment outfit
 *     // OR single garment:
 *     garment_image: string
 *     garment: { label: 'top'|'layer'|'pants'|'shoes', type?, name?, description? }
 *     step?:  number
 *     total?: number
 *   }
 *
 * Response:
 *   {
 *     success: true,
 *     resultUrl: string,        // data:image/png;base64,...
 *     methodUsed: 'inpainting_frozen_region_v2',
 *     fluxStepCount,            // how many steps got FLUX refinement
 *     step, total, garmentLabel, renderedGarments, elapsedMs
 *   }
 */

import express from 'express';
import { authenticateToken } from '../middleware/auth.js';
import { supabase } from '../lib/supabase.js';
import logger from '../utils/logger.js';
import {
  GARMENT_RENDER_ORDER,
  normalizeGarmentLabel,
  buildBaseCanvas,
  encodeCanvas,
} from '../services/tryonRenderer.js';
import {
  applyInpaintingStep,
} from '../services/strategies/inpainting.js';

const router = express.Router();

/**
 * Resolve the NVIDIA API key from Supabase app_config or env vars.
 */
async function getNvidiaKey() {
  try {
    const row = await supabase
      .from('app_config')
      .select('value')
      .eq('key', 'nvidia_token')
      .maybeSingle();
    const k = row.data?.value || process.env.NVIDIA_API_KEY_FLUX_1 || process.env.NVIDIA_API_KEY;
    if (!k) throw new Error('NVIDIA FLUX.1 Kontext-dev token is not configured');
    return k;
  } catch (err) {
    const fallback = process.env.NVIDIA_API_KEY_FLUX_1 || process.env.NVIDIA_API_KEY;
    if (fallback) return fallback;
    throw err;
  }
}

// ─── Route: POST /api/tryon-v2/render ──────────────────────────────────────

router.post('/render', authenticateToken, async (req, res) => {
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
    const step = Number(body.step ?? 1);
    const total = Number(body.total ?? 1);

    if (!mannequinSrc) {
      return res.status(400).json({ success: false, error: 'mannequin_image is required' });
    }
    if (!garmentSrc && garmentEntries.length === 0) {
      return res.status(400).json({ success: false, error: 'garment_image is required' });
    }

    const nvidiaKey = await getNvidiaKey();

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

      if (orderedGarments.length === 0) {
        return res.status(400).json({ success: false, error: 'No valid garments supplied' });
      }

      logger.info(
        `[tryon-v2/render] outfit render count=${orderedGarments.length} labels=${orderedGarments.map((g) => g.label).join(',')}`,
      );

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
      logger.info(
        `[tryon-v2/render] outfit done count=${orderedGarments.length} flux=${fluxStepCount}/${orderedGarments.length} in ${elapsedMs}ms`,
      );

      return res.json({
        success: true,
        resultUrl: encoded.imageDataUri,
        methodUsed: 'inpainting_frozen_region_v2',
        fluxStepCount,
        step: orderedGarments.length,
        total: orderedGarments.length,
        garmentLabel: finalLabel,
        renderedGarments: stepLabels,
        elapsedMs,
      });
    }

    // ── Single garment step ───────────────────────────────────────────────
    const label = normalizeGarmentLabel(rawLabel, step);
    logger.info(`[tryon-v2/render] step=${step}/${total} label=${label} (raw=${rawLabel})`);

    const basePx = await buildBaseCanvas(mannequinSrc);
    const stepResult = await applyInpaintingStep(basePx, garmentSrc, label, nvidiaKey);
    const encoded = await encodeCanvas(basePx, stepResult.label);

    const elapsedMs = Date.now() - startedAt;
    logger.info(
      `[tryon-v2/render] done step=${step}/${total} label=${stepResult.label} flux=${stepResult.fluxApplied} in ${elapsedMs}ms`,
    );

    return res.json({
      success: true,
      resultUrl: encoded.imageDataUri,
      methodUsed: 'inpainting_frozen_region_v2',
      step,
      total,
      garmentLabel: stepResult.label,
      elapsedMs,
    });
  } catch (err) {
    logger.error('[tryon-v2/render] failed:', err?.message || err);
    return res.status(500).json({
      success: false,
      error: err?.message || 'Render failed',
    });
  }
});

export default router;

/**
 * POST /api/tryon-v1/render
 *
 * Strategy v1 — Side-by-Side Reference + Mask-Locked Recomposition.
 * See @/Users/zohidvohidjonov/.windsurf/plans/tryon-v1-sidebyside-361a39.md
 *
 * Same request/response contract as /api/tryon/render, but routed through
 * services/strategies/sideBySide.js with: clean preStep mannequin in left
 * panel (no coarse composite), 2048x1024 reference, dilation +6 / feather 4,
 * binary hard-paste merge with thin feather seam.
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
  preprocessGarmentAndCache,
  applySideBySideStep,
} from '../services/strategies/sideBySide.js';

const router = express.Router();

async function loadNvidiaKey() {
  try {
    const row = await supabase.from('app_config').select('value').eq('key', 'nvidia_token').maybeSingle();
    const k = row.data?.value || process.env.NVIDIA_API_KEY_FLUX_1 || process.env.NVIDIA_API_KEY;
    if (!k) throw new Error('NVIDIA FLUX.1 Kontext-dev token is not configured');
    return k;
  } catch (err) {
    const fallback = process.env.NVIDIA_API_KEY_FLUX_1 || process.env.NVIDIA_API_KEY;
    if (fallback) return fallback;
    throw err;
  }
}

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

    const nvidiaKey = await loadNvidiaKey();

    if (garmentEntries.length > 0) {
      const orderedGarments = GARMENT_RENDER_ORDER
        .map((label) => {
          const entry = garmentEntries.find((g) => normalizeGarmentLabel(g?.label || g?.type || label) === label);
          if (!entry) return null;
          return {
            label,
            garmentSrc: entry?.garmentSrc || entry?.garment_image || entry?.image || entry?.imageUrl || entry?.url,
          };
        })
        .filter((g) => g?.garmentSrc);

      if (orderedGarments.length === 0) {
        return res.status(400).json({ success: false, error: 'No valid garments supplied' });
      }

      logger.info(`[tryon-v1/render] outfit count=${orderedGarments.length} labels=${orderedGarments.map((g) => g.label).join(',')}`);

      const basePx = await buildBaseCanvas(mannequinSrc);
      let fluxStepCount = 0;
      const stepLabels = [];
      for (const garment of orderedGarments) {
        const r = await applySideBySideStep(basePx, garment.garmentSrc, garment.label, nvidiaKey);
        if (r.fluxApplied) fluxStepCount += 1;
        stepLabels.push(r.label);
      }

      const finalLabel = stepLabels[stepLabels.length - 1] || 'outfit';
      const encoded = await encodeCanvas(basePx, finalLabel);
      const elapsedMs = Date.now() - startedAt;
      logger.info(`[tryon-v1/render] outfit done count=${orderedGarments.length} flux=${fluxStepCount}/${orderedGarments.length} in ${elapsedMs}ms`);

      return res.json({
        success: true,
        resultUrl: encoded.imageDataUri,
        methodUsed: 'side_by_side_v1',
        fluxStepCount,
        step: orderedGarments.length,
        total: orderedGarments.length,
        garmentLabel: finalLabel,
        renderedGarments: stepLabels,
        elapsedMs,
      });
    }

    const label = normalizeGarmentLabel(rawLabel, step);
    logger.info(`[tryon-v1/render] step=${step}/${total} label=${label} (raw=${rawLabel})`);

    const basePx = await buildBaseCanvas(mannequinSrc);
    const r = await applySideBySideStep(basePx, garmentSrc, label, nvidiaKey);
    const encoded = await encodeCanvas(basePx, r.label);
    const elapsedMs = Date.now() - startedAt;
    logger.info(`[tryon-v1/render] done step=${step}/${total} label=${r.label} flux=${r.fluxApplied} in ${elapsedMs}ms`);

    return res.json({
      success: true,
      resultUrl: encoded.imageDataUri,
      methodUsed: 'side_by_side_v1',
      step,
      total,
      garmentLabel: r.label,
      elapsedMs,
    });
  } catch (err) {
    logger.error('[tryon-v1/render] failed:', err?.message || err);
    return res.status(500).json({ success: false, error: err?.message || 'Render failed' });
  }
});

router.post('/preprocess', authenticateToken, async (req, res) => {
  try {
    const src = req.body?.garment_image;
    if (!src) {
      return res.status(400).json({ success: false, error: 'garment_image is required' });
    }
    const startedAt = Date.now();
    await preprocessGarmentAndCache(src);
    return res.json({ success: true, elapsedMs: Date.now() - startedAt });
  } catch (err) {
    logger.error('[tryon-v1/preprocess] failed:', err?.message || err);
    return res.status(500).json({ success: false, error: err?.message || 'Preprocess failed' });
  }
});

export default router;

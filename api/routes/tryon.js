/**
 * POST /api/tryon/render
 *
 * FLUX.1-Kontext-dev try-on pipeline (FLUX-only, no anchor compositor).
 *
 * For each garment (top → layer → pants → shoes):
 *   1. Snapshot the canvas BEFORE this garment (preStep).
 *   2. Send a side-by-side composite (current dressed mannequin | clean
 *      product photo) directly to FLUX.1-Kontext-dev with a strong
 *      dressing prompt. FLUX itself produces the realistic dressed
 *      mannequin — no manual placement, no anchor boxes, no per-garment
 *      templates.
 *   3. Extract FLUX's left half = the dressed mannequin candidate.
 *   4. Compute a pixel-diff mask between preStep and the FLUX candidate.
 *      Pixels that meaningfully changed = the garment region FLUX painted.
 *      Pixels that didn't change = mannequin / background that must stay.
 *   5. Drift guard: if the diff mask covers >85% of the canvas, FLUX has
 *      shifted the entire image — fail this step with a clear error.
 *   6. Dilate + feather the diff mask, then merge:
 *        - Inside the mask  → use FLUX pixels  (realistic drape).
 *        - Outside the mask → snap back to preStep (mannequin locked).
 *
 * Body:
 *   {
 *     mannequin_image: string  // data URI or HTTP(S) URL
 *     garments: [{ label, garment_image, ... }]    // multi-garment outfit
 *     // OR single garment:
 *     garment_image: string
 *     garment: { label: 'top'|'layer'|'pants'|'shoes', type?, name? }
 *     step?: number
 *     total?: number
 *   }
 *
 * Response:
 *   {
 *     success: true,
 *     resultUrl: string,        // data:image/png;base64,...
 *     methodUsed: 'flux_only_diff_mask',
 *     fluxStepCount,
 *     step, total, garmentLabel, renderedGarments, elapsedMs
 *   }
 */

import express from 'express';
import sharp from 'sharp';
import rateLimit, { ipKeyGenerator } from 'express-rate-limit';
import { authenticateToken } from '../middleware/auth.js';
import logger from '../utils/logger.js';
import {
  GARMENT_RENDER_ORDER,
  normalizeGarmentLabel,
  preprocessGarmentAndCache,
  buildBaseCanvas,
  encodeCanvas,
  encodeBasePxToDataUri,
} from '../services/tryonRenderer.js';
import {
  toDataUri,
  loadImageBuffer,
  getNvidiaKey,
  callFluxKontext,
  extractFluxLeftHalfRaw,
  maskMergeFluxIntoBase,
  dilateAndFeatherMask,
  computeDiffMask,
} from '../services/tryonShared.js';
import { callFashnVton, checkFashnHealth } from '../services/fashnClient.js';
import { callHuggingFaceVton, normalizeToFashnCategory as hfNormalizeCategory } from '../services/huggingfaceVtonClient.js';
import { idmVtonRender } from '../services/strategies/idmVton.js';
import { checkIdmVtonHealth } from '../services/idmVtonClient.js';
import { catvtonRender } from '../services/strategies/catvton.js';
import { callCatvton, checkCatvtonHealth } from '../services/catvtonClient.js';
import { mobileVtonRender } from '../services/strategies/mobileVton.js';

// ── Rate limiting for expensive FLUX.1 calls ────────────────────────────
// 5 requests per minute per user — FLUX.1 is costly and slow
const tryonLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 5,
  standardHeaders: true,
  legacyHeaders: false,
  validate: { xForwardedForHeader: false },
  keyGenerator: (req) => req.user?.id ? `u:${req.user.id}` : ipKeyGenerator(req),
  handler: (req, res) => {
    logger.warn(`Rate limit exceeded for user ${req.user?.id}`);
    res.status(429).json({
      success: false,
      error: 'Too many try-on requests. Please wait a minute and try again.',
      retryAfter: 60,
    });
  },
});

const router = express.Router();

// ── Side-by-side composite: [mannequin | garment] at 1536×1024 ──
async function buildKontextComposite(personImage, garmentImage) {
  const [personBuf, garmentBuf] = await Promise.all([
    loadImageBuffer(personImage),
    loadImageBuffer(garmentImage),
  ]);
  const [left, right] = await Promise.all([
    sharp(personBuf)
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

function buildDressingPrompt(label) {
  const garmentNoun =
    label === 'pants'
      ? 'pants / trousers'
      : label === 'shoes'
        ? 'pair of shoes'
        : label === 'layer'
          ? 'outer layer (jacket / coat / cardigan)'
          : 'top (shirt / t-shirt / sweater)';
  const garmentZone =
    label === 'pants'
      ? 'on the hips, thighs, knees, and legs of the mannequin'
      : label === 'shoes'
        ? 'on the feet of the mannequin'
        : label === 'layer'
          ? 'over the existing top, on the shoulders, chest, back, and sleeves of the mannequin'
          : 'on the torso, chest, shoulders, and arms of the mannequin';
  return [
    'You are a photorealistic virtual try-on engine. The input is a single image with TWO halves separated vertically:',
    'LEFT HALF: a smooth, headless, light-grey fashion mannequin on a clean white seamless studio background. The mannequin may already be wearing previously-applied garments — those are FINAL and must NOT change.',
    `RIGHT HALF: the exact product photo of a single new ${garmentNoun} that must be put on the mannequin.`,
    `TASK: produce the same scene but with the ${garmentNoun} from the right half now WORN and DRESSED ${garmentZone}. The garment must look like real clothing on a real person: natural drape, realistic fabric folds, correct hems and cuffs, settled sleeves and collars, proper occlusion where the garment meets the body, soft contact shadows where fabric touches the body. The fit, proportions, and layering must be photorealistic — NOT a flat overlay, NOT a sticker, NOT a screenshot of the product photo glued onto the mannequin.`,
    'ABSOLUTE HARD CONSTRAINTS — VIOLATING ANY OF THESE IS A FAILURE:',
    '1. The mannequin itself MUST be pixel-identical to the LEFT HALF input: same headless silhouette, same light-grey body color, exact same pose, exact same proportions, exact same height, exact same arm/leg position, exact same camera angle, exact same framing, exact same crop. The mannequin MAY NOT move, rotate, scale, or change shape in any way.',
    '2. The white seamless studio background MUST be pixel-identical. No new global shadows, no gradient changes, no color shift. Only subtle contact shadows directly under garment edges are allowed.',
    '3. Do NOT introduce a human face, hair, eyes, skin tone, ethnic features, extra limbs, hands with fingers, jewelry, or any accessory not present in the right-half product photo. The figure stays a faceless mannequin — never a real person.',
    '4. Any garments already worn on the mannequin in the LEFT HALF must remain visually identical in the output. This is a cumulative outfit build: previous layers do not change.',
    '5. Match the garment in the right half EXACTLY: identical color, identical pattern / print / logo, identical fabric texture, identical silhouette, identical design details (buttons, zippers, pockets, stitching, collar style, cuff style, hem style). Do not invent details.',
    '6. Do NOT crop, zoom, pan, or re-frame the mannequin. The output mannequin must occupy the same bounding box as the input mannequin.',
    '7. The RIGHT HALF of the output should be plain clean white — the product reference is no longer needed.',
    'STYLE: studio fashion catalog photograph, soft even lighting, sharp focus, photorealistic, premium e-commerce product photography quality. No artistic stylization, no painterly effects.',
  ].join(' ');
}

/**
 * Run FLUX.1-Kontext-dev once for a single garment step.
 * Returns the raw FLUX data URI (full side-by-side output).
 */
async function callFluxForStep({ mannequinImage, garmentImage, label }) {
  let cleanedGarmentImage = garmentImage;
  try {
    const cleaned = await preprocessGarmentAndCache(garmentImage, label);
    cleanedGarmentImage = toDataUri(cleaned, 'image/png');
  } catch (preprocessErr) {
    logger.warn(
      `[tryon/render] garment preprocess fallback label=${label}: ${preprocessErr?.message || preprocessErr}`,
    );
  }

  const nvidiaKey = await getNvidiaKey();
  const composite = await buildKontextComposite(mannequinImage, cleanedGarmentImage);
  return callFluxKontext({
    imageDataUri: composite,
    prompt: buildDressingPrompt(label),
    nvidiaKey,
    provider: process.env.FLUX_PROVIDER || 'nvidia_local',
  });
}

/**
 * Process one garment step (FLUX-only, no anchor compositor).
 */
async function applyGarmentStep(basePx, garmentSrc, label) {
  const normalized = normalizeGarmentLabel(label);
  const preStep = new Uint8ClampedArray(basePx);
  const stepStart = Date.now();

  const mannequinDataUri = await encodeBasePxToDataUri(basePx);
  const fluxResult = await callFluxForStep({
    mannequinImage: mannequinDataUri,
    garmentImage: garmentSrc,
    label: normalized,
  });

  const fluxRaw = await extractFluxLeftHalfRaw(fluxResult);
  if (!fluxRaw) {
    throw new Error(`FLUX.1 Kontext-dev output could not be decoded label=${normalized}`);
  }

  const { mask, coverage, drifted } = computeDiffMask(preStep, fluxRaw, {
    threshold: 14,
    maxCoverage: 0.85,
  });
  if (drifted) {
    throw new Error(
      `FLUX drift guard tripped label=${normalized} coverage=${(coverage * 100).toFixed(
        1,
      )}% — mannequin shifted, refusing to ship`,
    );
  }

  const mergeMask = await dilateAndFeatherMask(mask, 8, 4);
  maskMergeFluxIntoBase(basePx, preStep, fluxRaw, mergeMask);

  logger.info(
    `[tryon/render] step done label=${normalized} coverage=${(coverage * 100).toFixed(
      1,
    )}% in ${Date.now() - stepStart}ms`,
  );
  return { label: normalized, fluxApplied: true, coverage };
}

router.post('/render', authenticateToken, tryonLimiter, async (req, res) => {
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

    if (garmentEntries.length > 0) {
      const orderedGarments = GARMENT_RENDER_ORDER
        .map((label) => {
          const entry = garmentEntries.find(
            (garment) => normalizeGarmentLabel(garment?.label || garment?.type || label) === label,
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
        .filter((garment) => garment?.garmentSrc);

      if (orderedGarments.length === 0) {
        return res.status(400).json({ success: false, error: 'No valid garments supplied' });
      }

      logger.info(
        `[tryon/render] outfit render count=${orderedGarments.length} labels=${orderedGarments
          .map((g) => g.label)
          .join(',')}`,
      );

      const basePx = await buildBaseCanvas(mannequinSrc);
      let fluxStepCount = 0;
      const stepLabels = [];
      for (const garment of orderedGarments) {
        const stepResult = await applyGarmentStep(basePx, garment.garmentSrc, garment.label);
        if (stepResult.fluxApplied) fluxStepCount += 1;
        stepLabels.push(stepResult.label);
      }

      const finalLabel = stepLabels[stepLabels.length - 1] || 'outfit';
      const encoded = await encodeCanvas(basePx, finalLabel);
      const elapsedMs = Date.now() - startedAt;
      logger.info(
        `[tryon/render] outfit done count=${orderedGarments.length} flux=${fluxStepCount}/${orderedGarments.length} in ${elapsedMs}ms`,
      );

      return res.json({
        success: true,
        resultUrl: encoded.imageDataUri,
        methodUsed: 'flux_only_diff_mask',
        fluxStepCount,
        step: orderedGarments.length,
        total: orderedGarments.length,
        garmentLabel: finalLabel,
        renderedGarments: stepLabels,
        elapsedMs,
      });
    }

    const label = normalizeGarmentLabel(rawLabel, step);
    logger.info(`[tryon/render] step=${step}/${total} label=${label} (raw=${rawLabel})`);

    const basePx = await buildBaseCanvas(mannequinSrc);
    const stepResult = await applyGarmentStep(basePx, garmentSrc, label);
    const encoded = await encodeCanvas(basePx, stepResult.label);

    const elapsedMs = Date.now() - startedAt;
    logger.info(
      `[tryon/render] done step=${step}/${total} label=${stepResult.label} flux=${stepResult.fluxApplied} in ${elapsedMs}ms`,
    );

    return res.json({
      success: true,
      resultUrl: encoded.imageDataUri,
      methodUsed: 'flux_only_diff_mask',
      step,
      total,
      garmentLabel: stepResult.label,
      coverage: stepResult.coverage,
      elapsedMs,
    });
  } catch (err) {
    logger.error('[tryon/render] failed:', err?.message || err);
    return res.status(500).json({
      success: false,
      error: err?.message || 'FLUX.1 Kontext-dev render failed',
    });
  }
});

/**
 * POST /api/tryon/preprocess
 *
 * Warms the garment cache for a single source.
 */
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
    logger.error('[tryon/preprocess] failed:', err?.message || err);
    return res.status(500).json({ success: false, error: err?.message || 'Preprocess failed' });
  }
});

/**
 * POST /api/tryon/fashn
 *
 * FASHN VTON v1.5 try-on endpoint.
 *
 * This uses the FASHN VTON v1.5 model from Hugging Face for virtual try-on.
 * It's a Python-based model that handles pose detection automatically via DWPose.
 *
 * Body:
 *   {
 *     person_image: string  // data URI or HTTP(S) URL
 *     garment_image: string  // data URI or HTTP(S) URL
 *     category: string  // "tops", "bottoms", or "one-pieces"
 *     garment_photo_type?: string  // "model" or "flat-lay" (default: "model")
 *     num_samples?: number  // 1-4 (default: 1)
 *     num_timesteps?: number  // 20=fast, 30=balanced, 50=quality (default: 30)
 *     guidance_scale?: number  // classifier-free guidance (default: 1.5)
 *     seed?: number  // random seed (default: 42)
 *     segmentation_free?: boolean  // maskless mode (default: true)
 *   }
 *
 * Response:
 *   {
 *     success: true,
 *     resultUrl: string,  // data:image/png;base64,...
 *     methodUsed: 'fashn-vton-v1.5',
 *     elapsedMs
 *   }
 */
router.post('/fashn', authenticateToken, async (req, res) => {
  const startedAt = Date.now();
  try {
    const {
      person_image,
      garment_image,
      category,
      garment_photo_type = 'model',
      num_samples = 1,
      num_timesteps = 30,
      guidance_scale = 1.5,
      seed = 42,
      segmentation_free = true,
    } = req.body || {};

    if (!person_image) {
      return res.status(400).json({ success: false, error: 'person_image is required' });
    }
    if (!garment_image) {
      return res.status(400).json({ success: false, error: 'garment_image is required' });
    }
    if (!category) {
      return res.status(400).json({ success: false, error: 'category is required (tops/bottoms/one-pieces)' });
    }

    logger.info(`[tryon/fashn] category=${category} timesteps=${num_timesteps}`);

    const result = await callFashnVton({
      personImage: person_image,
      garmentImage: garment_image,
      category,
      garmentPhotoType: garment_photo_type,
      numSamples: num_samples,
      numTimesteps: num_timesteps,
      guidanceScale: guidance_scale,
      seed,
      segmentationFree: segmentation_free,
    });

    if (!result.success) {
      return res.status(500).json({
        success: false,
        error: result.error || 'FASHN VTON inference failed',
      });
    }

    const elapsedMs = Date.now() - startedAt;
    logger.info(`[tryon/fashn] done in ${elapsedMs}ms`);

    return res.json({
      success: true,
      resultUrl: result.resultImage,
      methodUsed: 'fashn-vton-v1.5',
      elapsedMs,
    });
  } catch (err) {
    logger.error('[tryon/fashn] failed:', err?.message || err);
    return res.status(500).json({
      success: false,
      error: err?.message || 'FASHN VTON try-on failed',
    });
  }
});

/**
 * GET /api/tryon/fashn/health
 *
 * Health check for FASHN VTON service.
 */
router.get('/fashn/health', async (req, res) => {
  try {
    const health = await checkFashnHealth();
    if (health.healthy) {
      return res.json({ success: true, status: 'ok', service: 'fashn-vton-v1.5' });
    }
    return res.status(503).json({
      success: false,
      error: 'FASHN VTON service unhealthy',
      details: health,
    });
  } catch (err) {
    logger.error('[tryon/fashn/health] failed:', err?.message || err);
    return res.status(503).json({
      success: false,
      error: err?.message || 'FASHN VTON health check failed',
    });
  }
});

/**
 * POST /api/tryon/idm-vton
 *
 * IDM-VTON try-on endpoint (via Replicate).
 *
 * IDM-VTON is a specialized virtual try-on model designed to preserve
 * person/mannequin identity while realistically draping garments.
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
 *     methodUsed: 'idm_vton_diff_mask',
 *     fluxStepCount, step, total, garmentLabel, renderedGarments, elapsedMs
 *   }
 */
router.post('/idm-vton', authenticateToken, async (req, res) => {
  const startedAt = Date.now();
  try {
    const result = await idmVtonRender(req.body || {});
    const elapsedMs = Date.now() - startedAt;

    if (!result.success) {
      return res.status(500).json({
        success: false,
        error: result.error || 'IDM-VTON render failed',
      });
    }

    logger.info(`[tryon/idm-vton] done in ${elapsedMs}ms`);
    return res.json({
      ...result,
      elapsedMs,
    });
  } catch (err) {
    logger.error('[tryon/idm-vton] failed:', err?.message || err);
    return res.status(500).json({
      success: false,
      error: err?.message || 'IDM-VTON try-on failed',
    });
  }
});

/**
 * GET /api/tryon/idm-vton/health
 *
 * Health check for IDM-VTON service (via Replicate).
 */
router.get('/idm-vton/health', async (req, res) => {
  try {
    const health = await checkIdmVtonHealth();
    if (health.healthy) {
      return res.json({ success: true, status: 'ok', service: 'idm-vton' });
    }
    return res.status(503).json({
      success: false,
      error: 'IDM-VTON service unhealthy',
      details: health,
    });
  } catch (err) {
    logger.error('[tryon/idm-vton/health] failed:', err?.message || err);
    return res.status(503).json({
      success: false,
      error: err?.message || 'IDM-VTON health check failed',
    });
  }
});

/**
 * POST /api/tryon/hf-vton
 *
 * Hugging Face Inference API for FASHN VTON v1.5 (free, no local installation).
 *
 * This uses Hugging Face's free Inference API to run FASHN VTON without
 * requiring local Python installation or GPU.
 *
 * Body:
 *   {
 *     person_image: string  // data URI or HTTP(S) URL
 *     garment_image: string  // data URI or HTTP(S) URL
 *     category: string  // "tops", "bottoms", or "one-pieces"
 *     num_timesteps?: number  // 20=fast, 30=balanced, 50=quality (default: 30)
 *     guidance_scale?: number  // classifier-free guidance (default: 1.5)
 *     seed?: number  // random seed (default: 42)
 *   }
 *
 * Response:
 *   {
 *     success: true,
 *     resultUrl: string,  // data:image/png;base64,...
 *     methodUsed: 'huggingface-vton',
 *     elapsedMs
 *   }
 */
router.post('/hf-vton', authenticateToken, async (req, res) => {
  const startedAt = Date.now();
  try {
    const {
      person_image,
      garment_image,
      category,
      num_timesteps = 30,
      guidance_scale = 1.5,
      seed = 42,
    } = req.body || {};

    if (!person_image) {
      return res.status(400).json({ success: false, error: 'person_image is required' });
    }
    if (!garment_image) {
      return res.status(400).json({ success: false, error: 'garment_image is required' });
    }
    if (!category) {
      return res.status(400).json({ success: false, error: 'category is required (tops/bottoms/one-pieces)' });
    }

    const fashnCategory = hfNormalizeCategory(category);
    logger.info(`[tryon/hf-vton] category=${fashnCategory} timesteps=${num_timesteps}`);

    const result = await callHuggingFaceVton({
      personImage: person_image,
      garmentImage: garment_image,
      category: fashnCategory,
      num_timesteps,
      guidance_scale,
      seed,
    });

    if (!result.success) {
      return res.status(500).json({
        success: false,
        error: result.error || 'Hugging Face VTON inference failed',
        retry: result.retry || false,
      });
    }

    const elapsedMs = Date.now() - startedAt;
    logger.info(`[tryon/hf-vton] done in ${elapsedMs}ms`);

    return res.json({
      success: true,
      resultUrl: result.resultImage,
      methodUsed: 'huggingface-vton',
      elapsedMs,
    });
  } catch (err) {
    logger.error('[tryon/hf-vton] failed:', err?.message || err);
    return res.status(500).json({
      success: false,
      error: err?.message || 'Hugging Face VTON try-on failed',
    });
  }
});

/**
 * POST /api/tryon/catvton
 *
 * CatVTON virtual try-on endpoint (ICLR 2025).
 *
 * This uses the CatVTON diffusion model via a Python FastAPI service.
 * CatVTON is designed to preserve person/mannequin identity while realistically
 * draping garments with natural folds and fabric behavior.
 *
 * Prerequisites:
 * - Python FastAPI service running at CATVTON_SERVICE_URL (default: http://localhost:8000)
 * - GPU with CUDA support (8GB+ VRAM recommended for 1024x768)
 * - CatVTON model installed with Detectron2 & DensePose
 *
 * Body:
 *   {
 *     mannequin_image: string  // data URI or HTTP(S) URL
 *     garments: [{ label, garment_image, ... }]    // multi-garment outfit
 *     // OR single garment:
 *     garment_image: string
 *     garment: { label: 'top'|'layer'|'pants'|'shoes', type?, name? }
 *     total?: number
 *   }
 *
 * Response:
 *   {
 *     success: true,
 *     resultUrl: string,        // data:image/png;base64,...
 *     methodUsed: 'catvton',
 *     elapsedMs
 *   }
 */
router.post('/catvton', authenticateToken, tryonLimiter, async (req, res) => {
  const startedAt = Date.now();
  try {
    const mannequinSrc = req.body?.mannequin_image;
    const garmentEntries = Array.isArray(req.body.garments) ? req.body.garments : [];
    const garmentSrc =
      req.body.garment_image ||
      req.body.garment?.image ||
      req.body.garment?.imageUrl ||
      req.body.garment?.url;
    const rawLabel = req.body.garment?.label || req.body.garment?.type || 'top';
    const total = Number(req.body.total ?? 1);

    if (!mannequinSrc) {
      return res.status(400).json({ success: false, error: 'mannequin_image is required' });
    }
    if (!garmentSrc && garmentEntries.length === 0) {
      return res.status(400).json({ success: false, error: 'garment_image is required' });
    }

    // Normalize garments array
    let garments = [];
    if (garmentEntries.length > 0) {
      garments = garmentEntries.map((entry) => ({
        label: normalizeGarmentLabel(entry?.label || entry?.type || 'top'),
        garment_image:
          entry?.garmentSrc ||
          entry?.garment_image ||
          entry?.image ||
          entry?.imageUrl ||
          entry?.url,
        type: entry?.type,
        name: entry?.name,
      }));
    } else {
      garments = [{
        label: normalizeGarmentLabel(rawLabel),
        garment_image: garmentSrc,
      }];
    }

    logger.info(
      `[tryon/catvton] outfit render count=${garments.length} labels=${garments
        .map((g) => g.label)
        .join(',')}`,
    );

    const result = await callCatvton({
      mannequin_image: mannequinSrc,
      garments,
      total: garments.length,
    });

    if (!result.success) {
      return res.status(500).json({
        success: false,
        error: result.error || 'CatVTON service failed',
      });
    }

    const elapsedMs = Date.now() - startedAt;
    logger.info(`[tryon/catvton] done in ${elapsedMs}ms`);

    return res.json({
      success: true,
      resultUrl: result.resultImage,
      methodUsed: result.methodUsed || 'catvton',
      elapsedMs,
    });
  } catch (err) {
    logger.error('[tryon/catvton] failed:', err?.message || err);
    return res.status(500).json({
      success: false,
      error: err?.message || 'CatVTON try-on failed',
    });
  }
});

/**
 * POST /api/tryon/mobile-vton
 *
 * Mobile-VTON try-on endpoint.
 */
router.post('/mobile-vton', authenticateToken, tryonLimiter, async (req, res) => {
  const startedAt = Date.now();
  try {
    const result = await mobileVtonRender(req.body || {});
    if (!result.success) {
      return res.status(500).json(result);
    }
    return res.json(result);
  } catch (err) {
    logger.error('[tryon/mobile-vton] failed:', err?.message || err);
    return res.status(500).json({
      success: false,
      error: err?.message || 'Mobile-VTON try-on failed',
    });
  }
});

/**
 * GET /api/tryon/catvton/health
 *
 * Health check for CatVTON service (via Hugging Face API).
 */
router.get('/catvton/health', async (req, res) => {
  try {
    const health = await checkCatvtonHealth();
    if (health.healthy) {
      return res.json({ success: true, status: 'ok', service: 'catvton-flux' });
    }
    return res.status(503).json({
      success: false,
      error: 'CatVTON service unhealthy',
      details: health,
    });
  } catch (err) {
    logger.error('[tryon/catvton/health] failed:', err?.message || err);
    return res.status(503).json({
      success: false,
      error: err?.message || 'CatVTON health check failed',
    });
  }
});

// Export the per-step helper so admin / debug scripts can run a render
// without going through Express middleware.
export { applyGarmentStep, buildKontextComposite, buildDressingPrompt };
export default router;

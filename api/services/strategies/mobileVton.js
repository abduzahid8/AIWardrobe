/**
 * Mobile-VTON rendering strategy.
 *
 * Primary:  FastAPI Mobile-VTON GPU backend (Modal) via mobileVtonClient.
 * Fallback: FLUX.1-Kontext-dev in-process renderer (tryonRenderer / tryonShared).
 *
 * The fallback activates automatically when the Mobile-VTON service is down
 * (network error, 429 billing limit, 5xx, etc.).  The client always gets a
 * successful try-on — they just won't know which engine ran it.
 */

import logger from '../../utils/logger.js';
import {
  normalizeGarmentLabel,
  GARMENT_RENDER_ORDER,
  buildBaseCanvas,
  encodeCanvas,
  encodeBasePxToDataUri,
  preprocessGarmentAndCache,
} from '../tryonRenderer.js';
import {
  toDataUri,
  loadImageBuffer,
  getNvidiaKey,
  callFluxKontext,
  extractFluxLeftHalfRaw,
  maskMergeFluxIntoBase,
  dilateAndFeatherMask,
  computeDiffMask,
} from '../tryonShared.js';
import { callMobileVton, callMobileVtonMulti, MobileVtonServiceError } from '../mobileVtonClient.js';
import sharp from 'sharp';

// ─── FLUX helpers (fallback) ─────────────────────────────────────────────────

function buildDressingPrompt(label) {
  const garmentNoun =
    label === 'pants'  ? 'pants / trousers' :
    label === 'shoes'  ? 'pair of shoes' :
    label === 'layer'  ? 'outer layer (jacket / coat / cardigan)' :
                         'top (shirt / t-shirt / sweater)';

  const garmentZone =
    label === 'pants'  ? 'on the hips, thighs, knees, and legs of the mannequin' :
    label === 'shoes'  ? 'on the feet of the mannequin' :
    label === 'layer'  ? 'over the existing top, on the shoulders, chest, back, and sleeves of the mannequin' :
                         'on the torso, chest, shoulders, and arms of the mannequin';

  return [
    'You are a photorealistic virtual try-on engine. The input is a single image with TWO halves separated vertically:',
    'LEFT HALF: a smooth, headless, light-grey fashion mannequin on a clean white seamless studio background.',
    `RIGHT HALF: the exact product photo of a single new ${garmentNoun} that must be put on the mannequin.`,
    `TASK: produce the same scene but with the ${garmentNoun} from the right half now WORN and DRESSED ${garmentZone}.`,
    'ABSOLUTE HARD CONSTRAINTS: mannequin pose unchanged, background unchanged, no human face or skin, all previously applied garments remain.',
    'STYLE: studio fashion catalog photograph, soft even lighting, sharp focus, photorealistic.',
  ].join(' ');
}

async function buildKontextComposite(personImage, garmentImage) {
  const [personBuf, garmentBuf] = await Promise.all([
    loadImageBuffer(personImage),
    loadImageBuffer(garmentImage),
  ]);
  const [left, right] = await Promise.all([
    sharp(personBuf).resize(768, 1024, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } }).png().toBuffer(),
    sharp(garmentBuf).resize(768, 1024, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } }).png().toBuffer(),
  ]);
  const composite = await sharp({ create: { width: 1536, height: 1024, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } } })
    .composite([{ input: left, left: 0, top: 0 }, { input: right, left: 768, top: 0 }])
    .png()
    .toBuffer();
  return `data:image/png;base64,${composite.toString('base64')}`;
}

async function applyGarmentStepFlux(basePx, garmentSrc, label) {
  const normalized = normalizeGarmentLabel(label);
  const preStep = new Uint8ClampedArray(basePx);
  const stepStart = Date.now();

  let cleanedGarmentImage = garmentSrc;
  try {
    const cleaned = await preprocessGarmentAndCache(garmentSrc, normalized);
    cleanedGarmentImage = toDataUri(cleaned, 'image/png');
  } catch (preprocessErr) {
    logger.warn(`[mobileVton/fallback] preprocess fallback label=${normalized}: ${preprocessErr?.message}`);
  }

  const nvidiaKey = await getNvidiaKey();
  const mannequinDataUri = await encodeBasePxToDataUri(basePx);
  const composite = await buildKontextComposite(mannequinDataUri, cleanedGarmentImage);

  const fluxResult = await callFluxKontext({
    imageDataUri: composite,
    prompt: buildDressingPrompt(normalized),
    nvidiaKey,
    provider: process.env.FLUX_PROVIDER || 'nvidia_local',
  });

  const fluxRaw = await extractFluxLeftHalfRaw(fluxResult);
  if (!fluxRaw) throw new Error(`FLUX output could not be decoded label=${normalized}`);

  const { mask, coverage, drifted } = computeDiffMask(preStep, fluxRaw, { threshold: 14, maxCoverage: 0.85 });
  if (drifted) {
    throw new Error(`FLUX drift guard tripped label=${normalized} coverage=${(coverage * 100).toFixed(1)}%`);
  }

  const mergeMask = await dilateAndFeatherMask(mask, 8, 4);
  maskMergeFluxIntoBase(basePx, preStep, fluxRaw, mergeMask);

  logger.info(`[mobileVton/fallback] FLUX step done label=${normalized} in ${Date.now() - stepStart}ms`);
  return normalized;
}

// ─── Multi-garment FLUX fallback ─────────────────────────────────────────────

async function fluxMultiFallback(mannequinSrc, ordered) {
  logger.info(`[mobileVton/fallback] Running FLUX fallback for ${ordered.length} garment(s)`);
  const basePx = await buildBaseCanvas(mannequinSrc);
  const renderedLabels = [];

  for (const g of ordered) {
    const label = await applyGarmentStepFlux(basePx, g.image, g.label);
    renderedLabels.push(label);
  }

  const lastLabel = renderedLabels[renderedLabels.length - 1] || 'outfit';
  const encoded = await encodeCanvas(basePx, lastLabel);

  return {
    success: true,
    resultUrl: encoded.imageDataUri,
    methodUsed: 'flux_fallback_diff_mask',
    fluxStepCount: ordered.length,
    step: ordered.length,
    total: ordered.length,
    garmentLabel: lastLabel,
    renderedGarments: renderedLabels,
    degraded: true,
    degradedReason: 'mobile_vton_unavailable',
  };
}

// ─── Main export ─────────────────────────────────────────────────────────────

/**
 * Render a single-garment or multi-garment outfit via Mobile-VTON.
 * Falls back to the FLUX.1-Kontext-dev renderer when Mobile-VTON is unreachable.
 */
export async function mobileVtonRender(params) {
  const {
    mannequin_image: mannequinSrc,
    garments: garmentEntries,
    garment_image: garmentSrc,
    garment,
    step,
    total,
    num_inference_steps = 25,
    guidance_scale = 7.5,
    seed,
    pipeline_version = 'sequential_v1',
    // Body-fit additions (Month 1) — forwarded to the Python service.
    body_profile: bodyProfile,
    fit_assessment: fitAssessment,
    fit_assessments: fitAssessments,
  } = params;

  if (!mannequinSrc) {
    return { success: false, error: 'mannequin_image is required' };
  }

  // ── Multi-garment ──────────────────────────────────────────────────────────
  if (Array.isArray(garmentEntries) && garmentEntries.length > 0) {
    const ordered = GARMENT_RENDER_ORDER
      .map((label) => {
        const entry = garmentEntries.find(
          (g) => normalizeGarmentLabel(g?.label || g?.type || label) === label,
        );
        if (!entry) return null;
        return {
          label,
          image: entry?.garmentSrc || entry?.garment_image || entry?.image || entry?.imageUrl || entry?.url,
          description: entry?.name || entry?.description || label,
        };
      })
      .filter((g) => g?.image);

    if (ordered.length === 0) {
      return { success: false, error: 'No valid garments supplied' };
    }

    logger.info(`[mobileVton] multi count=${ordered.length} labels=${ordered.map((g) => g.label).join(',')} pipeline=${pipeline_version}`);

    try {
      const garmentsPayload = ordered.map((g) => ({
        garment_image: g.image,
        description: g.description,
        label: g.label,
      }));

      const result = await callMobileVtonMulti({
        personImage: mannequinSrc,
        garments: garmentsPayload,
        guidanceScale: guidance_scale,
        numInferenceSteps: num_inference_steps,
        seed,
        pipelineVersion: pipeline_version,
        bodyProfile: bodyProfile || null,
        fitAssessments: fitAssessments || null,
      });

      return {
        success: true,
        resultUrl: result.resultImage,
        methodUsed: result.methodUsed || 'mobile_vton',
        pipelineVersion: result.pipelineVersion || result.pipeline_version || pipeline_version,
        diagnostics: result.diagnostics || null,
        fluxStepCount: ordered.length,
        step: ordered.length,
        total: ordered.length,
        garmentLabel: ordered[ordered.length - 1]?.label || 'outfit',
        renderedGarments: ordered.map((g) => g.label),
        elapsedMs: result.elapsedMs,
      };
    } catch (err) {
      // If Mobile-VTON is down, fall back to FLUX in-process
      if (err instanceof MobileVtonServiceError && err.isServiceDown) {
        logger.warn(`[mobileVton] service down, activating FLUX fallback: ${err.message}`);
        try {
          return await fluxMultiFallback(mannequinSrc, ordered);
        } catch (fluxErr) {
          logger.error(`[mobileVton] FLUX fallback also failed: ${fluxErr.message}`);
          return { success: false, error: `Try-on failed (both engines): ${fluxErr.message}` };
        }
      }
      logger.error(`[mobileVton] multi-garment failed: ${err.message}`);
      return { success: false, error: err.message };
    }
  }

  // ── Single garment ─────────────────────────────────────────────────────────
  const singleSrc = garmentSrc || garment?.image || garment?.imageUrl || garment?.url;
  const rawLabel = garment?.label || garment?.type || 'top';
  const label = normalizeGarmentLabel(rawLabel, step || 1);

  if (!singleSrc) {
    return { success: false, error: 'garment_image is required' };
  }

  logger.info(`[mobileVton] single label=${label} step=${step}/${total}`);

  try {
    const result = await callMobileVton({
      personImage: mannequinSrc,
      garmentImage: singleSrc,
      garmentDescription: garment?.name || label,
      guidanceScale: guidance_scale,
      numInferenceSteps: num_inference_steps,
      seed,
      bodyProfile: bodyProfile || null,
      fitAssessment: fitAssessment || null,
    });

    return {
      success: true,
      resultUrl: result.resultImage,
      methodUsed: 'mobile_vton',
      fluxStepCount: 1,
      step: step || 1,
      total: total || 1,
      garmentLabel: label,
      renderedGarments: [label],
      elapsedMs: result.elapsedMs,
    };
  } catch (err) {
    // Fallback for single garment as well
    if (err instanceof MobileVtonServiceError && err.isServiceDown) {
      logger.warn(`[mobileVton] service down (single), activating FLUX fallback: ${err.message}`);
      try {
        const ordered = [{ label, image: singleSrc, description: garment?.name || label }];
        const result = await fluxMultiFallback(mannequinSrc, ordered);
        return {
          ...result,
          step: step || 1,
          total: total || 1,
          garmentLabel: label,
          renderedGarments: [label],
          fluxStepCount: 1,
        };
      } catch (fluxErr) {
        logger.error(`[mobileVton] FLUX fallback also failed: ${fluxErr.message}`);
        return { success: false, error: `Try-on failed (both engines): ${fluxErr.message}` };
      }
    }
    logger.error(`[mobileVton] single garment failed: ${err.message}`);
    return { success: false, error: err.message };
  }
}

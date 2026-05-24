import logger from '../../utils/logger.js';
import {
  normalizeGarmentLabel,
  GARMENT_RENDER_ORDER,
} from '../tryonRenderer.js';
import { callMobileVton, callMobileVtonMulti } from '../mobileVtonClient.js';

/**
 * Render a single-garment or multi-garment outfit via Mobile-VTON.
 *
 * Mobile-VTON is a high-fidelity on-device virtual try-on model (CVPR 2026)
 * based on SD3.5 with custom UNets, supporting single garment try-on.
 *
 * For multi-garment: iterates through each garment sequentially, sending
 * the full canvas at each step.
 */
export async function mobileVtonRender(params) {
  const {
    mannequin_image: mannequinSrc,
    garments: garmentEntries,
    garment_image: garmentSrc,
    garment,
    step,
    total,
    num_inference_steps = 10,
    guidance_scale = 2.0,
    seed,
  } = params;

  if (!mannequinSrc) {
    return { success: false, error: 'mannequin_image is required' };
  }

  // Multi-garment outfit
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

    logger.info(`[mobileVton] multi-garment count=${ordered.length} labels=${ordered.map(g => g.label).join(',')}`);

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
      });

      return {
        success: true,
        resultUrl: result.resultImage,
        methodUsed: 'mobile_vton',
        fluxStepCount: ordered.length,
        step: ordered.length,
        total: ordered.length,
        garmentLabel: ordered[ordered.length - 1]?.label || 'outfit',
        renderedGarments: ordered.map((g) => g.label),
        elapsedMs: result.elapsedMs,
      };
    } catch (err) {
      logger.error(`[mobileVton] multi-garment failed: ${err.message}`);
      return { success: false, error: err.message };
    }
  }

  // Single garment
  const singleSrc = garmentSrc || garment?.image || garment?.imageUrl || garment?.url;
  const rawLabel = garment?.label || garment?.type || 'top';
  const label = normalizeGarmentLabel(rawLabel, step || 1);

  if (!singleSrc) {
    return { success: false, error: 'garment_image is required' };
  }

  logger.info(`[mobileVton] single garment label=${label} step=${step}/${total}`);

  try {
    const result = await callMobileVton({
      personImage: mannequinSrc,
      garmentImage: singleSrc,
      garmentDescription: garment?.name || label,
      guidanceScale: guidance_scale,
      numInferenceSteps: num_inference_steps,
      seed,
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
    logger.error(`[mobileVton] single garment failed: ${err.message}`);
    return { success: false, error: err.message };
  }
}

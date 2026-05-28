/**
 * Mobile-VTON Client
 * API Client for FastAPI Mobile-VTON GPU backend on Modal/Cloud Run.
 *
 * Wraps all Axios errors into structured, user-friendly ServiceUnavailableError
 * so the calling strategy can decide whether to fall back to FLUX.
 */

import axios from 'axios';
import logger from '../utils/logger.js';

// ─── Sentinel error class ────────────────────────────────────────────────────
/**
 * Thrown when the Mobile-VTON service is unreachable or returned an error.
 * The `isServiceDown` flag indicates a network-level failure (vs a 4xx response).
 */
export class MobileVtonServiceError extends Error {
  constructor(message, { isServiceDown = false, statusCode = null, details = null } = {}) {
    super(message);
    this.name = 'MobileVtonServiceError';
    this.isServiceDown = isServiceDown;   // true → network error / 5xx from Modal
    this.statusCode = statusCode;
    this.details = details;
  }
}

// ─── URL configuration ───────────────────────────────────────────────────────
const getBaseUrl = () =>
  process.env.MOBILE_VTON_SERVICE_URL ||
  process.env.MODAL_MOBILE_VTON_URL ||
  'https://karimdzanovzoha--aiwardrobe-mobile-vton-fastapi-app.modal.run';

/** Normalize an Axios / network error into a MobileVtonServiceError. */
function wrapError(error) {
  if (error instanceof MobileVtonServiceError) return error;

  // Network-level failure (ECONNREFUSED, ETIMEDOUT, ENOTFOUND, 429-billing, etc.)
  const isNetwork =
    error.code === 'ECONNREFUSED' ||
    error.code === 'ECONNABORTED' ||
    error.code === 'ETIMEDOUT'   ||
    error.code === 'ENOTFOUND'   ||
    !error.response; // no HTTP response means transport-level error

  const statusCode = error.response?.status ?? null;
  // Modal billing limit returns 429, server errors 5xx → treat as "service down"
  const isServiceDown = isNetwork || statusCode === 429 || (statusCode >= 500 && statusCode < 600);

  const detail = error.response?.data?.detail || error.response?.data?.error || error.message;
  const friendly = isServiceDown
    ? 'The try-on GPU service is currently unavailable. Falling back to AI renderer.'
    : `Try-on service error: ${detail}`;

  return new MobileVtonServiceError(friendly, { isServiceDown, statusCode, details: detail });
}

// ─── Single-garment try-on ───────────────────────────────────────────────────
/**
 * POST /tryon — single garment onto mannequin.
 * @throws {MobileVtonServiceError}
 */
export async function callMobileVton({
  personImage,
  garmentImage,
  garmentDescription = 'clothing',
  guidanceScale = 2.0,
  numInferenceSteps = 10,
  seed,
}) {
  const baseUrl = getBaseUrl();
  const startTime = Date.now();
  logger.info(`[mobileVtonClient] single try-on → ${baseUrl}/tryon`);

  try {
    const response = await axios.post(
      `${baseUrl}/tryon`,
      {
        person_image: personImage,
        garment_image: garmentImage,
        garment_description: garmentDescription,
        guidance_scale: Number(guidanceScale),
        num_inference_steps: Number(numInferenceSteps),
        seed: seed ? Number(seed) : 42,
      },
      { timeout: 180_000 },
    );

    const elapsedMs = Date.now() - startTime;
    logger.info(`[mobileVtonClient] single try-on OK in ${elapsedMs}ms`);

    const data = response.data || {};
    return {
      ...data,
      resultImage: data.result_image || data.resultImage,
      elapsedMs: data.elapsed_ms || data.elapsedMs || elapsedMs,
    };
  } catch (error) {
    const wrapped = wrapError(error);
    logger.error(`[mobileVtonClient] single try-on failed (serviceDown=${wrapped.isServiceDown}): ${wrapped.message}`);
    throw wrapped;
  }
}

// ─── Multi-garment try-on ────────────────────────────────────────────────────
/**
 * POST /tryon/multi or /tryon/multi-fused — multiple garments.
 * @throws {MobileVtonServiceError}
 */
export async function callMobileVtonMulti({
  personImage,
  garments,
  guidanceScale = 2.0,
  numInferenceSteps = 10,
  seed,
  pipelineVersion = 'sequential_v1',
}) {
  const baseUrl = getBaseUrl();
  const startTime = Date.now();

  const isFused = pipelineVersion === 'fused_v2' || pipelineVersion === 'fused_v3';
  const endpoint = isFused ? '/tryon/multi-fused' : '/tryon/multi';
  logger.info(`[mobileVtonClient] multi try-on (${pipelineVersion}) → ${baseUrl}${endpoint}`);

  try {
    const payload = {
      person_image: personImage,
      garments: garments.map((g) => ({
        garment_image: g.garment_image || g.image,
        description: g.description || 'clothing',
        label: g.label,
      })),
      guidance_scale: Number(guidanceScale),
      num_inference_steps: Number(numInferenceSteps),
      seed: seed ? Number(seed) : 42,
    };

    if (isFused) {
      payload.pipeline_version = pipelineVersion;
    }

    const response = await axios.post(`${baseUrl}${endpoint}`, payload, { timeout: 180_000 });

    const elapsedMs = Date.now() - startTime;
    logger.info(`[mobileVtonClient] multi try-on OK in ${elapsedMs}ms`);

    const data = response.data || {};
    return {
      ...data,
      resultImage: data.result_image || data.resultImage,
      elapsedMs: data.elapsed_ms || data.elapsedMs || elapsedMs,
    };
  } catch (error) {
    const wrapped = wrapError(error);
    logger.error(`[mobileVtonClient] multi try-on failed (serviceDown=${wrapped.isServiceDown}): ${wrapped.message}`);
    throw wrapped;
  }
}

// ─── Health check ────────────────────────────────────────────────────────────
export async function checkMobileVtonHealth() {
  try {
    const baseUrl = getBaseUrl();
    const response = await axios.get(`${baseUrl}/health`, { timeout: 8_000 });
    return {
      healthy: response.data?.status === 'ok' || response.status === 200,
      details: response.data,
    };
  } catch (err) {
    return { healthy: false, error: err.message };
  }
}

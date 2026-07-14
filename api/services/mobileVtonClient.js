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
  'https://zoxxid75--aiwardrobe-mobile-vton-fastapi-app.modal.run';

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

// ─── Internal retry helper ───────────────────────────────────────────────────
/**
 * Execute an async fn, retrying once after `delayMs` if the first call throws
 * a retriable error (network-level or 5xx). 4xx errors are not retried.
 */
async function withRetry(fn, { maxRetries = 1, delayMs = 5000, label = '' } = {}) {
  let lastErr;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      const wrapped = err instanceof MobileVtonServiceError ? err : wrapError(err);
      // Retry only on service-level errors (network / 5xx), not 4xx
      if (!wrapped.isServiceDown || attempt >= maxRetries) throw wrapped;
      logger.warn(`[mobileVtonClient] ${label} attempt ${attempt + 1} failed (${wrapped.message}), retrying in ${delayMs}ms…`);
      await new Promise((r) => setTimeout(r, delayMs));
    }
  }
  throw lastErr;
}

// ─── Single-garment try-on ───────────────────────────────────────────────────
/**
 * POST /tryon — single garment onto mannequin.
 * Retries once after 5 s on network / 5xx errors (covers Render cold-start).
 * @throws {MobileVtonServiceError}
 */
export async function callMobileVton({
  personImage,
  garmentImage,
  garmentDescription = 'clothing',
  guidanceScale = 7.5,
  numInferenceSteps = 25,
  seed,
  bodyProfile = null,        // ← Month 1: forwarded when present
  fitAssessment = null,      // ← Month 1: forwarded when present
}) {
  const baseUrl = getBaseUrl();
  const startTime = Date.now();
  logger.info(`[mobileVtonClient] single try-on → ${baseUrl}/tryon`);

  const payload = {
    person_image: personImage,
    garment_image: garmentImage,
    garment_description: garmentDescription,
    guidance_scale: Number(guidanceScale),
    num_inference_steps: Number(numInferenceSteps),
    seed: seed ? Number(seed) : 42,
  };

  // Body-fit context (optional — engines that don't support it ignore).
  if (bodyProfile) payload.body_profile = bodyProfile;
  if (fitAssessment) payload.fit_assessment = fitAssessment;

  const result = await withRetry(
    () => axios.post(`${baseUrl}/tryon`, payload, { timeout: 240_000 }),
    { maxRetries: 1, delayMs: 5_000, label: 'single try-on' },
  );

  const elapsedMs = Date.now() - startTime;
  logger.info(`[mobileVtonClient] single try-on OK in ${elapsedMs}ms`);

  const data = result.data || {};
  return {
    ...data,
    resultImage: data.result_image || data.resultImage,
    elapsedMs: data.elapsed_ms || data.elapsedMs || elapsedMs,
  };
}

// ─── Multi-garment try-on ────────────────────────────────────────────────────
/**
 * POST /tryon/multi or /tryon/multi-fused — multiple garments.
 * @throws {MobileVtonServiceError}
 */
export async function callMobileVtonMulti({
  personImage,
  garments,
  guidanceScale = 7.5,
  numInferenceSteps = 25,
  seed,
  pipelineVersion = 'sequential_v1',
  bodyProfile = null,    // ← Month 1: forwarded when present
  fitAssessments = null, // ← Month 1: array, one per garment
}) {
  const baseUrl = getBaseUrl();
  const startTime = Date.now();

  const isFused = pipelineVersion === 'fused_v2' || pipelineVersion === 'fused_v3';
  const endpoint = isFused ? '/tryon/multi-fused' : '/tryon/multi';
  logger.info(`[mobileVtonClient] multi try-on (${pipelineVersion}) → ${baseUrl}${endpoint}`);

  try {
    const payload = {
      person_image: personImage,
      garments: garments.map((g, i) => ({
        garment_image: g.garment_image || g.image,
        description: g.description || 'clothing',
        label: g.label,
        // Pass through per-garment fit context if the caller supplied it.
        selected_size: g.selected_size || g.selectedSize || null,
        physical_profile: g.physical_profile || g.physicalProfile || null,
        fit_assessment: g.fit_assessment || g.fitAssessment
          || (Array.isArray(fitAssessments) ? fitAssessments[i] : null),
      })),
      guidance_scale: Number(guidanceScale),
      num_inference_steps: Number(numInferenceSteps),
      seed: seed ? Number(seed) : 42,
    };

    if (isFused) {
      payload.pipeline_version = pipelineVersion;
    }

    // Top-level body profile + the per-garment assessments array.
    if (bodyProfile) payload.body_profile = bodyProfile;
    if (Array.isArray(fitAssessments) && fitAssessments.length > 0) {
      payload.fit_assessments = fitAssessments;
    }

    const result = await withRetry(
      () => axios.post(`${baseUrl}${endpoint}`, payload, { timeout: 240_000 }),
      { maxRetries: 1, delayMs: 5_000, label: `multi try-on (${pipelineVersion})` },
    );

    const elapsedMs = Date.now() - startTime;
    logger.info(`[mobileVtonClient] multi try-on OK in ${elapsedMs}ms`);

    const data = result.data || {};
    return {
      ...data,
      resultImage: data.result_image || data.resultImage,
      elapsedMs: data.elapsed_ms || data.elapsedMs || elapsedMs,
    };
  } catch (err) {
    throw wrapError(err);
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

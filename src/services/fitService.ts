/**
 * Fit service — client-side wrapper around the fit engine + remote /api/fit.
 *
 * Two execution paths:
 *   1. LOCAL: run the engine in-process (no network). Fast, no rate limit,
 *      works offline. Default.
 *   2. REMOTE: call POST /api/fit/assess. Useful when the engine version
 *      needs to be centralized (e.g. A/B testing ease thresholds) or when
 *      a backend-only signal (analytics) needs to be recorded.
 *
 * The try-on screen uses LOCAL by default (zero-latency fit verdict before
 * the user even hits "Try on") and only calls REMOTE when the user taps
 * "Get fresh recommendation" (which is a deliberately slower path).
 *
 * Same shape as the server's /api/fit/assess response: { success, assessment }.
 */

import { assessFit, recommendSize } from '../lib/fit/fitEngine';
import type { BodyProfile } from '../types/bodyProfile';
import type { GarmentPhysicalProfile } from '../types/garment';
import type { FitAssessment } from '../types/fitAssessment';
import { supabase } from '../lib/supabase';

const API_TIMEOUT_MS = 8_000;

export interface FitServiceResult {
  assessment: FitAssessment;
  source: 'local' | 'remote';
  /** Wall-clock ms for the call (local is sub-ms; remote is HTTP RTT). */
  elapsedMs: number;
}

/** Run the fit engine in-process. */
export function assessFitLocal(
  body: BodyProfile,
  garment: GarmentPhysicalProfile,
): FitServiceResult {
  const start = Date.now();
  return {
    assessment: assessFit(body, garment),
    source: 'local',
    elapsedMs: Date.now() - start,
  };
}

/** Call POST /api/fit/assess on the configured API. */
export async function assessFitRemote(
  body: BodyProfile,
  garment: GarmentPhysicalProfile,
  apiBase: string,
): Promise<FitServiceResult> {
  const start = Date.now();
  const session = (await supabase.auth.getSession()).data.session;
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (session?.access_token) headers.Authorization = `Bearer ${session.access_token}`;

  const resp = await fetch(`${apiBase.replace(/\/$/, '')}/api/fit/assess`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ bodyProfile: body, garmentId: garment.garmentId, sizeLabel: garment.sizeLabel }),
    signal: AbortSignal.timeout(API_TIMEOUT_MS),
  });

  if (!resp.ok) {
    throw new Error(`fit/assess HTTP ${resp.status}: ${await resp.text().catch(() => '')}`);
  }
  const json = await resp.json();
  if (!json?.success || !json?.assessment) {
    throw new Error('fit/assess returned no assessment');
  }
  return { assessment: json.assessment as FitAssessment, source: 'remote', elapsedMs: Date.now() - start };
}

/**
 * Find the right GarmentPhysicalProfile for a shop item + size, falling back
 * to the seed set if the live catalog doesn't have one. This is the function
 * try-on screens call once they know the user's selected size.
 */
export function resolveGarmentProfile(
  itemId: string,
  sizeLabel: string,
  shopProfile?: { physicalProfiles?: GarmentPhysicalProfile[] },
): GarmentPhysicalProfile | undefined {
  const fromShop = shopProfile?.physicalProfiles?.find(
    (p) => p.garmentId === itemId && p.sizeLabel === sizeLabel,
  );
  if (fromShop) return fromShop;

  // Lazy import to avoid bundling seed data when not needed.
  // The seed is small; inlining is fine for v1.
  const { SEED_GARMENT_PHYSICAL_PROFILES } = require('../types/garment');
  return SEED_GARMENT_PHYSICAL_PROFILES.find(
    (p: GarmentPhysicalProfile) => p.garmentId === itemId && p.sizeLabel === sizeLabel,
  );
}

/** Compare a selected size against alternatives and recommend a better one. */
export function recommendSizeLocal(
  body: BodyProfile,
  current: GarmentPhysicalProfile,
  alternatives: GarmentPhysicalProfile[],
): { recommendedSize?: string; reason?: string } {
  const assessment = assessFit(body, current);
  return recommendSize(assessment, current, alternatives, body) ?? {};
}

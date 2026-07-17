/**
 * Fit assessment — the fit engine's verdict.
 *
 * Returned by `assessFit(body, garment, size)` (see src/lib/fit/fitEngine.ts).
 * Designed to be:
 *   - Stable across refactors (golden tests assert this shape)
 *   - Serializable for both mobile UI rendering and backend analytics
 *   - Conservative — every claim is paired with a confidence so the UI
 *     can render badges ("high / medium / low") and the user knows when
 *     to trust the recommendation.
 */

import { Confidence } from './bodyProfile';
export type { Confidence } from './bodyProfile';

export type OverallFit =
  | 'too_small'
  | 'tight'
  | 'good_fit'
  | 'relaxed'
  | 'oversized'
  | 'too_large'
  | 'unknown';

export type ZoneStatus =
  | 'too_tight'
  | 'snug'
  | 'good'
  | 'loose'
  | 'too_loose'
  | 'too_short'
  | 'too_long'
  | 'unknown';

export type FitZone =
  | 'shoulders'
  | 'chest'
  | 'waist'
  | 'hips'
  | 'arms'
  | 'sleeves'
  | 'torso_length'
  | 'thigh'
  | 'inseam'
  | 'rise'
  | 'calf'
  | 'feet';

export interface FitZoneAssessment {
  zone: FitZone;
  status: ZoneStatus;
  /** Garment minus body, in cm. Positive = garment is larger (loose). */
  deltaCm?: number;
  /** Human-readable explanation, shown in the fit panel. */
  message: string;
}

export interface FitAssessment {
  garmentId: string;
  bodyProfileId: string;
  selectedSize: string;
  overall: OverallFit;
  confidence: Confidence;
  zones: FitZoneAssessment[];
  /** Optional recommendation, if the engine found a better size in the same garment. */
  sizeRecommendation?: {
    recommendedSize: string;
    reason: string;
  };
  /** Engine version, useful for analytics + golden tests. */
  engineVersion: string;
  /** When the assessment was generated. ISO-8601. */
  generatedAt: string;
}

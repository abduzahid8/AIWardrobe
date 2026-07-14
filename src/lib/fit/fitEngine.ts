/**
 * Fit Engine v1 — pure functions only, no side effects, no React, no network.
 *
 * Input:  BodyProfile + GarmentPhysicalProfile (for a single selected size)
 * Output: FitAssessment
 *
 * The engine compares body measurements vs garment measurements, computes a
 * per-zone status, rolls up an overall verdict, optionally recommends a
 * different size, and tags everything with a confidence score.
 *
 * Rules are calibrated against the typical "buy clothing online" decisions
 * (Month 1 of the 6-month plan). They are deliberately conservative — we'd
 * rather under-promise and over-deliver than mislead a user into a bad size.
 *
 * The engine is also forward-compatible: Month 5 will add stretch-aware
 * tolerance, layering-aware jacket ease, and men/women sizing differences.
 *
 * Engine version is exported so analytics + golden tests can lock behaviour.
 */

import {
  BodyProfile,
  BodyMeasurement,
  Confidence,
} from '../../types/bodyProfile';
import { GarmentPhysicalProfile, FitIntent, Stretch } from '../../types/garment';
import {
  FitAssessment,
  FitZone,
  FitZoneAssessment,
  OverallFit,
  ZoneStatus,
} from '../../types/fitAssessment';

export const FIT_ENGINE_VERSION = 'fit-engine/v1';

// ─── Ease targets per fit intent (cm of chest ease over body chest) ──────────
// Source: 6-month plan, "Fit Engine Initial Rules > Tops > General ease targets"
const CHEST_EASE_TARGETS: Record<FitIntent, [number, number]> = {
  compression: [0, 2],
  slim: [2, 5],
  regular: [5, 10],
  relaxed: [10, 16],
  oversized: [16, 30],
};

// Pants use hip ease as the primary anchor.
const HIP_EASE_TARGETS: Record<FitIntent, [number, number]> = {
  compression: [0, 2],
  slim: [1, 4],
  regular: [4, 8],
  relaxed: [8, 14],
  oversized: [14, 25],
};

// Stretch gives us extra tolerance. 'high' stretch lets a tighter garment still fit.
const STRETCH_TOLERANCE_CM: Record<Stretch, number> = {
  none: 0,
  low: 1,
  medium: 2,
  high: 4,
};

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Assess fit for a single body + garment + size combination.
 * Returns a FitAssessment with overall fit, per-zone notes, and confidence.
 *
 * Never throws — returns `overall: 'unknown'` and `confidence: 'low'` if
 * inputs are missing, so callers can render a "we don't have enough data"
 * message instead of crashing.
 */
export function assessFit(
  body: BodyProfile,
  garment: GarmentPhysicalProfile,
): FitAssessment {
  const generatedAt = new Date().toISOString();

  const measurements = body.measurements ?? {};
  const heightCm = body.height?.valueCm;

  // Bail gracefully if the basics aren't there.
  if (!heightCm || !body.bodyType) {
    return buildUnknown(body, garment, generatedAt, ['height or body type missing']);
  }

  const garmentMeasurements = garment.measurements ?? {};
  const fitIntent: FitIntent = garment.fitIntent ?? 'regular';
  const stretch: Stretch = garment.stretch ?? 'none';
  const stretchCm = STRETCH_TOLERANCE_CM[stretch];

  let zones: FitZoneAssessment[] = [];
  let overall: OverallFit = 'unknown';
  let confidence: Confidence = 'medium';

  switch (garment.category) {
    case 'top':
    case 'shirt':
    case 'dress':
      ({ zones, overall, confidence } = assessUpper(
        measurements, garmentMeasurements, fitIntent, stretchCm,
      ));
      break;

    case 'jacket':
    case 'coat':
      ({ zones, overall, confidence } = assessJacket(
        measurements, garmentMeasurements, fitIntent, stretchCm,
      ));
      break;

    case 'pants':
    case 'jeans':
    case 'skirt':
      ({ zones, overall, confidence } = assessLower(
        measurements, garmentMeasurements, fitIntent, stretchCm,
      ));
      break;

    case 'shoes':
      ({ zones, overall, confidence } = assessShoes(
        measurements, garmentMeasurements, stretchCm,
      ));
      break;
  }

  return {
    garmentId: garment.garmentId,
    bodyProfileId: body.id,
    selectedSize: garment.sizeLabel,
    overall,
    confidence,
    zones,
    engineVersion: FIT_ENGINE_VERSION,
    generatedAt,
  };
}

// ─── Per-category assessors ──────────────────────────────────────────────────

function assessUpper(
  bodyM: NonNullable<BodyProfile['measurements']>,
  gM: NonNullable<GarmentPhysicalProfile['measurements']>,
  fitIntent: FitIntent,
  stretchCm: number,
): { zones: FitZoneAssessment[]; overall: OverallFit; confidence: Confidence } {
  const zones: FitZoneAssessment[] = [];
  const [easeLo, easeHi] = CHEST_EASE_TARGETS[fitIntent];

  // ── Chest ──────────────────────────────────────────────────────────────
  if (bodyM.chest && gM.chest) {
    const ease = gM.chest.valueCm - bodyM.chest.valueCm;
    const { status, message } = evaluateEase(
      ease, easeLo - stretchCm, easeHi + stretchCm, 'Chest', 'cm',
    );
    zones.push({ zone: 'chest', status, deltaCm: round1(ease), message });
  }

  // ── Shoulders ──────────────────────────────────────────────────────────
  if (bodyM.shoulderWidth && gM.shoulderWidth) {
    const delta = gM.shoulderWidth.valueCm - bodyM.shoulderWidth.valueCm;
    let status: ZoneStatus = 'good';
    let message = 'Shoulder line matches.';
    if (delta < -3) {
      status = 'too_tight';
      message = 'Shoulders will feel restrictive — the garment is narrower than your frame.';
    } else if (delta < -1) {
      status = 'snug';
      message = 'Shoulders will be snug.';
    } else if (delta > 4) {
      status = 'too_loose';
      message = 'Shoulders will drop off the frame — the garment is much wider than you.';
    } else if (delta > 2) {
      status = 'loose';
      message = 'Shoulder line will be slightly dropped.';
    }
    zones.push({ zone: 'shoulders', status, deltaCm: round1(delta), message });
  }

  // ── Sleeve length ──────────────────────────────────────────────────────
  if (bodyM.armLength && gM.sleeveLength) {
    const delta = gM.sleeveLength.valueCm - bodyM.armLength.valueCm;
    let status: ZoneStatus = 'good';
    let message = 'Sleeve length matches your arm.';
    if (delta < -3) {
      status = 'too_short';
      message = 'Sleeves will be too short.';
    } else if (delta < -1) {
      status = 'too_short';
      message = 'Sleeves will be slightly short.';
    } else if (delta > 5) {
      status = 'too_long';
      message = 'Sleeves will be noticeably long.';
    } else if (delta > 3) {
      status = 'too_long';
      message = 'Sleeves will be slightly long.';
    }
    zones.push({ zone: 'sleeves', status, deltaCm: round1(delta), message });
  }

  // ── Torso length ───────────────────────────────────────────────────────
  if (bodyM.torsoLength && gM.bodyLength) {
    const delta = gM.bodyLength.valueCm - bodyM.torsoLength.valueCm;
    let status: ZoneStatus = 'good';
    let message = 'Body length is balanced.';
    if (delta < -4) {
      status = 'too_short';
      message = 'Hem will sit high — top will feel cropped.';
    } else if (delta > 6) {
      status = 'too_long';
      message = 'Hem will sit low — top will feel long.';
    }
    zones.push({ zone: 'torso_length', status, deltaCm: round1(delta), message });
  }

  return {
    zones,
    overall: rollupOverall(zones, fitIntent),
    confidence: confidenceFromZones(zones, ['chest', 'shoulders', 'sleeves', 'torso_length']),
  };
}

function assessJacket(
  bodyM: NonNullable<BodyProfile['measurements']>,
  gM: NonNullable<GarmentPhysicalProfile['measurements']>,
  fitIntent: FitIntent,
  stretchCm: number,
) {
  // Layering: a jacket/coat sits over a top, so it needs extra ease.
  // Plan says "add 2-6 cm over top depending on layer thickness".
  // We use a flat 4 cm layering ease — good enough for v1, refined in Month 5.
  const layeringEase = 4;
  const adjustedGarment: typeof gM = {
    ...gM,
    chest: gM.chest ? { ...gM.chest, valueCm: gM.chest.valueCm - layeringEase } : gM.chest,
  };
  const result = assessUpper(bodyM, adjustedGarment, fitIntent, stretchCm);
  // Add jacket-specific note
  result.zones.unshift({
    zone: 'chest',
    status: 'good',
    message: 'Includes ~4 cm layering ease for wearing over a top.',
  });
  return result;
}

function assessLower(
  bodyM: NonNullable<BodyProfile['measurements']>,
  gM: NonNullable<GarmentPhysicalProfile['measurements']>,
  fitIntent: FitIntent,
  stretchCm: number,
): { zones: FitZoneAssessment[]; overall: OverallFit; confidence: Confidence } {
  const zones: FitZoneAssessment[] = [];
  const [easeLo, easeHi] = HIP_EASE_TARGETS[fitIntent];

  // ── Waist ──────────────────────────────────────────────────────────────
  if (bodyM.waist && gM.waist) {
    const ease = gM.waist.valueCm - bodyM.waist.valueCm;
    // Pants: a negative ease is acceptable with stretch (e.g. jeans with elastane).
    const adjLo = easeLo - stretchCm - 2; // 2 cm slack for the waistband itself
    const adjHi = easeHi + stretchCm;
    const { status, message } = evaluateEase(
      ease, adjLo, adjHi, 'Waist', 'cm',
      { tightMsg: 'Waist will be too tight — sit-down comfort is at risk.' },
    );
    zones.push({ zone: 'waist', status, deltaCm: round1(ease), message });
  }

  // ── Hips ───────────────────────────────────────────────────────────────
  if (bodyM.hips && gM.hips) {
    const ease = gM.hips.valueCm - bodyM.hips.valueCm;
    const { status, message } = evaluateEase(
      ease, easeLo - stretchCm, easeHi + stretchCm, 'Hips', 'cm',
      { tightMsg: 'Hips will be too tight — sit-down and walking will be restricted.' },
    );
    zones.push({ zone: 'hips', status, deltaCm: round1(ease), message });
  }

  // ── Thigh ──────────────────────────────────────────────────────────────
  if (bodyM.thigh && gM.thigh) {
    const ease = gM.thigh.valueCm - bodyM.thigh.valueCm;
    const { status, message } = evaluateEase(
      ease, 1 - stretchCm, 6 + stretchCm, 'Thigh', 'cm',
      { tightMsg: 'Thigh will be too tight — slim/athletic cut needed.' },
    );
    zones.push({ zone: 'thigh', status, deltaCm: round1(ease), message });
  }

  // ── Inseam ─────────────────────────────────────────────────────────────
  if (bodyM.inseam && gM.inseam) {
    const delta = gM.inseam.valueCm - bodyM.inseam.valueCm;
    let status: ZoneStatus = 'good';
    let message = 'Inseam length matches your leg.';
    if (delta < -4) {
      status = 'too_short';
      message = 'Pants will be too short — high-water look.';
    } else if (delta < -2) {
      status = 'too_short';
      message = 'Pants will sit above the ankle.';
    } else if (delta > 6) {
      status = 'too_long';
      message = 'Pants will be too long — expect to hem or stack.';
    } else if (delta > 4) {
      status = 'too_long';
      message = 'Pants will bunch at the ankle.';
    }
    zones.push({ zone: 'inseam', status, deltaCm: round1(delta), message });
  }

  return {
    zones,
    overall: rollupOverall(zones, fitIntent),
    confidence: confidenceFromZones(zones, ['waist', 'hips', 'thigh', 'inseam']),
  };
}

function assessShoes(
  bodyM: NonNullable<BodyProfile['measurements']>,
  gM: NonNullable<GarmentPhysicalProfile['measurements']>,
  stretchCm: number,
): { zones: FitZoneAssessment[]; overall: OverallFit; confidence: Confidence } {
  const zones: FitZoneAssessment[] = [];

  if (bodyM.footLength && gM.shoeLength) {
    const delta = gM.shoeLength.valueCm - bodyM.footLength.valueCm;
    let status: ZoneStatus = 'good';
    let message = 'Shoe length matches your foot.';
    // Plan rule: foot length if available.
    // Standard fit: ~0.5-1.5 cm of toe room.
    if (delta < 0) {
      status = 'too_tight';
      message = 'Shoes are smaller than your foot — too tight.';
    } else if (delta < 0.3) {
      status = 'snug';
      message = 'Shoes will be very snug — almost no toe room.';
    } else if (delta < 0.6) {
      status = 'snug';
      message = 'Shoes will be snug.';
    } else if (delta > 2.5 + stretchCm) {
      status = 'too_loose';
      message = 'Shoes will slip at the heel — too big.';
    } else if (delta > 1.8 + stretchCm) {
      status = 'loose';
      message = 'Shoes will be slightly loose.';
    }
    zones.push({ zone: 'feet', status, deltaCm: round1(delta), message });
  }

  return {
    zones,
    overall: rollupOverall(zones, 'regular'),
    confidence: zones.length > 0 ? 'medium' : 'low',
  };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function evaluateEase(
  ease: number,
  lo: number,
  hi: number,
  label: string,
  unit: string,
  overrides: { tightMsg?: string } = {},
): { status: ZoneStatus; message: string } {
  if (ease < lo - 2) {
    return {
      status: 'too_tight',
      message: overrides.tightMsg ?? `${label} is too tight (${round1(ease)} ${unit} ease).`,
    };
  }
  if (ease < lo) {
    return { status: 'snug', message: `${label} is snug (${round1(ease)} ${unit} ease).` };
  }
  if (ease > hi + 2) {
    return { status: 'too_loose', message: `${label} is too loose (${round1(ease)} ${unit} ease).` };
  }
  if (ease > hi) {
    return { status: 'loose', message: `${label} is relaxed (${round1(ease)} ${unit} ease).` };
  }
  return { status: 'good', message: `${label} has good ease (${round1(ease)} ${unit}).` };
}

/**
 * Roll up per-zone statuses into a single overall verdict, biased by the
 * fit intent (e.g. a 'slim' garment is *meant* to be snug, so a snug chest
 * is still 'good_fit'; a 'relaxed' garment is *meant* to be loose).
 */
function rollupOverall(zones: FitZoneAssessment[], fitIntent: FitIntent): OverallFit {
  if (zones.length === 0) return 'unknown';

  // Weight: too_tight / too_loose / too_short / too_long are decisive.
  const hasTight = zones.some((z) => z.status === 'too_tight' || z.status === 'too_short');
  const hasLoose = zones.some((z) => z.status === 'too_loose' || z.status === 'too_long');
  const hasSnug = zones.some((z) => z.status === 'snug');
  const hasRelaxed = zones.some((z) => z.status === 'loose');

  // Fit-intent aware adjustments: a snug 'slim' garment is on-target.
  const isSlimIntent = fitIntent === 'slim' || fitIntent === 'compression';
  const isLooseIntent = fitIntent === 'relaxed' || fitIntent === 'oversized';

  // Hard floor: if a zone is so far too-tight that no human can wear it,
  // the garment is too_small regardless of intent. (A 'slim' shirt that
  // would crush you is not "snug" — it's the wrong size.)
  const hasSevereTight = zones.some((z) =>
    (z.status === 'too_tight' || z.status === 'too_short') &&
    z.deltaCm != null && z.deltaCm < -4,
  );
  // For loose-intent garments (relaxed/oversized), a large ease is on-target,
  // so we don't short-circuit to 'too_large' — the zone status is still 'good'
  // and the per-zone band check already handled the up-roll.
  const hasSevereLoose = !isLooseIntent && zones.some((z) =>
    (z.status === 'too_loose' || z.status === 'too_long') &&
    z.deltaCm != null && z.deltaCm > 8,
  );

  if (hasSevereTight) return 'too_small';
  if (hasSevereLoose) return 'too_large';
  if (hasTight && !isSlimIntent) return 'too_small';
  if (hasLoose && !isLooseIntent) return 'too_large';
  if (hasSnug && isSlimIntent) return 'good_fit';
  if (hasRelaxed && isLooseIntent) return 'good_fit';
  if (hasSnug) return 'tight';
  if (hasRelaxed) return 'relaxed';
  if (hasTight && isSlimIntent) return 'tight'; // still too tight, just framed honestly
  if (hasLoose && isLooseIntent) return 'oversized';
  return 'good_fit';
}

function confidenceFromZones(
  zones: FitZoneAssessment[],
  expected: FitZone[],
): Confidence {
  const coverage = expected.filter((z) =>
    zones.some((zone) => zone.zone === z),
  ).length;
  const ratio = coverage / expected.length;
  if (ratio >= 0.75) return 'high';
  if (ratio >= 0.4) return 'medium';
  return 'low';
}

function buildUnknown(
  body: BodyProfile,
  garment: GarmentPhysicalProfile,
  generatedAt: string,
  reasons: string[],
): FitAssessment {
  return {
    garmentId: garment.garmentId,
    bodyProfileId: body.id,
    selectedSize: garment.sizeLabel,
    overall: 'unknown',
    confidence: 'low',
    zones: reasons.map((reason) => ({
      zone: 'chest',
      status: 'unknown',
      message: reason,
    })),
    engineVersion: FIT_ENGINE_VERSION,
    generatedAt,
  };
}

function round1(n: number): number {
  return Math.round(n * 10) / 10;
}

// ─── Size recommendation ────────────────────────────────────────────────────

/**
 * Compare the current assessment against neighboring sizes (in the same
 * garment family) and recommend one if it scores better.
 *
 * Caller must provide a way to fetch the next-larger / next-smaller
 * GarmentPhysicalProfile. Returns undefined if the current size is best
 * or if we don't have alternatives to compare.
 */
export function recommendSize(
  current: FitAssessment,
  currentProfile: GarmentPhysicalProfile,
  alternatives: GarmentPhysicalProfile[],
  body: BodyProfile,
): FitAssessment['sizeRecommendation'] {
  if (!currentProfile || alternatives.length === 0) return undefined;

  // Score = number of zones in 'good' (higher better); -3 for any too_tight/too_short.
  const scoreOf = (assessment: FitAssessment) => {
    if (assessment.overall === 'too_small' || assessment.overall === 'too_large') return -10;
    let score = 0;
    for (const z of assessment.zones) {
      if (z.status === 'good') score += 1;
      if (z.status === 'snug' || z.status === 'loose') score += 0.3;
      if (z.status === 'too_tight' || z.status === 'too_loose'
          || z.status === 'too_short' || z.status === 'too_long') score -= 3;
    }
    return score;
  };

  const currentScore = scoreOf(current);
  let best = { size: currentProfile.sizeLabel, score: currentScore };
  for (const alt of alternatives) {
    const a = assessFit(body, alt);
    const s = scoreOf(a);
    if (s > best.score) best = { size: alt.sizeLabel, score: s };
  }

  if (best.size === currentProfile.sizeLabel) return undefined;
  return {
    recommendedSize: best.size,
    reason: `Size ${best.size} matches your measurements better across more zones.`,
  };
}

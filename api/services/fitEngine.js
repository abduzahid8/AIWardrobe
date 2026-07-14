/**
 * Server-side copy of the fit engine (api/services/fitEngine.js).
 *
 * The mobile side imports the TypeScript implementation from
 * src/lib/fit/fitEngine.ts. The Express process is plain ESM JS so we
 * keep a parallel JS implementation here with the same shape and rules.
 *
 * KEEP THESE IN SYNC. The golden tests on the mobile side lock the TS
 * behaviour; the server copy is verified manually + by integration tests.
 *
 * If the engines drift, the fix is to either:
 *   (a) Move this file to a shared package and import it from both sides, or
 *   (b) Generate it from the TS source via tsc/tsx at build time.
 *
 * Option (a) is the long-term plan. For now, a faithful port is fine.
 */

import {
  Confidence,
  FitAssessment,
  FitZone,
  FitZoneAssessment,
  OverallFit,
  ZoneStatus,
} from '../types-fit.js';

const FIT_ENGINE_VERSION = 'fit-engine/v1';

const CHEST_EASE_TARGETS = {
  compression: [0, 2],
  slim: [2, 5],
  regular: [5, 10],
  relaxed: [10, 16],
  oversized: [16, 30],
};

const HIP_EASE_TARGETS = {
  compression: [0, 2],
  slim: [1, 4],
  regular: [4, 8],
  relaxed: [8, 14],
  oversized: [14, 25],
};

const STRETCH_TOLERANCE_CM = { none: 0, low: 1, medium: 2, high: 4 };

export function assessFit(body, garment) {
  const generatedAt = new Date().toISOString();
  const measurements = body.measurements || {};
  const heightCm = body.height?.valueCm;
  if (!heightCm || !body.bodyType) {
    return buildUnknown(body, garment, generatedAt, ['height or body type missing']);
  }
  const gM = garment.measurements || {};
  const fitIntent = garment.fitIntent || 'regular';
  const stretch = garment.stretch || 'none';
  const stretchCm = STRETCH_TOLERANCE_CM[stretch];

  let zones = [];
  let overall = 'unknown';
  let confidence = 'medium';

  switch (garment.category) {
    case 'top':
    case 'shirt':
    case 'dress':
      ({ zones, overall, confidence } = assessUpper(measurements, gM, fitIntent, stretchCm));
      break;
    case 'jacket':
    case 'coat':
      ({ zones, overall, confidence } = assessJacket(measurements, gM, fitIntent, stretchCm));
      break;
    case 'pants':
    case 'jeans':
    case 'skirt':
      ({ zones, overall, confidence } = assessLower(measurements, gM, fitIntent, stretchCm));
      break;
    case 'shoes':
      ({ zones, overall, confidence } = assessShoes(measurements, gM, stretchCm));
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

function assessUpper(bodyM, gM, fitIntent, stretchCm) {
  const zones = [];
  const [easeLo, easeHi] = CHEST_EASE_TARGETS[fitIntent];

  if (bodyM.chest && gM.chest) {
    const ease = gM.chest.valueCm - bodyM.chest.valueCm;
    const r = evaluateEase(ease, easeLo - stretchCm, easeHi + stretchCm, 'Chest', 'cm');
    zones.push({ zone: 'chest', status: r.status, deltaCm: round1(ease), message: r.message });
  }
  if (bodyM.shoulderWidth && gM.shoulderWidth) {
    const delta = gM.shoulderWidth.valueCm - bodyM.shoulderWidth.valueCm;
    let status = 'good', message = 'Shoulder line matches.';
    if (delta < -3) { status = 'too_tight'; message = 'Shoulders will feel restrictive.'; }
    else if (delta < -1) { status = 'snug'; message = 'Shoulders will be snug.'; }
    else if (delta > 4) { status = 'too_loose'; message = 'Shoulders will drop off the frame.'; }
    else if (delta > 2) { status = 'loose'; message = 'Shoulder line will be slightly dropped.'; }
    zones.push({ zone: 'shoulders', status, deltaCm: round1(delta), message });
  }
  if (bodyM.armLength && gM.sleeveLength) {
    const delta = gM.sleeveLength.valueCm - bodyM.armLength.valueCm;
    let status = 'good', message = 'Sleeve length matches your arm.';
    if (delta < -3) { status = 'too_short'; message = 'Sleeves will be too short.'; }
    else if (delta < -1) { status = 'too_short'; message = 'Sleeves will be slightly short.'; }
    else if (delta > 5) { status = 'too_long'; message = 'Sleeves will be noticeably long.'; }
    else if (delta > 3) { status = 'too_long'; message = 'Sleeves will be slightly long.'; }
    zones.push({ zone: 'sleeves', status, deltaCm: round1(delta), message });
  }
  if (bodyM.torsoLength && gM.bodyLength) {
    const delta = gM.bodyLength.valueCm - bodyM.torsoLength.valueCm;
    let status = 'good', message = 'Body length is balanced.';
    if (delta < -4) { status = 'too_short'; message = 'Hem will sit high — top will feel cropped.'; }
    else if (delta > 6) { status = 'too_long'; message = 'Hem will sit low — top will feel long.'; }
    zones.push({ zone: 'torso_length', status, deltaCm: round1(delta), message });
  }

  return { zones, overall: rollupOverall(zones, fitIntent), confidence: confidenceFromZones(zones, ['chest', 'shoulders', 'sleeves', 'torso_length']) };
}

function assessJacket(bodyM, gM, fitIntent, stretchCm) {
  const layeringEase = 4;
  const adjusted = { ...gM, chest: gM.chest ? { ...gM.chest, valueCm: gM.chest.valueCm - layeringEase } : gM.chest };
  const result = assessUpper(bodyM, adjusted, fitIntent, stretchCm);
  result.zones.unshift({ zone: 'chest', status: 'good', message: 'Includes ~4 cm layering ease for wearing over a top.' });
  return result;
}

function assessLower(bodyM, gM, fitIntent, stretchCm) {
  const zones = [];
  const [easeLo, easeHi] = HIP_EASE_TARGETS[fitIntent];

  if (bodyM.waist && gM.waist) {
    const ease = gM.waist.valueCm - bodyM.waist.valueCm;
    const r = evaluateEase(ease, easeLo - stretchCm - 2, easeHi + stretchCm, 'Waist', 'cm', { tightMsg: 'Waist will be too tight.' });
    zones.push({ zone: 'waist', status: r.status, deltaCm: round1(ease), message: r.message });
  }
  if (bodyM.hips && gM.hips) {
    const ease = gM.hips.valueCm - bodyM.hips.valueCm;
    const r = evaluateEase(ease, easeLo - stretchCm, easeHi + stretchCm, 'Hips', 'cm', { tightMsg: 'Hips will be too tight.' });
    zones.push({ zone: 'hips', status: r.status, deltaCm: round1(ease), message: r.message });
  }
  if (bodyM.thigh && gM.thigh) {
    const ease = gM.thigh.valueCm - bodyM.thigh.valueCm;
    const r = evaluateEase(ease, 1 - stretchCm, 6 + stretchCm, 'Thigh', 'cm', { tightMsg: 'Thigh will be too tight.' });
    zones.push({ zone: 'thigh', status: r.status, deltaCm: round1(ease), message: r.message });
  }
  if (bodyM.inseam && gM.inseam) {
    const delta = gM.inseam.valueCm - bodyM.inseam.valueCm;
    let status = 'good', message = 'Inseam length matches your leg.';
    if (delta < -4) { status = 'too_short'; message = 'Pants will be too short — high-water look.'; }
    else if (delta < -2) { status = 'too_short'; message = 'Pants will sit above the ankle.'; }
    else if (delta > 6) { status = 'too_long'; message = 'Pants will be too long — expect to hem or stack.'; }
    else if (delta > 4) { status = 'too_long'; message = 'Pants will bunch at the ankle.'; }
    zones.push({ zone: 'inseam', status, deltaCm: round1(delta), message });
  }

  return { zones, overall: rollupOverall(zones, fitIntent), confidence: confidenceFromZones(zones, ['waist', 'hips', 'thigh', 'inseam']) };
}

function assessShoes(bodyM, gM, stretchCm) {
  const zones = [];
  if (bodyM.footLength && gM.shoeLength) {
    const delta = gM.shoeLength.valueCm - bodyM.footLength.valueCm;
    let status = 'good', message = 'Shoe length matches your foot.';
    if (delta < 0) { status = 'too_tight'; message = 'Shoes are smaller than your foot — too tight.'; }
    else if (delta < 0.3) { status = 'snug'; message = 'Shoes will be very snug.'; }
    else if (delta < 0.6) { status = 'snug'; message = 'Shoes will be snug.'; }
    else if (delta > 2.5 + stretchCm) { status = 'too_loose'; message = 'Shoes will slip at the heel.'; }
    else if (delta > 1.8 + stretchCm) { status = 'loose'; message = 'Shoes will be slightly loose.'; }
    zones.push({ zone: 'feet', status, deltaCm: round1(delta), message });
  }
  return { zones, overall: rollupOverall(zones, 'regular'), confidence: zones.length > 0 ? 'medium' : 'low' };
}

function evaluateEase(ease, lo, hi, label, unit, overrides = {}) {
  if (ease < lo - 2) {
    return { status: 'too_tight', message: overrides.tightMsg || `${label} is too tight (${round1(ease)} ${unit} ease).` };
  }
  if (ease < lo) return { status: 'snug', message: `${label} is snug (${round1(ease)} ${unit} ease).` };
  if (ease > hi + 2) return { status: 'too_loose', message: `${label} is too loose (${round1(ease)} ${unit} ease).` };
  if (ease > hi) return { status: 'loose', message: `${label} is relaxed (${round1(ease)} ${unit} ease).` };
  return { status: 'good', message: `${label} has good ease (${round1(ease)} ${unit}).` };
}

function rollupOverall(zones, fitIntent) {
  if (zones.length === 0) return 'unknown';
  const hasTight = zones.some((z) => z.status === 'too_tight' || z.status === 'too_short');
  const hasLoose = zones.some((z) => z.status === 'too_loose' || z.status === 'too_long');
  const hasSnug = zones.some((z) => z.status === 'snug');
  const hasRelaxed = zones.some((z) => z.status === 'loose');
  const isSlimIntent = fitIntent === 'slim' || fitIntent === 'compression';
  const isLooseIntent = fitIntent === 'relaxed' || fitIntent === 'oversized';

  if (hasTight && !isSlimIntent) return 'too_small';
  if (hasLoose && !isLooseIntent) return 'too_large';
  if (hasSnug && isSlimIntent) return 'good_fit';
  if (hasRelaxed && isLooseIntent) return 'good_fit';
  if (hasSnug) return 'tight';
  if (hasRelaxed) return 'relaxed';
  if (hasTight && isSlimIntent) return 'tight';
  if (hasLoose && isLooseIntent) return 'oversized';
  return 'good_fit';
}

function confidenceFromZones(zones, expected) {
  const coverage = expected.filter((z) => zones.some((zone) => zone.zone === z)).length;
  const ratio = coverage / expected.length;
  if (ratio >= 0.75) return 'high';
  if (ratio >= 0.4) return 'medium';
  return 'low';
}

function buildUnknown(body, garment, generatedAt, reasons) {
  return {
    garmentId: garment.garmentId,
    bodyProfileId: body.id,
    selectedSize: garment.sizeLabel,
    overall: 'unknown',
    confidence: 'low',
    zones: reasons.map((r) => ({ zone: 'chest', status: 'unknown', message: r })),
    engineVersion: FIT_ENGINE_VERSION,
    generatedAt,
  };
}

function round1(n) { return Math.round(n * 10) / 10; }

export function recommendSize(current, currentProfile, alternatives, body) {
  if (!currentProfile || !alternatives || alternatives.length === 0) return undefined;
  const scoreOf = (a) => {
    if (a.overall === 'too_small' || a.overall === 'too_large') return -10;
    let s = 0;
    for (const z of a.zones) {
      if (z.status === 'good') s += 1;
      if (z.status === 'snug' || z.status === 'loose') s += 0.3;
      if (z.status === 'too_tight' || z.status === 'too_loose' || z.status === 'too_short' || z.status === 'too_long') s -= 3;
    }
    return s;
  };
  const currentScore = scoreOf(current);
  let best = { size: currentProfile.sizeLabel, score: currentScore };
  for (const alt of alternatives) {
    const a = assessFit(body, alt);
    const s = scoreOf(a);
    if (s > best.score) best = { size: alt.sizeLabel, score: s };
  }
  if (best.size === currentProfile.sizeLabel) return undefined;
  return { recommendedSize: best.size, reason: `Size ${best.size} matches your measurements better across more zones.` };
}

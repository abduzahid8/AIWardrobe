/**
 * Stylist 6 — CPU-only, explainable whole-look editorial checkpoint.
 *
 * This is a bounded second reading of the existing seven-dimensional
 * outfitCoherence evidence, NOT an image critic or a human fashion verdict.
 * It never guesses body type, weather, material or unseen colors, cannot
 * waive outfit admission/budget/user exclusions, and makes no remote calls.
 *
 * Emphasize the intended occasion and the weakest styling relationship
 * before using novelty to select a third look.
 */
export const EDITORIAL_QUALITY_VERSION = 's6-editorial-quality-v1';

const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
const isKnown = (x) => typeof x === 'number' && Number.isFinite(x) && x >= 0 && x <= 1;

export function evaluateEditorialQuality({ dimensions, descriptors = [], context = {}, hasLayer = false } = {}) {
  if (!dimensions || typeof dimensions !== 'object') {
    return { score: null, adjustment: 0, evidence: [], version: EDITORIAL_QUALITY_VERSION };
  }
  // Structural relationships, rather than the price, popularity, novelty or
  // number of purchased pieces, define this editorial readiness estimate.
  const weights = {
    dressCode: 0.23,
    style: 0.21,
    footwear: 0.17,
    proportion: 0.15,
    palette: 0.13,
    texture: 0.11,
    ...(hasLayer ? { layer: 0.13 } : {}),
  };
  const entries = Object.entries(weights).filter(([key]) => isKnown(dimensions[key]));
  if (!entries.length) {
    return { score: null, adjustment: 0, evidence: [], version: EDITORIAL_QUALITY_VERSION };
  }
  const denom = entries.reduce((sum, [, weight]) => sum + weight, 0);
  const mean = entries.reduce((sum, [key, weight]) => sum + dimensions[key] * weight, 0) / denom;
  const weakest = ['dressCode', 'style', 'footwear', 'proportion']
    .filter(key => isKnown(dimensions[key]))
    .map(key => dimensions[key]);
  const weak = weakest.length ? Math.min(...weakest) : null;
  const evidence = [];
  let deduction = 0;
  if (weak !== null && weak < 0.60) {
    deduction += Math.min(0.11, (0.60 - weak) * 0.30);
    evidence.push('weak_occasion_style_or_silhouette_relationship');
  }

  // Pattern/graphic prominence and eye-catching shoes are not bad in all
  // aesthetics. For QUIET looks, their combination is a soft caution, not a
  // blanket ban — explicit favorites remain legitimate.
  const quiet = ['minimalist', 'old_money', 'classic', 'semi_classic'].includes(context.style);
  if (quiet && Array.isArray(descriptors)) {
    const accent = descriptors.find(d => d?.slot === 'shoes' && d?.vividFootwear === true);
    const favoriteColors = Array.isArray(context.preferences?.favoriteColors)
      ? context.preferences.favoriteColors.map(x => String(x).toLowerCase().trim()) : [];
    const favoriteAccent = accent && typeof accent.color === 'string' &&
      favoriteColors.includes(accent.color.toLowerCase().trim());
    if (accent && !favoriteAccent) {
      deduction += context.style === 'minimalist' ? 0.075 : 0.045;
      evidence.push('unrequested_statement_footwear_in_quiet_style');
    }
    const loudPieces = descriptors.filter(d => d && (d.loudBranding === true ||
      (d.patternProminence === 'prominent' && ['print', 'graphic'].includes(d.pattern))));
    if (loudPieces.length >= 2) {
      deduction += 0.055;
      evidence.push('multiple_statement_pieces_in_quiet_style');
    }
  }

  const score = clamp(mean - deduction);
  // Never allow this heuristic to dominate pre-existing whole-look safety,
  // personalization, hard budget, requested dress-code or visual evidence.
  const adjustment = Math.max(-0.08, Math.min(0.04, (score - 0.78) * 0.28));
  return {
    score: Number(score.toFixed(4)),
    adjustment: Number(adjustment.toFixed(4)),
    evidence,
    version: EDITORIAL_QUALITY_VERSION,
  };
}

function scoreOf(look) {
  const score = look?.fashion?.editorial?.score;
  return isKnown(score) ? score : null;
}

/**
 * When both quality estimates are available, novelty cannot license a
 * substantially worse outfit. Unknown-only synthetic/legacy fixtures keep
 * the existing behavior; a known signature never treats missing candidate
 * evidence as proof of editorial quality.
 */
export function editorialDirectionEligible(signature, candidate, mode = 'discovery') {
  const baseline = scoreOf(signature), proposed = scoreOf(candidate);
  if (baseline === null) return proposed === null;
  if (proposed === null) return false;
  const floor = mode === 'discovery' ? 0.71 : mode === 'refresh' ? 0.66 : 0.62;
  const maxDrop = mode === 'discovery' ? 0.055 : mode === 'refresh' ? 0.09 : 0.12;
  return proposed >= floor && proposed >= baseline - maxDrop;
}

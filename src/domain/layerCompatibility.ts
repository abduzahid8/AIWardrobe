import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
import type { StylingContext } from './stylingContext.ts';
export interface LayerCompatibility {
  valid: boolean;
  score: number;
  reasons: string[];
  evidence: string[];
  confidence: 'low' | 'medium' | 'high';
}
/** Directional: the second garment must plausibly fit OVER the first. */
export function layerCompatibility(
  top: GarmentEvidence,
  layer: GarmentEvidence,
  context: StylingContext
): LayerCompatibility {
  const a = describeGarment(top),
    b = describeGarment(layer);
  const reasons: string[] = [],
    evidence: string[] = [];
  let score = 0.7;
  if (a.bulk === 'heavy' && b.bulk === 'heavy') reasons.push('competing_heavy_layers');
  if (a.volume === 'wide' && b.volume === 'slim') reasons.push('wide_base_under_fitted_layer');
  if (context.dressCode === 'formal' && b.subtype !== 'blazer')
    reasons.push('formal_requires_structured_layer');
  if (context.dressCode === 'formal' && !['shirt', 'dress-shirt'].includes(a.subtype))
    reasons.push('formal_requires_dress_base');
  if (context.dressCode === 'athletic' && b.structured)
    reasons.push('structured_layer_restricts_workout');
  if (a.bulk === 'light' && b.closure === 'openable') {
    score += 0.15;
    evidence.push('light_base_under_openable_layer');
  }
  if (
    ['shirt', 'dress-shirt', 't-shirt', 'polo'].includes(a.subtype) &&
    ['blazer', 'cardigan', 'overshirt', 'jacket', 'coat', 'bomber'].includes(b.subtype)
  ) {
    score += 0.1;
    evidence.push('base_and_outer_layer_roles');
  }
  if (a.length === 'long' && b.length === 'cropped') {
    score -= 0.15;
    evidence.push('long_base_below_cropped_layer');
  }
  if (a.collar === 'shirt' && b.collar === 'high') {
    score -= 0.2;
    evidence.push('competing_necklines');
  }
  if (a.bulk === 'heavy' && b.subtype === 'pullover') {
    score -= 0.3;
    evidence.push('bulky_base_under_pullover');
  }
  if (
    (context.weather === null || (context.weather.temp != null && context.weather.temp > 22)) &&
    b.bulk === 'heavy'
  ) {
    score -= 0.15;
    evidence.push('heavy_layer_without_cold_need');
  }
  return {
    valid: reasons.length === 0,
    score: reasons.length ? 0 : Math.round(Math.max(0, Math.min(1, score)) * 1000) / 1000,
    reasons,
    evidence,
    confidence:
      a.subtype === 'unknown' || b.subtype === 'unknown'
        ? 'low'
        : a.bulk === 'unknown' && b.bulk === 'unknown'
          ? 'medium'
          : 'high',
  };
}

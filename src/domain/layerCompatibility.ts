import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
import { normalizeGarmentText } from './garmentText.ts';
import type { StylingContext } from './stylingContext.ts';
export interface LayerCompatibility {
  valid: boolean;
  score: number;
  reasons: string[];
  evidence: string[];
  confidence: 'low' | 'medium' | 'high';
}
const pairMemo = new WeakMap<StylingContext, WeakMap<object, WeakMap<object, LayerCompatibility>>>();
export function layerCompatibility(top: GarmentEvidence, layer: GarmentEvidence, context: StylingContext): LayerCompatibility {
  if (!Object.isFrozen(top) || !Object.isFrozen(layer)) return evaluateLayer(top,layer,context);
  let tops=pairMemo.get(context);
  if(!tops){tops=new WeakMap();pairMemo.set(context,tops);}
  let layers=tops.get(top);
  if(!layers){layers=new WeakMap();tops.set(top,layers);}
  let result=layers.get(layer);
  if(!result){result=evaluateLayer(top,layer,context);layers.set(layer,result);}
  return result;
}
/** Directional: the second garment must plausibly fit OVER the first. */
function evaluateLayer(
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
  // A tailored layer cannot physically go over gym-wear or a hood: the
  // shoulders collapse and the hood lumps at the neckline.
  const topText = normalizeGarmentText(
    [top.name, top.type, top.description].filter(Boolean).join(' ')
  );
  if (b.structured && a.athletic) reasons.push('tailored_layer_over_athletic_base');
  if (b.subtype === 'blazer' && /\bhood/.test(topText))
    reasons.push('hood_under_tailored_blazer');
  // A bulky knit under a tailored blazer reads stuffed unless the base is
  // provably slim. Unknown thickness is NOT proof of incompatibility: it
  // only soft-penalizes, and provably wide non-slim bases hard-fail.
  if (
    b.subtype === 'blazer' &&
    a.volume !== 'slim' &&
    (a.bulk === 'heavy' || (a.subtype === 'sweater' && a.volume === 'wide'))
  )
    reasons.push('bulky_base_under_tailored_blazer');
  if (
    b.subtype === 'blazer' &&
    a.subtype === 'sweater' &&
    a.bulk === 'unknown' &&
    a.volume === 'unknown'
  ) {
    score -= 0.1;
    evidence.push('unverified_knit_bulk_under_blazer');
  }
  // A formal base with sporty outerwear is a register clash even outside
  // strict formal dress codes.
  if (a.formality === 'formal' && b.athletic)
    reasons.push('athletic_outerwear_over_formal_base');
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
  // Fine knitwear is an explicitly legitimate blazer base — differentiated
  // from bulky knits by the bulk evidence above, never by stereotype.
  if (a.subtype === 'fine-knit' && b.subtype === 'blazer') {
    score += 0.1;
    evidence.push('fine_knit_under_blazer');
  }
  if (a.bulk === 'heavy' && b.bulk === 'light' && !b.structured) {
    score -= 0.1;
    evidence.push('heavy_base_under_light_unstructured_layer');
  }
  if (a.length === 'long' && b.length === 'cropped') {
    score -= 0.15;
    evidence.push('long_base_below_cropped_layer');
  }
  if (a.collar === 'shirt' && b.collar === 'high') {
    score -= 0.2;
    evidence.push('competing_necklines');
  }
  // The base may fit physically while its drape/volume weakens the jacket's
  // shoulder line. Keep a plain regular tee with soft tailoring available.
  if (b.subtype === 'blazer' && !b.relaxedTailoring && a.volume === 'wide'
    && (a.loudBranding || a.fluidFabric)) {
    score -= 0.25;
    evidence.push('expressive_wide_base_under_structured_tailoring');
  }
  if (a.bulk === 'heavy' && b.subtype === 'pullover') {
    score -= 0.3;
    evidence.push('bulky_base_under_pullover');
  }
  if (
    context.weather?.temp != null && context.weather.temp > 22 &&
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

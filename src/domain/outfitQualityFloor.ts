import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
import type { OutfitCoherence } from './outfitCoherence.ts';
import type { StylingContext } from './stylingContext.ts';

export interface OutfitQualityFloor {
  passed: boolean;
  score: number;
  minimum: number;
  reason: string | null;
}
/** Acceptance is separate from hard eligibility and recency ranking. Unknown
 * metadata stays neutral; a low result never licenses a hard-rule waiver. */
export function outfitQualityFloor(
  items: GarmentEvidence[],
  context: StylingContext,
  coherence: OutfitCoherence
): OutfitQualityFloor {
  const refinedWork =
    context.dressCode === 'smart' &&
    /\b(work|office|business)\b/.test(context.occasion) &&
    ['old_money', 'classic', 'semi_classic', 'business_casual'].includes(context.style);
  const formal = context.dressCode === 'formal';
  const minimalistCasual = context.style === 'minimalist' && context.dressCode === 'casual';
  const minimum = formal || refinedWork ? 0.75 : minimalistCasual ? 0.65 : 0;
  let score = coherence.score;
  let reason: string | null = null;
  if (
    (formal || refinedWork) &&
    Math.min(
      coherence.dimensions.layer,
      coherence.dimensions.footwear,
      coherence.dimensions.style
    ) < 0.5
  ) {
    score = Math.min(score, 0.49);
    reason = 'weak_refined_occasion_compatibility';
  }
  if (refinedWork || formal) {
    const top = items.find(item => item.macroCategory === 'top');
    const layer = items.find(item => item.macroCategory === 'outerwear');
    const bottom = items.find(item => item.macroCategory === 'bottom');
    const a = top ? describeGarment(top) : undefined, b = layer ? describeGarment(layer) : undefined;
    const trousers = bottom ? describeGarment(bottom) : undefined;
    if (a?.utilityConstruction && b?.subtype === 'blazer') {
      score = Math.min(score, 0.49); reason = 'utility_shirt_conflicts_with_office_tailoring';
    } else if (b?.technicalLeisure && trousers?.structured) {
      score = Math.min(score, 0.49); reason = 'technical_leisure_conflicts_with_refined_tailoring';
    } else if ((layer && bottom && /\b(?:pure|100%)\s+linen\b/i.test(layer.name || '') && /\bflannel\b/i.test(`${bottom.name || ''} ${bottom.description || ''}`))
      || (top && layer && /\blinen\b/i.test(top.name || '') && /\bvelvet\b/i.test(layer.name || ''))
      || coherence.evidence.includes('heavy_and_airy_texture_contrast')) {
      score = Math.min(score, 0.49); reason = 'incompatible_heavy_and_airy_fabrics';
    } else if (a && b && ['stripe','check','print'].includes(a.pattern)
      && ['stripe','check','print'].includes(b.pattern) && a.pattern !== b.pattern) {
      score = Math.min(score, 0.49); reason = 'competing_patterns_in_office_tailoring';
    }
  }
  if (minimalistCasual) {
    const layer = items.find((item) => item.macroCategory === 'outerwear');
    const bottom = items.find((item) => item.macroCategory === 'bottom');
    const text = `${bottom?.name || ''} ${bottom?.description || ''}`.toLowerCase();
    const layerText = `${layer?.name || ''} ${layer?.description || ''}`.toLowerCase();
    const severeDistress =
      /\b(ripped|torn|destroyed|shredded|holes?)\b|\b(heavily|heavy|overly|extremely|severely)[ -]+distressed\b/.test(
        text
      );
    if (
      layer &&
      bottom &&
      describeGarment(layer).subtype === 'blazer' &&
      describeGarment(bottom).subtype === 'jeans' &&
      severeDistress &&
      !/\b(unstructured|casual|unlined|jersey|knitted)\b/.test(layerText)
    ) {
      score = Math.min(score, 0.49);
      reason = 'business_tailoring_with_severely_distressed_denim';
    }
  }
  const passed = coherence.valid && score >= minimum;
  return {
    passed,
    score,
    minimum,
    reason: passed ? null : reason || 'below_occasion_style_quality_floor',
  };
}

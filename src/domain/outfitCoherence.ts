import { garmentContextReason, scoreGarmentForContext } from './garmentContext.ts';
import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
import { layerCompatibility } from './layerCompatibility.ts';
import type { StylingContext } from './stylingContext.ts';
export interface OutfitCoherence {
  valid: boolean;
  score: number;
  reasons: string[];
  evidence: string[];
  dimensions: {
    layer: number;
    dressCode: number;
    palette: number;
    proportion: number;
    texture: number;
    footwear: number;
    style: number;
  };
}
/** Scores relationships; hard authority (source/budget/ownership) stays with existing gates. */
export function outfitCoherence(
  items: GarmentEvidence[],
  context: StylingContext
): OutfitCoherence {
  const descriptors = items.map(describeGarment),
    bySlot = (s: string) => descriptors.find((d) => d.slot === s);
  const top = items.find((i) => i.macroCategory === 'top'),
    layer = items.find((i) => i.macroCategory === 'outerwear');
  const reasons: string[] = [],
    evidence: string[] = [];
  const dimensions = {
    layer: 0.7,
    dressCode: 0.7,
    palette: 0.7,
    proportion: 0.7,
    texture: 0.7,
    footwear: 0.7,
    style: 0.7,
  };
  if (top && layer) {
    const pair = layerCompatibility(top, layer, context);
    dimensions.layer = pair.score;
    reasons.push(...pair.reasons);
    evidence.push(...pair.evidence);
  }
  const bottom = bySlot('bottom'),
    shoe = bySlot('shoes');
  if (
    bySlot('outerwear')?.insulated &&
    shoe &&
    ['espadrille', 'sandal', 'driver'].includes(shoe.subtype)
  )
    reasons.push('insulated_layer_with_summer_footwear');
  for (const item of items) {
    const reason = garmentContextReason(item, context);
    if (reason) reasons.push(reason);
  }
  if (context.dressCode === 'formal')
    dimensions.dressCode = reasons.some((r) => r.startsWith('formal_')) ? 0 : 1;
  if (context.dressCode !== 'formal') {
    dimensions.dressCode =
      items.reduce((sum, item) => sum + scoreGarmentForContext(item, context), 0) /
      Math.max(1, items.length);
    if (
      descriptors.some((d) => d.evening) &&
      !/black[ -]?tie|tuxedo|gala|смокинг/i.test(`${context.occasion} ${context.request}`)
    ) {
      dimensions.dressCode = Math.min(dimensions.dressCode, 0.2);
      evidence.push('evening_tailoring_overdresses_request');
    }
  }
  if (bottom && shoe) {
    if (
      bottom.structured &&
      ['oxford', 'derby', 'monk', 'loafer', 'chelsea', 'dress-boot'].includes(shoe.subtype)
    ) {
      dimensions.footwear = 1;
      evidence.push('dress_footwear_with_tailored_bottom');
    }
    if (bottom.structured && shoe.rugged) dimensions.footwear = 0.15;
    if (bottom.volume === 'wide' && shoe.subtype === 'driver') {
      dimensions.proportion = 0.45;
      evidence.push('wide_hem_over_delicate_driver');
    }
  }
  const volumes = descriptors.filter((d) => d.volume === 'wide').length;
  if (volumes >= 3) {
    dimensions.proportion = 0.5;
    evidence.push('multiple_high_volume_pieces');
  }
  const heavy = descriptors.some((d) => d.bulk === 'heavy'),
    airy = descriptors.some((d) => d.materials.includes('linen'));
  if (heavy && airy) {
    dimensions.texture = 0.4;
    evidence.push('heavy_and_airy_texture_contrast');
  }
  const colors = new Set(descriptors.map((d) => d.color).filter((c) => c !== 'unknown'));
  if (colors.size > 0) dimensions.palette = 0.9;
  const accents = [...colors].filter(
    (c) =>
      ![
        'white',
        'black',
        'grey',
        'gray',
        'navy',
        'brown',
        'beige',
        'earth',
        'neutral',
        'cream',
      ].includes(c)
  );
  if (colors.size >= 4 && accents.length >= 3) dimensions.palette = 0.55;
  if (context.style === 'minimalist') {
    const noisy = descriptors.filter((d) => /graphic|logo|floral|camo/.test(d.pattern)).length;
    const patternUnknown = descriptors.some((d) => d.pattern === 'unknown');
    dimensions.style = noisy ? 0.35 : patternUnknown ? 0.7 : 0.9;
    if (colors.size <= 2 && colors.size > 0) {
      dimensions.style = noisy ? 0.35 : patternUnknown ? 0.8 : 1;
      evidence.push('restrained_minimalist_palette');
    }
  } else if (context.style === 'old_money') {
    const classic = descriptors.filter((d) =>
      [
        'blazer',
        'cardigan',
        'polo',
        'shirt',
        'dress-shirt',
        'chino',
        'tailored-trouser',
        'loafer',
      ].includes(d.subtype)
    ).length;
    dimensions.style = 0.5 + classic * 0.125;
  } else if (context.style === 'semi_classic')
    dimensions.style = descriptors.filter((d) => d.structured).length >= 2 ? 0.95 : 0.65;
  else if (context.style === 'casual')
    dimensions.style = descriptors.some((d) =>
      ['overshirt', 'jeans', 'sneaker', 'minimal-sneaker'].includes(d.subtype)
    )
      ? 0.95
      : 0.7;
  dimensions.style =
    (dimensions.style +
      items.reduce((sum, item) => sum + scoreGarmentForContext(item, context), 0) /
        Math.max(1, items.length)) /
    2;
  const valid = !reasons.length;
  return {
    valid,
    score: valid ? Object.values(dimensions).reduce((s, n) => s + n, 0) / 7 : 0,
    reasons: [...new Set(reasons)],
    evidence: [...new Set(evidence)],
    dimensions,
  };
}

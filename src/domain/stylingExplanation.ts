import { outfitCoherence } from './outfitCoherence.ts';
import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
import type { StylingContext } from './stylingContext.ts';
/** Existing UI translates these grounded reasons; domain never chooses a language. */
export function stylingExplanationKeys(
  items: GarmentEvidence[],
  context: StylingContext
): string[] {
  const q = outfitCoherence(items, context),
    keys: string[] = [];
  const descriptors = items.map(describeGarment);
  const layer = descriptors.find(d => d.slot === 'outerwear');
  if (context.dressCode === 'formal' && q.valid) keys.push('formalStructure');
  if (/\b(travel|trip)\b/.test(`${context.occasion} ${context.request}`) && layer?.closure === 'openable' && layer.bulk !== 'heavy') keys.push('travelLayer');
  if (q.evidence.includes('light_base_under_openable_layer')) keys.push('lightBase');
  if (q.evidence.includes('dress_footwear_with_tailored_bottom')) keys.push('tailoredFootwear');
  if (q.evidence.includes('restrained_minimalist_palette')) keys.push('restrainedPalette');
  const d = items.map(describeGarment);
  if (
    !keys.length &&
    d.some((i) => i.subtype === 'overshirt') &&
    d.some((i) => i.subtype === 't-shirt')
  )
    keys.push('casualLayer');
  if (!context.weather && layer?.closure === 'openable' && layer.bulk !== 'heavy') keys.push('adaptableLayer');
  if (!keys.length) keys.push('fitCheck');
  return keys.slice(0, 2).map((key) => `outfitMaker.qualityExplanation.${key}`);
}

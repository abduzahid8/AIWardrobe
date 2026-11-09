import { outfitCoherence } from './outfitCoherence.ts';
import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
import type { StylingContext } from './stylingContext.ts';
/** Existing UI translates these grounded reasons; domain never chooses a language.
 * Every key is derived from verified garment evidence (never from a template
 * recitation), so a tip always adds information the description did not state
 * — and `distinctStylingTips` hides it when it does not. */
export function stylingExplanationKeys(
  items: GarmentEvidence[],
  context: StylingContext
): string[] {
  const q = outfitCoherence(items, context),
    keys: string[] = [];
  const descriptors = items.map(describeGarment);
  const layer = descriptors.find(d => d.slot === 'outerwear');
  const base = descriptors.find(d => d.slot === 'top');
  const bottom = descriptors.find(d => d.slot === 'bottom');
  const shoe = descriptors.find(d => d.slot === 'shoes');
  const tailoredFootwear = !!bottom && !!shoe && bottom.structured && !shoe.casualSole
    && ['loafer', 'derby', 'oxford', 'monk', 'chelsea', 'dress-boot'].includes(shoe.subtype);
  const lightOpenLayer = base?.bulk === 'light' && layer?.closure === 'openable';
  const textureContrast = !!layer && !!base && layer.texture !== 'unknown'
    && layer.texture !== 'smooth' && base.texture === 'smooth';
  if (context.dressCode === 'formal' && q.valid) keys.push('formalStructure');
  if (/\b(travel|trip)\b/.test(`${context.occasion} ${context.request}`) && layer?.closure === 'openable' && layer.bulk !== 'heavy') keys.push('travelLayer');
  if (lightOpenLayer) keys.push('lightBase');
  if (tailoredFootwear || q.evidence.includes('dress_footwear_with_tailored_bottom')) keys.push('tailoredFootwear');
  if (context.style === 'minimalist' && q.valid) keys.push('restrainedPalette');
  if (textureContrast) keys.push('textureDepth');
  if (
    !keys.length &&
    descriptors.some((i) => i.subtype === 'overshirt') &&
    descriptors.some((i) => i.subtype === 't-shirt')
  )
    keys.push('casualLayer');
  if (!context.weather && layer?.closure === 'openable' && layer.bulk !== 'heavy') keys.push('adaptableLayer');
  if (!keys.length) keys.push('fitCheck');
  return keys.slice(0, 2).map((key) => `outfitMaker.qualityExplanation.${key}`);
}

import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
import { valueIntelligence } from './editorialIntelligence.ts';
import type { StylingContext } from './stylingContext.ts';
/** Structured facts shared by localized UI and server fallback. No inferred
 * comfort, fibers, weather or claim that the reranker saw photographs. */
export function lookExplanationFacts(items: GarmentEvidence[], context: StylingContext) {
  const top = items.find((i) => i.macroCategory === 'top'),
    layer = items.find((i) => i.macroCategory === 'outerwear'),
    bottom = items.find((i) => i.macroCategory === 'bottom'),
    shoe = items.find((i) => i.macroCategory === 'shoes');
  const values = valueIntelligence(items, context),
    descriptors = items.map(describeGarment);
  const palette = [
    ...new Set(descriptors.map((d) => d.color).filter((c) => c !== 'unknown')),
  ].slice(0, 3);
  const base = top ? describeGarment(top) : undefined;
  const outer = layer ? describeGarment(layer) : undefined;
  const trouser = bottom ? describeGarment(bottom) : undefined;
  const footwear = shoe ? describeGarment(shoe) : undefined;
  const named = (i: GarmentEvidence | undefined) => i?.name || '';
  return {
    tailoredFootwear:
      !!bottom &&
      !!shoe &&
      describeGarment(bottom).structured &&
      !describeGarment(shoe).casualSole &&
      ['loafer', 'derby', 'oxford', 'monk', 'chelsea', 'dress-boot'].includes(
        describeGarment(shoe).subtype
      ),
    lightOpenLayer: base?.bulk === 'light' && outer?.closure === 'openable',
    textureContrast: !!outer && !!base && outer.texture !== 'unknown' && outer.texture !== 'smooth'
      && base.texture === 'smooth',
    casualFootwear: trouser?.formality === 'casual' && footwear?.formality === 'casual',
    knownColors: descriptors.filter(d => !['unknown','neutral',''].includes(d.color)).length,
    top: named(top),
    layer: named(layer),
    bottom: named(bottom),
    shoes: named(shoe),
    palette: palette.join(' / '),
    texture: layer ? describeGarment(layer).texture : 'unknown',
    shape: bottom ? describeGarment(bottom).volume : 'unknown',
    valueCompared: items.some((i) => values(i).peers >= 4),
    imageFields: descriptors.reduce(
      (n, d) => n + d.evidence.filter((e) => e.startsWith('image:')).length,
      0
    ),
  };
}
export function groundedLookExplanation(items: GarmentEvidence[], context: StylingContext): string {
  const f = lookExplanationFacts(items, context);
  const relationship = f.tailoredFootwear
    ? `The line of ${f.bottom} is continued by the dress profile of ${f.shoes}.`
    : f.lightOpenLayer ? `${f.top} provides a light base inside the openable ${f.layer}.`
    : f.textureContrast ? `${f.layer} adds ${f.texture} texture against the smoother ${f.top}.`
    : f.casualFootwear ? `${f.bottom} and ${f.shoes} share a casual register.`
    : f.layer ? `${f.top} sits under ${f.layer}; ${f.bottom} finishes with ${f.shoes}.`
    : `${f.top} pairs with ${f.bottom} and ${f.shoes}, with no additional layer.`;
  return relationship
    + (f.palette && f.knownColors >= 2 ? ` The ${f.palette} palette links the known colors.` : '')
    + (f.shape !== 'unknown' ? ` The ${f.shape} trouser cut defines the silhouette.` : '');
}

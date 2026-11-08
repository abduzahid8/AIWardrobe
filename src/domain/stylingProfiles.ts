import type { GarmentDescriptor } from './garmentDescriptor.ts';
import type { StylingContext } from './stylingContext.ts';

/** Explainable soft preferences; dress-code hard gates live in garmentContext. */
export interface StylingProfile {
  preferred: readonly string[];
  textures: readonly string[];
  restrained: boolean;
}
export const STYLING_PROFILES: Readonly<Record<string, StylingProfile>> = {
  minimalist: {
    preferred: [
      'shirt',
      'fine-knit',
      'cardigan',
      'overshirt',
      'tailored-trouser',
      'chino',
      'minimal-sneaker',
      'chelsea',
      'dress-boot',
    ],
    textures: ['smooth', 'fine-knit'],
    restrained: true,
  },
  old_money: {
    preferred: ['polo', 'dress-shirt', 'cardigan', 'blazer', 'chino', 'tailored-trouser', 'loafer'],
    textures: ['suede', 'knit', 'fine-knit', 'wool'],
    restrained: true,
  },
  classic: {
    preferred: ['dress-shirt', 'shirt', 'polo', 'blazer', 'cardigan', 'coat', 'tailored-trouser', 'chino', 'oxford', 'derby', 'loafer', 'monk'],
    textures: ['smooth', 'wool'], restrained: true,
  },
  business_casual: {
    preferred: ['polo', 'shirt', 'fine-knit', 'blazer', 'cardigan', 'chino', 'tailored-trouser', 'loafer'],
    textures: ['knit', 'wool'], restrained: true,
  },
  semi_classic: {
    preferred: [
      'dress-shirt',
      'shirt',
      'polo',
      'cardigan',
      'blazer',
      'tailored-trouser',
      'derby',
      'oxford',
      'monk',
      'dress-boot',
    ],
    textures: ['smooth', 'wool'],
    restrained: false,
  },
  casual: {
    preferred: [
      't-shirt',
      'polo',
      'shirt',
      'overshirt',
      'jacket',
      'bomber',
      'jeans',
      'chino',
      'sneaker',
      'minimal-sneaker',
    ],
    textures: ['denim', 'cotton'],
    restrained: false,
  },
};


/** Positive relationships, rather than a second set of admission rules. Each
 * formula describes a whole silhouette. Unlisted combinations remain usable. */
export interface StylingFormula {
  id: string;
  slots: readonly (readonly string[])[];
}
const formula = (id: string, top: string[], layer: string[], bottom: string[], shoes: string[]): StylingFormula =>
  ({id, slots: [top, layer, bottom, shoes]});
const tailoring = formula('tailored', ['dress-shirt', 'shirt'], ['blazer', 'coat'],
  ['tailored-trouser'], ['oxford', 'derby', 'monk', 'dress-boot', 'loafer']);
const ivy = formula('ivy-knit', ['polo', 'dress-shirt', 'shirt'], ['cardigan', 'blazer'],
  ['chino', 'tailored-trouser'], ['loafer', 'chelsea']);
const soft = formula('soft-tailoring', ['fine-knit', 'polo', 'sweater'], ['blazer', 'coat'],
  ['tailored-trouser', 'chino'], ['loafer', 'chelsea', 'derby']);
const relaxed = formula('relaxed-layers', ['t-shirt', 'polo'], ['overshirt', 'jacket', 'bomber'],
  ['jeans', 'chino', 'cargo'], ['minimal-sneaker', 'sneaker', 'boot']);
const texture = formula('textured-knit', ['t-shirt', 'shirt', 'fine-knit'], ['cardigan', 'pullover'],
  ['jeans', 'chino', 'trouser'], ['minimal-sneaker', 'loafer', 'chelsea']);
const modern = formula('modern-tailoring', ['t-shirt', 'fine-knit', 'polo'], ['blazer', 'cardigan'],
  ['tailored-trouser', 'chino', 'jeans'], ['minimal-sneaker', 'chelsea', 'loafer']);
const clean = formula('clean-lines', ['t-shirt', 'fine-knit', 'shirt'], ['overshirt', 'coat', 'cardigan'],
  ['tailored-trouser', 'chino', 'jeans'], ['minimal-sneaker', 'chelsea']);
const active = formula('active', ['t-shirt', 'polo'], ['jacket', 'hoodie', 'bomber'],
  ['athletic-trouser', 'cargo'], ['sport-sneaker', 'hiking-boot']);
const STYLE_FORMULAS: Readonly<Record<string, readonly StylingFormula[]>> = {
  old_money: [ivy, soft, tailoring], classic: [tailoring, soft, ivy],
  semi_classic: [modern, soft, tailoring], business_casual: [modern, ivy, soft],
  casual: [relaxed, texture, modern], minimalist: [clean, modern, texture],
  streetwear: [relaxed, texture, modern], sport: [active, relaxed],
};
export function stylingFormulas(context: StylingContext): readonly StylingFormula[] {
  if (context.dressCode === 'athletic') return [active];
  if (context.dressCode === 'formal') return [tailoring];
  return STYLE_FORMULAS[context.style] || [relaxed, modern, texture];
}
export const FORMULA_SLOTS = ['top', 'outerwear', 'bottom', 'shoes'] as const;
export function formulaGarmentAffinity(d: GarmentDescriptor, f: StylingFormula): number {
  const slot = FORMULA_SLOTS.indexOf(d.slot as typeof FORMULA_SLOTS[number]);
  if (slot < 0 || d.subtype === 'unknown') return 0.5;
  return f.slots[slot].includes(d.subtype) ? 1 : 0.25;
}
/** The weakest relationship matters: three individually classic pieces do
 * not establish an Ivy outfit when the fourth piece has no connection. */
export function outfitStyleIntent(descriptors: GarmentDescriptor[], context: StylingContext): {id: string; score: number} {
  let best = {id: 'open', score: 0.5};
  for (const [priority, f] of stylingFormulas(context).entries()) {
    const affinities = descriptors.map(d => formulaGarmentAffinity(d, f));
    const score = affinities.reduce((sum, n) => sum + n, 0) / Math.max(1, affinities.length) * 0.7
      + (affinities.length ? Math.min(...affinities) : 0.5) * 0.3;
    const identityScore = score * (1 - priority * 0.08);
    if (identityScore > best.score) best = {id: f.id, score: identityScore};
  }
  return best;
}

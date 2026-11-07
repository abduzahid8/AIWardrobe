import {
  describeGarment,
  type GarmentDescriptor,
  type GarmentEvidence,
} from './garmentDescriptor.ts';
import { formulaGarmentAffinity, stylingFormulas, type StylingFormula } from './stylingProfiles.ts';
import type { StylingContext } from './stylingContext.ts';
export const OUTFIT_CONCEPT_VERSION = 1;
export interface OutfitConcept extends StylingFormula {
  title: string;
  palette: readonly string[];
  silhouette: 'balanced' | 'relaxed' | 'tailored';
  textures: readonly string[];
  register: readonly string[];
  story: string;
}
const concept = (
  id: string,
  title: string,
  slots: string[][],
  palette: string[],
  silhouette: OutfitConcept['silhouette'],
  textures: string[],
  register: string[],
  story: string
): OutfitConcept => ({ id, title, slots, palette, silhouette, textures, register, story });
const tailored = ['tailored-trouser', 'trouser'];
const dressShoes = ['oxford', 'derby', 'monk', 'dress-boot'];
const ivy = concept(
  'refined-ivy',
  'Refined Ivy',
  [
    ['dress-shirt', 'shirt'],
    ['blazer', 'cardigan'],
    ['chino', 'tailored-trouser'],
    ['loafer', 'derby'],
  ],
  ['navy', 'cream', 'blue', 'brown'],
  'balanced',
  ['knit', 'tweed', 'smooth'],
  ['smart'],
  'A crisp base, easy tailoring and grounded leather footwear.'
);
const textured = concept(
  'textured-tailoring',
  'Textured tailoring',
  [
    ['shirt', 'fine-knit', 'dress-shirt'],
    ['blazer', 'coat'],
    tailored,
    ['derby', 'loafer', 'monk'],
  ],
  ['brown', 'beige', 'cream', 'grey'],
  'tailored',
  ['tweed', 'suede', 'corduroy', 'knit'],
  ['smart', 'formal'],
  'Quiet color relationships give textured tailoring room to lead.'
);
const smartLayer = concept(
  'smart-knit-layering',
  'Smart knit layering',
  [
    ['shirt', 'dress-shirt', 'polo'],
    ['cardigan', 'overshirt'],
    ['tailored-trouser', 'chino'],
    ['loafer', 'chelsea'],
  ],
  ['cream', 'beige', 'navy', 'brown'],
  'balanced',
  ['knit', 'smooth'],
  ['smart'],
  'A softer open layer keeps the proportions refined and approachable.'
);
const office = concept(
  'precise-office',
  'Precise office tailoring',
  [['dress-shirt', 'shirt'], ['blazer', 'coat'], tailored, dressShoes],
  ['navy', 'grey', 'blue', 'white'],
  'tailored',
  ['smooth'],
  ['formal', 'smart'],
  'A clear shirt line and structured trousers carry the professional register.'
);
const separates = concept(
  'classic-separates',
  'Classic separates',
  [
    ['shirt', 'dress-shirt'],
    ['blazer'],
    ['tailored-trouser', 'chino'],
    ['derby', 'loafer', 'monk'],
  ],
  ['blue', 'brown', 'cream', 'navy'],
  'balanced',
  ['tweed', 'smooth', 'suede'],
  ['smart', 'formal'],
  'Contrast between jacket and trousers gives traditional separates definition.'
);
const evening = concept(
  'tonal-evening',
  'Tonal evening',
  [
    ['fine-knit', 'polo', 'shirt'],
    ['blazer', 'cardigan'],
    tailored,
    ['chelsea', 'loafer', 'derby'],
  ],
  ['black', 'grey', 'navy', 'burgundy'],
  'tailored',
  ['smooth', 'knit'],
  ['smart'],
  'A darker tonal palette and clean lines make an understated evening look.'
);
const softDate = concept(
  'soft-evening-texture',
  'Soft evening texture',
  [
    ['fine-knit', 'shirt', 'polo'],
    ['cardigan', 'blazer'],
    ['tailored-trouser', 'chino'],
    ['loafer', 'chelsea'],
  ],
  ['brown', 'cream', 'beige', 'burgundy'],
  'balanced',
  ['knit', 'suede', 'corduroy'],
  ['smart'],
  'Warm neutrals and tactile layers bring softness to a polished date look.'
);
const modern = concept(
  'modern-evening',
  'Modern evening separates',
  [
    ['t-shirt', 'fine-knit', 'shirt'],
    ['jacket', 'overshirt', 'blazer'],
    ['tailored-trouser', 'jeans', 'chino'],
    ['chelsea', 'loafer', 'derby'],
  ],
  ['black', 'navy', 'grey', 'cream'],
  'balanced',
  ['smooth', 'denim'],
  ['smart', 'casual'],
  'A simple base lets modern outerwear and a clean trouser line define the look.'
);
const travel = concept(
  'easy-transit',
  'Easy transit',
  [
    ['t-shirt', 'polo'],
    ['overshirt', 'jacket'],
    ['chino', 'jeans'],
    ['minimal-sneaker', 'sneaker'],
  ],
  ['navy', 'grey', 'cream', 'blue'],
  'relaxed',
  ['smooth', 'denim', 'knit'],
  ['casual'],
  'Easy proportions, walkable footwear and an adaptable layer for travel.'
);
const utility = concept(
  'soft-utility',
  'Soft utility',
  [
    ['t-shirt', 'shirt'],
    ['jacket', 'overshirt', 'bomber'],
    ['cargo', 'chino', 'jeans'],
    ['sneaker', 'boot', 'minimal-sneaker'],
  ],
  ['olive', 'beige', 'brown', 'cream'],
  'relaxed',
  ['denim', 'smooth'],
  ['casual'],
  'Earth tones and practical construction keep the whole look grounded.'
);
const weekend = concept(
  'relaxed-knit',
  'Relaxed knit layers',
  [
    ['t-shirt', 'polo', 'shirt'],
    ['cardigan', 'pullover'],
    ['jeans', 'chino', 'trouser'],
    ['minimal-sneaker', 'sneaker', 'loafer'],
  ],
  ['cream', 'grey', 'blue', 'brown'],
  'relaxed',
  ['knit', 'denim', 'suede'],
  ['casual', 'smart'],
  'Soft knit texture balances relaxed trousers and comfortable footwear.'
);
export function outfitConcepts(context: StylingContext): readonly OutfitConcept[] {
  if (['formal', 'athletic'].includes(context.dressCode))
    return stylingFormulas(context).map((f) => ({
      ...f,
      title: f.id === 'active' ? 'Active layers' : 'Formal tailoring',
      palette: [],
      silhouette: 'balanced',
      textures: [],
      register: [context.dressCode],
      story: 'A complete look aligned with the requested dress code.',
    }));
  if (context.style === 'old_money') return [ivy, textured, smartLayer];
  if (context.style === 'classic') return [office, separates, textured];
  if (context.style === 'semi_classic' && /date|свидан/.test(context.occasion))
    return [evening, softDate, modern];
  if (context.style === 'semi_classic' || context.style === 'business_casual')
    return [separates, smartLayer, office];
  if (context.style === 'casual' || /travel|trip|путеш/.test(context.occasion))
    return [travel, utility, weekend];
  return stylingFormulas(context).map((f) => ({
    ...f,
    title: f.id.replaceAll('-', ' '),
    palette: [],
    silhouette: 'balanced',
    textures: [],
    register: [],
    story: 'Simple proportions and considered layers form a complete look.',
  }));
}
export function conceptGarmentAffinity(d: GarmentDescriptor, c: OutfitConcept): number {
  const palette = d.color === 'unknown' ? 0.5 : c.palette.includes(d.color) ? 1 : 0.3;
  const texture = d.texture === 'unknown' ? 0.5 : c.textures.includes(d.texture) ? 1 : 0.4;
  const register = d.formality === 'unknown' ? 0.5 : c.register.includes(d.formality) ? 1 : 0.2;
  const proportion =
    d.volume === 'unknown'
      ? 0.5
      : c.silhouette === 'relaxed'
        ? d.volume === 'slim'
          ? 0.3
          : 1
        : d.volume === 'wide'
          ? 0.4
          : 1;
  return (
    0.55 * formulaGarmentAffinity(d, c) +
    0.15 * palette +
    0.1 * texture +
    0.1 * register +
    0.1 * proportion
  );
}
export function lookConcept(
  items: GarmentEvidence[],
  context: StylingContext
): { concept: OutfitConcept; score: number } {
  const descriptors = items.map(describeGarment);
  return outfitConcepts(context)
    .map((concept) => {
      const values = descriptors.map((d) => conceptGarmentAffinity(d, concept));
      return {
        concept,
        score:
          (0.7 * values.reduce((sum, n) => sum + n, 0)) / Math.max(1, values.length) +
          0.3 * Math.min(...values),
      };
    })
    .sort((a, b) => b.score - a.score)[0];
}

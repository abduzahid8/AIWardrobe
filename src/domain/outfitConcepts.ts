import { resolveOutfitComposition, hasColdLayerIntent } from './outfitComposition.ts';
import { colorFamilyOf } from './outfitDiversity.ts';
import {
  describeGarment,
  type GarmentDescriptor,
  type GarmentEvidence,
} from './garmentDescriptor.ts';
import { formulaGarmentAffinity, stylingFormulas, type StylingFormula } from './stylingProfiles.ts';
import type { StylingContext } from './stylingContext.ts';
export const OUTFIT_CONCEPT_VERSION = 2;
export interface OutfitConcept extends StylingFormula {
  title: string;
  footwearPurpose?: 'city' | 'dress' | 'active';
  layerPurpose?: 'light-shell' | 'weather-shell';
  layerRole?: 'physical' | 'absent';
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
    ['cardigan'],
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
  [['t-shirt'], ['overshirt'], ['chino', 'jeans'], ['minimal-sneaker', 'sneaker']],
  ['navy', 'grey', 'cream', 'blue'],
  'relaxed',
  ['smooth', 'denim', 'knit'],
  ['casual'],
  'Easy proportions, understated footwear and an adaptable layer for travel.'
);
const utility = concept(
  'soft-utility',
  'Soft utility',
  [
    ['shirt', 'polo'],
    ['jacket', 'bomber'],
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
    ['polo', 'shirt'],
    ['cardigan'],
    ['jeans', 'chino', 'trouser'],
    ['minimal-sneaker', 'sneaker', 'loafer'],
  ],
  ['cream', 'grey', 'blue', 'brown'],
  'relaxed',
  ['knit', 'denim', 'suede'],
  ['casual', 'smart'],
  'Soft knit texture balances relaxed trousers and understated footwear.'
);
function baseConcepts(context: StylingContext): readonly OutfitConcept[] {
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
  if (context.style === 'minimalist')
    return [
      {
        ...modern,
        id: 'clean-lines',
        title: 'Clean lines',
        palette: ['black', 'white', 'grey', 'navy'],
        textures: ['smooth', 'denim'],
        slots: [
          ['t-shirt'],
          ['overshirt'],
          ['jeans', 'chino', 'trouser'],
          ['minimal-sneaker', 'sneaker', 'chelsea'],
        ],
      },
      {
        ...weekend,
        id: 'textured-knit',
        title: 'Quiet knit texture',
        palette: ['cream', 'grey', 'beige', 'brown'],
        silhouette: 'balanced',
        slots: [
          ['shirt', 'dress-shirt'],
          ['cardigan'],
          ['chino', 'trouser', 'tailored-trouser'],
          ['minimal-sneaker', 'loafer', 'chelsea'],
        ],
      },
      {
        ...modern,
        id: 'modern-tailoring',
        title: 'Tonal modern separates',
        palette: ['black', 'navy', 'grey', 'cream'],
        textures: ['smooth', 'knit'],
        slots: [
          ['polo', 'fine-knit'],
          ['blazer', 'jacket'],
          ['chino', 'tailored-trouser', 'trouser'],
          ['loafer', 'minimal-sneaker', 'chelsea'],
        ],
      },
    ];
  if (context.style === 'old_money') return [ivy, textured, smartLayer];
  if (context.style === 'classic')
    return [
      office,
      smartLayer,
      {
        ...textured,
        slots: [textured.slots[0], ['coat', 'blazer'], textured.slots[2], textured.slots[3]],
      },
    ];
  if (context.style === 'semi_classic' && /date|свидан/.test(context.occasion))
    return [evening, softDate, { ...modern, layerPurpose: 'light-shell' }];
  if (context.style === 'semi_classic' || context.style === 'business_casual')
    return [separates, smartLayer, office];
  if (context.style === 'smart_casual')
    return [
      {
        ...separates,
        id: 'smart-casual-tailored',
        title: 'Tailored smart casual',
        palette: ['navy', 'white', 'grey', 'brown'],
        slots: [
          ['shirt', 'fine-knit', 'dress-shirt'],
          ['blazer', 'jacket'],
          ['tailored-trouser', 'chino'],
          ['loafer', 'derby', 'chelsea', 'minimal-sneaker'],
        ],
      },
      smartLayer,
      {
        ...modern,
        id: 'smart-casual-knit',
        title: 'Refined casual separates',
        palette: ['cream', 'beige', 'olive', 'brown'],
        textures: ['knit', 'smooth', 'suede'],
        slots: [
          ['polo', 'knit-polo', 't-shirt'],
          ['overshirt', 'jacket'],
          ['chino', 'jeans', 'trouser'],
          ['minimal-sneaker', 'loafer', 'sneaker'],
        ],
      },
    ];
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
  const palette =
    !c.palette.length || d.color === 'unknown'
      ? 0.5
      : c.palette.some((color) => color === d.color || colorFamilyOf({ color }) === d.color)
        ? 1
        : 0.3;
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
/**
 * Neutral fallbacks: preserve the context's composition (slots, layer and
 * footwear roles) but make no texture, color, or knit claims beyond verified
 * garment evidence, so the title always describes the actual garments. The
 * variant is chosen from verified subtypes and register — never from color
 * words, product names, or slot concatenations. Base object is spread, never
 * mutated.
 */
export function groundedConceptFallback(
  base: OutfitConcept,
  items?: readonly GarmentEvidence[],
): OutfitConcept {
  const descriptors = (items || []).map(describeGarment);
  const slot = (s: string) => descriptors.find((d) => d.slot === s);
  const layer = slot('outerwear'), bottom = slot('bottom'), shoe = slot('shoes');
  const knitLayer = layer && (layer.texture === 'knit'
    || ['cardigan', 'pullover', 'sweater', 'fine-knit'].includes(layer.subtype));
  const dressyShoes = shoe && !shoe.casualSole
    && ['oxford', 'derby', 'monk', 'dress-boot', 'loafer', 'chelsea'].includes(shoe.subtype);
  const tailoredLayer = layer && (layer.subtype === 'blazer' || layer.subtype === 'coat');
  const denimEase = bottom?.subtype === 'jeans'
    && shoe && ['sneaker', 'minimal-sneaker', 'boot'].includes(shoe.subtype);
  if (knitLayer && dressyShoes !== false && bottom && ['chino', 'tailored-trouser', 'trouser'].includes(bottom.subtype))
    return {
      ...base, id: 'grounded-knit-layers', title: 'Soft Knit Layers', textures: [],
      story: 'Soft knit layering keeps the look polished and approachable.',
    };
  if (tailoredLayer && bottom?.structured && dressyShoes)
    return {
      ...base, id: 'grounded-tailoring', title: 'Refined Tailoring', textures: [],
      story: 'Structured tailoring and dress footwear carry a refined register.',
    };
  if (denimEase)
    return {
      ...base, id: 'grounded-denim-ease', title: 'Relaxed Denim Ease', textures: [],
      story: 'Relaxed denim and understated footwear keep the look easy.',
    };
  return {
    ...base,
    id: 'grounded-separates',
    title: 'Clean separates',
    textures: [],
    story: 'Coordinated separates form a complete look.',
  };
}
export function lookConcept(
  items: GarmentEvidence[],
  context: StylingContext
): { concept: OutfitConcept; score: number } {
  const descriptors = items.map(describeGarment);
  // A concept title must describe the actual selected garments: a knit-titled
  // concept is only fulfillable when at least one item shows knit evidence —
  // knit texture or a knit-construction subtype (unknown construction stays
  // eligible; only proven absence disqualifies).
  const knitSubtypes = new Set(['sweater', 'cardigan', 'pullover', 'fine-knit', 'knit-polo', 'knit']);
  const all = outfitConcepts(context);
  const fulfillable = all.filter((concept) => {
    if (!concept.textures.length) return true;
    const needsKnit = concept.textures.includes('knit') || /knit/i.test(concept.title);
    if (!needsKnit) {
      const known = descriptors.filter((d) => d.texture !== 'unknown');
      if (!known.length) return true;
      return known.some((d) => concept.textures.includes(d.texture));
    }
    return descriptors.some((d) => d.texture === 'knit' || knitSubtypes.has(d.subtype));
  });
  // Zero fulfillable concepts must never resurrect an ungrounded knit/texture
  // title: score a single neutral concept built on this context's composition
  // and grounded in the actual garments.
  const pool = fulfillable.length ? fulfillable : [groundedConceptFallback(all[0], items)];
  return pool
    .map((concept) => {
      const values = descriptors.map((d) => conceptRoleCompatible(d,concept,context) ? conceptGarmentAffinity(d, concept) : 0.1);
      return {
        concept,
        score:
          (0.7 * values.reduce((sum, n) => sum + n, 0)) / Math.max(1, values.length) +
          0.3 * Math.min(...values),
      };
    })
    .sort((a, b) => b.score - a.score)[0];
}

/** Capability and real weather/intent decide whether a layer is mandated.
 * No garment or placeholder represents the absence of a physical layer. */
export function outfitConcepts(context: StylingContext): readonly OutfitConcept[] {
  const activeFootwear = context.dressCode === 'athletic' || (context.dressCode === 'casual'
    && /\b(hik(?:e|ing)|trail|trek(?:king)?|training|basketball)\b/i.test(`${context.activity} ${context.request}`));
  const cityTravel = /\b(travel|trip|sightseeing)\b/i.test(`${context.occasion} ${context.request}`);
  const cold = hasColdLayerIntent(context.weather);
  const concepts: OutfitConcept[] = baseConcepts(context).map(c => ({
    ...c,
    footwearPurpose: activeFootwear ? 'active' as const : cityTravel ? 'city' as const : c.footwearPurpose,
    ...(activeFootwear && context.dressCode === 'casual' ? {slots:[c.slots[0],c.slots[1],c.slots[2],['sport-sneaker','hiking-boot','boot']]} : {}),
    ...(cold && context.dressCode !== 'formal' && context.dressCode !== 'athletic' ? {
      layerRole: 'physical' as const,
      layerPurpose: 'weather-shell' as const,
      slots: [c.slots[0], ['coat','jacket'], c.slots[2], ['chelsea','dress-boot','boot','hiking-boot']],
    } : {}),
  }));
  const policy = resolveOutfitComposition(
    context.style,
    context.weather,
    context.request,
    context.optionalLayer === true
  );
  if (!context.optionalLayer) return concepts;
  if (policy.layered) {
    const cool = context.weather?.temp != null && context.weather.temp < 18;
    const winter = /winter|freezing|snow|зим|мороз/i.test(context.weather?.condition || '');
    if (context.dressCode === 'casual' && (context.style === 'casual' || cityTravel)
      && !cool && !winter && concepts.length) {
      return [{...concepts[0],id:'clean-transit',title:'Clean transit',layerRole:'absent',
        slots:[concepts[0].slots[0],[],concepts[0].slots[2],concepts[0].slots[3]],
        story:'A simple base and trousers with understated city footwear.'},...concepts.slice(1)];
    }
    return concepts;
  }
  return concepts.map((c, index) => ({
    ...c,
    id: `unlayered-separates-${index + 1}`,
    title: ['Clean separates', 'Soft separates', 'Textured separates'][index] || 'Clean separates',
    layerRole: 'absent' as const,
    slots: [c.slots[0], [], c.slots[2], c.slots[3]],
    story: 'A complete look with no additional physical layer.',
  }));
}

/** A role requirement belongs to a brief, not a global catalog veto. A known
 * insulated shell can serve a cold-weather brief, but not a light evening layer.
 * Missing construction evidence remains unknown and eligible. */
export function conceptRoleCompatible(
  d: GarmentDescriptor,
  c: OutfitConcept,
  context: StylingContext
): boolean {
  if (d.slot === 'outerwear' && c.layerPurpose === 'light-shell' && d.insulated) return false;
  const winterTraction = context.weather?.temp != null && context.weather.temp < 5
    && /snow|ice|снег|л[её]д/i.test(context.weather.condition || '');
  if (d.slot === 'shoes' && c.footwearPurpose === 'city'
    && (d.trainingFootwear || d.footwearPurpose === 'training' || (!winterTraction && (d.rugged || d.footwearPurpose === 'trail')))) return false;
  return true;
}

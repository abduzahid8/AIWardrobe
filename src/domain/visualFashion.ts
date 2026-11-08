import type { GarmentDescriptor, GarmentEvidence } from './garmentDescriptor.ts';

export const VISUAL_FASHION_VERSION = 1;
export interface VisualObservation {
  value: string | boolean;
  confidence: number;
  margin: number;
  evidence: 'image-model' | 'image-pixels' | 'image-ocr' | 'human';
}
/** Offline evidence is tied to the exact source image and catalog revision.
 * Similarity scores are not calibrated probabilities; both score and margin
 * must clear a conservative gate. Unknown attributes are absent, never guessed. */
export interface VisualFashionProfile {
  version: number;
  sourceHash: string;
  imageUrl: string;
  imageSha256: string;
  model: string;
  revision: string;
  attributes: Record<string, VisualObservation>;
  imageKind?: VisualObservation;
  imageQuality?: { width: number; height: number; textCoverage?: number };
}
const allowed: Partial<Record<keyof GarmentDescriptor, readonly (string | boolean)[]>> = {
  insulated: [true,false],
  footwearPurpose: ['walking', 'training', 'trail', 'dress'],
  accentHue: ['black', 'white', 'cream', 'beige', 'brown', 'navy', 'blue', 'grey', 'green', 'red'],
  hemShape: ['straight', 'tapered', 'barrel', 'flared'],
  slot: ['top', 'outerwear', 'bottom', 'shoes','outfits'],
  subtype: ['complete-suit',
    'dress-shirt',
    'shirt',
    'field-shirt',
    'polo',
    't-shirt',
    'fine-knit',
    'sweater',
    'blazer',
    'cardigan',
    'overshirt',
    'coat',
    'bomber',
    'jacket',
    'hoodie',
    'pullover',
    'tailored-trouser',
    'chino',
    'trouser',
    'jeans',
    'knit-trouser',
    'athletic-trouser',
    'cargo',
    'oxford',
    'derby',
    'monk',
    'loafer',
    'chelsea',
    'dress-boot',
    'boot',
    'driver',
    'sandal',
    'espadrille',
    'minimal-sneaker',
    'sneaker',
    'sport-sneaker',
    'hiking-boot',
  ],
  color: [
    'black',
    'white',
    'cream',
    'beige',
    'brown',
    'navy',
    'blue',
    'grey',
    'green',
    'olive',
    'burgundy',
    'red',
    'pink',
    'purple',
    'yellow',
    'orange',
  ],
  volume: ['slim', 'regular', 'wide'],
  bulk: ['light', 'heavy'],
  length: ['cropped', 'regular', 'long'],
  texture: ['smooth', 'knit', 'ribbed', 'denim', 'suede', 'corduroy', 'tweed', 'technical'],
  pattern: ['solid', 'stripe', 'check', 'print'],
  patternProminence: ['subtle', 'prominent'],
  formality: ['athletic', 'casual', 'smart', 'formal'],
  soleProfile: ['dress', 'comfort', 'sneaker', 'rugged'],
  closure: ['openable', 'pullover'],
  collar: ['shirt', 'high', 'open', 'crew'],
  sportyZip: [true, false],
  utilityConstruction: [true, false],
  loudBranding: [true, false],
};
export function confidentVisual(o: VisualObservation | undefined): boolean {
  return (
    !!o &&
    Number.isFinite(o.confidence) &&
    o.confidence >= 0.8 &&
    o.confidence <= 1 &&
    Number.isFinite(o.margin) &&
    o.margin >= 0.1 &&
    o.margin <= 1 &&
    ['image-model', 'image-pixels', 'image-ocr', 'human'].includes(o.evidence)
  );
}
export function currentVisual(item: GarmentEvidence): VisualFashionProfile | undefined {
  const p = item.visualFashion;
  if (
    !p ||
    p.version !== VISUAL_FASHION_VERSION ||
    !p.model ||
    !p.revision ||
    !/^[a-f0-9]{64}$/.test(p.imageSha256) ||
    !/^[a-f0-9]{32}$/.test(p.sourceHash) ||
    p.sourceHash !== item.visualSourceHash ||
    p.imageUrl !== (item.image_url || item.imageUrl) ||
    !p.attributes ||
    typeof p.attributes !== 'object'
  )
    return;
  return p;
}
export function applyVisualFashion(
  base: GarmentDescriptor,
  item: GarmentEvidence
): GarmentDescriptor {
  const p = currentVisual(item);
  if (!p) return base;
  const sourceSlot = base.slot;
  const slot = p.attributes.slot;
  // Vision may expose catalog errors, but cannot turn a shirt into purchasable
  // footwear or rewrite source-category authority. Conflicts are quarantined.
  if (confidentVisual(slot) && sourceCategoryConflict(sourceSlot, String(slot.value))) return base;
  const result = { ...base, evidence: [...base.evidence] };
  for (const [key, values] of Object.entries(allowed)) {
    if (key === 'slot') continue;
    const observation = p.attributes[key];
    if (!confidentVisual(observation) || !values.includes(observation.value)) continue;
    // Subtype must belong to the source slot, even if the model's slot is absent.
    if (key === 'subtype' && subtypeSlot(String(observation.value)) !== sourceSlot) continue;
    (result as unknown as Record<string, unknown>)[key] = observation.value;
    result.evidence.push(`image:${key}:${p.model}@${p.revision}`);
  }
  if (result.footwearPurpose === 'training') result.trainingFootwear = true;
  if (result.footwearPurpose === 'trail') result.rugged = true;
  result.casualSole = ['comfort', 'sneaker', 'rugged'].includes(result.soleProfile) || (result.soleProfile === 'unknown' && base.casualSole);
  result.athletic =
    result.formality === 'athletic' ||
    ['athletic-trouser', 'sport-sneaker'].includes(result.subtype);
  result.structured =
    result.subtype === 'tailored-trouser' || result.subtype === 'blazer' || base.structured;
  return result;
}
export function subtypeSlot(subtype: string): string {
  if(subtype==='complete-suit')return 'outfits';
  if (
    ['dress-shirt', 'shirt', 'field-shirt', 'polo', 't-shirt', 'fine-knit', 'sweater'].includes(
      subtype
    )
  )
    return 'top';
  if (
    ['blazer', 'cardigan', 'overshirt', 'coat', 'bomber', 'jacket', 'hoodie', 'pullover'].includes(
      subtype
    )
  )
    return 'outerwear';
  if (
    [
      'tailored-trouser',
      'chino',
      'trouser',
      'jeans',
      'knit-trouser',
      'athletic-trouser',
      'cargo',
    ].includes(subtype)
  )
    return 'bottom';
  return 'shoes';
}
/** A knit may serve as a base or a layer; that role ambiguity is not proof
 * of a bad catalog category. Cross-family conflicts remain quarantined. */
function sourceCategoryConflict(source: string, observed: string): boolean {
  if (source === observed) return false;
  return !(['top', 'outerwear'].includes(source) && ['top', 'outerwear'].includes(observed));
}
export function visualAdmissionReason(item: GarmentEvidence): string | null {
  const p = currentVisual(item);
  if (!p) return null;
  if (
    confidentVisual(p.attributes.slot) &&
    sourceCategoryConflict(String(item.macroCategory), String(p.attributes.slot.value))
  )
    return 'visual_source_category_conflict';
  if (
    confidentVisual(p.imageKind) &&
    ['human', 'image-ocr', 'image-pixels'].includes(p.imageKind!.evidence) &&
    ['diagram', 'promotion', 'unusable'].includes(String(p.imageKind!.value))
  )
    return 'visual_image_unusable';
  return null;
}
/** Only an image actually enriched for this product may replace a thumbnail. */
export function verifiedRecommendationImage(item: GarmentEvidence): string | undefined {
  const p = currentVisual(item);
  return p && confidentVisual(p.imageKind) && p.imageKind!.value === 'product'
    ? p.imageUrl
    : undefined;
}

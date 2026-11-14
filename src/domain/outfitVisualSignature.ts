import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
/** Versioned semantic identity. Physical products keep their separate authority. */
export function outfitVisualSignature(items: GarmentEvidence[]): string {
  return (
    'sq1:' +
    ['top', 'outerwear', 'bottom', 'shoes']
      .map((slot) => {
        const item = items.find((i) => i.macroCategory === slot);
        if (!item) return `${slot}:missing`;
        const d = describeGarment(item);
        return [slot, d.subtype, d.color, d.volume, d.texture, d.formality, d.length].join(':');
      })
      .join('|')
  );
}
/**
 * Maximum token-overlap similarity (0..1) between a look and the bounded
 * recent-show history. Pure description: 1.0 means "same styling idea",
 * 0.0 means nothing comparable was shown. Unknown-heavy looks score 0 —
 * absence of evidence is not similarity.
 */
export function recentLookSimilarity(
  items: GarmentEvidence[],
  recent: readonly string[] = []
): number {
  if (items.filter((item) => describeGarment(item).subtype !== 'unknown').length < 2) return 0;
  const signature = outfitVisualSignature(items),
    parts = signature.slice(4).split('|');
  let maximum = 0;
  for (const previous of recent.slice(-36)) {
    if (!previous.startsWith('sq1:')) continue;
    const other = previous.slice(4).split('|');
    let equal = 0,
      weight = 0,
      known = 0;
    // Formula, palette and fit carry more evidence than shared generic
    // formality/length values. Keep the persisted sq1 format compatible.
    const weights = [0, 0.35, 0.25, 0.2, 0.1, 0.03, 0.07];
    for (let i = 0; i < parts.length; i++) {
      const a = parts[i].split(':'),
        b = (other[i] || '').split(':');
      for (let j = 1; j < a.length; j++)
        if (a[j] !== 'unknown' && b[j] && b[j] !== 'unknown') {
          known++;
          weight += weights[j] || 0;
          if (a[j] === b[j]) equal += weights[j] || 0;
        }
    }
    const similarity = known >= 8 && weight > 0 ? equal / weight : 0;
    maximum = Math.max(maximum, similarity);
  }
  return maximum;
}
export function recentStylingPenalty(
  items: GarmentEvidence[],
  recent: readonly string[] = []
): number {
  const similarity = recentLookSimilarity(items, recent);
  return similarity >= 0.8 ? (similarity - 0.8) * 0.15 : 0;
}

/** Semantic change, independent of brands/IDs. Unknown attributes provide no
 * novelty credit. Cache this per pair in the existing bounded deck search. */
export type OutfitStylingFeatures = readonly (readonly (string | undefined)[] | null)[];
const STYLE_DISTANCE_WEIGHTS = [0.35, 0.2, 0.2, 0.15, 0.1];
/** Compile once for the graph: no repeated garment extraction per look pair. */
export function outfitStylingFeatures(
  items: GarmentEvidence[],
  anchorId?: string | null
): OutfitStylingFeatures {
  return ['top', 'outerwear', 'bottom', 'shoes'].map((slot) => {
    const item = items.find((item) => item.macroCategory === slot);
    if (!item || (anchorId && String(item.id) === anchorId)) return null;
    const d = describeGarment(item);
    return [d.subtype, d.volume, d.color, d.texture, d.pattern];
  });
}
export function stylingFeatureDistance(a: OutfitStylingFeatures, b: OutfitStylingFeatures): number {
  let changed = 0,
    possible = 0;
  for (let slot = 0; slot < 4; slot++) {
    const first = a[slot],
      second = b[slot];
    if (!first || !second) continue;
    for (let field = 0; field < STYLE_DISTANCE_WEIGHTS.length; field++) {
      const x = first[field],
        y = second[field];
      if (!x || !y || x === 'unknown' || y === 'unknown' || x === 'neutral' || y === 'neutral')
        continue;
      possible += STYLE_DISTANCE_WEIGHTS[field];
      if (x !== y) changed += STYLE_DISTANCE_WEIGHTS[field];
    }
  }
  return possible >= 0.7 ? changed / possible : 0;
}
export function outfitStylingDistance(
  a: GarmentEvidence[],
  b: GarmentEvidence[],
  anchorId?: string | null
): number {
  return stylingFeatureDistance(
    outfitStylingFeatures(a, anchorId),
    outfitStylingFeatures(b, anchorId)
  );
}

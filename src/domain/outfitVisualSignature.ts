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
    let equal = 0, weight = 0, known = 0;
    // Formula, palette and fit carry more evidence than shared generic
    // formality/length values. Keep the persisted sq1 format compatible.
    const weights = [0, 0.35, 0.25, 0.20, 0.10, 0.03, 0.07];
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
export function outfitStylingDistance(a: GarmentEvidence[], b: GarmentEvidence[], anchorId?: string | null): number {
  let changed = 0, possible = 0;
  for (const slot of ['top', 'outerwear', 'bottom', 'shoes']) {
    const first = a.find(item => item.macroCategory === slot), second = b.find(item => item.macroCategory === slot);
    if (!first || !second || (anchorId && (String(first.id) === anchorId || String(second.id) === anchorId))) continue;
    const da = describeGarment(first), db = describeGarment(second);
    for (const [key, weight] of [['subtype', 0.35], ['volume', 0.2], ['color', 0.2], ['texture', 0.15], ['pattern', 0.1]] as const) {
      if (['unknown', 'neutral', ''].includes(da[key]) || ['unknown', 'neutral', ''].includes(db[key])) continue;
      possible += weight;
      if (da[key] !== db[key]) changed += weight;
    }
  }
  // Require comparable evidence across at least two garments.
  return possible >= 0.7 ? changed / possible : 0;
}

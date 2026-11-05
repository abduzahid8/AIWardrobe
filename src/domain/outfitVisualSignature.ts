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
export function recentStylingPenalty(
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
      known = 0;
    for (let i = 0; i < parts.length; i++) {
      const a = parts[i].split(':'),
        b = (other[i] || '').split(':');
      for (let j = 1; j < a.length; j++)
        if (a[j] !== 'unknown' && b[j] && b[j] !== 'unknown') {
          known++;
          if (a[j] === b[j]) equal++;
        }
    }
    const similarity = known >= 8 ? equal / known : 0;
    maximum = Math.max(maximum, similarity >= 0.8 ? (similarity - 0.8) * 0.15 : 0);
  }
  return maximum;
}

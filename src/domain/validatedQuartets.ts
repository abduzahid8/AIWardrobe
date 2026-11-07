import { garmentContextReason, scoreGarmentForContext } from './garmentContext.ts';
import { layerCompatibility } from './layerCompatibility.ts';
import { outfitCoherence } from './outfitCoherence.ts';
import { outfitQualityFloor } from './outfitQualityFloor.ts';
import { productIdentityKeys, spreadSuitableProducts, brandKey, garmentNearDuplicateKey } from './outfitDiversity.ts';
import type { GarmentEvidence } from './garmentDescriptor.ts';
import type { StylingContext } from './stylingContext.ts';
/** Bounded pair-first exploration. Caller keeps final budget/source/ownership authority. */
export function validatedQuartets<T extends GarmentEvidence & { id: string }>(
  items: T[],
  context: StylingContext,
  accepts: (items: T[]) => boolean,
  options: { requiredIds?: string[]; maximum?: number; searchLimit?: number; perSlot?: number; reserveMultiplier?: number } = {}
): { id: string; items: T[] }[] {
  const required = options.requiredIds || [],
    maximum = options.maximum ?? 64,
    limit = options.searchLimit ?? 6000;
  const reserveLimit = Math.min(256, maximum * Math.min(4, Math.max(1, options.reserveMultiplier ?? 4)));
  const perSlot = Math.min(32, Math.max(3, options.perSlot ?? 24));
  const slots = ['top', 'outerwear', 'bottom', 'shoes'];
  const pools = slots.map((slot) => {
    const all = items.filter(
        (item) => item.macroCategory === slot && !garmentContextReason(item, context)
      ),
      locked = all.filter((item) => required.includes(item.id));
    return locked.length
      ? locked
      : spreadSuitableProducts(all.map(item=>({item,score:scoreGarmentForContext(item,context)})), entry=>entry.item, entry=>entry.score)
          .slice(0, perSlot).map(entry=>entry.item);
  });
  if (pools.some((pool) => !pool.length)) return [];
  const pairs: { items: T[]; score: number; ordinal: number }[] = [];
  for (const top of pools[0])
    for (const layer of pools[1]) {
      const pair = layerCompatibility(top, layer, context);
      if (pair.valid) pairs.push({ items: [top, layer], score: pair.score, ordinal: pairs.length });
    }
  pairs.sort((a, b) => b.score - a.score || a.ordinal - b.ordinal);
  // Keep diversity of base/layer products before the pair beam cap.
  const usedTop = new Set<string>(),
    usedLayer = new Set<string>();
  const novel = pairs.filter((pair) => {
    const t = pair.items[0].id,
      l = pair.items[1].id;
    if (usedTop.has(t) && usedLayer.has(l)) return false;
    usedTop.add(t);
    usedLayer.add(l);
    return true;
  });
  const beam = [...novel, ...pairs.filter((p) => !novel.includes(p))].slice(
    0,
    Math.min(96, pairs.length)
  );
  const result: { id: string; items: T[]; score: number; order: number }[] = [];
  const signatures = new Set<string>();
  // Rows are immutable during this search: normalize identity once per row.
  const identities = new Map(items.map(item => [item, productIdentityKeys(item)]));
  let examined = 0;
  // Visit all retained top/layer pairs before extending one pair repeatedly.
  for (let step = 0; step < pools[2].length * pools[3].length && examined < limit; step++) {
    for (let p = 0; p < beam.length && examined < limit; p++) {
      examined++;
      const parts = [
        ...beam[p].items,
        pools[2][(step + p) % pools[2].length],
        pools[3][(Math.floor(step / pools[2].length) + p) % pools[3].length],
      ];
      const ids = new Set(parts.map((i) => i.id));
      if (ids.size !== 4 || required.some((id) => !ids.has(id))) continue;
      const keys = new Set<string>();
      let repeated = false;
      for (const part of parts) {
        const own = identities.get(part)!;
        if (own.some((k) => keys.has(k))) repeated = true;
        own.forEach((k) => keys.add(k));
      }
      if (repeated) continue;
      const signature = [...ids].sort().join('|');
      if (signatures.has(signature)) continue;
      signatures.add(signature);
      if (!accepts(parts)) continue;
      const quality = outfitCoherence(parts, context);
      if (!quality.valid || !outfitQualityFloor(parts, context, quality).passed) continue;
      result.push({ id: '', items: parts, score: quality.score, order: result.length });
      // Bounded validated reserve; enough room for global physical-identity matching.
      if (result.length >= reserveLimit) break;
    }
    if (result.length >= reserveLimit) break;
  }
  // Reserve diverse physical alternatives before quality trim, keeping the
  // whole-look scorer as the order within each coverage round.
  result.sort((a, b) => b.score - a.score || a.order - b.order);
  const reserve: typeof result = [],
    seen = new Set<string>();
  const ordered=spreadSuitableProducts(result, candidate=>({brand:candidate.items.filter(i=>!required.includes(i.id)).map(brandKey).join('/'),
      name:''}),candidate=>candidate.score,0.08,candidate=>candidate.items.map(garmentNearDuplicateKey).join('/'));
  for (const candidate of ordered) {
    const companions = candidate.items.filter((i) => !required.includes(i.id));
    if (companions.some((i) => !seen.has(i.id))) {
      reserve.push(candidate);
      companions.forEach((i) => seen.add(i.id));
    }
    if (reserve.length >= maximum) break;
  }
  const chosen = [...reserve, ...result.filter((r) => !reserve.includes(r))].slice(0, maximum);
  return chosen.map((c, i) => ({ id: `look${String(i + 1).padStart(3, '0')}`, items: c.items }));
}

import { stylingCandidateRepresentatives } from './catalogCandidates.ts';
import { describeGarment } from './garmentDescriptor.ts';
import { formulaGarmentAffinity, stylingFormulas } from './stylingProfiles.ts';
import { garmentContextReason, scoreGarmentForContext } from './garmentContext.ts';
import { layerCompatibility } from './layerCompatibility.ts';
import { outfitCoherence } from './outfitCoherence.ts';
import { outfitQualityFloor } from './outfitQualityFloor.ts';
import { productIdentityKeys } from './outfitDiversity.ts';
import type { GarmentEvidence } from './garmentDescriptor.ts';
import type { StylingContext } from './stylingContext.ts';
/** Bounded pair-first exploration. Caller keeps final budget/source/ownership authority. */
export function validatedQuartets<T extends GarmentEvidence & { id: string }>(
  items: T[],
  context: StylingContext,
  accepts: (items: T[]) => boolean,
  options: { requiredIds?: string[]; maximum?: number; searchLimit?: number; perSlot?: number; reserveMultiplier?: number } = {}
): { id: string; items: T[] }[] {
  const required = [...new Set([...(options.requiredIds || []), ...(context.anchorId ? [context.anchorId] : [])])],
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
      : stylingCandidateRepresentatives(all, item=>item, item=>scoreGarmentForContext(item,context), context, perSlot);
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
  const descriptors = new Map(items.map(item => [item, describeGarment(item)]));
  const formulas = stylingFormulas(context);
  // Each intent receives a share of the existing beam. A generic pair beam
  // remains available for good combinations outside the supplied formulas.
  const generic = [...novel, ...pairs.filter(p => !novel.includes(p))];
  const beams = formulas.map(f => [...pairs].sort((a,b) => {
    const affinity = (pair: typeof a) => pair.score * 0.5 + pair.items.reduce((n,item) =>
      n + formulaGarmentAffinity(descriptors.get(item)!, f), 0) * 0.25;
    return affinity(b) - affinity(a) || a.ordinal - b.ordinal;
  }));
  const beam: typeof pairs = [], usedPairs = new Set<typeof pairs[number]>();
  const beamIntent = new Map<typeof pairs[number], number>();
  for (let i = 0; i < generic.length && beam.length < 96; i++) {
    for (const [intent,list] of [...beams, generic].entries()) {
      const pair = list[i];
      if (pair && !usedPairs.has(pair)) {beam.push(pair); usedPairs.add(pair); beamIntent.set(pair,intent);}
      if (beam.length >= 96) break;
    }
  }
  const extensions = [...formulas.map(f => [pools[2],pools[3]].map(pool => [...pool].sort((a,b) =>
    formulaGarmentAffinity(descriptors.get(b)!,f) - formulaGarmentAffinity(descriptors.get(a)!,f)))), [pools[2],pools[3]]];
  const result: { id: string; items: T[]; score: number; order: number }[] = [];
  const signatures = new Set<string>();
  // Rows are immutable during this search: normalize identity once per row.
  const identities = new Map(items.map(item => [item, productIdentityKeys(item)]));
  let examined = 0;
  // Visit all retained top/layer pairs before extending one pair repeatedly.
  for (let step = 0; step < pools[2].length * pools[3].length && examined < limit; step++) {
    for (let p = 0; p < beam.length && examined < limit; p++) {
      examined++;
      const [bottoms, shoes] = extensions[beamIntent.get(beam[p])!];
      const parts = [
        ...beam[p].items,
        bottoms[(step + p) % bottoms.length],
        shoes[(Math.floor(step / bottoms.length) + p) % shoes.length],
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
  const chosen: typeof result = [], remaining = [...result];
  const seenProducts = new Set<string>(), seenRoles = new Set<string>();
  const roles = (candidate: typeof result[number]) => candidate.items.filter(item=>!required.includes(item.id)).map(item=>{
    const d=descriptors.get(item)!;
    return [d.slot,d.subtype,d.color,d.volume,d.texture,d.pattern].join(':');
  });
  // Retain distinct styling decisions inside a narrow quality band, while
  // keeping independent products available for the final joint deck match.
  while (remaining.length && chosen.length < maximum) {
    const best=remaining[0].score;
    const comparable=remaining.filter(candidate=>candidate.score>=best-0.08);
    const novelty=(candidate: typeof result[number]) => {
      const companions=candidate.items.filter(item=>!required.includes(item.id));
      return roles(candidate).filter(role=>!seenRoles.has(role)).length * 0.025
        + companions.filter(item=>!seenProducts.has(item.id)).length * 0.02;
    };
    comparable.sort((a,b)=>(b.score+novelty(b))-(a.score+novelty(a)) || a.order-b.order);
    const candidate=comparable[0];
    chosen.push(candidate);
    roles(candidate).forEach(role=>seenRoles.add(role));
    candidate.items.forEach(item=>seenProducts.add(item.id));
    remaining.splice(remaining.indexOf(candidate),1);
  }
  return chosen.map((c, i) => ({ id: `look${String(i + 1).padStart(3, '0')}`, items: c.items }));
}

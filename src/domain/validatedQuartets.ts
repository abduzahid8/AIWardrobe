import { outfitStylingDistance } from './outfitVisualSignature.ts';
import { stylingCandidateRepresentatives } from './catalogCandidates.ts';
import { describeGarment, prepareGarment } from './garmentDescriptor.ts';
import { formulaGarmentAffinity, stylingFormulas } from './stylingProfiles.ts';
import { garmentContextReason, scoreGarmentForContext } from './garmentContext.ts';
import { layerCompatibility } from './layerCompatibility.ts';
import { outfitCoherence } from './outfitCoherence.ts';
import { outfitQualityFloor } from './outfitQualityFloor.ts';
import { productIdentityKeys } from './outfitDiversity.ts';
import type { GarmentEvidence } from './garmentDescriptor.ts';
import type { StylingContext } from './stylingContext.ts';
/** Compatibility adapter: all callers now share the same bounded planner. */
export function validatedQuartets<T extends GarmentEvidence & { id: string }>(
  items: T[],
  context: StylingContext,
  accepts: (items: T[]) => boolean,
  options: {
    requiredIds?: string[];
    maximum?: number;
    searchLimit?: number;
    perSlot?: number;
    reserveMultiplier?: number;
  } = {}
): { id: string; items: T[] }[] {
  const plan = createOutfitPlan(
    items,
    context,
    (parts) => {
      if (!accepts(parts)) return { valid: false, score: 0 };
      const coherence = outfitCoherence(parts, context);
      return {
        valid: coherence.valid && outfitQualityFloor(parts, context, coherence).passed,
        score: coherence.score,
      };
    },
    {
      requiredIds: options.requiredIds || [],
      target: 3,
      priority: () => [],
      cost: () => 0,
      maxEvaluations: Math.min(
        options.searchLimit ?? 256,
        (options.maximum ?? 64) * (options.reserveMultiplier ?? 4)
      ),
    }
  );
  return plan.alternatives
    .slice(0, options.maximum ?? 64)
    .map((look) => ({ id: look.id, items: look.items }));
}

/** A single immutable plan owns construction, physical matching and shoe
 * rotation. Provider ranking can reorder these alternatives, never rebuild
 * them. All per-look authority remains supplied by the caller's evaluator. */
export interface PlannedLook<T> {
  id: string;
  items: T[];
  quality: number;
  priority: readonly number[];
}
export function createOutfitPlan<T extends GarmentEvidence & { id: string }>(
  items: T[],
  context: StylingContext,
  evaluate: (items: T[]) => { valid: boolean; score: number },
  options: {
    requiredIds: string[];
    target: number;
    priority: (items: T[]) => readonly number[];
    cost: (item: T) => number;
    budget?: number;
    maxEvaluations?: number;
    maxSyncMs?: number;
  }
) {
  const start = performance.now(),
    maximum = options.maxEvaluations ?? 256;
  items = items.map((item) => (Object.isFrozen(item) ? item : prepareGarment(item)));
  const required = new Set([
    ...options.requiredIds,
    ...(context.anchorId ? [context.anchorId] : []),
  ]);
  const descriptors = new Map(items.map((item) => [item, describeGarment(item)]));
  const identities = new Map(items.map((item) => [item, productIdentityKeys(item)]));
  const formulas = stylingFormulas(context),
    slots = ['top', 'outerwear', 'bottom', 'shoes'];
  const pools = slots.map((slot) => {
    const pool = items.filter(
      (item) => item.macroCategory === slot && !garmentContextReason(item, context)
    );
    const locked = pool.filter((item) => required.has(item.id));
    return locked.length
      ? locked
      : stylingCandidateRepresentatives(
          pool,
          (i) => i,
          (i) => scoreGarmentForContext(i, context),
          context,
          12
        );
  });
  const counts = Object.fromEntries(slots.map((slot, i) => [slot, pools[i].length]));
  let evaluations = 0,
    proposals = 0,
    rejected = 0,
    budgetRejected = 0,
    cpuRisk = false;
  const candidates: PlannedLook<T>[] = [];
  const evaluated = new Map<string, { valid: boolean; score: number }>();
  // Bit masks use exact identity equivalence, including aliases linked by
  // any product key. No name/color family is used as physical uniqueness.
  const parent = items.map((_, i) => i),
    keyOwner = new Map<string, number>();
  const root = (n: number): number => (parent[n] === n ? n : (parent[n] = root(parent[n])));
  items.forEach((item, i) => {
    for (const key of identities.get(item)!) {
      const prior = keyOwner.get(key);
      if (prior !== undefined) parent[root(i)] = root(prior);
      else keyOwner.set(key, i);
    }
  });
  const group = new Map(items.map((item, i) => [item, root(i)]));
  const forcedSlots = slots.filter(
    (slot, i) =>
      !pools[i].some((item) => required.has(item.id)) &&
      new Set(pools[i].map((item) => group.get(item))).size < options.target
  );
  const mask = (parts: T[]) =>
    parts.reduce(
      (m, item) =>
        required.has(item.id) || forcedSlots.includes(String(item.macroCategory))
          ? m
          : m | (1n << BigInt(group.get(item)!)),
      0n
    );
  const masks = new Map<PlannedLook<T>, bigint>();
  const admit = (parts: T[]) => {
    proposals++;
    const ids = new Set(parts.map((item) => item.id));
    if (ids.size !== 4 || [...required].some((id) => !ids.has(id))) return;
    const signature = [...ids].sort().join('|');
    if (evaluated.has(signature)) return;
    const physical = new Set<string>();
    for (const item of parts) {
      const keys = identities.get(item)!;
      if (keys.some((key) => physical.has(key))) return;
      keys.forEach((key) => physical.add(key));
    }
    if (
      options.budget !== undefined &&
      parts.reduce((n, item) => n + options.cost(item), 0) > options.budget + 0.000001
    ) {
      budgetRejected++;
      return;
    }
    if (evaluations >= maximum) return;
    if (performance.now() - start > (options.maxSyncMs ?? 650)) {
      cpuRisk = true;
      return;
    }
    evaluations++;
    const result = evaluate(parts);
    evaluated.set(signature, result);
    if (!result.valid) {
      rejected++;
      return;
    }
    const look = {
      id: `look${String(candidates.length + 1).padStart(3, '0')}`,
      items: parts,
      quality: result.score,
      priority: options.priority(parts),
    };
    candidates.push(look);
    masks.set(look, mask(parts));
  };
  // Shoe/bottom relationships are planned before any provider call. Each
  // formula gets its own ranked lists plus a generic path for creative looks.
  const configurations = [
    ...formulas.map((f) =>
      pools.map((pool) =>
        [...pool].sort(
          (a, b) =>
            formulaGarmentAffinity(descriptors.get(b)!, f) -
            formulaGarmentAffinity(descriptors.get(a)!, f)
        )
      )
    ),
    pools,
  ];
  const upper = configurations.map((pool) =>
    pool[0]
      .flatMap((top) =>
        pool[1].map((layer) => ({
          parts: [top, layer],
          pair: layerCompatibility(top, layer, context),
        }))
      )
      .filter((p) => p.pair.valid)
      .sort((a, b) => b.pair.score - a.pair.score)
  );
  // Cover all physical shoes, bases and layers in the first rounds. Latin
  // offsets then explore other relationships, with one shared work bound.
  const largest = Math.max(0, ...pools.map((pool) => pool.length));
  for (let round = 0; round < largest * largest && evaluations < maximum && !cpuRisk; round++) {
    for (let f = 0; f < configurations.length && evaluations < maximum && !cpuRisk; f++) {
      const pool = configurations[f],
        pairs = upper[f];
      if (pool.some((p) => !p.length) || !pairs.length) continue;
      for (let shoe = 0; shoe < pool[3].length && evaluations < maximum && !cpuRisk; shoe++) {
        const pair = pairs[(round * pool[3].length + shoe) % pairs.length];
        admit([...pair.parts, pool[2][(round + shoe) % pool[2].length], pool[3][shoe]]);
        if (proposals >= 4096) break;
      }
      if (proposals >= 4096) break;
    }
    if (proposals >= 4096) break;
  }
  // Precompute semantic differences and rotation priorities once; the exact
  // triple search below performs no garment extraction or outfit scoring.
  const reserve: PlannedLook<T>[] = [];
  const productCoverage = new Set<number>();
  const sorted = [...candidates].sort((a, b) => b.quality - a.quality);
  for (const look of sorted) {
    if (
      look.items.some((item) => !required.has(item.id) && !productCoverage.has(group.get(item)!))
    ) {
      reserve.push(look);
      look.items.forEach((item) => productCoverage.add(group.get(item)!));
    }
    if (reserve.length >= 48) break;
  }
  for (const look of sorted) {
    if (reserve.length >= 48) break;
    if (!reserve.includes(look)) reserve.push(look);
  }
  const distances = new Map<string, number>();
  const distance = (a: PlannedLook<T>, b: PlannedLook<T>) => {
    const key = [a.id, b.id].sort().join(':');
    let value = distances.get(key);
    if (value === undefined) {
      value = outfitStylingDistance(a.items, b.items, context.anchorId);
      distances.set(key, value);
    }
    return value;
  };
  const select = (choices: PlannedLook<T>[], ranks: ReadonlyMap<string, number> = new Map()) => {
    let best: PlannedLook<T>[] = [],
      bestPriority: number[] = [],
      bestQuality = -1,
      bestVariety = -1,
      bestRank = Infinity;
    const consider = (deck: PlannedLook<T>[]) => {
      const priority: number[] = [];
      for (const look of deck)
        look.priority.forEach((n, i) => (priority[i] = (priority[i] || 0) + n));
      let order = 0;
      for (let i = 0; i < Math.max(priority.length, bestPriority.length); i++) {
        order = (priority[i] || 0) - (bestPriority[i] || 0);
        if (order) break;
      }
      const quality = deck.reduce((n, look) => n + look.quality, 0);
      let variety = 1;
      for (let a = 0; a < deck.length; a++)
        for (let b = a + 1; b < deck.length; b++)
          variety = Math.min(variety, distance(deck[a], deck[b]));
      const rank = deck.reduce((n, look) => n + (ranks.get(look.id) ?? 100), 0);
      if (
        !best.length ||
        order > 0 ||
        (order === 0 &&
          (quality > bestQuality + 0.08 * deck.length ||
            (quality >= bestQuality - 0.08 * deck.length &&
              (variety > bestVariety + 0.05 ||
                (Math.abs(variety - bestVariety) <= 0.05 &&
                  (rank < bestRank || (rank === bestRank && quality > bestQuality)))))))
      ) {
        best = deck;
        bestPriority = priority;
        bestQuality = quality;
        bestVariety = variety;
        bestRank = rank;
      }
    };
    // Exact feasible matching over the bounded reserve: diversity cannot
    // turn an existing valid triple into an unexplained 422.
    for (let a = 0; a < choices.length; a++) {
      if (options.target === 1) {
        consider([choices[a]]);
        continue;
      }
      for (let b = a + 1; b < choices.length; b++) {
        if (masks.get(choices[a])! & masks.get(choices[b])!) continue;
        if (options.target === 2) {
          consider([choices[a], choices[b]]);
          continue;
        }
        const used = masks.get(choices[a])! | masks.get(choices[b])!;
        for (let c = b + 1; c < choices.length; c++)
          if (!(used & masks.get(choices[c])!)) consider([choices[a], choices[b], choices[c]]);
      }
    }
    return best;
  };
  let seed = select(reserve);
  if (seed.length !== options.target && candidates.length > reserve.length) {
    // Find any feasible seed over every evaluated option using compatibility
    // bitsets. This avoids cubic re-evaluation and reserve-trim false failures.
    const adjacency = candidates.map(() => 0n),
      bitIndex = new Map<bigint, number>();
    candidates.forEach((_, i) => bitIndex.set(1n << BigInt(i), i));
    for (let a = 0; a < candidates.length; a++)
      for (let b = a + 1; b < candidates.length; b++)
        if (!(masks.get(candidates[a])! & masks.get(candidates[b])!))
          adjacency[a] |= 1n << BigInt(b);
    search: for (let a = 0; a < candidates.length; a++)
      for (let b = a + 1; b < candidates.length; b++) {
        if (!(adjacency[a] & (1n << BigInt(b)))) continue;
        if (options.target === 2) {
          seed = [candidates[a], candidates[b]];
          break search;
        }
        const shared = adjacency[a] & adjacency[b];
        if (shared) {
          const c = bitIndex.get(shared & -shared)!;
          seed = [candidates[a], candidates[b], candidates[c]];
          break search;
        }
      }
  }

  // Always carry a feasible seed into the provider shortlist.
  const shortlist = [...seed];
  for (const look of reserve) {
    if (shortlist.length >= 9) break;
    if (!shortlist.includes(look)) shortlist.push(look);
  }
  return {
    candidates: shortlist,
    alternatives: reserve,
    seed,
    select: (ranks?: ReadonlyMap<string, number>) => select(shortlist, ranks),
    diagnostics: {
      counts,
      evaluations,
      proposals,
      rejected,
      budgetRejected,
      cpuRisk,
      forcedSlots,
      validCandidates: candidates.length,
      exploredCompletely: proposals < 4096 && evaluations < maximum && !cpuRisk,
      syncMs: performance.now() - start,
    },
  };
}

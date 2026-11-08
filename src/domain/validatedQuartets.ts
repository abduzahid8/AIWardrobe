import {hasColdLayerIntent} from './outfitComposition.ts';
import {
  valueIntelligence,
  garmentRelationship,
  lookRelationships,
  imageEvidenceStrength,
  editorialIdentity,
} from './editorialIntelligence.ts';
import { outfitStylingDistance } from './outfitVisualSignature.ts';
import { stylingCandidateRepresentatives } from './catalogCandidates.ts';
import { describeGarment, prepareGarment } from './garmentDescriptor.ts';
import { outfitConcepts, conceptGarmentAffinity, conceptRoleCompatible, type OutfitConcept } from './outfitConcepts.ts';
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
  concept: OutfitConcept;
  conceptScore: number;
  valueScore: number;
  identity: string;
  relationships: number;
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
  const valueOf = valueIntelligence(items, context);
  let formulas = outfitConcepts(context);
  const slots = ['top', 'outerwear', 'bottom', 'shoes'];
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
          24
        );
  });
  const canFill = (f: OutfitConcept) =>
    pools.every(
      (pool, index) =>
        (index === 1 && f.layerRole === 'absent' && context.optionalLayer) ||
        pool.some(
          (item) =>
            f.slots[index].includes(descriptors.get(item)!.subtype) &&
            conceptRoleCompatible(descriptors.get(item)!, f, context)
        )
    );
  formulas = formulas.filter(canFill);
  if (!formulas.length && pools.every((pool) => pool.length)) {
    const roles = pools.map((pool) =>
      [...new Set(pool.map((item) => descriptors.get(item)!.subtype))].filter(
        (type) => type !== 'unknown'
      )
    );
    if (roles.every((role) => role.length))
      formulas = [
        {
          id: 'available-separates',
          layerPurpose:
            context.dressCode === 'smart' && !hasColdLayerIntent(context.weather)
              ? 'light-shell'
              : undefined,
          title: 'Inventory separates',
          slots: roles,
          palette: [],
          silhouette: 'balanced',
          textures: [],
          register: [context.dressCode],
          story: 'A complete alternative built from the available garment roles.',
        },
      ];
  }
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
      !(slot === 'outerwear' && formulas.every(f => f.layerRole === 'absent')) &&
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
  const admit = (parts: T[], concept: OutfitConcept) => {
    proposals++;
    const ids = new Set(parts.map((item) => item.id));
    if (
      ids.size !== parts.length ||
      ![3, 4].includes(parts.length) ||
      [...required].some((id) => !ids.has(id))
    )
      return;
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
    const affinities = parts.map((item) => conceptGarmentAffinity(descriptors.get(item)!, concept));
    const conceptFit = { concept, score: affinities.reduce((n, v) => n + v, 0) / parts.length };
    const valueScore = parts.reduce((n, item) => n + valueOf(item).score, 0) / parts.length;
    const relationships = lookRelationships(parts, context);
    const feedback = Math.max(
      -1,
      Math.min(1, context.conceptFeedback?.[conceptFit.concept.id] || 0)
    );
    const look = {
      id: `look${String(candidates.length + 1).padStart(3, '0')}`,
      items: parts,
      quality: Math.max(
        0,
        Math.min(
          1,
          0.74 * result.score +
            0.16 * relationships +
            0.08 * valueScore +
            (0.02 * parts.reduce((n, item) => n + imageEvidenceStrength(item), 0)) / parts.length +
            0.01 * feedback
        )
      ),
      valueScore,
      relationships,
      identity: editorialIdentity(parts),
      concept: conceptFit.concept,
      conceptScore: conceptFit.score,
      priority: options.priority(parts),
    };
    candidates.push(look);
    masks.set(look, mask(parts));
  };
  // Briefs exist BEFORE selecting products. Each slot is retrieved for its
  // stated role, so a blazer cannot retrospectively become a knit concept.
  // Two compatibility graphs -> bounded, diverse pair beams -> whole-look
  // evaluation. At most 3 * 24^2 cheap joins and 256 full evaluations.
  const feasible: { concept: OutfitConcept; joins: { parts: T[]; score: number }[] }[] = [];
  for (const concept of formulas) {
    const withoutLayer = concept.layerRole === 'absent' && context.optionalLayer === true;
    const rolePools = pools.map((pool, index) => {
      if (index === 1 && withoutLayer) return [];
      const wanted = pool.filter(
        (item) =>
          concept.slots[index].includes(descriptors.get(item)!.subtype) &&
          conceptRoleCompatible(descriptors.get(item)!, concept, context)
      );
      return wanted
        .sort((a, b) => {
          const score = (item: T) =>
            0.6 * conceptGarmentAffinity(descriptors.get(item)!, concept) +
            0.22 * scoreGarmentForContext(item, context) +
            0.15 * valueOf(item).score +
            0.03 * imageEvidenceStrength(item);
          return score(b) - score(a);
        })
        .slice(0, 12);
    });
    if (rolePools.some((p, i) => !p.length && !(i === 1 && withoutLayer))) continue;
    const pairBeam = (pairs: { parts: T[]; score: number }[]) => {
      const uses = new Map<T, number>();
      const chosen: typeof pairs = [];
      for (const pair of pairs.sort((a, b) => b.score - a.score)) {
        if (pair.parts.some((item) => (uses.get(item) || 0) >= 4)) continue;
        chosen.push(pair);
        pair.parts.forEach((item) => uses.set(item, (uses.get(item) || 0) + 1));
        if (chosen.length >= 24) break;
      }
      return chosen;
    };
    const uppers = withoutLayer
      ? rolePools[0].map((top) => ({
          parts: [top],
          score: conceptGarmentAffinity(descriptors.get(top)!, concept),
        }))
      : pairBeam(
          rolePools[0]
            .flatMap((top) =>
              rolePools[1].map((layer) => {
                const pair = layerCompatibility(top, layer, context);
                return {
                  parts: [top, layer],
                  score: pair.valid
                    ? (pair.score +
                        garmentRelationship(
                          descriptors.get(top)!,
                          descriptors.get(layer)!,
                          context
                        )) /
                      2
                    : -1,
                };
              })
            )
            .filter((p) => p.score >= 0)
        );
    const lowers = pairBeam(
      rolePools[2].flatMap((bottom) =>
        rolePools[3].map((shoe) => ({
          parts: [bottom, shoe],
          score: garmentRelationship(descriptors.get(bottom)!, descriptors.get(shoe)!, context),
        }))
      )
    );
    const joins = uppers
      .flatMap((upper) =>
        lowers.map((lower) => {
          const parts = [...upper.parts, ...lower.parts];
          const affinity =
            parts.reduce(
              (n, item) => n + conceptGarmentAffinity(descriptors.get(item)!, concept),
              0
            ) / parts.length;
          const value = parts.reduce((n, item) => n + valueOf(item).score, 0) / parts.length;
          return {
            parts,
            score:
              (0.45 * (upper.score + lower.score)) / 2 +
              0.3 * affinity +
              0.15 * value +
              0.1 *
                garmentRelationship(
                  descriptors.get(upper.parts.at(-1)!)!,
                  descriptors.get(lower.parts[0])!,
                  context
                ),
          };
        })
      )
      .sort((a, b) => b.score - a.score);
    feasible.push({ concept, joins });
  }
  // Round-robin gives every feasible brief an evaluation budget. Within a
  // brief, interleave physical products rather than 80 versions of one pair.
  for (const f of feasible) {
    const counts = new Map<string, number>();
    f.joins = f.joins
      .map((join) => {
        const penalty = join.parts.reduce((n, item) => n + (counts.get(item.id) || 0), 0);
        join.parts.forEach((item) => counts.set(item.id, (counts.get(item.id) || 0) + 1));
        return { ...join, score: join.score - 0.002 * penalty };
      })
      .sort((a, b) => b.score - a.score);
  }
  for (let index = 0; index < 576 && evaluations < maximum && !cpuRisk; index++) {
    for (const f of feasible) {
      if (evaluations >= maximum || cpuRisk) break;
      const join = f.joins[index];
      if (join) admit(join.parts, f.concept);
    }
  }
  // Precompute semantic differences and rotation priorities once; the exact
  // triple search below performs no garment extraction or outfit scoring.
  const reserve: PlannedLook<T>[] = [];
  const productCoverage = new Set<number>();
  const sorted = [...candidates].sort((a, b) => b.quality - a.quality);
  for (const concept of formulas) {
    const best = sorted.filter((look) => look.concept.id === concept.id).slice(0, 4);
    for (const look of best)
      if (!reserve.includes(look)) {
        reserve.push(look);
        look.items.forEach((item) => productCoverage.add(group.get(item)!));
      }
  }
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
  const availablePerSlot = pools.map(pool => new Set(pool.map(item => group.get(item))).size);
  const lockedSlots = pools.map(pool => pool.some(item => required.has(item.id)));
  const select = (choices: PlannedLook<T>[], ranks: ReadonlyMap<string, number> = new Map()) => {
    const decks: {looks: PlannedLook<T>[]; quality: number; utility: number}[] = [];
    let bestQuality = -Infinity;
    const consider = (deck: PlannedLook<T>[]) => {
      for (let i = 0; i < slots.length; i++) {
        if (
          lockedSlots[i] || availablePerSlot[i] < deck.length
        )
          continue;
        const products = deck.flatMap((look) =>
          look.items
            .filter((item) => item.macroCategory === slots[i])
            .map((item) => group.get(item))
        );
        if (new Set(products).size !== products.length) return;
      }
      const priority: number[] = [];
      for (const look of deck)
        look.priority.forEach((n, i) => (priority[i] = (priority[i] || 0) + n));
      const quality = deck.reduce((n, look) => n + look.quality, 0);
      const conceptVariety =
        new Set(deck.filter((look) => look.conceptScore >= 0.65).map((look) => look.concept.id))
          .size / Math.max(1, deck.length);
      let variety = 1;
      for (let a = 0; a < deck.length; a++)
        for (let b = a + 1; b < deck.length; b++)
          variety = Math.min(variety, distance(deck[a], deck[b]));
      variety = 0.55 * variety + 0.45 * conceptVariety;
      const rank = deck.reduce((n, look) => n + (ranks.get(look.id) ?? 100), 0);
      const mean = quality / Math.max(1, deck.length),
        floor = Math.min(...deck.map((look) => look.quality));
      const primary = 0.65 * mean + 0.35 * floor;
      const rankValue = ranks.size ? 1 - rank / Math.max(1, ranks.size * deck.length) : 0;
      const recency =
        priority.reduce((n, value) => n + Math.tanh(value / Math.max(1, deck.length)), 0) /
        Math.max(1, priority.length);
      const actualRoles = ['top', 'outerwear', 'bottom', 'shoes'].map(
        (slot) =>
          new Set(
            deck.map((look) => {
              const item = look.items.find((item) => item.macroCategory === slot);
              if (!item) return 'absent';
              const d = descriptors.get(item)!;
              if (slot === 'top' && ['shirt', 'dress-shirt'].includes(d.subtype)) return 'shirt';
              if (slot === 'shoes')
                return d.athletic
                  ? 'athletic'
                  : d.casualSole
                    ? 'casual'
                    : d.subtype === 'loafer'
                      ? 'loafer'
                      : 'dress';
              if (slot === 'bottom')
                return [d.subtype, d.volume === 'unknown' ? 'unverified' : d.volume].join(':');
              return d.subtype;
            })
          ).size / Math.max(1, deck.length)
      );
      const roleVariety =
        actualRoles[0] * 0.2 + actualRoles[1] * 0.45 + actualRoles[2] * 0.2 + actualRoles[3] * 0.15;
      variety = 0.35 * variety + 0.65 * roleVariety;
      const utility = primary + 0.15 * variety + 0.007 * rankValue + 0.005 * recency;
      bestQuality = Math.max(bestQuality, primary);
      decks.push({looks: deck, quality: primary, utility});
    };
    const distinct = (a: PlannedLook<T>, b: PlannedLook<T>) =>
      a.identity !== b.identity && distance(a, b) >= 0.08;
    // Exact feasible matching over the bounded reserve: diversity cannot
    // turn an existing valid triple into an unexplained 422.
    for (let a = 0; a < choices.length; a++) {
      if (options.target === 1) {
        consider([choices[a]]);
        continue;
      }
      for (let b = a + 1; b < choices.length; b++) {
        if (masks.get(choices[a])! & masks.get(choices[b])! || !distinct(choices[a], choices[b]))
          continue;
        if (options.target === 2) {
          consider([choices[a], choices[b]]);
          continue;
        }
        const used = masks.get(choices[a])! | masks.get(choices[b])!;
        for (let c = b + 1; c < choices.length; c++)
          if (
            !(used & masks.get(choices[c])!) &&
            distinct(choices[a], choices[c]) &&
            distinct(choices[b], choices[c])
          )
            consider([choices[a], choices[b], choices[c]]);
      }
    }
    // Fix the quality floor across the entire feasible set before applying
    // diversity, provider or recency preferences. Selection order cannot move it.
    let best: typeof decks[number] | undefined;
    for (const deck of decks) {
      if (deck.quality < bestQuality - 0.035) continue;
      if (!best || deck.utility > best.utility || (deck.utility === best.utility && deck.quality > best.quality)) best = deck;
    }
    return best?.looks || [];
  };
  let seed = select(reserve);
  // Reserve trimming must not turn a feasible three-look inventory into
  // a false shortage. A bounded compatibility bitset search recovers a seed
  // from already evaluated, quality-approved looks, preserving identity and
  // semantic differences. It never evaluates additional garment combinations.
  if (!seed.length && options.target === 3 && candidates.length > reserve.length) {
    const ordered = [...candidates].sort((a, b) => b.quality - a.quality),
      adjacency = ordered.map(() => 0n),
      bitIndex = new Map<bigint, number>();
    ordered.forEach((_, i) => bitIndex.set(1n << BigInt(i), i));
    matching: for (let a = 0; a < ordered.length; a++)
      for (let b = a + 1; b < ordered.length; b++) {
        if (
          masks.get(ordered[a])! & masks.get(ordered[b])! ||
          ordered[a].identity === ordered[b].identity
        )
          continue;
        if (performance.now() - start > (options.maxSyncMs ?? 650)) {
          cpuRisk = true;
          break matching;
        }
        if (distance(ordered[a], ordered[b]) >= 0.08) adjacency[a] |= 1n << BigInt(b);
      }
    search: for (let a = 0; a < ordered.length && !cpuRisk; a++)
      for (let b = a + 1; b < ordered.length; b++) {
        if (!(adjacency[a] & (1n << BigInt(b)))) continue;
        const shared = adjacency[a] & adjacency[b];
        if (shared) {
          seed = [ordered[a], ordered[b], ordered[bitIndex.get(shared & -shared)!]];
          break search;
        }
      }
    for (const look of seed) if (!reserve.includes(look)) reserve.push(look);
    if (seed.length) seed = select(reserve);
  }
  if (!seed.length && options.target > 1) {
    const target = options.target;
    options = { ...options, target: target - 1 };
    seed = select(reserve);
    if (!seed.length && options.target > 1) {
      options = { ...options, target: 1 };
      seed = select(reserve);
    }
  }

  // Always carry a feasible seed into the provider shortlist.
  const shortlist = [...seed];
  for (const concept of formulas) {
    for (const look of reserve.filter((look) => look.concept.id === concept.id).slice(0, 2))
      if (!shortlist.includes(look)) shortlist.push(look);
  }
  for (const look of reserve) {
    if (shortlist.length >= 12) break;
    if (!shortlist.includes(look)) shortlist.push(look);
  }
  return {
    candidates: shortlist,
    alternatives: reserve,
    seed,
    select: (ranks?: ReadonlyMap<string, number>) => select(shortlist, ranks),
    diagnostics: {
      counts,
      conceptsTarget: 3,
      conceptsFeasible: feasible.length,
      conceptsAchieved: new Set(seed.map((look) => look.concept.id)).size,
      editorialIdentities: new Set(seed.map((look) => look.identity)).size,
      photoProducts: items.filter((item) => imageEvidenceStrength(item) > 0).length,
      selectedQuality: seed.map((look) => look.quality),
      selectedValue: seed.map((look) => look.valueScore),
      evaluations,
      proposals,
      rejected,
      budgetRejected,
      cpuRisk,
      forcedSlots,
      validCandidates: candidates.length,
      conceptCounts: Object.fromEntries(
        formulas.map((f) => [f.id, candidates.filter((look) => look.concept.id === f.id).length])
      ),
      exploredCompletely: proposals < 4096 && evaluations < maximum && !cpuRisk,
      syncMs: performance.now() - start,
    },
  };
}

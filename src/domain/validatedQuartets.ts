import { createDeckSelector } from './outfitDeckSelection.ts';
import { hasColdLayerIntent } from './outfitComposition.ts';
import {
  valueIntelligence,
  garmentRelationship,
  lookRelationships,
  imageEvidenceStrength,
  editorialIdentity,
} from './editorialIntelligence.ts';
import { outfitStylingFeatures, stylingFeatureDistance } from './outfitVisualSignature.ts';
import { stylingCandidateRepresentatives } from './catalogCandidates.ts';
import { describeGarment, prepareGarment } from './garmentDescriptor.ts';
import {
  outfitConcepts,
  conceptGarmentAffinity,
  conceptRoleCompatible,
  type OutfitConcept,
} from './outfitConcepts.ts';
import { garmentContextReason, scoreGarmentForContext } from './garmentContext.ts';
import { layerCompatibility } from './layerCompatibility.ts';
import { outfitCoherence } from './outfitCoherence.ts';
import { outfitQualityFloor } from './outfitQualityFloor.ts';
import { productIdentityKeys } from './outfitDiversity.ts';
import type { DiversitySlot } from './outfitDiversity.ts';
import type { GarmentEvidence } from './garmentDescriptor.ts';
import type { StylingContext } from './stylingContext.ts';

/** Provider ranking may break a near-quality tie but cannot override the
 * planner's 0.035 quality acceptance band. */
export const NVIDIA_RANK_UTILITY_WEIGHT = 0.012;
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

export function selectedPlanDiagnostics<T extends { concept: { id: string }; identity: string }>(
  looks: readonly T[]
) {
  return {
    conceptsAchieved: new Set(looks.map((look) => look.concept.id)).size,
    editorialIdentities: new Set(looks.map((look) => look.identity)).size,
  };
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
    forcedRepeatSlots?: readonly DiversitySlot[];
  }
) {
  const start = performance.now(),
    maximum = Math.min(512, Math.max(0, options.maxEvaluations ?? 256));
  items = items.map((item) => (Object.isFrozen(item) ? item : prepareGarment(item)));
  const required = new Set([
    ...options.requiredIds,
    ...(context.anchorId ? [context.anchorId] : []),
  ]);
  const descriptors = new Map(items.map((item) => [item, describeGarment(item)]));
  const identities = new Map(items.map((item) => [item, productIdentityKeys(item)]));
  const valueOf = valueIntelligence(items, context);
  let formulas = outfitConcepts(context).flatMap((formula) =>
    context.optionalLayer && formula.layerRole !== 'absent'
      ? [
          formula,
          {
            ...formula,
            id: `${formula.id}-no-layer`,
            title: `${formula.title} without a layer`,
            layerRole: 'absent' as const,
            slots: [
              formula.slots[0],
              [],
              formula.slots[2],
              formula.slots[3],
            ] as OutfitConcept['slots'],
            story: 'The same styling direction with no physical outer layer.',
          },
        ]
      : [formula]
  );
  const slots = ['top', 'outerwear', 'bottom', 'shoes'];
  const pools = slots.map((slot) => {
    const pool = items.filter(
      (item) => item.macroCategory === slot && !garmentContextReason(item, context)
    );
    const locked = pool.filter((item) => required.has(item.id));
    if (locked.length) return locked;
    const formulas = outfitConcepts(context);
    const reserved = formulas.flatMap((formula) =>
      stylingCandidateRepresentatives(
        pool.filter((item) => {
          const descriptor = describeGarment(item);
          const slotIndex = slots.indexOf(slot);
          return (
            formula.slots[slotIndex]?.includes(descriptor.subtype) &&
            conceptRoleCompatible(descriptor, formula, context)
          );
        }),
        (i) => i,
        (i) => scoreGarmentForContext(i, context),
        context,
        6
      )
    );
    const unique = new Map<string, T>();
    for (const item of [
      ...reserved,
      ...stylingCandidateRepresentatives(
        pool,
        (i) => i,
        (i) => scoreGarmentForContext(i, context),
        context,
        24
      ),
    ]) {
      if (unique.size >= 24) break;
      unique.set(productIdentityKeys(item).join('|') || item.id, item);
    }
    return [...unique.values()];
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
  // Older clients require three physical four-slot alternatives. A feasible
  // brief may cover only one trouser subtype despite adequate compatible
  // inventory. Retain a reserve of the available roles under the same whole-
  // outfit evaluator instead of treating that brief as catalog authority.
  const legacyRoleGap =
    context.legacyFourSlot &&
    pools.some((pool, index) => {
      if (pool.some((item) => required.has(item.id))) return false;
      const covered = pool.filter((item) =>
        formulas.some(
          (f) =>
            f.slots[index].includes(descriptors.get(item)!.subtype) &&
            conceptRoleCompatible(descriptors.get(item)!, f, context)
        )
      );
      return covered.length < Math.min(options.target, pool.length);
    });
  if ((!formulas.length || legacyRoleGap) && pools.every((pool) => pool.length)) {
    const roles = pools.map((pool) =>
      [...new Set(pool.map((item) => descriptors.get(item)!.subtype))].filter(
        (type) => type !== 'unknown'
      )
    );
    if (roles.every((role) => role.length))
      formulas = [
        ...formulas,
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
  const lookBySignature = new Map<string, PlannedLook<T>>();
  // Resolve exact identity equivalence, including aliases linked by
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
  const forcedSlots = [
    ...new Set([
      ...slots.filter(
        (slot, i) =>
          !(slot === 'outerwear' && formulas.every((f) => f.layerRole === 'absent')) &&
          !pools[i].some((item) => required.has(item.id)) &&
          new Set(pools[i].map((item) => group.get(item))).size < options.target
      ),
      ...(options.forcedRepeatSlots || []),
    ]),
  ];
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
    if (evaluated.has(signature)) {
      // A shared combination is evaluated once, but its editorial identity
      // must not depend on which compatible brief happened to visit it first.
      const prior = lookBySignature.get(signature);
      if (prior) {
        const fit =
          parts.reduce(
            (sum, item) => sum + conceptGarmentAffinity(descriptors.get(item)!, concept),
            0
          ) / parts.length;
        if (fit > prior.conceptScore) {
          prior.concept = concept;
          prior.conceptScore = fit;
        }
      }
      return;
    }
    // Equivalence is transitive across listings, not just direct key overlap.
    if (new Set(parts.map((item) => group.get(item))).size !== parts.length) return;
    if (options.budget !== undefined) {
      const costs = parts.map((item) => options.cost(item));
      if (
        costs.some((cost) => !Number.isFinite(cost) || cost < 0) ||
        costs.reduce((sum, cost) => sum + cost, 0) > options.budget + 0.000001
      ) {
        budgetRejected++;
        return;
      }
    }
    if (evaluations >= maximum) return;
    if (performance.now() - start > (options.maxSyncMs ?? 650)) {
      cpuRisk = true;
      return;
    }
    evaluations++;
    const result = evaluate(parts);
    evaluated.set(signature, result);
    if (!result.valid || !Number.isFinite(result.score)) {
      rejected++;
      return;
    }
    const affinities = parts.map((item) => conceptGarmentAffinity(descriptors.get(item)!, concept));
    const conceptFit = { concept, score: affinities.reduce((n, v) => n + v, 0) / parts.length };
    const valueScore = parts.reduce((n, item) => n + valueOf(item).score, 0) / parts.length;
    const relationships = lookRelationships(parts, context);
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
            (0.02 * parts.reduce((n, item) => n + imageEvidenceStrength(item), 0)) / parts.length
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
    lookBySignature.set(signature, look);
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
  // One graph owns every cross-look constraint over the full evaluated set.
  // The provider shortlist is a presentation boundary, never a feasibility cap.
  const indices = new Map(candidates.map((look, i) => [look, i]));
  const features = candidates.map((look) => outfitStylingFeatures(look.items, context.anchorId));
  const distances = new Float64Array(candidates.length * candidates.length).fill(NaN);
  const distance = (a: PlannedLook<T>, b: PlannedLook<T>) => {
    const x = indices.get(a)!,
      y = indices.get(b)!,
      key = Math.min(x, y) * candidates.length + Math.max(x, y);
    if (Number.isNaN(distances[key]))
      distances[key] = stylingFeatureDistance(features[x], features[y]);
    return distances[key];
  };
  const roles = new Map(
    candidates.map((look) => [
      look,
      slots.map((slot) => {
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
          return `${d.subtype}:${d.volume === 'unknown' ? 'unverified' : d.volume}`;
        return d.subtype;
      }),
    ])
  );
  const editorialUtility = (deck: PlannedLook<T>[], ranks: ReadonlyMap<string, number>) => {
    const mean = deck.reduce((sum, look) => sum + look.quality, 0) / deck.length;
    const primary = 0.65 * mean + 0.35 * Math.min(...deck.map((look) => look.quality));
    const concepts =
      new Set(deck.filter((look) => look.conceptScore >= 0.65).map((look) => look.concept.id))
        .size / deck.length;
    let gap = 1;
    for (let a = 0; a < deck.length; a++)
      for (let b = a + 1; b < deck.length; b++) gap = Math.min(gap, distance(deck[a], deck[b]));
    const roleVariety = slots.reduce(
      (sum, _, i) =>
        sum +
        ([0.2, 0.45, 0.2, 0.15][i] * new Set(deck.map((look) => roles.get(look)![i])).size) /
          deck.length,
      0
    );
    const variety = 0.35 * (0.55 * gap + 0.45 * concepts) + 0.65 * roleVariety;
    const rank = ranks.size
      ? deck.reduce(
          (sum, look) => sum + (ranks.has(look.id) ? 1 - ranks.get(look.id)! / ranks.size : 0),
          0
        ) / deck.length
      : 0;
    // Personal taste belongs after intrinsic quality acceptance. Changing a
    // combination's best-fitting brief cannot change its quality certificate.
    const feedback =
      deck.reduce(
        (sum, look) =>
          sum + Math.max(-1, Math.min(1, context.conceptFeedback?.[look.concept.id] || 0)),
        0
      ) / deck.length;
    return primary + 0.15 * variety + NVIDIA_RANK_UTILITY_WEIGHT * rank + 0.01 * feedback;
  };
  const selector = createDeckSelector(
    candidates.map((look) => ({
      look,
      quality: look.quality,
      priority: look.priority,
      products: slots.map((slot) => {
        const item = look.items.find((item) => item.macroCategory === slot);
        return !item || required.has(item.id) ? -1 : group.get(item)!;
      }),
    })),
    {
      target: options.target,
      scarceSlots: forcedSlots.map((slot) => slots.indexOf(slot)),
      compatible: (a, b) =>
        context.legacyFourSlot === true || (a.identity !== b.identity && distance(a, b) >= 0.08),
      utility: editorialUtility,
    }
  );
  let selection = selector.select();
  const seed = selection.looks;
  const reserve = [...candidates].sort((a, b) => b.quality - a.quality);
  const shortlist = [...seed];
  for (const concept of formulas)
    for (const look of reserve.filter((look) => look.concept.id === concept.id).slice(0, 2))
      if (!shortlist.includes(look) && shortlist.length < 12) shortlist.push(look);
  for (const look of reserve) {
    if (shortlist.length >= 12) break;
    if (!shortlist.includes(look)) shortlist.push(look);
  }
  return {
    candidates: shortlist,
    alternatives: reserve,
    seed,
    select: (ranks?: ReadonlyMap<string, number>) => {
      selection = selector.select(ranks);
      return selection.looks;
    },
    get selection() {
      return selection.diagnostics;
    },
    diagnostics: {
      counts,
      conceptsTarget: 3,
      conceptsFeasible: feasible.length,
      ...selectedPlanDiagnostics(seed),
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
      exploredCompletely: feasible.every((f) =>
        f.joins.every((join) =>
          evaluated.has(
            join.parts
              .map((item) => item.id)
              .sort()
              .join('|')
          )
        )
      ),
      selection: selection.diagnostics,
      syncMs: performance.now() - start,
    },
  };
}

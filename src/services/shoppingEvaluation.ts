import { inferItemAttributes } from '../../features/outfit-generator/utils/styleInference';
import type { ShopCatalogItem } from '../types/shop';
import {
  catalogTitleColor,
  normalizeIdentityText,
  visualFamilyIdentityOf,
  physicalProductIdentityOf,
} from '../utils/catalogIdentity';
import { inferShopRowMacro } from '../utils/shopCatalogOrder';
import {
  compareShoppingAlternative,
  duplicateRiskEvidence,
  rankDiverseShoppingCandidates,
} from './shoppingIntelligence';

export type AlternativeGoal = 'overall' | 'cheaper' | 'closest-style' | 'wardrobe' | 'fit';
export interface AlternativeSignals {
  /** Caller-computed evidence from validated outfit comparisons, not combinatorics. */
  wardrobeGain?: number;
  fit?: { score: number; confidence: number; evidence: readonly string[] };
  relevance?: number;
}
const unit = (value: number | undefined) =>
  typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : undefined;
const tokens = (item: ShopCatalogItem) => new Set(normalizeIdentityText(item.name).split(' '));
function textSimilarity(a: ShopCatalogItem, b: ShopCatalogItem): number {
  const left = tokens(a),
    right = tokens(b);
  return (
    [...left].filter((token) => right.has(token)).length /
    Math.max(1, new Set([...left, ...right]).size)
  );
}
export function rankShoppingAlternatives(
  reference: ShopCatalogItem,
  candidates: readonly ShopCatalogItem[],
  options: {
    goal: AlternativeGoal;
    eligible: (item: ShopCatalogItem) => boolean;
    signals?: (item: ShopCatalogItem) => AlternativeSignals;
    limit?: number;
  }
) {
  const evidenceById = new Map<
    string,
    { evidence: string[]; score: number; confidence: 'low' | 'medium' | 'high' }
  >();
  const ranked = candidates.flatMap((item) => {
    if (
      item.id === reference.id ||
      (['dresses', 'outfit', 'accessory'].includes(reference.garmentType) &&
        item.garmentType !== reference.garmentType) ||
      !options.eligible(item) ||
      inferShopRowMacro(item.garmentType, item.name) !==
        inferShopRowMacro(reference.garmentType, reference.name)
    )
      return [];
    const comparison = compareShoppingAlternative(reference, item),
      signals = options.signals?.(item) || {};
    const fit =
      signals.fit && signals.fit.confidence >= 0.7 && signals.fit.evidence.length
        ? unit(signals.fit.score)
        : undefined;
    const wardrobe =
      typeof signals.wardrobeGain === 'number' && Number.isFinite(signals.wardrobeGain)
        ? Math.max(0, signals.wardrobeGain)
        : undefined;
    const saving = comparison.priceSaving;
    if (options.goal === 'cheaper' && !(saving !== undefined && saving > 0)) return [];
    if (options.goal === 'fit' && fit === undefined) return [];
    if (options.goal === 'wardrobe' && !(wardrobe !== undefined && wardrobe > 0)) return [];
    const similarity =
      (comparison.sameRole ? 0.6 : 0) +
      0.3 * textSimilarity(reference, item) +
      (comparison.sameColor ? 0.1 : 0);
    const relevance = unit(signals.relevance) ?? similarity;
    const value =
      saving === undefined ? undefined : Math.max(-1, Math.min(1, saving / reference.price));
    const wardrobeScore = wardrobe === undefined ? undefined : wardrobe / (wardrobe + 3);
    const score =
      options.goal === 'fit'
        ? fit!
        : options.goal === 'wardrobe'
          ? wardrobeScore!
          : options.goal === 'cheaper'
            ? 0.6 * similarity + 0.4 * (value || 0)
            : options.goal === 'closest-style'
              ? similarity
              : 0.55 * relevance +
                0.25 * similarity +
                0.1 * (value || 0) +
                0.05 * (wardrobeScore || 0) +
                0.05 * (fit || 0);
    const evidence = [
      ...comparison.evidence,
      `title similarity ${textSimilarity(reference, item).toFixed(2)}`,
      ...(saving !== undefined ? [`observed saving ${saving} ${reference.currency}`] : []),
      ...(wardrobe !== undefined ? [`${wardrobe} validated sampled outfit gains`] : []),
      ...(fit !== undefined ? signals.fit!.evidence : []),
    ];
    evidenceById.set(item.id, {
      score,
      evidence,
      confidence:
        options.goal === 'fit' && fit !== undefined && signals.fit!.confidence >= 0.85
          ? 'high'
          : comparison.confidence,
    });
    return [{ item, score }];
  });
  return rankDiverseShoppingCandidates(ranked, {
    limit: options.limit || 10,
    eligible: options.eligible,
  }).map((item) => ({ item, goal: options.goal, ...evidenceById.get(item.id)! }));
}

export interface OutfitUnlockContext {
  id: string;
  occasion: string;
}
export interface UsefulOutfit {
  items: ShopCatalogItem[];
  contextId: string;
  signature: string;
}
/** Observable look similarity, separate from deck physical identity. Unknown
 * color/role uses concrete evidence rather than collapsing unrelated looks. */
export function shoppingLookKey(item: ShopCatalogItem): string {
  const role = visualFamilyIdentityOf(item).visualFamilyKey,
    color = catalogTitleColor(item);
  const materials = normalizeIdentityText(item.name)
    .split(' ')
    .filter((word) =>
      ['leather', 'suede', 'cotton', 'linen', 'wool', 'denim', 'silk'].includes(word)
    )
    .sort()
    .join(',');
  const patterns = inferItemAttributes(item).patterns.sort().join(',');
  const silhouette = normalizeIdentityText(item.name)
    .split(' ')
    .filter((word) =>
      ['slim', 'relaxed', 'oversized', 'tailored', 'wide', 'straight'].includes(word)
    )
    .sort()
    .join(',');
  return color && role.includes(':') && !role.includes('other')
    ? `${role}:${color}:${materials}:${patterns}:${silhouette}`
    : physicalProductIdentityOf(item).canonicalKey;
}
export function estimateUsefulOutfitUnlocks(
  candidate: ShopCatalogItem,
  owned: readonly ShopCatalogItem[],
  options: {
    contexts: readonly OutfitUnlockContext[];
    eligible: (item: ShopCatalogItem, context: OutfitUnlockContext) => boolean;
    validateOutfit: (items: ShopCatalogItem[], context: OutfitUnlockContext) => boolean;
    maxEvaluations?: number;
    maxPerSlot?: number;
    maxResults?: number;
  }
) {
  const slots = ['outerwear', 'top', 'bottom', 'shoes'];
  const candidateSlot = inferShopRowMacro(candidate.garmentType, candidate.name);
  const maxEvaluations = Math.max(1, Math.min(512, options.maxEvaluations || 128));
  const maxPerSlot = Math.max(1, Math.min(8, options.maxPerSlot || 4));
  const maxResults = Math.max(1, Math.min(50, options.maxResults || 20));
  const outfits: UsefulOutfit[] = [],
    seen = new Set<string>();
  let evaluations = 0,
    truncated = options.contexts.length > 12;
  if (
    ['dresses', 'outfit', 'accessory'].includes(candidate.garmentType) ||
    !slots.includes(candidateSlot)
  )
    return { outfits, evaluations, truncated, confidence: 'low' as const };
  const candidateIdentity = new Set(physicalProductIdentityOf(candidate).keys);
  if (
    owned.some((item) =>
      physicalProductIdentityOf(item).keys.some((key) => candidateIdentity.has(key))
    )
  )
    return { outfits, evaluations, truncated, confidence: 'low' as const };
  for (const context of options.contexts.slice(0, 12)) {
    if (!options.eligible(candidate, context)) continue;
    let beam: ShopCatalogItem[][] = [[]];
    for (const slot of slots) {
      const bucket =
        slot === candidateSlot
          ? [candidate]
          : owned.filter(
              (item) =>
                !['dresses', 'outfit', 'accessory'].includes(item.garmentType) &&
                inferShopRowMacro(item.garmentType, item.name) === slot &&
                options.eligible(item, context)
            );
      const distinct = [...new Map(bucket.map((item) => [shoppingLookKey(item), item])).values()];
      if (distinct.length > maxPerSlot) truncated = true;
      const next: ShopCatalogItem[][] = [];
      for (const prefix of beam)
        for (const item of distinct.slice(0, maxPerSlot)) {
          const identity = physicalProductIdentityOf(item);
          if (
            prefix.some((other) =>
              physicalProductIdentityOf(other).keys.some((key) => identity.keys.includes(key))
            )
          )
            continue;
          if (next.length < maxEvaluations) next.push([...prefix, item]);
          else truncated = true;
        }
      beam = next;
    }
    for (const items of beam) {
      if (evaluations >= maxEvaluations || outfits.length >= maxResults) {
        truncated = true;
        break;
      }
      evaluations++;
      if (!options.validateOutfit(items, context)) continue;
      // Test an observed equivalent owned piece in this exact look. A valid
      // equivalent means this purchase adds no meaningfully new outfit.
      const equivalent = owned.filter(
        (item) =>
          inferShopRowMacro(item.garmentType, item.name) === candidateSlot &&
          shoppingLookKey(item) === shoppingLookKey(candidate)
      );
      let alreadyPossible = false;
      for (const item of equivalent) {
        if (evaluations >= maxEvaluations) {
          truncated = true;
          alreadyPossible = true;
          break;
        }
        evaluations++;
        if (
          options.eligible(item, context) &&
          options.validateOutfit(
            items.map((piece) => (piece.id === candidate.id ? item : piece)),
            context
          )
        ) {
          alreadyPossible = true;
          break;
        }
      }
      const signature = `${context.id}:${items.map(shoppingLookKey).join('|')}`;
      if (!alreadyPossible && !seen.has(signature)) {
        seen.add(signature);
        outfits.push({ items, contextId: context.id, signature });
      }
    }
    if (evaluations >= maxEvaluations || outfits.length >= maxResults) {
      truncated = true;
      break;
    }
  }
  return { outfits, evaluations, truncated, confidence: 'low' as const };
}
export function evaluateWardrobeValue(
  candidate: ShopCatalogItem,
  owned: readonly ShopCatalogItem[],
  unlocks: ReturnType<typeof estimateUsefulOutfitUnlocks>
) {
  const duplicateRisk = duplicateRiskEvidence(candidate, owned);
  const count = unlocks.outfits.length,
    contexts = new Set(unlocks.outfits.map((outfit) => outfit.contextId)).size;
  const fillsGap = !owned.some(
    (item) =>
      inferShopRowMacro(item.garmentType, item.name) ===
      inferShopRowMacro(candidate.garmentType, candidate.name)
  );
  const score = Math.max(
    0,
    Math.min(
      1,
      0.12 * Math.min(5, count) +
        0.1 * Math.min(3, contexts) +
        (fillsGap && count ? 0.2 : 0) -
        (duplicateRisk.length ? 0.15 : 0)
    )
  );
  return {
    level:
      score >= 0.65 ? ('High' as const) : score >= 0.3 ? ('Medium' as const) : ('Low' as const),
    score,
    sampledNewOutfits: count,
    contextsWithNewLooks: contexts,
    fillsGap,
    duplicateRisk,
    confidence: unlocks.confidence,
    truncated: unlocks.truncated,
    evidence: [
      `${count} distinct validated sampled looks`,
      ...(fillsGap ? ['missing wardrobe slot'] : []),
      ...(duplicateRisk.length ? ['similar owned role and color'] : []),
    ],
  };
}

import type { ShopCatalogItem } from '../types/shop';
import {
  catalogTitleColor,
  visualFamilyIdentityOf,
  normalizeIdentityText,
} from '../utils/catalogIdentity';
import { buildCatalogFamilyViews } from './catalogFamilyView';

export interface ShoppingCandidate {
  item: ShopCatalogItem;
  score: number;
  budgetRepresentative?: boolean;
  anchor?: boolean;
}
/** Eligibility must use the caller's existing deterministic request gates. Rotation
 * is a soft penalty only. Mandatory anchors/budget representatives survive trimming.
 */
export function rankDiverseShoppingCandidates(
  candidates: readonly ShoppingCandidate[],
  options: {
    limit: number;
    eligible: (item: ShopCatalogItem) => boolean;
    recentIds?: ReadonlySet<string>;
  }
): ShopCatalogItem[] {
  const unique = new Map<string, ShoppingCandidate>();
  for (const candidate of candidates)
    if (!unique.has(candidate.item.id) && options.eligible(candidate.item))
      unique.set(candidate.item.id, candidate);
  const limit = Math.max(0, Math.min(100, Math.floor(options.limit)));
  if (!Number.isFinite(limit)) throw new Error('invalid shopping result limit');
  const ranked = [...unique.values()]
    .map((candidate) => ({
      ...candidate,
      effectiveScore: candidate.score - (options.recentIds?.has(candidate.item.id) ? 0.03 : 0),
    }))
    .filter((candidate) => Number.isFinite(candidate.effectiveScore))
    .sort((a, b) => b.effectiveScore - a.effectiveScore);
  const mandatory = ranked.filter(
    (candidate) => candidate.anchor || candidate.budgetRepresentative
  );
  if (mandatory.length > limit)
    throw new Error('candidate limit cannot preserve anchors and budget representatives');
  const familyById = new Map<string, string>();
  for (const family of buildCatalogFamilyViews(ranked.map((candidate) => candidate.item)).families)
    for (const variant of family.variants) familyById.set(variant.variantId, family.familyKey);
  const result = mandatory.map((candidate) => candidate.item),
    used = new Set(result.map((item) => item.id));
  const counts = new Map<string, number>();
  const dimensions = (item: ShopCatalogItem): Array<[string, number]> => [
    [`family:${familyById.get(item.id)}`, 0.12],
    [`role:${visualFamilyIdentityOf(item).visualFamilyKey}`, 0.18],
    ...(item.brand
      ? [[`brand:${normalizeIdentityText(item.brand)}`, 0.06] as [string, number]]
      : []),
    ...(catalogTitleColor(item)
      ? [[`color:${catalogTitleColor(item)}`, 0.025] as [string, number]]
      : []),
  ];
  const metadata = new Map(
    ranked.map((candidate) => [candidate.item.id, dimensions(candidate.item)])
  );
  const record = (item: ShopCatalogItem) => {
    for (const [key] of metadata.get(item.id) || []) counts.set(key, (counts.get(key) || 0) + 1);
  };
  result.forEach(record);
  // Greedy selection is bounded to at most 100 outputs. Redundancy penalties
  // can only choose within 0.15 of the best remaining relevance score.
  while (result.length < limit) {
    const best = ranked.find((candidate) => !used.has(candidate.item.id));
    if (!best) break;
    let winner = best,
      value = -Infinity;
    for (const candidate of ranked) {
      if (used.has(candidate.item.id)) continue;
      if (candidate.effectiveScore < best.effectiveScore - 0.15) break;
      const penalty = (metadata.get(candidate.item.id) || []).reduce(
        (sum, [key, weight]) => sum + weight * Math.min(3, counts.get(key) || 0),
        0
      );
      const adjusted = candidate.effectiveScore - penalty;
      if (adjusted > value) {
        winner = candidate;
        value = adjusted;
      }
    }
    result.push(winner.item);
    used.add(winner.item.id);
    record(winner.item);
  }
  return result;
}

/** Evidence for comparison, never a quality/fit/price-history claim. Different
 * currencies are incomparable until the authoritative budget FX layer is used.
 */
export function compareShoppingAlternative(reference: ShopCatalogItem, candidate: ShopCatalogItem) {
  const role = visualFamilyIdentityOf(reference).visualFamilyKey;
  const sameRole = role === visualFamilyIdentityOf(candidate).visualFamilyKey;
  const specificRole = sameRole && role.includes(':') && !role.includes('other');
  const referenceColor = catalogTitleColor(reference),
    color = catalogTitleColor(candidate);
  const comparablePrice =
    !!reference.currency &&
    reference.currency === candidate.currency &&
    reference.price > 0 &&
    candidate.price > 0 &&
    Number.isFinite(reference.price) &&
    Number.isFinite(candidate.price);
  return {
    sameRole,
    sameColor: referenceColor && color ? referenceColor === color : undefined,
    priceSaving: comparablePrice ? reference.price - candidate.price : undefined,
    evidence: [
      ...(sameRole ? ['same inferred garment role'] : []),
      ...(comparablePrice ? ['same currency observed prices'] : []),
    ],
    confidence: specificRole && referenceColor && color ? ('medium' as const) : ('low' as const),
  };
}
export function duplicateRiskEvidence(
  candidate: ShopCatalogItem,
  owned: readonly ShopCatalogItem[]
) {
  return owned.flatMap((item) => {
    const comparison = compareShoppingAlternative(item, candidate);
    return comparison.sameRole &&
      comparison.sameColor === true &&
      comparison.confidence === 'medium'
      ? [
          {
            ownedId: item.id,
            evidence: ['same inferred role and title color'],
            confidence: 'medium' as const,
          },
        ]
      : [];
  });
}
/** Caller supplies IDs only after validating complete distinct outfits using the
 * existing outfit rules. This is relative usefulness, never financial ROI. */
export function wardrobeValueInputs(
  validNewOutfitIds: readonly string[],
  unlockedOccasions: readonly string[],
  fillsGap: boolean
) {
  return {
    validatedDistinctOutfits: new Set(validNewOutfitIds).size,
    unlockedOccasions: [...new Set(unlockedOccasions)],
    fillsGap,
  };
}

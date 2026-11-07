import type { ShopCatalogItem } from '../types/shop';
import {
  catalogTitleColor,
  catalogBaseName,
  familyIdentityCandidate,
  type GroupingConfidence,
} from '../utils/catalogIdentity';
import { normalizeCatalogImageKey } from '../utils/shopCatalogOrder';

export interface CatalogVariantView {
  variantId: string;
  originalCatalogId: string;
  price: number;
  currency?: string;
  image: ShopCatalogItem['imageUrl'];
  sourceUrl?: string;
  source?: string;
  originalRow: ShopCatalogItem;
  listingId: string;
  color?: string;
  item: ShopCatalogItem;
}
export interface CatalogProductFamilyView {
  familyKey: string;
  brand: string;
  displayName: string;
  canonicalName: string;
  category: string;
  garmentType: ShopCatalogItem['garmentType'];
  variants: CatalogVariantView[];
  preferredVariantId: string;
  groupingConfidence: GroupingConfidence;
  evidence: string[];
}
export interface FamilyPreferences {
  selectedVariantIds?: readonly string[];
  avoidedColors?: readonly string[];
  preferredColors?: readonly string[];
}

/** Linear indexed pass over loaded pages only. No row is destructively collapsed. */
export function buildCatalogFamilyViews(
  items: readonly ShopCatalogItem[],
  preferences: FamilyPreferences = {}
) {
  const buckets = new Map<
    string,
    { identity: ReturnType<typeof familyIdentityCandidate>; items: ShopCatalogItem[] }
  >();
  const seen = new Set<string>();
  for (const item of items) {
    if (seen.has(item.id)) continue;
    seen.add(item.id);
    const identity = familyIdentityCandidate(item);
    const bucket = buckets.get(identity.familyKey);
    if (bucket) bucket.items.push(item);
    else buckets.set(identity.familyKey, { identity, items: [item] });
  }
  const families: CatalogProductFamilyView[] = [];
  const selected = new Map(
    (preferences.selectedVariantIds || []).map((id, index) => [id, 1000 - index])
  );
  const avoided = new Set(preferences.avoidedColors || []);
  const preferred = new Set(preferences.preferredColors || []);
  for (const { identity, items: rows } of buckets.values()) {
    const colors = rows.map(catalogTitleColor);
    const images = rows.map((r) => normalizeCatalogImageKey(r.imageUrl));
    const prices = rows.map((r) => r.price);
    const compatible =
      identity.confidence === 'high' ||
      (rows.length > 1 &&
        colors.every(Boolean) &&
        new Set(colors).size > 1 &&
        images.every(Boolean) &&
        new Set(images).size === rows.length &&
        rows.every((r) => !!r.currency && r.currency === rows[0].currency) &&
        prices.every((p) => Number.isFinite(p) && p > 0) &&
        Math.max(...prices) / Math.min(...prices) <= 1.5);
    const groups = compatible ? [rows] : rows.map((r) => [r]);
    for (const group of groups) {
      const variants = group.map((item) => ({
        variantId: item.id,
        originalCatalogId: item.id,
        price: item.price,
        currency: item.currency,
        image: item.imageUrl,
        sourceUrl: item.sourceUrl,
        source: item.source,
        originalRow: item,
        listingId: item.id,
        color: catalogTitleColor(item),
        item,
      }));
      // Stable incoming order wins ties; appending a page cannot change a pin.
      const score = (v: CatalogVariantView) =>
        selected.has(v.variantId)
          ? selected.get(v.variantId)!
          : (avoided.has(v.color || '') ? -10 : 0) + (preferred.has(v.color || '') ? 1 : 0);
      const winner = variants.reduce((a, b) => (score(b) > score(a) ? b : a));
      families.push({
        familyKey: compatible ? identity.familyKey : JSON.stringify(['listing', group[0].id]),
        brand: group[0].brand,
        displayName: group[0].name,
        canonicalName: catalogBaseName(group[0]),
        category: group[0].category || group[0].garmentType,
        garmentType: group[0].garmentType,
        variants,
        preferredVariantId: winner.variantId,
        groupingConfidence: group.length > 1 ? identity.confidence : 'low',
        evidence: group.length > 1 ? identity.evidence : ['standalone listing; no safe merge'],
      });
    }
  }
  // Keep the original card order even when an early bucket has ungrouped rows.
  const positions = new Map<string, number>();
  items.forEach((item, index) => {
    if (!positions.has(item.id)) positions.set(item.id, index);
  });
  families.sort(
    (a, b) =>
      (positions.get(a.variants[0].variantId) || 0) - (positions.get(b.variants[0].variantId) || 0)
  );
  return {
    families,
    diagnostics: {
      rawListings: items.length,
      variants: seen.size,
      families: families.length,
      confidence: families.reduce(
        (counts, family) => {
          counts[family.groupingConfidence]++;
          return counts;
        },
        { high: 0, medium: 0, low: 0 }
      ),
    },
  };
}

export function selectCatalogVariant(
  family: CatalogProductFamilyView,
  variantId = family.preferredVariantId
): ShopCatalogItem | undefined {
  return family.variants.find((v) => v.variantId === variantId)?.item;
}

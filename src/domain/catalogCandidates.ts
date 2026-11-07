import { productIdentityKeys } from './outfitDiversity.ts';
import { describeGarment, type GarmentEvidence } from './garmentDescriptor.ts';
import { conceptGarmentAffinity, outfitConcepts } from './outfitConcepts.ts';
import type { StylingContext } from './stylingContext.ts';

/** Neutral retrieval wire contract shared by mobile discovery and Edge.
 * These candidates are evidence, never authority to bypass outfit or FX gates.
 */
export interface CatalogCandidateRpcRequest {
  p_slot: string;
  p_source: string;
  p_search: string;
  p_seed: string;
  p_limit: number;
  p_offset: number;
  p_currency: string | null;
  p_max_price: number | null;
  p_context?: Record<string, unknown>;
}
export interface CatalogCandidateRetrievalDiagnostics {
  seed: string;
  pages: number;
  degraded: boolean;
  candidateCount: number;
}

/** Rank a fixed retrieved pool by usable styling roles, with a quality band
 * preventing near-identical garments from consuming the shortlist. Floors,
 * anchors and LRU are retained by callers; this function never admits rows. */
export function stylingCandidateRepresentatives<T>(
  rows: T[],
  itemOf: (row: T) => GarmentEvidence,
  scoreOf: (row: T) => number,
  context: StylingContext,
  maximum = rows.length
): T[] {
  const formulas = outfitConcepts(context);
  const entries = rows
    .map((row, order) => {
      const item = itemOf(row),
        d = describeGarment(item);
      const known = (value: string) => !['unknown', 'neutral', ''].includes(value);
      const facets = [d.subtype, d.color, d.volume, d.pattern, d.texture];
      const brand = String((item as GarmentEvidence & { brand?: string }).brand || 'unknown');
      // Sparse descriptions are not proof of sameness.
      const family = facets.filter(known).length >= 2 ? facets.join(':') : `sparse:${order}`;
      return {
        row,
        order,
        d,
        family,
        brand,
        keys: productIdentityKeys(item),
        score: scoreOf(row),
        owned: item.isShopItem !== true && !String(item.id || '').startsWith('shop_'),
      };
    })
    .sort((a, b) => b.score - a.score || a.order - b.order);
  // Alias listings are one physical product, irrespective of styling facets.
  // Collapse before the representative cap so duplicate rows cannot starve it.
  const physical = new Set<string>();
  const selected: typeof entries = [],
    remaining = entries.filter((entry) => {
      if (entry.keys.some((key) => physical.has(key))) return false;
      entry.keys.forEach((key) => physical.add(key));
      return true;
    });
  const families = new Map<string, number>();
  const brands = new Map<string, number>();
  while (remaining.length && selected.length < maximum) {
    const best = remaining[0].score;
    const f = formulas[selected.length % formulas.length];
    const comparable = remaining.filter((entry) => entry.score >= best - 0.08);
    comparable.sort((a, b) => {
      const value = (e: typeof a) =>
        e.score +
        0.06 * conceptGarmentAffinity(e.d, f) -
        0.07 * (families.get(e.family) || 0) -
        0.02 * (brands.get(e.brand) || 0) +
        (context.sourceMode === 'mixed' && e.owned ? 0.025 : 0);
      return value(b) - value(a) || a.order - b.order;
    });
    const chosen = comparable[0];
    selected.push(chosen);
    brands.set(chosen.brand, (brands.get(chosen.brand) || 0) + 1);
    families.set(chosen.family, (families.get(chosen.family) || 0) + 1);
    remaining.splice(remaining.indexOf(chosen), 1);
  }
  return selected.map((entry) => entry.row);
}

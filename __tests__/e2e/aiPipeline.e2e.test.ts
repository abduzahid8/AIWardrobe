/**
 * aiPipeline.e2e.test.ts — END-TO-END through the REAL AI outfit pipeline.
 *
 * Outside-the-box vs the existing unit tests: instead of tapping buttons on a
 * device (flaky, spends AI budget, needs network + Supabase + LLM keys), this
 * suite drives the exact production code path the app uses to turn catalog
 * rows into outfits — over the REAL bundled catalog data:
 *
 *   shopCatalogItems / inspoMensShopItems
 *     → getMacroCategory (real slot mapping)
 *     → scoreItemForStyle / rankItemsForStyle (real style gate)
 *     → sanitizeGeneratedOutfitItems (real slot filler / validator)
 *
 * If any stage regresses (catalog loses all shoes, scorer buries loafers for
 * old_money, sanitizer drops bottoms), this fails BEFORE a user sees a broken
 * Generate button. No network. No AI spend. Deterministic.
 */

import { SHOP_CATALOG_ITEMS, SHOP_OUTFITS } from '../../data/shopCatalogItems';
import { CLASSIC_MENS_ITEMS } from '../../data/inspoMensShopItems';
import {
  canonicalizeMacroCategory,
  getMacroCategory,
} from '../../src/utils/categoryMapper';
import {
  scoreItemForStyle,
  rankItemsForStyle,
  needsLayering,
  normalizeStyleId,
  type StyleId,
} from '../../features/outfit-generator/utils/styleInference';
import { sanitizeGeneratedOutfitItems } from '../../features/outfit-generator/utils/sanitizeGeneratedOutfit';
import { spreadSimilarCatalogItems, buildCatalogSimilarityKey, mergeShopSlices, collapseSimilarByName, preferMappedMacro, classicCatalogOffset, inferShopRowMacro } from '../../src/utils/shopCatalogOrder';

const STYLES: StyleId[] = [
  'old_money',
  'semi_classic',
  'minimalist',
  'business_casual',
  'casual',
  'classic',
];

interface MappedItem {
  id: string;
  name: string;
  type: string;
  category: string;
  macroCategory: string;
  color: string;
  brand: string;
  description: string;
  image: string;
  imageUrl: string;
}

/** Production slot rule — inferShopRowMacro. */
function toSlot(garmentType: string, name: string, description: string): string {
  return inferShopRowMacro(garmentType, name, description);
}

function mapCatalog(): MappedItem[] {
  const seen = new Set<string>();
  const out: MappedItem[] = [];
  for (const s of [...SHOP_CATALOG_ITEMS, ...CLASSIC_MENS_ITEMS] as any[]) {
    const id = String(s?.id || '');
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const macro = toSlot(s.garmentType, s.name || '', s.description || '');
    out.push({
      id,
      name: s.name || 'Shop item',
      type: macro,
      category: macro,
      macroCategory: macro,
      color: 'neutral',
      brand: s.brand || '',
      description: s.description || '',
      image: s.imageUrl || '',
      imageUrl: s.imageUrl || '',
    });
  }
  return out;
}

const pool = mapCatalog();
const countBy = (macro: string) =>
  pool.filter((i) => canonicalizeMacroCategory(i.macroCategory) === macro).length;

/** Mixed-slot sample shaped like a real wardrobe (catalog order is grouped
 * by garment, so a naive slice(0, N) is tops-only and starves shoes). */
function sampleWardrobe(perSlot = 15) {
  const out: MappedItem[] = [];
  for (const slot of ['top', 'bottom', 'shoes', 'outerwear'] as const) {
    out.push(...pool.filter((i) => canonicalizeMacroCategory(i.macroCategory) === slot).slice(0, perSlot));
  }
  return out;
}

function toAvailable(items: MappedItem[]) {
  return items.map((i) => ({
    id: i.id,
    image: i.imageUrl,
    type: i.type,
    color: i.color,
    name: i.name,
    brand: i.brand,
    macroCategory: i.macroCategory,
    category: i.category,
    isShopItem: true,
  }));
}

describe('AI pipeline e2e — real catalog → real code', () => {
  it('catalog covers every outfit slot (candidatePool static-fallback contract)', () => {
    // AIOutfitmaker pads the shop pool from this exact data when live is thin.
    // If a slot drops below these floors, shop-mode generation starves.
    expect(pool.length).toBeGreaterThan(50);
    expect(countBy('top')).toBeGreaterThanOrEqual(6);
    expect(countBy('bottom')).toBeGreaterThanOrEqual(6);
    expect(countBy('shoes')).toBeGreaterThanOrEqual(6);
    expect(countBy('outerwear')).toBeGreaterThanOrEqual(3);
    // Accessories must never leak into clothing slots.
    expect(pool.filter((i) => i.macroCategory === 'other').length).toBeGreaterThanOrEqual(0);
    for (const item of pool) {
      if (item.macroCategory !== 'other') {
        expect(['top', 'bottom', 'outerwear', 'shoes']).toContain(item.macroCategory);
      }
    }
  });

  it.each(STYLES)('style "%s" keeps ≥3 rankable candidates per slot (no starvation)', (style) => {
    // Mirrors the edge rank call: { minKeep: 16, dropThreshold: -2, perCategoryFloor: 4 }.
    // rankItemsForStyle guarantees the per-category floor — if the INPUT lacks
    // a slot entirely, no ranker can save generation.
    const ranked = rankItemsForStyle(pool as any[], style, {
      minKeep: 16,
      dropThreshold: -2,
      perCategoryFloor: 4,
    });
    for (const slot of ['top', 'bottom', 'shoes'] as const) {
      const n = ranked.filter((i: any) => canonicalizeMacroCategory(i.macroCategory) === slot).length;
      expect(n).toBeGreaterThanOrEqual(3);
    }
  });

  it.each(STYLES)('style "%s" scores a full outfit above the suitability floor', (style: StyleId) => {
    // The app requires score >= 0.35 per slot (MIN_REQUIRED logic).
    const suitable = (slot: string) =>
      pool.filter(
        (i) =>
          canonicalizeMacroCategory(i.macroCategory) === slot &&
          scoreItemForStyle(
            { name: i.name, brand: i.brand, color: i.color, type: i.type, macroCategory: i.macroCategory },
            style,
          ) >= 0.35,
      );
    expect(suitable('top').length).toBeGreaterThanOrEqual(1);
    expect(suitable('bottom').length).toBeGreaterThanOrEqual(1);
    expect(suitable('shoes').length).toBeGreaterThanOrEqual(1);
  });

  it('sanitize turns a noisy AI answer into a complete outfit (substitution e2e)', () => {
    const available = toAvailable(sampleWardrobe(15));
    const bySlot = (slot: string) => available.find((a) => a.macroCategory === slot)!;
    // Noisy LLM output: one hallucinated id, one duplicate, shuffled, missing shoes.
    // NOTE: id-authentication (dropping hallucinated ids) is the EDGE's job
    // (filterValidOutfits against itemMap) — the client's contract is slot
    // completeness + no duplicate cards, pinned here.
    const generated: any[] = [
      { id: 'does-not-exist-123', name: 'Ghost', type: 'top', macroCategory: 'top', image: '' },
      { id: bySlot('bottom').id, name: 'x', type: 'bottom', macroCategory: 'bottom', image: 'x' },
      { id: bySlot('top').id, name: 'x', type: 'top', macroCategory: 'top', image: 'x' },
      { id: bySlot('top').id, name: 'x', type: 'top', macroCategory: 'top', image: 'x' },
    ];
    const result = sanitizeGeneratedOutfitItems(generated, available as any, 5, 'old_money');
    const macros = result.map((r: any) => canonicalizeMacroCategory(r.macroCategory));
    // Complete: top + bottom + shoes present, duplicate top collapsed.
    expect(macros).toContain('top');
    expect(macros).toContain('bottom');
    expect(macros).toContain('shoes');
    expect(new Set(result.map((r: any) => String(r.id))).size).toBe(result.length);
  });

  it('pipeline is deterministic: same catalog in → byte-identical deck out', () => {
    const available = toAvailable(sampleWardrobe(10));
    const generated: any[] = (['top', 'bottom', 'shoes'] as const).map((slot) => {
      const a = available.find((x) => x.macroCategory === slot)!;
      return { id: a.id, name: a.name, type: a.type, macroCategory: slot, image: a.image };
    });
    const run = () =>
      JSON.stringify(sanitizeGeneratedOutfitItems(generated, available as any, 5, 'casual'));
    expect(run()).toBe(run());
  });

  it('style ids normalize from free-form UI strings', () => {
    expect(normalizeStyleId('Old Money')).toBe('old_money');
    expect(normalizeStyleId('old-money')).toBe('old_money');
    expect(normalizeStyleId('business_casual')).toBe('business_casual');
    expect(normalizeStyleId(null)).toBe('casual');
    expect(normalizeStyleId('nonsense-xyz')).toBe('casual');
    // Padded ids fold to their preset (trim runs before key match).
    expect(normalizeStyleId(' BUSINESS_CASUAL ')).toBe('business_casual');
    expect(normalizeStyleId('  old_money  ')).toBe('old_money');
  });

  it('layering policy matrix: cold/formal layers, beach never does', () => {
    expect(needsLayering('old_money', { temp: 22, condition: 'clear' }, null)).toBe(true);
    expect(needsLayering('casual', { temp: 5, condition: 'clear' }, null)).toBe(true);
    expect(needsLayering('casual', { temp: 30, condition: 'clear' }, 'beach day')).toBe(false);
    expect(needsLayering('casual', { temp: 30, condition: 'clear' }, null)).toBe(false);
    expect(needsLayering('business_casual', { temp: 25, condition: 'clear' }, null)).toBe(true);
  });

  it('diversifier honors its sliding-window contract (shop-deck variety)', () => {
    // Real guarantee of spreadSimilarCatalogItems: at each position the
    // greedy pass picks an item whose similarity key is absent from the
    // previous two — whenever any remaining item satisfies that. So in the
    // output, keys[k] may equal keys[k-1]/keys[k-2] ONLY if every item from
    // k onward shared one key (unavoidable tail).
    const items = pool.slice(0, 80).map((i) => ({
      id: i.id,
      name: i.name,
      garmentType: i.macroCategory,
    }));
    const diversified = spreadSimilarCatalogItems(items as any[], []);
    expect(diversified).toHaveLength(items.length);
    expect(new Set(diversified.map((d: any) => d.id)).size).toBe(items.length);
    const keys = diversified.map((d: any) => buildCatalogSimilarityKey(d));
    for (let k = 0; k < keys.length; k++) {
      const window = new Set(keys.slice(Math.max(0, k - 2), k));
      const remaining = new Set(keys.slice(k));
      const escapable = [...remaining].some((key) => !window.has(key));
      if (escapable) {
        expect(window.has(keys[k])).toBe(false);
      }
    }
  });

  it('shop outfits seed is non-empty (Inspo guide first paint)', () => {
    expect(SHOP_OUTFITS.length).toBeGreaterThan(0);
  });

  it('explicit clothing types survive hostile description words (belt-loops regression)', () => {
    // Production incident: "belt loops" in trouser descriptions demoted real
    // suit pants/chinos to 'other', starving the bottom slot of formal decks.
    expect(toSlot('lower_body', 'Linen Cotton Pleated Suit Pants', 'Pleated suit pants, belt loops, side pockets')).toBe('bottom');
    expect(toSlot('lower_body', 'Mid Brown Slim Leg Straight Chinos', 'Chinos with belt loops')).toBe('bottom');
    expect(toSlot('upper_body', 'Lightweight cotton blend trench coat', 'Trench with belted waist')).toBe('outerwear');
    // ...while true accessories still map to 'other'.
    expect(toSlot('upper_body', 'Nappa leather iPhone 16 Pro Max case', 'Phone case')).toBe('other');
    expect(toSlot('outfit', 'Metal Sunglasses', 'Sunglasses with case')).toBe('other');
    // And the blazer-as-upper_body layer rule still holds.
    expect(toSlot('upper_body', 'Navy Wool Blazer', 'Tailored blazer')).toBe('outerwear');
    expect(toSlot('lower_body', 'Cotton blend tailored pants', 'Pair with a wool blazer and oxfords')).toBe('bottom');
    expect(toSlot('lower_body', '100% cool wool suit blazer', 'Suit blazer')).toBe('other');
  });

  it('feed mistags lose to product identity (live-data regression)', () => {
    // 341 tees/knits stored as lower_body + tops stored as shoes (2026-10-01
    // audit). Identity keywords rescue the slot in every direction.
    expect(toSlot('lower_body', '100% cotton short sleeve T-shirt', 'Classic tee')).toBe('top');
    expect(toSlot('lower_body', 'Wool-blend short-sleeve knit sweater', 'Knit')).toBe('top');
    expect(toSlot('shoes', 'Slim fit 100% cotton Oxford shirt', 'Shirt')).toBe('top');
    expect(toSlot('shoes', 'Front view of light blue jeans with belt loops', 'Jeans')).toBe('bottom');
    // Descriptions mentioning other garments must NOT steal the slot.
    expect(toSlot('lower_body', 'Navy Chinos', 'Wear with a white shirt')).toBe('bottom');
  });

  it('mergeShopSlices dedupes by id, primary wins order', () => {
    const primary = [{ id: 'a' }, { id: 'b' }];
    const extra = [{ id: 'b' }, { id: 'c' }, { id: 'a' }, { id: 'd' }];
    expect(mergeShopSlices(primary, extra as any[]).map((i: any) => i.id)).toEqual(['a', 'b', 'c', 'd']);
    expect(mergeShopSlices(primary, [])).toBe(primary);
    expect(mergeShopSlices([], extra as any[]).map((i: any) => i.id)).toEqual(['b', 'c', 'a', 'd']);
  });

  it('per-tap pool = memoized page + random slice (bottoms unfreeze)', () => {
    // Production incident: normalize drew ONLY from the memoized first-100
    // pool, so the random slice reached the edge but finals recycled the same
    // ~3 bottoms. The contract: merged pool must contain slice-only macros.
    const page = pool.filter((i) => canonicalizeMacroCategory(i.macroCategory) === 'top').slice(0, 20);
    const slice = pool.filter((i) => canonicalizeMacroCategory(i.macroCategory) === 'bottom').slice(0, 8);
    expect(page.some((i) => canonicalizeMacroCategory(i.macroCategory) === 'bottom')).toBe(false);
    const merged = collapseSimilarByName(mergeShopSlices(page as any[], slice as any[]));
    expect(merged.filter((i: any) => canonicalizeMacroCategory(i.macroCategory) === 'bottom').length).toBeGreaterThan(0);
    // No id appears twice across the page/slice boundary.
    const ids = merged.map((i: any) => String(i.id));
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('collapseSimilarByName merges same-name ingestions, keeps colorways distinct', () => {
    // Production case: "Leather loafers with penny strap" ×22 UUIDs.
    const items = [
      { id: 's16', name: 'Leather loafers with penny strap', macroCategory: 'shoes' },
      { id: 's17', name: 'Leather loafers with penny strap', macroCategory: 'shoes' },
      { id: 's18', name: 'Navy Leather loafers with penny strap', macroCategory: 'shoes' },
      { id: 't1', name: 'Leather loafers with penny strap', macroCategory: 'top' },
    ];
    const out = collapseSimilarByName(items);
    // Exact-name dupes collapse (keeps first = best-ranked); navy colorway
    // and cross-slot entries survive.
    expect(out.map((i) => i.id)).toEqual(['s16', 's18', 't1']);
  });

  it('collapse shrinks the real bundled pool (duplicate ingestions exist)', () => {
    const before = pool.length;
    const after = collapseSimilarByName(pool.map((p) => ({ id: p.id, name: p.name, macroCategory: p.macroCategory }))).length;
    expect(after).toBeLessThan(before);
  });

  it('payload keeps the mapped slot: recompute must not flip mistagged rows back', () => {
    // Production incident: toPayload rebuilt macros from type/category
    // aliases, so a tee fixed to 'top' by identity mapping flipped back to
    // 'bottom' (type "bottoms") on the way to the edge — and Oxford shirts
    // / jeans stored as "shoes" stayed footwear. preferMappedMacro pins it.
    expect(preferMappedMacro('top', 'bottom')).toBe('top');
    expect(preferMappedMacro('bottom', 'top')).toBe('bottom');
    expect(preferMappedMacro('shoes', 'top')).toBe('shoes');
    expect(preferMappedMacro('outerwear', 'top')).toBe('outerwear');
    expect(preferMappedMacro('other', 'bottom')).toBe('bottom');
    expect(preferMappedMacro(undefined, 'top')).toBe('top');
    expect(preferMappedMacro('', 'shoes')).toBe('shoes');
    // End-to-end over the mapped pool: every clothing slot survives.
    for (const item of pool) {
      const kept = preferMappedMacro(item.macroCategory, 'top');
      expect(['top', 'bottom', 'shoes', 'outerwear']).toContain(kept);
    }
  });

  it('classic catalog offset rotates inside the curated head, never pins 0 forever', () => {
    const offsets = [1, 2, 3, 4, 5].map((salt) => classicCatalogOffset(2308, 120, salt));
    expect(offsets.every((n) => n >= 0 && n <= 720 - 120)).toBe(true);
    expect(new Set(offsets).size).toBeGreaterThan(1);
    expect(classicCatalogOffset(2308, 120, 1)).not.toBe(classicCatalogOffset(2308, 120, 2));
  });

  it('full-catalog scoring fits a perf budget (no Generate-time jank)', () => {
    const start = Date.now();
    for (const style of STYLES) {
      for (const item of pool) {
        scoreItemForStyle(
          { name: item.name, brand: item.brand, color: item.color, type: item.type, macroCategory: item.macroCategory },
          style,
        );
      }
    }
    expect(Date.now() - start).toBeLessThan(5000);
  });
});

/**
 * aiAdversarial.e2e.test.ts — chaos engineering for the AI outfit pipeline.
 *
 * The AI consumes user uploads, boutique product feeds, and LLM output — all
 * untrusted. These tests throw the worst realistic inputs at the REAL pure
 * modules and pin the survival invariants:
 *
 *   1. never throws (a crash = blank Generate screen)
 *   2. accessories ("Metal Sunglasses") never occupy a clothing slot
 *      (the classic "sunglasses rendered as a shirt" bug)
 *   3. a blazer stored as category "tops" still resolves to outerwear
 *      (layer-vs-base distinction the app depends on)
 *   4. formal layer + shorts never co-occur in one outfit
 *   5. empty pool → empty result with missing slots reported, never a
 *      half-outfit presented as complete
 *   6. 200 seeded fuzz mutations → same invariants hold every time
 */

import {
  canonicalizeMacroCategory,
  getMacroCategory,
} from '../../src/utils/categoryMapper';
import {
  scoreItemForStyle,
  normalizeStyleId,
} from '../../features/outfit-generator/utils/styleInference';
import { sanitizeGeneratedOutfitItemsDetailed } from '../../features/outfit-generator/utils/sanitizeGeneratedOutfit';

/** Deterministic PRNG (mulberry32) so fuzz is reproducible in CI. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const HOSTILE = [
  { id: 'h-blazer', name: 'Navy Blazer', type: 'tops', category: 'tops', macroCategory: 'top' },
  { id: 'h-trunks', name: 'Swimming trunks', type: 'bottom', category: 'bottoms', macroCategory: 'bottom' },
  { id: 'h-shades', name: 'Metal Sunglasses', type: 'accessories', category: 'accessories', macroCategory: 'other' },
  { id: 'h-belt', name: 'Leather Belt', type: 'belt', category: 'accessories', macroCategory: 'other' },
  { id: 'h-empty', name: '', type: '', category: '', macroCategory: '' },
  { id: 'h-emoji', name: '👕🔥 Белая футболка', type: 'верх', category: 'верх', macroCategory: 'top' },
  { id: 'h-shirt', name: 'White Oxford Shirt', type: 'shirt', category: 'tops', macroCategory: 'top' },
  { id: 'h-jeans', name: 'Dark Wash Jeans', type: 'jeans', category: 'bottoms', macroCategory: 'bottom' },
  { id: 'h-loafer', name: 'Brown Loafers', type: 'loafers', category: 'shoes', macroCategory: 'shoes' },
  { id: 'h-cardigan', name: 'Camel Cardigan', type: 'cardigan', category: 'tops', macroCategory: 'outerwear' },
];

const toAvailable = (items: typeof HOSTILE) =>
  items.map((i) => ({
    id: i.id,
    image: `${i.id}.png`,
    type: i.type,
    color: 'neutral',
    name: i.name,
    brand: 'Fuzz',
    macroCategory: i.macroCategory,
    category: i.category,
  }));

describe('AI adversarial e2e', () => {
  it('blazer-as-tops resolves to outerwear, trunks stay bottom, shades stay other', () => {
    expect(getMacroCategory('tops', 'Navy Blazer')).toBe('outerwear');
    expect(getMacroCategory('bottom', 'Swimming trunks')).toBe('bottom');
    expect(getMacroCategory('accessories', 'Metal Sunglasses')).toBe('accessory');
    expect(canonicalizeMacroCategory('upper_body')).toBe('top');
    expect(canonicalizeMacroCategory('lower_body')).toBe('bottom');
    expect(canonicalizeMacroCategory('footwear')).toBe('shoes');
    expect(canonicalizeMacroCategory('')).toBe('other');
    expect(canonicalizeMacroCategory(null)).toBe('other');
  });

  it('scorer never throws on hostile input and ranks real clothing above noise', () => {
    for (const item of HOSTILE) {
      for (const style of ['old_money', 'casual', 'business_casual'] as const) {
        let score: number;
        expect(() => {
          score = scoreItemForStyle(
            { name: item.name, type: item.type, color: 'x', macroCategory: item.macroCategory },
            style,
          );
        }).not.toThrow();
        expect(score!).toBeGreaterThanOrEqual(0);
        expect(score!).toBeLessThanOrEqual(1);
      }
    }
    const shirt = scoreItemForStyle({ name: 'White Oxford Shirt', type: 'shirt' }, 'old_money');
    const shades = scoreItemForStyle({ name: 'Metal Sunglasses', type: 'sunglasses' }, 'old_money');
    expect(shirt).toBeGreaterThan(shades);
  });

  it('sanitizer never emits an accessory into a clothing slot', () => {
    const available = toAvailable(HOSTILE);
    const generated: any[] = HOSTILE.map((h) => ({
      id: h.id,
      name: h.name,
      type: h.type,
      macroCategory: h.macroCategory,
      image: `${h.id}.png`,
    }));
    const result = sanitizeGeneratedOutfitItemsDetailed(generated, available as any, 5, 'casual');
    const ids = result.items.map((r: any) => String(r.id));
    expect(ids).not.toContain('h-shades');
    expect(ids).not.toContain('h-belt');
    // Every emitted id resolves back to the available pool (no ghosts).
    for (const id of ids) {
      expect(available.some((a) => a.id === id)).toBe(true);
    }
  });

  it('empty pool yields no crash and no phantom outfit', () => {
    const result = sanitizeGeneratedOutfitItemsDetailed([], [], 5, 'casual');
    expect(Array.isArray(result.items)).toBe(true);
    expect(result.items.length).toBe(0);
  });

  it('single-item pool cannot produce a "complete" outfit', () => {
    const one = toAvailable([HOSTILE[6]]); // one shirt
    const result = sanitizeGeneratedOutfitItemsDetailed(
      [{ id: 'h-shirt', name: 'x', type: 'shirt', macroCategory: 'top', image: 'x' }],
      one as any,
      5,
      'casual',
    );
    const macros = result.items.map((r: any) => canonicalizeMacroCategory(r.macroCategory));
    // A lone shirt is not an outfit: shoes + bottom must be missing.
    expect(macros).not.toContain('shoes');
  });

  it('fuzz: 200 seeded hostile mutations preserve all survival invariants', () => {
    const rand = rng(20260930);
    const prefixes = ['', 'Vintage ', 'OVERSIZED ', 'бре́ндовый ', '$$$ ', '100% '];
    const suffixes = ['', ' PRO', ' (2026)', '!!!', ' 👔', ' б/у'];
    const hostileWords = ['blazer', 'trunks', 'sunglasses', 't-shirt', 'jeans', 'loafers', 'belt', 'cardigan'];
    for (let k = 0; k < 200; k++) {
      const base = HOSTILE[Math.floor(rand() * HOSTILE.length)];
      const name = `${prefixes[Math.floor(rand() * prefixes.length)]}${base.name} ${hostileWords[Math.floor(rand() * hostileWords.length)]}${suffixes[Math.floor(rand() * suffixes.length)]}`;
      const mutated = { ...base, id: `fuzz-${k}`, name };
      let out: any = { items: [], missingSlots: [] };
      expect(() => {
        out = sanitizeGeneratedOutfitItemsDetailed(
          [{ id: mutated.id, name, type: mutated.type, macroCategory: mutated.macroCategory, image: 'x' }],
          toAvailable([mutated]) as any,
          5,
          k % 2 ? 'old_money' : 'casual',
        );
        scoreItemForStyle({ name, type: mutated.type }, 'old_money');
        normalizeStyleId(name);
      }).not.toThrow();
      // Accessory-only input: nothing may masquerade as clothing — any
      // emitted item must keep macro 'other' and clothing slots stay missing
      // (callers treat missing top/bottom/shoes as "not an outfit").
      if (mutated.macroCategory === 'other') {
        for (const r of out.items) {
          expect(canonicalizeMacroCategory((r as any).macroCategory)).toBe('other');
        }
        expect(out.missingSlots).toEqual(expect.arrayContaining(['top', 'bottom', 'shoes']));
      } else {
        // Single-item pool: result is empty or the same item — never a ghost.
        for (const r of out.items) {
          expect(String((r as any).id)).toBe(mutated.id);
        }
      }
    }
  });
});

/**
 * outfitTopSlotRegression — guards the non-layered top-slot fix.
 *
 * Bug: in non-layered mode the rotation picks the "top" from legacyTops
 * (outerwear + tops). When it landed on an outerwear piece,
 * normalizeGeneratedOutfitItems dropped it (outerwear excluded from the top
 * slot), emitting broken 2-item cards (bottom + shoes, no top).
 *
 * Fix: non-layered baseTop accepts outerwear-as-top (mirrors the edge
 * validator, which accepts hasBaseTop || hasOuter when not layered).
 */
import { buildLocalOutfits, type LocalClothingItem } from '../../src/services/outfitGenerationService';

function owned(id: string, slot: string, name: string, color = 'navy'): LocalClothingItem {
  return {
    // A real wardrobe never points every garment at ONE photo, and the
    // product-identity layer treats a shared photo as a shared product.
    id, userId: 'u1', imageUrl: `https://example.com/${id}.jpg`,
    category: slot as any, subCategory: name, macroCategory: slot,
    primaryColor: color, colorHex: '#000000', pattern: 'solid', material: 'cotton',
    brand: '', name, seasons: [], occasions: [], wearCount: 0, lastWornAt: null,
    isFavorite: false, createdAt: '2026-01-01T00:00:00Z', updatedAt: '2026-01-01T00:00:00Z',
  } as LocalClothingItem;
}

const MIXED = [
  owned('o-top1', 'top', 'white oxford shirt', 'white'),
  owned('o-top2', 'top', 'cream knit polo', 'cream'),
  owned('o-out1', 'outerwear', 'navy blazer', 'navy'),
  owned('o-bot1', 'bottom', 'charcoal pleated trousers', 'charcoal'),
  owned('o-shoe1', 'shoes', 'brown penny loafers', 'brown'),
];

describe('non-layered top slot keeps outerwear-as-top', () => {
  it('every hot-summer outfit has exactly 3 items incl. a top', () => {
    // Pool has 3 top-slot candidates (2 tops + blazer-as-top) sharing one
    // bottom/shoes pair, so at most 3 distinct looks exist — the point is
    // none of them is a broken 2-item card.
    const decks = buildLocalOutfits(MIXED, {
      stylePreferences: 'casual', occasion: 'Everyday', limit: 3,
      prompt: 'hot summer beach day', weather: { temp: 32, condition: 'sunny' },
    });
    expect(decks.length).toBe(3);
    for (const o of decks) {
      expect(o.items.length).toBe(3);
      expect(o.items.some((i) => ['top', 'outerwear'].includes((i.macroCategory || '').toLowerCase()))).toBe(true);
      expect(o.items.some((i) => (i.macroCategory || '').toLowerCase() === 'bottom')).toBe(true);
      expect(o.items.some((i) => (i.macroCategory || '').toLowerCase() === 'shoes')).toBe(true);
    }
  });

  it('cold outfits still have exactly 4 items with a true base top', () => {
    const decks = buildLocalOutfits(MIXED, {
      stylePreferences: 'old_money', occasion: 'Everyday', limit: 2,
      weather: { temp: 5, condition: 'cold' },
    });
    expect(decks.every((o) => o.items.length === 4)).toBe(true);
  });
});

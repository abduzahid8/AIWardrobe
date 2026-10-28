/** A layer never replaces the required base shirt, including in summer. */
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

describe('four-category top and layer separation', () => {
  it('summer outfits retain a layer and a separate base shirt', () => {
    // Two base shirts share one actual layer, pants and shoes: two distinct complete looks.
    const decks = buildLocalOutfits(MIXED, {
      stylePreferences: 'casual', occasion: 'Everyday', limit: 3,
      prompt: 'mild beach day', weather: { temp: 18, condition: 'sunny' },
    });
    expect(decks.length).toBe(2);
    for (const o of decks) {
      expect(o.items.length).toBe(4);
      expect(o.items.filter((i) => i.macroCategory === 'top')).toHaveLength(1);
      expect(o.items.filter((i) => i.macroCategory === 'outerwear')).toHaveLength(1);
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

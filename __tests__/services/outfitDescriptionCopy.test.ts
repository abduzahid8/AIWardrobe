/**
 * Regression tests for outfit description copy: the style slot must render a
 * human-readable label ("Old Money", not "old_money") with the correct
 * indefinite article ("An Old Money…", not "A old_money…").
 */
import {
    buildLocalOutfits,
    type LocalClothingItem,
} from '../../src/services/outfitGenerationService';

let idCounter = 0;

function makeShopItem(slot: 'top' | 'bottom' | 'shoes' | 'outerwear'): LocalClothingItem {
    idCounter += 1;
    const subCategory =
        slot === 'top' ? 'shirt'
        : slot === 'bottom' ? 'trousers'
        : slot === 'shoes' ? 'loafers'
        : 'blazer';
    return {
        id: `shop_${idCounter}`,
        userId: 'user-1',
        name: `shop ${subCategory}`,
        imageUrl: 'https://example.com/shop.jpg',
        category: slot as any,
        subCategory,
        primaryColor: 'navy',
        colorHex: '#1e3a8a',
        pattern: 'solid',
        material: 'wool',
        seasons: [],
        occasions: [],
        wearCount: 0,
        lastWornAt: null,
        isFavorite: false,
        createdAt: '2026-01-01T00:00:00Z',
        updatedAt: '2026-01-01T00:00:00Z',
        macroCategory: slot,
        isShopItem: true,
    } as LocalClothingItem;
}

function shopPool(): LocalClothingItem[] {
    idCounter = 0;
    return [
        makeShopItem('outerwear'),
        makeShopItem('top'),
        makeShopItem('top'),
        makeShopItem('bottom'),
        makeShopItem('bottom'),
        makeShopItem('shoes'),
        makeShopItem('shoes'),
    ];
}

describe('outfit description copy', () => {
    it('shop deck renders a humanized style label with correct article', () => {
        const outfits = buildLocalOutfits(shopPool(), {
            stylePreferences: 'old_money',
            occasion: 'Team Collaboration',
            weather: { temp: 18, condition: 'clear' },
            limit: 1,
        });
        expect(outfits.length).toBeGreaterThan(0);
        expect(outfits[0].description).toBe('An Old Money look curated from the shop.');
    });

    it('business_casual maps to its display label', () => {
        const outfits = buildLocalOutfits(shopPool(), {
            stylePreferences: 'business_casual',
            occasion: 'Work',
            weather: { temp: 18, condition: 'clear' },
            limit: 1,
        });
        expect(outfits.length).toBeGreaterThan(0);
        expect(outfits[0].description).toBe('A Modern Professional look curated from the shop.');
    });

    it('wardrobe deck uses the same label rules', () => {
        const owned = shopPool().map((item, i) => ({ ...item, id: `w${i}`, isShopItem: false }));
        const outfits = buildLocalOutfits(owned, {
            stylePreferences: 'old_money',
            occasion: 'Work',
            weather: { temp: 18, condition: 'clear' },
            limit: 1,
        });
        expect(outfits.length).toBeGreaterThan(0);
        expect(outfits[0].description).toBe('An Old Money look built around your wardrobe.');
    });
});

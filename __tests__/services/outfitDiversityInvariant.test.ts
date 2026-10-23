/**
 * Production-path diversity invariant for buildLocalOutfits (the local
 * fallback + Home daily-deck builder).
 *
 * The invariant is PRODUCT uniqueness, not archetype variety:
 *   - with enough valid inventory, no non-anchor product appears twice;
 *   - the same archetype is valid when every actual product is unique;
 *   - genuinely thin inventory may repeat, but still produces a deck.
 */
import {
    buildLocalOutfits,
    type LocalClothingItem,
} from '../../src/services/outfitGenerationService';
import {
    analyzeDeckProductDiversity,
    productIdentityKeys,
} from '../../features/outfit-generator/utils/outfitDiversity';

let n = 0;
function item(slot: 'top' | 'bottom' | 'shoes' | 'outerwear', name: string): LocalClothingItem {
    n += 1;
    return {
        id: `d${n}`,
        userId: 'u1',
        name,
        imageUrl: `https://example.com/${n}.jpg`,
        category: slot as any,
        subCategory: slot === 'top' ? 'shirt' : slot === 'bottom' ? 'trousers' : slot === 'shoes' ? 'loafers' : 'blazer',
        primaryColor: 'navy',
        colorHex: '#001f3f',
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

function largePool(): LocalClothingItem[] {
    n = 0;
    const pool: LocalClothingItem[] = [];
    for (const p of ['alpha', 'beta', 'gamma', 'delta']) {
        pool.push(
            item('outerwear', `${p} wool blazer`),
            item('top', `${p} oxford shirt`),
            item('bottom', `${p} wool trousers`),
            item('shoes', `${p} leather loafers`),
        );
    }
    return pool;
}

/** A pool with genuinely different garment TYPES per slot. */
function variedPool(): LocalClothingItem[] {
    n = 0;
    const pool: LocalClothingItem[] = [];
    for (const p of ['alpha', 'beta', 'gamma']) {
        pool.push(
            item('outerwear', `${p} wool blazer`),
            item('outerwear', `${p} suede jacket`),
            item('outerwear', `${p} cardigan`),
            item('top', `${p} oxford shirt`),
            item('top', `${p} knit polo`),
            item('top', `${p} fine gauge knit`),
            item('bottom', `${p} wool trousers`),
            item('bottom', `${p} chinos`),
            item('bottom', `${p} dark jeans`),
            item('shoes', `${p} leather loafers`),
            item('shoes', `${p} derby shoes`),
            item('shoes', `${p} minimal sneakers`),
        );
    }
    return pool;
}

describe('buildLocalOutfits diversity invariant', () => {
    it('3 looks from a large pool reuse zero non-anchor products', () => {
        for (let run = 0; run < 5; run++) {
            const outfits = buildLocalOutfits(largePool(), {
                stylePreferences: 'classic',
                occasion: 'work',
                weather: { temp: 12, condition: 'cool' },
                limit: 3,
            });
            expect(outfits).toHaveLength(3);
            const diversity = analyzeDeckProductDiversity(
                outfits.map((outfit) => outfit.items as any[]),
            );
            expect(diversity.nonAnchorProductReuseCount).toBe(0);
            const keys = new Set(outfits.flatMap((outfit) =>
                outfit.items.flatMap((item) => productIdentityKeys(item)),
            ));
            expect(keys.size).toBeGreaterThan(0);
        }
    });

    it('3 looks from a varied pool stay product-unique even when they share an archetype', () => {
        for (let run = 0; run < 5; run++) {
            const outfits = buildLocalOutfits(variedPool(), {
                stylePreferences: 'casual',
                occasion: 'Everyday',
                weather: { temp: 16, condition: 'mild' },
                limit: 3,
            });
            expect(outfits).toHaveLength(3);
            const diversity = analyzeDeckProductDiversity(
                outfits.map((outfit) => outfit.items as any[]),
            );
            expect(diversity.nonAnchorProductReuseCount).toBe(0);
        }
    });

    it('a pool with only one garment type per slot still produces a deck', () => {
        n = 0;
        const pool: LocalClothingItem[] = [];
        for (const p of ['alpha', 'beta', 'gamma']) {
            pool.push(
                item('outerwear', `${p} wool blazer`),
                item('top', `${p} oxford shirt`),
                item('bottom', `${p} wool trousers`),
                item('shoes', `${p} leather loafers`),
            );
        }
        const outfits = buildLocalOutfits(pool, {
            stylePreferences: 'classic',
            occasion: 'work',
            weather: { temp: 12, condition: 'cool' },
            limit: 3,
        });
        expect(outfits.length).toBeGreaterThan(0);
        for (const outfit of outfits) {
            expect(outfit.items.length).toBeGreaterThanOrEqual(3);
        }
    });
});

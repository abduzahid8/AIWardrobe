/**
 * Production-path diversity invariant for buildLocalOutfits (the local
 * fallback + Home daily-deck builder): with enough valid inventory, any
 * two looks share at most 1 item — never the same blazer AND trousers.
 */
import {
    buildLocalOutfits,
    type LocalClothingItem,
} from '../../src/services/outfitGenerationService';

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

describe('buildLocalOutfits diversity invariant', () => {
    it('3 looks from a large pool share at most 1 item pairwise', () => {
        for (let run = 0; run < 5; run++) {
            const outfits = buildLocalOutfits(largePool(), {
                stylePreferences: 'classic',
                occasion: 'work',
                weather: { temp: 12, condition: 'cool' },
                limit: 3,
            });
            expect(outfits).toHaveLength(3);
            const idSets = outfits.map((o) => new Set(o.items.map((i) => String(i.id))));
            for (let i = 0; i < 3; i++) {
                for (let j = i + 1; j < 3; j++) {
                    const shared = [...idSets[i]].filter((id) => idSets[j].has(id));
                    expect(shared.length).toBeLessThanOrEqual(1);
                }
            }
            // No item appears in all three looks.
            const inAll = [...idSets[0]].filter((id) => idSets.every((s) => s.has(id)));
            expect(inAll).toHaveLength(0);
        }
    });
});

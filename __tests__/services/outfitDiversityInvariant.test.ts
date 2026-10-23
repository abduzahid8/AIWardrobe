/**
 * Production-path diversity invariant for buildLocalOutfits (the local
 * fallback + Home daily-deck builder).
 *
 * Two layers are asserted:
 *   1. PRODUCT diversity — with enough valid inventory, any two looks share
 *      at most 1 item (never the same blazer AND trousers).
 *   2. VISUAL diversity — the three looks must not be three colourways of
 *      one outfit formula (blazer + shirt + tailored trousers + loafers).
 *      Different product ids are NOT enough.
 */
import {
    buildLocalOutfits,
    type LocalClothingItem,
} from '../../src/services/outfitGenerationService';
import {
    buildOutfitFingerprint,
    MAX_VISUAL_SIMILARITY,
    sameArchetype,
    visualSimilarity,
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

    it('3 looks from a varied pool are three different STYLING DIRECTIONS', () => {
        for (let run = 0; run < 5; run++) {
            const outfits = buildLocalOutfits(variedPool(), {
                stylePreferences: 'casual',
                occasion: 'Everyday',
                weather: { temp: 16, condition: 'mild' },
                limit: 3,
            });
            expect(outfits).toHaveLength(3);
            const fingerprints = outfits.map((o) => buildOutfitFingerprint(o.items as any[]));
            const archetypes = fingerprints.map((f) => f.archetype);
            // Not three colourways of one formula.
            expect(new Set(archetypes).size).toBeGreaterThan(1);
            for (let i = 0; i < 3; i++) {
                for (let j = i + 1; j < 3; j++) {
                    expect(sameArchetype(fingerprints[i], fingerprints[j])).toBe(false);
                    expect(visualSimilarity(fingerprints[i], fingerprints[j]))
                        .toBeLessThan(MAX_VISUAL_SIMILARITY);
                }
            }
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

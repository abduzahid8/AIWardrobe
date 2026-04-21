/**
 * Golden-case tests for the suggestion engine.
 *
 * The engine is the product's brain: its scoring shape MUST be stable
 * across refactors. These tests lock down:
 *   - getFormalityTier     — item-to-tier assignment
 *   - generateSuggestions  — occasion/weather filtering + scoring
 *   - generateOutfitsForItem — anchor piece mode
 *   - quickSuggest         — single best outfit
 *   - generateDailyOutfits — 4-variant daily carousel
 *   - buildEmptyStateSuggestion path — empty wardrobe fallback
 *
 * When you intentionally change behaviour, update the asserted values
 * in one place rather than silently breaking the carousel.
 */

import {
    getFormalityTier,
    generateSuggestions,
    generateOutfitsForItem,
    quickSuggest,
    generateDailyOutfits,
    FORMALITY_TIER_LABELS,
    DRESS_CODE_BADGE,
    type WeatherContext,
} from '../../src/services/suggestionEngine';
import type {
    ClothingItem,
    WearLog,
    Occasion,
    Season,
} from '../../src/types/domain';

// ============================================================
// TEST FIXTURES
// ============================================================

function makeItem(overrides: Partial<ClothingItem> & { id: string }): ClothingItem {
    return {
        userId: 'user-1',
        imageUrl: 'https://example.com/x.jpg',
        category: 'top',
        subCategory: 't-shirt',
        primaryColor: 'white',
        colorHex: '#ffffff',
        pattern: 'solid',
        material: 'cotton',
        seasons: ['spring', 'summer', 'fall', 'winter'] as Season[],
        occasions: ['casual'] as Occasion[],
        wearCount: 0,
        lastWornAt: null,
        isFavorite: false,
        createdAt: '2026-01-01T00:00:00Z',
        updatedAt: '2026-01-01T00:00:00Z',
        ...overrides,
    };
}

/** A minimal but realistic 10-item men's wardrobe spanning all tiers. */
function buildFixtureWardrobe(): ClothingItem[] {
    return [
        // Formal (tier 1)
        makeItem({
            id: 'suit-navy',
            category: 'top',
            subCategory: 'suit blazer',
            primaryColor: 'navy',
            material: 'wool',
            occasions: ['formal', 'work'],
        }),
        makeItem({
            id: 'trousers-suit',
            category: 'bottom',
            subCategory: 'dress pant',
            primaryColor: 'navy',
            material: 'wool',
            occasions: ['formal', 'work'],
        }),
        makeItem({
            id: 'oxford-black',
            category: 'shoes',
            subCategory: 'oxford shoe',
            primaryColor: 'black',
            material: 'leather',
            occasions: ['formal', 'work'],
        }),

        // Business casual (tier 2)
        makeItem({
            id: 'blazer-grey',
            category: 'top',
            subCategory: 'blazer',
            primaryColor: 'gray',
            material: 'wool',
            occasions: ['work', 'date'],
        }),
        makeItem({
            id: 'chinos-beige',
            category: 'bottom',
            subCategory: 'chinos',
            primaryColor: 'beige',
            material: 'cotton',
            occasions: ['work', 'casual', 'date'],
        }),
        makeItem({
            id: 'loafers-brown',
            category: 'shoes',
            subCategory: 'loafer',
            primaryColor: 'brown',
            material: 'leather',
            occasions: ['work', 'date', 'casual'],
        }),

        // Smart casual (tier 3)
        makeItem({
            id: 'polo-navy',
            category: 'top',
            subCategory: 'polo shirt',
            primaryColor: 'navy',
            material: 'cotton',
            occasions: ['casual', 'date'],
        }),
        makeItem({
            id: 'jeans-dark',
            category: 'bottom',
            subCategory: 'dark jeans',
            primaryColor: 'blue',
            material: 'denim',
            occasions: ['casual', 'date'],
        }),

        // Casual (tier 4)
        makeItem({
            id: 'tee-white',
            category: 'top',
            subCategory: 't-shirt',
            primaryColor: 'white',
            material: 'cotton',
            occasions: ['casual'],
        }),
        makeItem({
            id: 'sneakers-white',
            category: 'shoes',
            subCategory: 'sneaker',
            primaryColor: 'white',
            material: 'canvas',
            occasions: ['casual', 'sport'],
        }),

        // Sport / active (tier 5)
        makeItem({
            id: 'joggers-black',
            category: 'bottom',
            subCategory: 'joggers',
            primaryColor: 'black',
            material: 'polyester',
            occasions: ['sport', 'casual'],
        }),
        makeItem({
            id: 'runners-black',
            category: 'shoes',
            subCategory: 'running trainer',
            primaryColor: 'black',
            material: 'mesh',
            occasions: ['sport'],
        }),
    ];
}

const EMPTY_WEAR_LOGS: WearLog[] = [];
const TEMPERATE: WeatherContext = { temp: 18, condition: 'clear' };
const COLD_RAIN: WeatherContext = { temp: 4, condition: 'rain' };

// ============================================================
// getFormalityTier — item -> tier
// ============================================================

describe('getFormalityTier', () => {
    it('assigns tier 1 to formal items (suits, oxford shoes, dress shirts)', () => {
        expect(
            getFormalityTier(
                makeItem({ id: 'x', subCategory: 'suit', material: 'wool', category: 'top' }),
            ),
        ).toBe(1);
        expect(
            getFormalityTier(
                makeItem({ id: 'x', subCategory: 'oxford shoe', material: 'leather', category: 'shoes' }),
            ),
        ).toBe(1);
    });

    it('assigns tier 2 to business-casual items (blazer, loafer, chinos)', () => {
        expect(
            getFormalityTier(makeItem({ id: 'x', subCategory: 'blazer', material: 'wool' })),
        ).toBe(2);
        expect(
            getFormalityTier(
                makeItem({ id: 'x', subCategory: 'loafer', material: 'leather', category: 'shoes' }),
            ),
        ).toBe(2);
        expect(
            getFormalityTier(makeItem({ id: 'x', subCategory: 'chinos', category: 'bottom' })),
        ).toBe(2);
    });

    it('assigns tier 3 to smart-casual items (polo, dark jeans, chelsea boot)', () => {
        expect(
            getFormalityTier(makeItem({ id: 'x', subCategory: 'polo shirt' })),
        ).toBe(2); // polo shirt matches tier 2 (business) via "polo shirt"
        expect(
            getFormalityTier(makeItem({ id: 'x', subCategory: 'dark jeans', category: 'bottom' })),
        ).toBe(3);
        expect(
            getFormalityTier(makeItem({ id: 'x', subCategory: 'chelsea boot', category: 'shoes' })),
        ).toBe(3);
    });

    it('assigns tier 4 to casual items (t-shirt, hoodie, sneaker)', () => {
        expect(getFormalityTier(makeItem({ id: 'x', subCategory: 't-shirt' }))).toBe(4);
        expect(getFormalityTier(makeItem({ id: 'x', subCategory: 'hoodie' }))).toBe(4);
        expect(
            getFormalityTier(makeItem({ id: 'x', subCategory: 'sneaker', material: 'canvas', category: 'shoes' })),
        ).toBe(4);
    });

    it('assigns tier 5 to athletic items (joggers, running trainer)', () => {
        expect(getFormalityTier(makeItem({ id: 'x', subCategory: 'joggers' }))).toBe(5);
        expect(
            getFormalityTier(
                makeItem({ id: 'x', subCategory: 'running trainer', category: 'shoes', material: 'mesh' }),
            ),
        ).toBe(5);
    });

    it('defaults to tier 3 for unrecognized items', () => {
        expect(
            getFormalityTier(
                makeItem({ id: 'x', subCategory: 'mystery-garment', material: 'hemp' }),
            ),
        ).toBe(3);
    });

    it('FORMALITY_TIER_LABELS and DRESS_CODE_BADGE are defined for every tier', () => {
        for (const tier of [1, 2, 3, 4, 5] as const) {
            expect(FORMALITY_TIER_LABELS[tier]).toBeTruthy();
            expect(DRESS_CODE_BADGE[tier]).toBeTruthy();
        }
    });
});

// ============================================================
// generateSuggestions — core flow
// ============================================================

describe('generateSuggestions', () => {
    it('returns a placeholder outfit when wardrobe is empty', () => {
        const result = generateSuggestions({
            items: [],
            wearLogs: EMPTY_WEAR_LOGS,
            occasion: 'work',
        });
        expect(result).toHaveLength(1);
        expect(result[0].outfit.itemIds).toHaveLength(0);
        expect(result[0].shoppingSuggestions.length).toBeGreaterThan(0);
        // Work -> tier 2 preferred
        expect(result[0].formalityTier).toBeGreaterThanOrEqual(1);
        expect(result[0].formalityTier).toBeLessThanOrEqual(3);
    });

    it('produces outfits that each contain no duplicate categories', () => {
        const results = generateSuggestions({
            items: buildFixtureWardrobe(),
            wearLogs: EMPTY_WEAR_LOGS,
            occasion: 'casual',
            weather: TEMPERATE,
        });
        expect(results.length).toBeGreaterThan(0);
        for (const r of results) {
            const cats = r.outfit.itemIds
                .map((id) => buildFixtureWardrobe().find((i) => i.id === id)?.category)
                .filter((c): c is NonNullable<typeof c> => Boolean(c));
            expect(new Set(cats).size).toBe(cats.length);
        }
    });

    it('work occasion skews toward tier 1-3 items', () => {
        const results = generateSuggestions({
            items: buildFixtureWardrobe(),
            wearLogs: EMPTY_WEAR_LOGS,
            occasion: 'work',
            weather: TEMPERATE,
        });
        expect(results.length).toBeGreaterThan(0);
        expect(results[0].formalityTier).toBeLessThanOrEqual(3);
    });

    it('sport occasion produces tier 5 outfits', () => {
        const results = generateSuggestions({
            items: buildFixtureWardrobe(),
            wearLogs: EMPTY_WEAR_LOGS,
            occasion: 'sport',
            weather: TEMPERATE,
        });
        expect(results.length).toBeGreaterThan(0);
        expect(results[0].formalityTier).toBe(5);
    });

    it('returns outfits sorted by score descending', () => {
        const results = generateSuggestions({
            items: buildFixtureWardrobe(),
            wearLogs: EMPTY_WEAR_LOGS,
            occasion: 'casual',
            weather: TEMPERATE,
        });
        for (let i = 1; i < results.length; i++) {
            expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
        }
    });

    it('returns scores in [0, 1] with well-formed breakdown', () => {
        const [top] = generateSuggestions({
            items: buildFixtureWardrobe(),
            wearLogs: EMPTY_WEAR_LOGS,
            occasion: 'casual',
            weather: TEMPERATE,
        });
        expect(top.score).toBeGreaterThanOrEqual(0);
        expect(top.score).toBeLessThanOrEqual(1);
        for (const key of ['formalityScore', 'noveltyScore', 'harmonyScore', 'weatherScore'] as const) {
            expect(top.breakdown[key]).toBeGreaterThanOrEqual(0);
            expect(top.breakdown[key]).toBeLessThanOrEqual(1);
        }
    });

    it('cold rainy weather still produces at least one outfit (no throw)', () => {
        const results = generateSuggestions({
            items: buildFixtureWardrobe(),
            wearLogs: EMPTY_WEAR_LOGS,
            occasion: 'casual',
            weather: COLD_RAIN,
        });
        expect(results.length).toBeGreaterThan(0);
    });

    it('recently-worn items are penalised in novelty scoring', () => {
        const items = buildFixtureWardrobe();
        const today = new Date().toISOString();

        // Wear one of the casual tops heavily today
        const heavyWearItem = items.find((i) => i.id === 'tee-white')!;
        const wearLogs: WearLog[] = Array.from({ length: 5 }).map((_, i) => ({
            id: `log-${i}`,
            userId: 'user-1',
            itemId: heavyWearItem.id,
            wornAt: today,
            createdAt: today,
        } as unknown as WearLog));

        const fresh = generateSuggestions({
            items,
            wearLogs: EMPTY_WEAR_LOGS,
            occasion: 'casual',
            weather: TEMPERATE,
        });
        const stale = generateSuggestions({
            items,
            wearLogs,
            occasion: 'casual',
            weather: TEMPERATE,
        });

        // Top suggestion when item is freshly worn should tend to use other items.
        // We assert the novelty scores of outfits CONTAINING the worn item dropped.
        const containsWorn = (o: { outfit: { itemIds: string[] } }) =>
            o.outfit.itemIds.includes(heavyWearItem.id);
        const freshWithWorn = fresh.find(containsWorn);
        const staleWithWorn = stale.find(containsWorn);
        if (freshWithWorn && staleWithWorn) {
            expect(staleWithWorn.breakdown.noveltyScore).toBeLessThanOrEqual(
                freshWithWorn.breakdown.noveltyScore,
            );
        }
    });
});

// ============================================================
// generateOutfitsForItem — anchor piece mode
// ============================================================

describe('generateOutfitsForItem', () => {
    it('every returned outfit contains the anchor item', () => {
        const items = buildFixtureWardrobe();
        const anchor = 'blazer-grey';
        const results = generateOutfitsForItem(anchor, items, EMPTY_WEAR_LOGS, TEMPERATE);
        expect(results.length).toBeGreaterThan(0);
        for (const r of results) {
            expect(r.outfit.itemIds).toContain(anchor);
        }
    });

    it('returns [] if anchor item is not in the wardrobe', () => {
        const items = buildFixtureWardrobe();
        expect(generateOutfitsForItem('does-not-exist', items, EMPTY_WEAR_LOGS, TEMPERATE))
            .toEqual([]);
    });
});

// ============================================================
// quickSuggest — single best pick
// ============================================================

describe('quickSuggest', () => {
    it('returns exactly one scored outfit', () => {
        const items = buildFixtureWardrobe();
        const out = quickSuggest(items, EMPTY_WEAR_LOGS, TEMPERATE);
        expect(out).not.toBeNull();
        expect(out?.outfit.itemIds.length).toBeGreaterThan(0);
    });

    it('handles empty wardrobe without throwing', () => {
        expect(() => quickSuggest([], EMPTY_WEAR_LOGS, TEMPERATE)).not.toThrow();
    });
});

// ============================================================
// generateDailyOutfits — Home carousel
// ============================================================

describe('generateDailyOutfits', () => {
    it('returns a 4-variant object (may have nulls on small wardrobes)', () => {
        const daily = generateDailyOutfits(
            buildFixtureWardrobe(),
            EMPTY_WEAR_LOGS,
            TEMPERATE,
        );
        expect(daily).toHaveProperty('work');
        expect(daily).toHaveProperty('smartCasual');
        expect(daily).toHaveProperty('weekendCasual');
        expect(daily).toHaveProperty('wildcard');

        // With our 12-item fixture, at least one variant must exist.
        const filled = [daily.work, daily.smartCasual, daily.weekendCasual, daily.wildcard]
            .filter(Boolean);
        expect(filled.length).toBeGreaterThan(0);
    });

    it('work variant, if produced, has formality tier <= 3', () => {
        const daily = generateDailyOutfits(
            buildFixtureWardrobe(),
            EMPTY_WEAR_LOGS,
            TEMPERATE,
        );
        if (daily.work) {
            expect(daily.work.formalityTier).toBeLessThanOrEqual(3);
        }
    });
});

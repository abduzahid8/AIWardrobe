/**
 * Repro / regression tests for "only a single look gets generated, the
 * clothing doesn't change" with ~10 wardrobe items.
 *
 * Drives the REAL buildLocalOutfits() builder (same code path as the
 * generateOutfitsLocally fallback and Home daily decks) with realistic
 * wardrobes across styles/weather and asserts deck size + pairwise variety.
 */
import {
    buildLocalOutfits,
    type LocalClothingItem,
    type GenerateOutfitsParams,
} from '../../src/services/outfitGenerationService';

let idCounter = 0;

function makeItem(
    slot: 'top' | 'bottom' | 'shoes' | 'outerwear',
    name: string,
    extra: Partial<LocalClothingItem> = {},
): LocalClothingItem {
    idCounter += 1;
    const subCategory =
        slot === 'top' ? 't-shirt'
        : slot === 'bottom' ? 'jeans'
        : slot === 'shoes' ? 'sneakers'
        : 'jacket';
    return {
        id: `w${idCounter}`,
        userId: 'user-1',
        imageUrl: `https://example.com/${name.replace(/\s+/g, '-')}.jpg`,
        category: slot as any,
        subCategory,
        primaryColor: 'blue',
        colorHex: '#1e3a8a',
        pattern: 'solid',
        material: 'cotton',
        seasons: [],
        occasions: [],
        wearCount: 0,
        lastWornAt: null,
        isFavorite: false,
        createdAt: '2026-01-01T00:00:00Z',
        updatedAt: '2026-01-01T00:00:00Z',
        macroCategory: slot,
        ...extra,
    } as LocalClothingItem;
}

/** The user's scenario: ~10 different owned items across all slots. */
function tenItemWardrobe(): LocalClothingItem[] {
    idCounter = 0;
    return [
        makeItem('top', 'white tee'),
        makeItem('top', 'navy polo'),
        makeItem('top', 'grey shirt'),
        makeItem('bottom', 'blue jeans'),
        makeItem('bottom', 'khaki chinos'),
        makeItem('bottom', 'black trousers'),
        makeItem('shoes', 'white sneakers'),
        makeItem('shoes', 'brown loafers'),
        makeItem('outerwear', 'denim jacket'),
        makeItem('outerwear', 'beige trench'),
    ];
}

function sharedCount(a: string[], b: Set<string>): number {
    let n = 0;
    for (const id of a) if (b.has(id)) n += 1;
    return n;
}

function maxShared(ids: string[][]): number {
    let max = 0;
    for (let i = 0; i < ids.length; i++) {
        for (let j = i + 1; j < ids.length; j++) {
            max = Math.max(max, sharedCount(ids[i], new Set(ids[j])));
        }
    }
    return max;
}

const baseParams = (overrides: Partial<GenerateOutfitsParams> = {}): GenerateOutfitsParams => ({
    stylePreferences: 'casual',
    occasion: 'Everyday',
    limit: 3,
    ...overrides,
});

describe('buildLocalOutfits variety (10-item wardrobe)', () => {
    it('warm weather: yields 3 complete 3-item looks', () => {
        const outfits = buildLocalOutfits(
            tenItemWardrobe(),
            baseParams({ weather: { temp: 24, condition: 'clear' } }),
        );
        expect(outfits.length).toBe(3);
        for (const o of outfits) {
            expect(o.items.length).toBe(3);
        }
    });

    it('warm weather: looks differ (no frozen clothing)', () => {
        const outfits = buildLocalOutfits(
            tenItemWardrobe(),
            baseParams({ weather: { temp: 24, condition: 'clear' } }),
        );
        const ids = outfits.map((o) => o.items.map((i: any) => String(i.id)));
        // 3-item looks sharing 2+ pieces read as "the same outfit".
        expect(maxShared(ids)).toBeLessThanOrEqual(1);
    });

    it('cool weather: yields 3 complete 4-item layered looks', () => {
        const outfits = buildLocalOutfits(
            tenItemWardrobe(),
            baseParams({ weather: { temp: 10, condition: 'cool' } }),
        );
        expect(outfits.length).toBe(3);
        for (const o of outfits) {
            expect(o.items.length).toBe(4);
        }
    });

    it('cool weather: layered looks differ by 2+ items', () => {
        const outfits = buildLocalOutfits(
            tenItemWardrobe(),
            baseParams({ weather: { temp: 10, condition: 'cool' } }),
        );
        const ids = outfits.map((o) => o.items.map((i: any) => String(i.id)));
        // 4-item looks sharing 3+ pieces read as "the same outfit".
        expect(maxShared(ids)).toBeLessThanOrEqual(2);
    });

    it('sparse wardrobe (1 per slot): still yields at least 1 complete look', () => {
        idCounter = 100;
        const sparse = [
            makeItem('top', 'lone tee'),
            makeItem('bottom', 'lone jeans'),
            makeItem('shoes', 'lone sneakers'),
        ];
        const outfits = buildLocalOutfits(
            sparse,
            baseParams({ weather: { temp: 24, condition: 'clear' } }),
        );
        expect(outfits.length).toBeGreaterThanOrEqual(1);
        expect(outfits[0].items.length).toBe(3);
    });

    it.each(['casual', 'minimalist', 'old_money', 'streetwear'])(
        'style %s with 10 items: yields 3 complete looks',
        (style) => {
            const outfits = buildLocalOutfits(
                tenItemWardrobe(),
                baseParams({ stylePreferences: style, weather: { temp: 24, condition: 'clear' } }),
            );
            expect(outfits.length).toBe(3);
        },
    );

    it('business_casual never fills a shop-only deck with owned wardrobe pieces', () => {
        expect(buildLocalOutfits(tenItemWardrobe(), baseParams({
            stylePreferences: 'business_casual', weather: { temp: 24, condition: 'clear' },
        }))).toEqual([]);
    });

    it('repeated taps: deck still covers distinct items (no single frozen look)', () => {
        const decks = [0, 1, 2].map(() =>
            buildLocalOutfits(
                tenItemWardrobe(),
                baseParams({ weather: { temp: 10, condition: 'cool' } }),
            ).map((o) => o.items.map((i: any) => String(i.id)).sort().join('|')),
        );
        const allSigs = new Set(decks.flat());
        // 3 taps x 3 looks should surface more than 1 unique combination.
        expect(allSigs.size).toBeGreaterThan(1);
    });

    it('thin slots (6 tops/2 bottoms/1 shoes/1 outerwear): still yields 3 looks', () => {
        idCounter = 200;
        const thin = [
            makeItem('top', 'tee one'),
            makeItem('top', 'tee two'),
            makeItem('top', 'polo three'),
            makeItem('top', 'shirt four'),
            makeItem('top', 'henley five'),
            makeItem('top', 'knit six'),
            makeItem('bottom', 'jeans one'),
            makeItem('bottom', 'chinos two'),
            makeItem('shoes', 'only sneakers'),
            makeItem('outerwear', 'only jacket'),
        ];
        const outfits = buildLocalOutfits(
            thin,
            baseParams({ weather: { temp: 10, condition: 'cool' } }),
        );
        expect(outfits.length).toBe(3);
    });

    it('thin slots (4/2/2/2): still yields 3 looks', () => {
        idCounter = 300;
        const thin = [
            makeItem('top', 'tee one'),
            makeItem('top', 'polo two'),
            makeItem('top', 'shirt three'),
            makeItem('top', 'henley four'),
            makeItem('bottom', 'jeans one'),
            makeItem('bottom', 'chinos two'),
            makeItem('shoes', 'sneakers one'),
            makeItem('shoes', 'loafers two'),
            makeItem('outerwear', 'jacket one'),
            makeItem('outerwear', 'coat two'),
        ];
        const outfits = buildLocalOutfits(
            thin,
            baseParams({ weather: { temp: 10, condition: 'cool' } }),
        );
        expect(outfits.length).toBe(3);
    });
});

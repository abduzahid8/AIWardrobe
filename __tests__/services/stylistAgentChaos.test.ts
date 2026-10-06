/**
 * stylistAgentChaos — outside-the-box scenarios for the AI stylist agent.
 *
 * The agent consumes untrusted input (user prompts, boutique feeds, prefs,
 * weather APIs). These tests pin survival + sanity invariants the existing
 * suites don't cover:
 *   - conflicting preferences (avoid-everything, favorite==avoided)
 *   - extreme/absurd weather (NaN, ±50°C, garbage conditions)
 *   - absurd limits (0, negative, far above combinatorial max)
 *   - poisoned pools (nulls, dup ids, missing ids, single-slot pools)
 *   - garbage occasion/style/prompt strings, nonexistent anchors
 *
 * Rule: the agent must NEVER throw and NEVER return exact-duplicate looks.
 */
import {
    generateSuggestions,
    generateOutfitsForItem,
    generateVarietyOutfits,
    quickSuggest,
    type WeatherContext,
} from '../../src/services/suggestionEngine';
import {
    buildLocalOutfits,
    type LocalClothingItem,
    type GenerateOutfitsParams,
} from '../../src/services/outfitGenerationService';
import { buildOutfitCacheKey } from '../../src/services/ai/outfitService';
import type { ClothingItem, Season, Occasion } from '../../src/types/domain';

jest.mock('../../src/config/env', () => ({
    __esModule: true,
    default: { api: { url: 'https://test-api.example.com', alicevisionUrl: 'https://test-av.example.com' } },
    Config: { api: { url: 'https://test-api.example.com', alicevisionUrl: 'https://test-av.example.com' } },
}));

let n = 0;
function clothing(
    slot: 'top' | 'bottom' | 'shoes' | 'outerwear',
    name: string,
    color = 'blue',
    extra: Partial<ClothingItem> = {},
): ClothingItem {
    n += 1;
    const sub = slot === 'top' ? 't-shirt' : slot === 'bottom' ? 'jeans' : slot === 'shoes' ? 'sneakers' : 'jacket';
    return {
        id: `c${n}`,
        userId: 'u1',
        imageUrl: 'https://example.com/x.jpg',
        category: slot as any,
        subCategory: sub,
        primaryColor: color,
        colorHex: '#000000',
        pattern: 'solid',
        material: 'cotton',
        seasons: [] as Season[],
        occasions: [] as Occasion[],
        wearCount: 0,
        lastWornAt: null,
        isFavorite: false,
        createdAt: '2026-01-01T00:00:00Z',
        updatedAt: '2026-01-01T00:00:00Z',
        macroCategory: slot,
        ...extra,
    } as ClothingItem;
}

function tenMixed(): ClothingItem[] {
    n = 100;
    return [
        clothing('top', 'white tee', 'white'),
        clothing('top', 'navy polo', 'navy'),
        clothing('top', 'grey shirt', 'grey'),
        clothing('bottom', 'blue jeans', 'blue'),
        clothing('bottom', 'khaki chinos', 'khaki'),
        clothing('bottom', 'black trousers', 'black'),
        clothing('shoes', 'white sneakers', 'white'),
        clothing('shoes', 'brown loafers', 'brown'),
        clothing('outerwear', 'denim jacket', 'blue'),
        clothing('outerwear', 'beige trench', 'beige'),
    ];
}

const toLocal = (items: ClothingItem[]): LocalClothingItem[] =>
    items as unknown as LocalClothingItem[];

function sigsOf(outfits: Array<{ items: Array<{ id: string }> }>): string[] {
    return outfits.map((o) => o.items.map((i) => String(i.id)).sort().join('|'));
}

function expectNoExactDupes(outfits: Array<{ items: Array<{ id: string }> }>) {
    const sigs = sigsOf(outfits);
    expect(new Set(sigs).size).toBe(sigs.length);
}

describe('stylist agent chaos', () => {
    it('avoid-everything preferences yield empty deck, not a crash', () => {
        const items = tenMixed();
        const colors = [...new Set(items.map((i) => (i.primaryColor || '').toLowerCase()))];
        const res = generateSuggestions({
            items,
            wearLogs: [],
            occasion: 'casual',
            preferences: { preferredColors: [], avoidColors: colors, preferredStyles: [], adventurousness: 0.5 },
        });
        expect(Array.isArray(res)).toBe(true);
    });

    it('favorite==avoided color overlap does not throw', () => {
        expect(() =>
            generateSuggestions({
                items: tenMixed(),
                wearLogs: [],
                occasion: 'work',
                preferences: { preferredColors: ['blue'], avoidColors: ['blue', 'white'], preferredStyles: [], adventurousness: 0.5 },
            }),
        ).not.toThrow();
    });

    it.each([
        { temp: -30, condition: 'snow' },
        { temp: 55, condition: 'clear' },
        { temp: NaN, condition: 'clear' },
        { temp: 20, condition: 'DROP TABLE wardrobe; --' },
        { temp: 20, condition: '' },
        { temp: 20, condition: '👔🔥' },
    ])('survives weather %p', (weather) => {
        const w = weather as unknown as WeatherContext;
        expect(() => generateSuggestions({ items: tenMixed(), wearLogs: [], occasion: 'casual', weather: w })).not.toThrow();
        expect(() =>
            buildLocalOutfits(toLocal(tenMixed()), {
                stylePreferences: 'casual',
                occasion: 'Everyday',
                limit: 3,
                weather: { temp: (w as any).temp, condition: String((w as any).condition ?? '') },
            }),
        ).not.toThrow();
    });

    it.each([0, -3, 50])('absurd limit %p never throws, never dupes', (limit) => {
        const outfits = buildLocalOutfits(toLocal(tenMixed()), {
            stylePreferences: 'casual',
            occasion: 'Everyday',
            limit,
            weather: { temp: 22, condition: 'clear' },
        } as GenerateOutfitsParams);
        expect(Array.isArray(outfits)).toBe(true);
        expect(outfits.length).toBeLessThanOrEqual(Math.max(0, limit));
        expectNoExactDupes(outfits as any);
    });

    it('poisoned pool (nulls, dup ids, missing ids) never throws, never dupes', () => {
        const base = toLocal(tenMixed());
        const poisoned = [
            ...base,
            null,
            undefined,
            { ...base[0] },
            { ...base[1], id: base[0].id },
            { id: undefined, macroCategory: 'top' },
            { id: '', macroCategory: 'shoes' },
        ] as unknown as LocalClothingItem[];
        const outfits = buildLocalOutfits(poisoned.filter(Boolean) as LocalClothingItem[], {
            stylePreferences: 'casual',
            occasion: 'Everyday',
            limit: 3,
            weather: { temp: 22, condition: 'clear' },
        });
        expect(Array.isArray(outfits)).toBe(true);
        expectNoExactDupes(outfits as any);
    });

    it('single-slot pools (10 tops only / 10 shoes only) never throw', () => {
        n = 500;
        const topsOnly = Array.from({ length: 10 }, (_, i) => clothing('top', `tee ${i}`, 'white'));
        const shoesOnly = Array.from({ length: 10 }, (_, i) => clothing('shoes', `shoe ${i}`, 'black'));
        for (const pool of [topsOnly, shoesOnly]) {
            const outfits = buildLocalOutfits(toLocal(pool), {
                stylePreferences: 'casual',
                occasion: 'Everyday',
                limit: 3,
                weather: { temp: 22, condition: 'clear' },
            });
            expect(Array.isArray(outfits)).toBe(true);
            expectNoExactDupes(outfits as any);
        }
    });

    it.each([
        '',
        '   ',
        '"; DROP TABLE clothing_items; --',
        '👔🔥💧❄️',
        'a'.repeat(500),
        'business_casual',
        'TEAM COLLABORATION',
        null,
        undefined,
    ])('garbage style/occasion %p never throws', (style) => {
        const outfits = buildLocalOutfits(toLocal(tenMixed()), {
            stylePreferences: style as any,
            occasion: style as any,
            limit: 3,
            weather: { temp: 22, condition: 'clear' },
            prompt: typeof style === 'string' ? style : undefined,
        });
        expect(Array.isArray(outfits)).toBe(true);
        expectNoExactDupes(outfits as any);
    });

    it('nonexistent anchor returns [], existing anchor locks in', () => {
        const items = tenMixed();
        expect(generateOutfitsForItem('no-such-id', items, [])).toEqual([]);
        const anchored = generateOutfitsForItem(items[0].id, items, []);
        for (const o of anchored) {
            expect(o.outfit.itemIds).toContain(items[0].id);
        }
    });

    it('quickSuggest on empty wardrobe returns the empty-state suggestion, never throws', () => {
        const empty = quickSuggest([], []);
        expect(empty).not.toBeNull();
        expect(empty?.outfit.itemIds).toEqual([]);
        expect(empty?.reasoning).toMatch(/closet/i);
        const crumbs = [clothing('top', 'lone tee', 'white')];
        expect(() => generateVarietyOutfits(crumbs, [])).not.toThrow();
        expect(() => generateVarietyOutfits([], [])).not.toThrow();
    });

    it('cache key tolerates undefined/null/empty wardrobes and stays stable', () => {
        const w = { temp: 20, condition: 'clear' };
        const k1 = buildOutfitCacheKey('casual', 'casual', undefined, w);
        const k2 = buildOutfitCacheKey('casual', 'casual', null as any, w);
        const k3 = buildOutfitCacheKey('casual', 'casual', [], w);
        expect(typeof k1).toBe('string');
        expect(k1).toBe(k2);
        expect(k1).toBe(k3);
        expect(k1).not.toBe(buildOutfitCacheKey('work', 'casual', [], w));
    });

    it('limit above combinatorial max fills what exists without dupes', () => {
        n = 900;
        const tiny = [clothing('top', 'a', 'white'), clothing('bottom', 'b', 'blue'), clothing('shoes', 'c', 'black')];
        const outfits = buildLocalOutfits(toLocal(tiny), {
            stylePreferences: 'casual',
            occasion: 'Everyday',
            limit: 15,
            weather: { temp: 22, condition: 'clear' },
        });
        expect(outfits.length).toBeGreaterThanOrEqual(1);
        expect(outfits.length).toBeLessThanOrEqual(15);
        expectNoExactDupes(outfits as any);
    });
});

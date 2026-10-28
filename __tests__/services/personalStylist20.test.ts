import {
    buildLocalOutfits,
    type LocalClothingItem,
} from '../../src/services/outfitGenerationService';
import {
    AGENT_PIPELINE,
    STYLE_PILLARS,
    STYLIST_GREETING,
    STYLIST_NAME,
    buildStylistSystemPrompt,
    getPillar,
    personaReply,
    resolvePersonaLang,
} from '../../src/services/ai/stylistPersona';
import { parseStylistIntent } from '../../features/outfit-generator/utils/stylistIntent';
import {
    rejectForClimate,
    rejectOutfitClimate,
    rejectOutfitIncoherence,
    styleJudgeReason,
} from '../../features/outfit-generator/utils/occasionRules';
import { generateSuggestions } from '../../src/services/suggestionEngine';
import { outfitShopSpend } from '../../src/utils/outfitBudget';

let sequence = 0;

const SLOT_SUBCATEGORY = {
    top: 'shirt',
    bottom: 'trousers',
    shoes: 'loafers',
    outerwear: 'blazer',
} as const;

type Slot = keyof typeof SLOT_SUBCATEGORY;

function garment(
    scope: 'own' | 'shop',
    slot: Slot,
    name: string,
    extra: Partial<LocalClothingItem> = {},
): LocalClothingItem {
    sequence += 1;
    return {
        id: `${scope}_${sequence}`,
        userId: 'stylist-suite',
        name,
        imageUrl: `https://cdn.example.com/${scope}-${sequence}.jpg`,
        category: slot as any,
        subCategory: SLOT_SUBCATEGORY[slot],
        primaryColor: 'navy',
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
        isShopItem: scope === 'shop',
        ...extra,
    } as LocalClothingItem;
}

function mixedWardrobe(): LocalClothingItem[] {
    return [
        garment('own', 'outerwear', 'Navy unstructured blazer'),
        garment('own', 'top', 'White oxford shirt'),
        garment('own', 'top', 'Navy knit polo'),
        garment('own', 'bottom', 'Charcoal wool trousers'),
        garment('own', 'bottom', 'Beige chinos'),
        garment('own', 'shoes', 'Brown penny loafers'),
        garment('own', 'shoes', 'White leather sneakers'),
        garment('shop', 'outerwear', 'Camel wool topcoat', { price: 120, currency: 'USD' }),
        garment('shop', 'top', 'Light blue oxford shirt', { price: 40, currency: 'USD' }),
        garment('shop', 'top', 'Grey merino sweater', { price: 60, currency: 'USD' }),
        garment('shop', 'bottom', 'Navy chinos', { price: 45, currency: 'USD' }),
        garment('shop', 'bottom', 'Dark rinse jeans', { price: 50, currency: 'USD' }),
        garment('shop', 'shoes', 'Black derby shoes', { price: 70, currency: 'USD' }),
        garment('shop', 'shoes', 'Minimal white sneakers', { price: 55, currency: 'USD' }),
    ];
}

function shopOnlyPool(): LocalClothingItem[] {
    return [
        garment('shop', 'outerwear', 'shop blazer', { price: 80, currency: 'USD' }),
        garment('shop', 'top', 'shop shirt', { price: 30, currency: 'USD' }),
        garment('shop', 'top', 'shop polo', { price: 25, currency: 'USD' }),
        garment('shop', 'bottom', 'shop trousers', { price: 40, currency: 'USD' }),
        garment('shop', 'bottom', 'shop chinos', { price: 35, currency: 'USD' }),
        garment('shop', 'shoes', 'shop loafers', { price: 60, currency: 'USD' }),
        garment('shop', 'shoes', 'shop sneakers', { price: 50, currency: 'USD' }),
    ];
}

function budgetPool(): LocalClothingItem[] {
    return [
        garment('shop', 'outerwear', 'light cotton jacket', { price: 20, currency: 'USD' }),
        garment('shop', 'top', 'cheap polo', { price: 25, currency: 'USD' }),
        garment('shop', 'top', 'luxury knit', { price: 400, currency: 'USD' }),
        garment('shop', 'bottom', 'cheap chinos', { price: 30, currency: 'USD' }),
        garment('shop', 'bottom', 'luxury trousers', { price: 400, currency: 'USD' }),
        garment('shop', 'shoes', 'cheap loafers', { price: 20, currency: 'USD' }),
        garment('shop', 'shoes', 'luxury loafers', { price: 400, currency: 'USD' }),
    ];
}

describe('personal stylist — 20 acceptance tests', () => {
    it('1. understands an English work request with style and budget', () => {
        expect(parseStylistIntent('Work tomorrow, old money, budget $200')).toEqual({
            style: 'old_money',
            occasion: 'work',
            budget: { amount: 200, currency: 'USD' },
        });
    });

    it('2. understands a Russian work request with style and budget', () => {
        const intent = parseStylistIntent('На работу, классика, 150$');
        expect(intent.style).toBe('classic');
        expect(intent.occasion).toBe('work');
        expect(intent.budget).toMatchObject({ amount: 150, currency: 'USD' });
    });

    it('3. understands an Uzbek travel request with a budget', () => {
        const intent = parseStylistIntent('sayohat uchun $300');
        expect(intent.style).toBeNull();
        expect(intent.occasion).toBe('travel');
        expect(intent.budget).toMatchObject({ amount: 300, currency: 'USD' });
    });

    it('4. separates smart-casual occasion from an explicit budget', () => {
        expect(parseStylistIntent('weekend casual brunch under $200')).toEqual({
            style: 'casual',
            occasion: 'casual',
            budget: { amount: 200, currency: 'USD' },
        });
    });

    it('5. treats empty or whitespace input as a safe empty intent', () => {
        const empty = { style: null, occasion: null, budget: null };
        expect(parseStylistIntent('')).toEqual(empty);
        expect(parseStylistIntent('   ')).toEqual(empty);
    });

    it('6. exposes a named stylist with a wardrobe-first greeting', () => {
        expect(STYLIST_NAME).toBe('Ava');
        expect(STYLIST_GREETING).toContain(STYLIST_NAME);
        expect(STYLIST_GREETING).toMatch(/wardrobe/i);
    });

    it('7. answers a Russian work question in Russian with three actions', () => {
        const reply = personaReply('Дайте мне рабочий образ', 'casual', 'ru');
        expect(reply.lang).toBe('ru');
        expect(reply.text).toMatch(/работ/i);
        expect(reply.suggestions).toHaveLength(3);
        expect(reply.suggestions.every((suggestion) => /[А-Яа-я]/.test(suggestion))).toBe(true);
    });

    it('8. answers an Uzbek budget question in Uzbek with three actions', () => {
        const reply = personaReply('Menga byudjet obraz kerak', 'classic', 'uz');
        expect(reply.lang).toBe('uz');
        expect(reply.text).toMatch(/byudjetingizda/i);
        expect(reply.suggestions).toHaveLength(3);
        expect(reply.suggestions[0]).toMatch(/gacha/);
    });

    it('9. resolves regional locales and falls back to English', () => {
        expect(resolvePersonaLang('ru-RU')).toBe('ru');
        expect(resolvePersonaLang('uz-UZ')).toBe('uz');
        expect(resolvePersonaLang('de-DE')).toBe('en');
        expect(resolvePersonaLang(undefined)).toBe('en');
    });

    it('10. defines all six style pillars with complete guidance', () => {
        expect(Object.keys(STYLE_PILLARS).sort()).toEqual([
            'business_casual',
            'casual',
            'classic',
            'minimalist',
            'old_money',
            'semi_classic',
        ]);
        for (const pillar of Object.values(STYLE_PILLARS)) {
            expect(pillar.label.length).toBeGreaterThan(0);
            expect(pillar.vibe.length).toBeGreaterThan(0);
            expect(pillar.palette.length).toBeGreaterThanOrEqual(5);
            expect(pillar.fabrics.length).toBeGreaterThanOrEqual(4);
            expect(pillar.pairing.length).toBeGreaterThan(0);
            expect(pillar.avoid.length).toBeGreaterThan(0);
        }
    });

    it('11. normalizes style aliases and defaults unknown styles to casual', () => {
        expect(getPillar('OLD MONEY').id).toBe('old_money');
        expect(getPillar('old-money').id).toBe('old_money');
        expect(getPillar('semi classic').id).toBe('semi_classic');
        expect(getPillar('goth-ninja').id).toBe('casual');
        expect(getPillar(null).id).toBe('casual');
    });

    it('12. builds a system prompt with identity, voice, and the full pipeline', () => {
        const prompt = buildStylistSystemPrompt({ style: 'classic' });
        expect(prompt).toContain(`You are ${STYLIST_NAME}`);
        expect(prompt).toContain('Style direction (Classic Menswear)');
        expect(prompt).toContain('Palette:');
        for (const step of AGENT_PIPELINE) {
            expect(prompt).toContain(step);
        }
    });

    it('13. injects occasion, weather, color limits, and budget into the prompt', () => {
        const prompt = buildStylistSystemPrompt({
            style: 'old_money',
            occasion: 'work',
            weather: { temp: 6, condition: 'cold' },
            preferences: { favoriteColors: ['navy'], avoidColors: ['red'] },
            budgetHint: 'HARD BUDGET: $200',
        });
        expect(prompt).toContain('Occasion: work.');
        expect(prompt).toContain('Weather: 6C cold.');
        expect(prompt).toContain('Never use: red.');
        expect(prompt).toContain('Favor: navy.');
        expect(prompt).toContain('HARD BUDGET: $200');
    });

    it('14. offers a shopping path when the wardrobe is empty', () => {
        const [suggestion] = generateSuggestions({
            items: [],
            wearLogs: [],
            occasion: 'work',
        });
        expect(suggestion.outfit.itemIds).toEqual([]);
        expect(suggestion.shoppingSuggestions.map((item) => item.category)).toEqual([
            'top',
            'bottom',
            'shoes',
        ]);
        expect(suggestion.reasoning).toMatch(/closet/i);
    });

    it('15. builds three complete, distinct looks that keep owned pieces', () => {
        const outfits = buildLocalOutfits(mixedWardrobe(), {
            stylePreferences: 'casual',
            occasion: 'Everyday',
            weather: { temp: 10, condition: 'cool' },
            limit: 3,
        });
        expect(outfits).toHaveLength(3);
        for (const outfit of outfits) {
            const slots = outfit.items.map((item) => item.macroCategory);
            expect(outfit.items).toHaveLength(4);
            expect(slots).toContain('outerwear');
            expect(slots).toContain('bottom');
            expect(slots).toContain('shoes');
            expect(slots).toContain('top');
            expect(outfit.items.some((item) => String(item.id).startsWith('own_'))).toBe(true);
        }
        const signatures = new Set(
            outfits.map((outfit) => outfit.items.map((item) => item.id).sort().join('|')),
        );
        expect(signatures.size).toBe(3);
    });

    it('15b. consistently fills a three-look deck from a mixed wardrobe', () => {
        const originalRandom = Math.random;
        try {
            for (let attempt = 0; attempt < 200; attempt += 1) {
                let state = (attempt + 1) >>> 0;
                Math.random = () => {
                    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
                    return state / 0x1_0000_0000;
                };
                const outfits = buildLocalOutfits(mixedWardrobe(), {
                    stylePreferences: 'casual',
                    occasion: 'Everyday',
                    weather: { temp: 10, condition: 'cool' },
                    limit: 3,
                });
                expect(outfits).toHaveLength(3);
                for (const outfit of outfits) {
                    expect(outfit.items.map((item) => item.macroCategory).sort()).toEqual([
                        'bottom',
                        'outerwear',
                        'shoes',
                        'top',
                    ]);
                }
            }
        } finally {
            Math.random = originalRandom;
        }
    });

    it('16. enforces a shop budget and never picks luxury over-cap pieces', () => {
        const outfits = buildLocalOutfits(budgetPool(), {
            stylePreferences: 'casual',
            occasion: 'Weekend',
            prompt: 'under $100',
            weather: { temp: 22, condition: 'clear' },
            limit: 3,
        });
        expect(outfits.length).toBeGreaterThan(0);
        for (const outfit of outfits) {
            expect(outfitShopSpend(outfit.items)).toBeLessThanOrEqual(100);
            expect(outfit.items.some((item) => /luxury/i.test(item.name || ''))).toBe(false);
        }
    });

    it('17. keeps the Team Collaboration deck shop-only even with a full wardrobe', () => {
        const outfits = buildLocalOutfits([...shopOnlyPool(), ...mixedWardrobe()], {
            stylePreferences: 'business_casual',
            occasion: 'Team Collaboration',
            weather: { temp: 22, condition: 'clear' },
            limit: 3,
        });
        expect(outfits.length).toBeGreaterThan(0);
        for (const outfit of outfits) {
            for (const item of outfit.items) {
                expect(String(item.id).startsWith('shop_')).toBe(true);
                expect(item.isShopItem).toBe(true);
            }
        }
    });

    it('18. accepts athletic clothing for a workout and rejects office clothing', () => {
        const athletic = [
            { name: 'Performance training tee', macroCategory: 'top' },
            { name: 'Running athletic shorts', macroCategory: 'bottom' },
            { name: 'Training athletic sneakers', macroCategory: 'shoes' },
        ];
        const office = [
            { name: 'White cotton shirt', macroCategory: 'top' },
            { name: 'Navy trousers', macroCategory: 'bottom' },
            { name: 'Brown loafers', macroCategory: 'shoes' },
        ];
        expect(styleJudgeReason(athletic, 'casual', 'gym workout')).toBeNull();
        expect(styleJudgeReason(office, 'old_money', 'gym workout')).toMatch(/workout/);
    });

    it('19. accepts a real tuxedo for black tie and rejects an ordinary suit', () => {
        const tuxedo = [
            { name: 'Black tuxedo dinner jacket', macroCategory: 'outerwear' },
            { name: 'White tuxedo formal shirt', macroCategory: 'top' },
            { name: 'Black tuxedo trousers', macroCategory: 'bottom' },
            { name: 'Black patent Oxford shoes', macroCategory: 'shoes' },
        ];
        const suit = [
            { name: 'White cotton shirt', macroCategory: 'top' },
            { name: 'Navy trousers', macroCategory: 'bottom' },
            { name: 'Brown loafers', macroCategory: 'shoes' },
            { name: 'Navy suit blazer', macroCategory: 'outerwear' },
        ];
        expect(styleJudgeReason(tuxedo, 'classic', 'black tie')).toBeNull();
        expect(styleJudgeReason(suit, 'classic', 'black-tie')).toMatch(/tuxedo/);
    });

    it('20. enforces coherence and cold-weather guardrails', () => {
        expect(
            rejectOutfitIncoherence(
                [
                    { name: 'Navy suit blazer', macroCategory: 'outerwear' },
                    { name: 'White oxford shirt', macroCategory: 'top' },
                    { name: 'Charcoal wool suit trousers', macroCategory: 'bottom' },
                    { name: 'White leather sneakers', macroCategory: 'shoes' },
                ],
                'casual',
                'casual',
            ),
        ).toBe('sneakers never pair with tailored trousers');
        expect(
            rejectForClimate(
                { name: '100% linen shirt', macroCategory: 'top' },
                { temp: 10, condition: 'cold' },
            ),
        ).toMatch(/too cold for linen/);
        expect(
            rejectOutfitClimate(
                [
                    { name: 'Cotton t-shirt', macroCategory: 'top' },
                    { name: 'Beige chinos', macroCategory: 'bottom' },
                    { name: 'White sneakers', macroCategory: 'shoes' },
                ],
                { temp: 10, condition: 'cold' },
            ),
        ).toMatch(/needs a warm layer/);
    });
});

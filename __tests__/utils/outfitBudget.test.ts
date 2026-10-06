import {
    parseBudgetFromText,
    outfitShopSpend,
    itemShopPrice,
    buildAgentClothingContext,
    formatMoney,
} from '../../src/utils/outfitBudget';
import {
    buildLocalOutfits,
    type LocalClothingItem,
} from '../../src/services/outfitGenerationService';

jest.mock('../../src/config/env', () => ({
    __esModule: true,
    default: { api: { url: 'https://test-api.example.com', alicevisionUrl: 'https://test-av.example.com' } },
    Config: { api: { url: 'https://test-api.example.com', alicevisionUrl: 'https://test-av.example.com' } },
}));

describe('parseBudgetFromText', () => {
    it('reads 250$ and $250 and Work: 250$', () => {
        expect(parseBudgetFromText('250$')).toEqual({ amount: 250, currency: 'USD' });
        expect(parseBudgetFromText('$250')).toEqual({ amount: 250, currency: 'USD' });
        expect(parseBudgetFromText('Work: 250$')).toEqual({ amount: 250, currency: 'USD' });
    });

    it('reads budget 80 euros', () => {
        expect(parseBudgetFromText('budget 80 euros please')).toEqual({ amount: 80, currency: 'EUR' });
    });

    it('reads max €200', () => {
        expect(parseBudgetFromText('max €200')).toEqual({ amount: 200, currency: 'EUR' });
    });

    it('ignores weather units', () => {
        expect(parseBudgetFromText('outfit under 20 C')).toBeNull();
    });

    it('returns null without a spend cap', () => {
        expect(parseBudgetFromText('date night old money look')).toBeNull();
    });
});

describe('outfit shop spend', () => {
    it('counts Massimo/Zara catalog ids as shop spend', () => {
        expect(itemShopPrice({ id: 'md-63001892', price: 89.9 })).toBe(89.9);
        expect(itemShopPrice({ id: 'zara-06077001', price: 40 })).toBe(40);
        expect(outfitShopSpend([
            { id: 'md-1', price: 100 },
            { id: 'zara-2', price: 50 },
            { id: 'uuid-owned', isWardrobe: true, price: 999 },
        ])).toBe(150);
    });
});

describe('buildAgentClothingContext', () => {
    it('lists shop prices so the agent can see them', () => {
        const ctx = buildAgentClothingContext({
            owned: [{ name: 'navy polo', primaryColor: 'navy' }],
            shop: [{ name: 'cream trousers', brand: 'Massimo Dutti', price: 89.9, currency: 'EUR' }],
            userMessage: 'outfit under €100',
        });
        expect(ctx).toMatch(/cream trousers/i);
        expect(ctx).toMatch(/€89\.90|€89.9/);
        expect(ctx).toMatch(/HARD BUDGET/);
        expect(ctx).toMatch(/Owned wardrobe/);
    });
});

describe('formatMoney', () => {
    it('uses currency symbols', () => {
        expect(formatMoney(80, 'EUR')).toBe('€80');
        expect(formatMoney(80.5, 'USD')).toBe('$80.50');
    });
});

let idCounter = 0;
function shopPiece(slot: 'top' | 'bottom' | 'shoes', name: string, price: number): LocalClothingItem {
    idCounter += 1;
    return {
        id: `shop_${slot}_${idCounter}`,
        userId: 'u1',
        imageUrl: 'https://example.com/x.jpg',
        category: slot as any,
        subCategory: slot === 'top' ? 'polo' : slot === 'bottom' ? 'trousers' : 'loafers',
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
        name,
        isShopItem: true,
        price,
    } as LocalClothingItem;
}

describe('buildLocalOutfits respects budget', () => {
    it('never lets shop spend exceed the user cap', () => {
        idCounter = 0;
        const pool = [
            shopPiece('top', 'cheap polo', 25),
            shopPiece('top', 'luxury knit', 400),
            shopPiece('bottom', 'cheap chinos', 30),
            shopPiece('bottom', 'luxury trousers', 400),
            shopPiece('shoes', 'cheap loafers', 20),
            shopPiece('shoes', 'luxury loafers', 400),
        ];
        const outfits = buildLocalOutfits(pool, {
            stylePreferences: 'casual',
            occasion: 'Weekend',
            prompt: 'under $100',
            weather: { temp: 22, condition: 'clear' },
            limit: 3,
        });
        expect(outfits.length).toBeGreaterThan(0);
        for (const o of outfits) {
            expect(outfitShopSpend(o.items)).toBeLessThanOrEqual(100);
            expect(o.items.some((i) => /luxury/i.test(i.name || ''))).toBe(false);
        }
    });
});

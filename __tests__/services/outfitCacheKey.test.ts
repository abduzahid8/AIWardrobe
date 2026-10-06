/**
 * Regression test for "the clothing doesn't change": the outfit-suggestion
 * cache key used to hash only `wardrobeItems.length`, so two different
 * 10-item wardrobes (or the same wardrobe after swapping pieces) collided
 * and returned identical cached looks.
 */
import { buildOutfitCacheKey } from '../../src/services/ai/outfitService';

jest.mock('../../src/config/env', () => ({
    __esModule: true,
    default: {
        api: {
            url: 'https://test-api.example.com',
            alicevisionUrl: 'https://test-alicevision.example.com',
        },
    },
    Config: {
        api: {
            url: 'https://test-api.example.com',
            alicevisionUrl: 'https://test-alicevision.example.com',
        },
    },
}));

const weather = { temp: 22, condition: 'clear' };
const itemsA = Array.from({ length: 10 }, (_, i) => ({ id: `a${i}` }));
const itemsB = Array.from({ length: 10 }, (_, i) => ({ id: `b${i}` }));

describe('buildOutfitCacheKey', () => {
    it('differs for different wardrobes with the same item count', () => {
        expect(buildOutfitCacheKey('casual', 'casual', itemsA, weather)).not.toBe(
            buildOutfitCacheKey('casual', 'casual', itemsB, weather),
        );
    });

    it('differs after swapping a single piece', () => {
        const swapped = [...itemsA.slice(0, 9), { id: 'zzz-new' }];
        expect(buildOutfitCacheKey('casual', 'casual', itemsA, weather)).not.toBe(
            buildOutfitCacheKey('casual', 'casual', swapped, weather),
        );
    });

    it('is stable for the same wardrobe regardless of order', () => {
        const shuffled = [...itemsA].reverse();
        expect(buildOutfitCacheKey('casual', 'casual', itemsA, weather)).toBe(
            buildOutfitCacheKey('casual', 'casual', shuffled, weather),
        );
    });

    it('differs across weather temps', () => {
        expect(buildOutfitCacheKey('casual', 'casual', itemsA, weather)).not.toBe(
            buildOutfitCacheKey('casual', 'casual', itemsA, { temp: 8, condition: 'clear' }),
        );
    });
});

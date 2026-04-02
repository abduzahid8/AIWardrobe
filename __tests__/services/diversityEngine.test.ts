/**
 * Tests for diversityEngine service
 */

import {
    scoreDiversity,
    getUnwornAlert,
    surpriseMe,
    getColorDistribution,
    getCategoryBreakdown,
    getDefaultLayer,
} from '../../src/services/diversityEngine';

import type { ClothingItem, WearLog } from '../../src/types/domain';

// ============================================
// HELPERS
// ============================================

const makeItem = (overrides: Partial<ClothingItem> = {}): ClothingItem => ({
    id: `item_${Math.random().toString(36).substr(2, 5)}`,
    userId: 'user1',
    imageUrl: 'https://example.com/img.jpg',
    category: 'top',
    subCategory: 't-shirt',
    primaryColor: 'blue',
    colorHex: '#0000FF',
    pattern: 'solid',
    material: 'cotton',
    seasons: ['spring', 'summer'],
    occasions: ['casual'],
    wearCount: 0,
    lastWornAt: null,
    isFavorite: false,
    createdAt: new Date(Date.now() - 30 * 86400000).toISOString(),
    updatedAt: new Date().toISOString(),
    ...overrides,
});

const makeLog = (itemIds: string[], daysAgo = 0): WearLog => ({
    id: `log_${Math.random().toString(36).substr(2, 5)}`,
    userId: 'user1',
    itemIds,
    date: new Date(Date.now() - daysAgo * 86400000).toISOString().split('T')[0],
    createdAt: new Date(Date.now() - daysAgo * 86400000).toISOString(),
});

// ============================================
// TESTS
// ============================================

describe('diversityEngine', () => {
    describe('scoreDiversity', () => {
        it('returns 0 for empty wardrobe', () => {
            expect(scoreDiversity([], [])).toBe(0);
        });

        it('returns 0 when no items have been worn', () => {
            const items = [makeItem(), makeItem(), makeItem()];
            expect(scoreDiversity(items, [])).toBe(0);
        });

        it('returns higher score when items are evenly worn', () => {
            const items = [
                makeItem({ id: 'a', wearCount: 3, category: 'top' }),
                makeItem({ id: 'b', wearCount: 3, category: 'bottom' }),
                makeItem({ id: 'c', wearCount: 3, category: 'shoes' }),
            ];
            const logs = [
                makeLog(['a', 'b', 'c'], 1),
                makeLog(['a', 'b', 'c'], 2),
                makeLog(['a', 'b', 'c'], 3),
            ];
            const score = scoreDiversity(items, logs);
            expect(score).toBeGreaterThan(50);
        });

        it('returns lower score when only one item is worn', () => {
            const items = [
                makeItem({ id: 'a', wearCount: 10 }),
                makeItem({ id: 'b', wearCount: 0 }),
                makeItem({ id: 'c', wearCount: 0 }),
                makeItem({ id: 'd', wearCount: 0 }),
                makeItem({ id: 'e', wearCount: 0 }),
            ];
            const logs = [makeLog(['a'], 1)];
            const score = scoreDiversity(items, logs);
            expect(score).toBeLessThan(30);
        });

        it('score is between 0 and 100', () => {
            const items = Array.from({ length: 20 }, (_, i) =>
                makeItem({ id: `item_${i}`, wearCount: Math.floor(Math.random() * 10) })
            );
            const logs = items.slice(0, 10).map((item, i) => makeLog([item.id], i));
            const score = scoreDiversity(items, logs);
            expect(score).toBeGreaterThanOrEqual(0);
            expect(score).toBeLessThanOrEqual(100);
        });
    });

    describe('getUnwornAlert', () => {
        it('returns empty array when all items have been worn recently', () => {
            const items = [makeItem({ id: 'a' }), makeItem({ id: 'b' })];
            const logs = [makeLog(['a', 'b'], 1)];
            expect(getUnwornAlert(items, logs, 30)).toHaveLength(0);
        });

        it('returns items not worn in N days', () => {
            const items = [
                makeItem({ id: 'a' }),
                makeItem({ id: 'b' }),
                makeItem({ id: 'c' }),
            ];
            const logs = [makeLog(['a'], 1)]; // Only 'a' worn
            const unworn = getUnwornAlert(items, logs, 30);
            expect(unworn).toHaveLength(2);
            expect(unworn.map(u => u.item.id)).toContain('b');
            expect(unworn.map(u => u.item.id)).toContain('c');
        });

        it('marks never-worn items', () => {
            const items = [makeItem({ id: 'a', wearCount: 0 })];
            const unworn = getUnwornAlert(items, [], 30);
            expect(unworn[0].neverWorn).toBe(true);
        });
    });

    describe('surpriseMe', () => {
        it('returns null for empty wardrobe', () => {
            expect(surpriseMe([], [])).toBeNull();
        });

        it('returns itemIds from available items', () => {
            const items = [
                makeItem({ id: 'a', category: 'top', wearCount: 0 }),
                makeItem({ id: 'b', category: 'bottom', wearCount: 0 }),
            ];
            const result = surpriseMe(items, []);
            expect(result).not.toBeNull();
            expect(result!.itemIds.length).toBeGreaterThanOrEqual(1);
            expect(result!.reasoning).toBeTruthy();
        });

        it('favors less-worn items', () => {
            const items = [
                makeItem({ id: 'heavy', category: 'top', wearCount: 50 }),
                makeItem({ id: 'light', category: 'top', wearCount: 0 }),
            ];
            // Run multiple times; the light item should appear more often
            let lightCount = 0;
            for (let i = 0; i < 100; i++) {
                const result = surpriseMe(items, []);
                if (result?.itemIds.includes('light')) lightCount++;
            }
            expect(lightCount).toBeGreaterThan(60); // Should be strongly biased
        });
    });

    describe('getColorDistribution', () => {
        it('returns empty for no items', () => {
            expect(getColorDistribution([])).toHaveLength(0);
        });

        it('aggregates colors correctly', () => {
            const items = [
                makeItem({ primaryColor: 'blue', colorHex: '#0000FF' }),
                makeItem({ primaryColor: 'blue', colorHex: '#0000FF' }),
                makeItem({ primaryColor: 'red', colorHex: '#FF0000' }),
            ];
            const dist = getColorDistribution(items);
            expect(dist[0].name).toBe('blue');
            expect(dist[0].count).toBe(2);
            expect(dist[1].name).toBe('red');
            expect(dist[1].count).toBe(1);
        });
    });

    describe('getCategoryBreakdown', () => {
        it('counts categories correctly', () => {
            const items = [
                makeItem({ category: 'top' }),
                makeItem({ category: 'top' }),
                makeItem({ category: 'bottom' }),
                makeItem({ category: 'shoes' }),
            ];
            const breakdown = getCategoryBreakdown(items);
            const topEntry = breakdown.find(b => b.category === 'top');
            expect(topEntry?.count).toBe(2);
        });
    });

    describe('getDefaultLayer', () => {
        it('maps top to base', () => {
            expect(getDefaultLayer('top')).toBe('base');
        });

        it('maps outerwear to outer', () => {
            expect(getDefaultLayer('outerwear')).toBe('outer');
        });

        it('maps accessory to accessory', () => {
            expect(getDefaultLayer('accessory')).toBe('accessory');
        });
    });
});

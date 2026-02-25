/**
 * Tests for retentionService
 */

import {
    calculateStreak,
    getClosetUtilization,
    getUnwornItems,
    generateStyleInsights,
    shouldNudge,
    getNudgeType,
} from '../../src/services/retentionService';
import type { ClothingItem, WearLog } from '../../src/types/domain';

// ============================================
// HELPERS
// ============================================

const makeItem = (overrides: Partial<ClothingItem> = {}): ClothingItem => ({
    id: `item_${Math.random().toString(36).substr(2, 6)}`,
    userId: 'user1',
    imageUrl: 'https://example.com/img.jpg',
    category: 'top',
    subCategory: 'shirt',
    primaryColor: 'Blue',
    colorHex: '#2563EB',
    pattern: 'solid',
    material: 'cotton',
    seasons: ['summer'],
    occasions: ['casual'],
    wearCount: 0,
    lastWornAt: null,
    isFavorite: false,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    ...overrides,
});

const makeLog = (date: string, itemIds: string[]): WearLog => ({
    id: `log_${Math.random().toString(36).substr(2, 6)}`,
    userId: 'user1',
    itemIds,
    date,
    createdAt: new Date().toISOString(),
});

const today = () => new Date().toISOString().split('T')[0];
const daysAgo = (n: number) =>
    new Date(Date.now() - n * 86400000).toISOString().split('T')[0];

// ============================================
// TESTS
// ============================================

describe('calculateStreak', () => {
    it('should return 0 for empty logs', () => {
        expect(calculateStreak([])).toBe(0);
    });

    it('should return 1 for a log today', () => {
        const logs = [makeLog(today(), ['item1'])];
        expect(calculateStreak(logs)).toBe(1);
    });

    it('should return 1 for a log yesterday', () => {
        const logs = [makeLog(daysAgo(1), ['item1'])];
        expect(calculateStreak(logs)).toBe(1);
    });

    it('should return 0 for a log 2 days ago only', () => {
        const logs = [makeLog(daysAgo(2), ['item1'])];
        expect(calculateStreak(logs)).toBe(0);
    });

    it('should count consecutive days', () => {
        const logs = [
            makeLog(today(), ['item1']),
            makeLog(daysAgo(1), ['item1']),
            makeLog(daysAgo(2), ['item1']),
        ];
        expect(calculateStreak(logs)).toBe(3);
    });

    it('should break on gaps', () => {
        const logs = [
            makeLog(today(), ['item1']),
            makeLog(daysAgo(1), ['item1']),
            // gap on daysAgo(2)
            makeLog(daysAgo(3), ['item1']),
        ];
        expect(calculateStreak(logs)).toBe(2);
    });

    it('should handle duplicate dates', () => {
        const logs = [
            makeLog(today(), ['item1']),
            makeLog(today(), ['item2']), // same day, different items
            makeLog(daysAgo(1), ['item1']),
        ];
        expect(calculateStreak(logs)).toBe(2);
    });
});

describe('getClosetUtilization', () => {
    it('should return 0 for empty closet', () => {
        expect(getClosetUtilization([], [])).toBe(0);
    });

    it('should return 100 if all items worn recently', () => {
        const items = [makeItem({ id: 'a' }), makeItem({ id: 'b' })];
        const logs = [makeLog(today(), ['a', 'b'])];
        expect(getClosetUtilization(items, logs, 30)).toBe(100);
    });

    it('should return 50 if half items worn', () => {
        const items = [makeItem({ id: 'a' }), makeItem({ id: 'b' })];
        const logs = [makeLog(today(), ['a'])];
        expect(getClosetUtilization(items, logs, 30)).toBe(50);
    });

    it('should not count old logs beyond cutoff', () => {
        const items = [makeItem({ id: 'a' })];
        const logs = [makeLog(daysAgo(31), ['a'])]; // older than 30 days
        expect(getClosetUtilization(items, logs, 30)).toBe(0);
    });
});

describe('getUnwornItems', () => {
    it('should return all items if no logs', () => {
        const items = [makeItem({ id: 'a' }), makeItem({ id: 'b' })];
        expect(getUnwornItems(items, [], 30)).toHaveLength(2);
    });

    it('should exclude recently worn items', () => {
        const items = [makeItem({ id: 'a' }), makeItem({ id: 'b' })];
        const logs = [makeLog(today(), ['a'])];
        const unworn = getUnwornItems(items, logs, 30);
        expect(unworn).toHaveLength(1);
        expect(unworn[0].id).toBe('b');
    });
});

describe('generateStyleInsights', () => {
    it('should return empty array for no data', () => {
        expect(generateStyleInsights([], [])).toEqual([]);
    });

    it('should generate utilization insight when items exist', () => {
        const items = [makeItem({ id: 'a' })];
        const insights = generateStyleInsights(items, []);
        expect(insights.some((i) => i.type === 'utilization')).toBe(true);
    });

    it('should generate unworn_nudge when many items unworn', () => {
        const items = [
            makeItem({ id: 'a' }),
            makeItem({ id: 'b' }),
            makeItem({ id: 'c' }),
            makeItem({ id: 'd' }),
        ];
        const insights = generateStyleInsights(items, []);
        expect(insights.some((i) => i.type === 'unworn_nudge')).toBe(true);
    });

    it('should generate streak insight when streak >= 3', () => {
        const items = [makeItem({ id: 'a' })];
        const logs = [
            makeLog(today(), ['a']),
            makeLog(daysAgo(1), ['a']),
            makeLog(daysAgo(2), ['a']),
        ];
        const insights = generateStyleInsights(items, logs);
        expect(insights.some((i) => i.type === 'streak')).toBe(true);
    });
});

describe('shouldNudge', () => {
    it('should return false for null lastActive', () => {
        expect(shouldNudge(null)).toBe(false);
    });

    it('should return false for recent activity', () => {
        expect(shouldNudge(new Date().toISOString())).toBe(false);
    });

    it('should return true for inactive user', () => {
        const fourDaysAgo = new Date(Date.now() - 4 * 86400000).toISOString();
        expect(shouldNudge(fourDaysAgo, 3)).toBe(true);
    });
});

describe('getNudgeType', () => {
    it('should return none when everything is fine', () => {
        const items = [makeItem({ id: 'a' })];
        const logs = [makeLog(today(), ['a'])];
        expect(getNudgeType(items, logs, 1)).toBe('none');
    });

    it('should return streak_at_risk when streak exists but not logged today', () => {
        const items = [makeItem({ id: 'a' })];
        const logs = [makeLog(daysAgo(1), ['a'])];
        expect(getNudgeType(items, logs, 1)).toBe('streak_at_risk');
    });
});

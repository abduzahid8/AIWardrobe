/**
 * Diversity Engine — Outfit variety scoring and smart suggestions
 *
 * Features:
 *   - scoreDiversity(): 0-100 score based on how evenly items are worn
 *   - getUnwornAlert(): Items not worn in N days with age info
 *   - surpriseMe(): Picks outfit weighted toward least-worn items
 *   - getColorDistribution(): Aggregated color palette from wardrobe
 *   - getCategoryBreakdown(): Items per category
 */

import type { ClothingItem, WearLog, Outfit, ClothingCategory, Occasion } from '../../src/types/domain';

// ============================================
// DIVERSITY SCORE
// ============================================

/**
 * Compute a 0-100 diversity score based on how evenly items have been worn.
 *
 * Algorithm:
 * 1. Gini coefficient of wear counts (0 = perfectly equal, 1 = all on one item)
 * 2. Invert and scale to 0-100
 * 3. Bonus for using items from multiple categories
 * 4. Penalty for >50% unworn items
 */
export function scoreDiversity(items: ClothingItem[], wearLogs: WearLog[], days = 30): number {
    if (items.length <= 1) return items.length === 1 && items[0].wearCount > 0 ? 50 : 0;

    // Filter recent logs
    const cutoff = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
    const recentLogs = wearLogs.filter(log => log.date >= cutoff);
    const wornItemIds = new Set(recentLogs.flatMap(log => log.itemIds));

    // 1. Compute wear count distribution
    const counts = items.map(i => i.wearCount);
    const totalWears = counts.reduce((s, c) => s + c, 0);

    if (totalWears === 0) return 0;

    // 2. Gini coefficient
    const sorted = [...counts].sort((a, b) => a - b);
    const n = sorted.length;
    let giniSum = 0;
    for (let i = 0; i < n; i++) {
        giniSum += (2 * (i + 1) - n - 1) * sorted[i];
    }
    const gini = giniSum / (n * totalWears);
    const equalityScore = Math.round((1 - gini) * 60); // Max 60 points from equality

    // 3. Category diversity bonus (max 20 points)
    const usedCategories = new Set(
        items.filter(i => wornItemIds.has(i.id)).map(i => i.category)
    );
    const categoryBonus = Math.min(usedCategories.size * 5, 20);

    // 4. Utilization bonus (max 20 points)
    const utilizationPct = wornItemIds.size / items.length;
    const utilizationBonus = Math.round(utilizationPct * 20);

    return Math.min(100, Math.max(0, equalityScore + categoryBonus + utilizationBonus));
}

// ============================================
// UNWORN ALERT
// ============================================

export interface UnwornAlertItem {
    item: ClothingItem;
    daysSinceAdded: number;
    neverWorn: boolean;
}

/**
 * Get items not worn in the last N days, sorted by days since added.
 */
export function getUnwornAlert(items: ClothingItem[], wearLogs: WearLog[], days = 30): UnwornAlertItem[] {
    const cutoff = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
    const recentLogs = wearLogs.filter(log => log.date >= cutoff);
    const wornItemIds = new Set(recentLogs.flatMap(log => log.itemIds));

    const now = Date.now();

    return items
        .filter(item => !wornItemIds.has(item.id))
        .map(item => ({
            item,
            daysSinceAdded: Math.floor((now - new Date(item.createdAt).getTime()) / 86400000),
            neverWorn: item.wearCount === 0,
        }))
        .sort((a, b) => b.daysSinceAdded - a.daysSinceAdded);
}

// ============================================
// SURPRISE ME
// ============================================

/**
 * Pick an outfit from the least-worn items, avoiding recent repeats.
 *
 * Strategy:
 * 1. Score each item inversely by wearCount (less worn = more likely)
 * 2. Filter out items worn in the last 3 days
 * 3. Pick one from each needed category (top, bottom, shoes optional)
 * 4. Return as a pseudo-outfit
 */
export function surpriseMe(
    items: ClothingItem[],
    wearLogs: WearLog[],
    recentDays = 3,
): { itemIds: string[]; reasoning: string } | null {
    if (items.length === 0) return null;

    // Get items worn in the last N days
    const cutoff = new Date(Date.now() - recentDays * 86400000).toISOString().split('T')[0];
    const recentLogs = wearLogs.filter(log => log.date >= cutoff);
    const recentlyWornIds = new Set(recentLogs.flatMap(log => log.itemIds));

    // Score items: lower wearCount = higher weight
    const scoredItems = items
        .filter(item => !recentlyWornIds.has(item.id))
        .map(item => ({
            ...item,
            weight: 1 / (item.wearCount + 1), // +1 to avoid division by zero
        }));

    if (scoredItems.length === 0) {
        // All items worn recently — fall back to full pool
        return surpriseMeFallback(items);
    }

    // Pick one from each category using weighted random
    const categories: ClothingCategory[] = ['top', 'bottom', 'dress', 'shoes', 'outerwear', 'accessory'];
    const picked: string[] = [];
    const pickedNames: string[] = [];

    for (const cat of categories) {
        const candidates = scoredItems.filter(i => i.category === cat);
        if (candidates.length === 0) continue;

        const selected = weightedRandom(candidates);
        if (selected) {
            picked.push(selected.id);
            pickedNames.push(selected.name || selected.subCategory || cat);
        }
    }

    if (picked.length === 0) return null;

    const unwornPicked = picked.filter(id => {
        const item = items.find(i => i.id === id);
        return item && item.wearCount === 0;
    });

    const reasoning = unwornPicked.length > 0
        ? `🎲 Surprise! This outfit features ${unwornPicked.length} item(s) you've never worn. Time to try something new!`
        : `🎲 Mixed it up with your least-worn pieces. Fresh vibes ahead!`;

    return { itemIds: picked, reasoning };
}

function surpriseMeFallback(items: ClothingItem[]): { itemIds: string[]; reasoning: string } | null {
    const categories: ClothingCategory[] = ['top', 'bottom'];
    const picked: string[] = [];

    for (const cat of categories) {
        const candidates = items.filter(i => i.category === cat);
        if (candidates.length > 0) {
            const idx = Math.floor(Math.random() * candidates.length);
            picked.push(candidates[idx].id);
        }
    }

    return picked.length > 0
        ? { itemIds: picked, reasoning: '🎲 Random pick from your closet!' }
        : null;
}

function weightedRandom<T extends { weight: number }>(items: T[]): T | null {
    if (items.length === 0) return null;
    const totalWeight = items.reduce((sum, i) => sum + i.weight, 0);
    let random = Math.random() * totalWeight;

    for (const item of items) {
        random -= item.weight;
        if (random <= 0) return item;
    }
    return items[items.length - 1];
}

// ============================================
// COLOR DISTRIBUTION
// ============================================

export interface ColorDistEntry {
    name: string;
    color: string;
    count: number;
}

/**
 * Aggregate items by primaryColor, returning sorted distribution.
 */
export function getColorDistribution(items: ClothingItem[]): ColorDistEntry[] {
    const map = new Map<string, { color: string; count: number }>();

    for (const item of items) {
        const name = item.primaryColor || 'Unknown';
        const existing = map.get(name);
        if (existing) {
            existing.count++;
        } else {
            map.set(name, { color: item.colorHex || '#808080', count: 1 });
        }
    }

    return Array.from(map.entries())
        .map(([name, { color, count }]) => ({ name, color, count }))
        .sort((a, b) => b.count - a.count);
}

// ============================================
// CATEGORY BREAKDOWN
// ============================================

export interface CategoryBreakdownEntry {
    category: string;
    count: number;
}

/**
 * Count items per category.
 */
export function getCategoryBreakdown(items: ClothingItem[]): CategoryBreakdownEntry[] {
    const map = new Map<string, number>();

    for (const item of items) {
        map.set(item.category, (map.get(item.category) || 0) + 1);
    }

    return Array.from(map.entries())
        .map(([category, count]) => ({ category, count }))
        .sort((a, b) => b.count - a.count);
}

// ============================================
// SEASON BREAKDOWN
// ============================================

export interface SeasonBreakdownEntry {
    season: string;
    count: number;
}

export function getSeasonBreakdown(items: ClothingItem[]): SeasonBreakdownEntry[] {
    const map = new Map<string, number>();

    for (const item of items) {
        if (item.seasons) {
            item.seasons.forEach((s) => {
                map.set(s, (map.get(s) || 0) + 1);
            });
        }
    }

    return Array.from(map.entries())
        .map(([season, count]) => ({ season, count }))
        .sort((a, b) => b.count - a.count);
}

// ============================================
// OCCASION BREAKDOWN
// ============================================

export interface OccasionBreakdownEntry {
    occasion: string;
    count: number;
}

export function getOccasionBreakdown(items: ClothingItem[]): OccasionBreakdownEntry[] {
    const map = new Map<string, number>();

    for (const item of items) {
        if (item.occasions) {
            item.occasions.forEach((o) => {
                map.set(o, (map.get(o) || 0) + 1);
            });
        }
    }

    return Array.from(map.entries())
        .map(([occasion, count]) => ({ occasion, count }))
        .sort((a, b) => b.count - a.count);
}

// ============================================
// WARDROBE HEALTH SCORE
// ============================================

export interface WardrobeHealthScore {
    overall: number;
    utilization: number;
    diversity: number;
    maintenance: number;
    freshness: number;
}

export function getWardrobeHealthScore(items: ClothingItem[], wearLogs: WearLog[]): WardrobeHealthScore {
    const totalItems = items.length;
    if (totalItems === 0) {
        return { overall: 0, utilization: 0, diversity: 0, maintenance: 0, freshness: 0 };
    }

    // Utilization score (30%)
    const cutoff = new Date(Date.now() - 30 * 86400000).toISOString().split('T')[0];
    const recentLogs = wearLogs.filter((log) => log.date >= cutoff);
    const wornItemIds = new Set(recentLogs.flatMap((log) => log.itemIds));
    const utilizationPct = wornItemIds.size / totalItems;
    const utilizationScore = Math.round(utilizationPct * 100);

    // Diversity score (30%)
    const diversityScore = scoreDiversity(items, wearLogs);

    // Maintenance score (20%) - items without wear are okay, but favor items with some use
    const neverWorn = items.filter((i) => i.wearCount === 0).length;
    const maintenanceScore = Math.round(Math.max(0, (1 - neverWorn / totalItems)) * 100);

    // Freshness score (20%) - recently added items indicate active wardrobe management
    const thirtyDaysAgo = new Date(Date.now() - 30 * 86400000).toISOString();
    const recentItems = items.filter((i) => i.createdAt >= thirtyDaysAgo).length;
    const freshnessScore = Math.min(100, Math.round((recentItems / Math.max(totalItems, 1)) * 200));

    const overall = Math.round(
        utilizationScore * 0.3 + diversityScore * 0.3 + maintenanceScore * 0.2 + freshnessScore * 0.2
    );

    return { overall: Math.min(100, overall), utilization: utilizationScore, diversity: diversityScore, maintenance: maintenanceScore, freshness: Math.min(100, freshnessScore) };
}

// ============================================
// DEFAULT LAYER ASSIGNMENT
// ============================================

type ClothingLayer = 'outer' | 'mid' | 'base' | 'accessory';

/** Auto-assign a layer based on item category */
export function getDefaultLayer(category: ClothingCategory): ClothingLayer {
    switch (category) {
        case 'top':
        case 'bottom':
        case 'dress':
        case 'shoes':
            return 'base';
        case 'outerwear':
            return 'outer';
        case 'accessory':
            return 'accessory';
        case 'other':
        default:
            return 'mid';
    }
}

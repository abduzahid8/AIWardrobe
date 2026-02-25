/**
 * Retention Service — Analytics and engagement engine
 *
 * Pure functions for calculating retention metrics.
 * No side effects, no API calls — just math on local data.
 */

import type { ClothingItem, WearLog, StyleInsight } from '../types/domain';

// ============================================
// STREAK
// ============================================

/**
 * Calculate consecutive-day streak from wear logs.
 * A streak is maintained if the user logged a wear today or yesterday.
 */
export function calculateStreak(wearLogs: WearLog[]): number {
    if (wearLogs.length === 0) return 0;

    const dates = [...new Set(wearLogs.map((log) => log.date))].sort().reverse();
    const today = new Date().toISOString().split('T')[0];
    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];

    if (dates[0] !== today && dates[0] !== yesterday) return 0;

    let streak = 1;
    for (let i = 1; i < dates.length; i++) {
        const prev = new Date(dates[i - 1]);
        const curr = new Date(dates[i]);
        const diffDays = (prev.getTime() - curr.getTime()) / 86400000;

        if (Math.round(diffDays) === 1) {
            streak++;
        } else {
            break;
        }
    }

    return streak;
}

// ============================================
// CLOSET UTILIZATION
// ============================================

/**
 * Calculate what % of the closet has been worn in the last N days.
 * Returns 0-100.
 */
export function getClosetUtilization(
    items: ClothingItem[],
    wearLogs: WearLog[],
    days: number = 30
): number {
    if (items.length === 0) return 0;

    const cutoff = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
    const recentLogs = wearLogs.filter((log) => log.date >= cutoff);
    const wornItemIds = new Set(recentLogs.flatMap((log) => log.itemIds));

    return Math.round((wornItemIds.size / items.length) * 100);
}

// ============================================
// UNWORN ITEMS
// ============================================

/**
 * Get items that haven't been worn in the last N days.
 * Useful for "Try something new" nudges.
 */
export function getUnwornItems(
    items: ClothingItem[],
    wearLogs: WearLog[],
    days: number = 30
): ClothingItem[] {
    const cutoff = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
    const recentLogs = wearLogs.filter((log) => log.date >= cutoff);
    const wornItemIds = new Set(recentLogs.flatMap((log) => log.itemIds));

    return items.filter((item) => !wornItemIds.has(item.id));
}

// ============================================
// STYLE INSIGHTS
// ============================================

/**
 * Generate style insights from wear data.
 * Returns an array of human-readable insights.
 */
export function generateStyleInsights(
    items: ClothingItem[],
    wearLogs: WearLog[]
): StyleInsight[] {
    const insights: StyleInsight[] = [];
    const now = new Date().toISOString();

    // 1. Closet utilization
    const utilization = getClosetUtilization(items, wearLogs, 30);
    if (items.length > 0) {
        insights.push({
            type: 'utilization',
            title: `${utilization}% Closet Utilization`,
            description:
                utilization >= 80
                    ? 'You\'re making great use of your wardrobe!'
                    : utilization >= 50
                        ? `You've worn ${utilization}% of your closet this month. Try mixing in some forgotten pieces.`
                        : `Only ${utilization}% of your closet was worn this month. There's a lot to rediscover!`,
            data: { utilization, totalItems: items.length },
            generatedAt: now,
        });
    }

    // 2. Unworn nudge
    const unworn = getUnwornItems(items, wearLogs, 30);
    if (unworn.length >= 3) {
        insights.push({
            type: 'unworn_nudge',
            title: `${unworn.length} Items Waiting`,
            description: `You have ${unworn.length} items you haven't worn in 30 days. Here are some to try.`,
            data: {
                unwornCount: unworn.length,
                sampleItems: unworn.slice(0, 3).map((i) => ({
                    id: i.id,
                    name: i.name || i.subCategory,
                    category: i.category,
                })),
            },
            generatedAt: now,
        });
    }

    // 3. Color patterns
    if (wearLogs.length >= 5) {
        const colorFrequency: Record<string, number> = {};
        wearLogs.forEach((log) => {
            log.itemIds.forEach((itemId) => {
                const item = items.find((i) => i.id === itemId);
                if (item?.primaryColor) {
                    const color = item.primaryColor.toLowerCase();
                    colorFrequency[color] = (colorFrequency[color] || 0) + 1;
                }
            });
        });

        const sorted = Object.entries(colorFrequency).sort((a, b) => b[1] - a[1]);
        if (sorted.length >= 2) {
            const [topColor, topCount] = sorted[0];
            insights.push({
                type: 'color_pattern',
                title: `${topColor.charAt(0).toUpperCase() + topColor.slice(1)} is Your Go-To`,
                description: `You've worn ${topColor} items ${topCount} times recently. Your top 3 colors are ${sorted.slice(0, 3).map(([c]) => c).join(', ')}.`,
                data: { colorFrequency: Object.fromEntries(sorted.slice(0, 5)) },
                generatedAt: now,
            });
        }
    }

    // 4. Day-of-week pattern
    if (wearLogs.length >= 14) {
        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        const dayColorMap: Record<number, Record<string, number>> = {};

        wearLogs.forEach((log) => {
            const dayOfWeek = new Date(log.date).getDay();
            if (!dayColorMap[dayOfWeek]) dayColorMap[dayOfWeek] = {};

            log.itemIds.forEach((itemId) => {
                const item = items.find((i) => i.id === itemId);
                if (item?.primaryColor) {
                    const color = item.primaryColor.toLowerCase();
                    dayColorMap[dayOfWeek][color] = (dayColorMap[dayOfWeek][color] || 0) + 1;
                }
            });
        });

        // Find day with strongest single-color preference
        let bestDay = -1;
        let bestColor = '';
        let bestRatio = 0;

        Object.entries(dayColorMap).forEach(([dayStr, freqMap]) => {
            const day = parseInt(dayStr);
            const entries = Object.entries(freqMap);
            const total = entries.reduce((sum, [, count]) => sum + count, 0);
            if (total < 3) return; // Need enough data

            entries.forEach(([color, count]) => {
                const ratio = count / total;
                if (ratio > bestRatio && ratio > 0.4) {
                    bestDay = day;
                    bestColor = color;
                    bestRatio = ratio;
                }
            });
        });

        if (bestDay >= 0) {
            insights.push({
                type: 'variety',
                title: `${dayNames[bestDay]}day ${bestColor.charAt(0).toUpperCase() + bestColor.slice(1)}`,
                description: `You tend to wear ${bestColor} on ${dayNames[bestDay]}days. Interesting pattern!`,
                data: { day: bestDay, dayName: dayNames[bestDay], color: bestColor, ratio: bestRatio },
                generatedAt: now,
            });
        }
    }

    // 5. Streak milestone
    const streak = calculateStreak(wearLogs);
    if (streak >= 3) {
        insights.push({
            type: 'streak',
            title: `🔥 ${streak} Day Streak!`,
            description:
                streak >= 30
                    ? 'Incredible! A full month of daily style logging.'
                    : streak >= 14
                        ? 'Two weeks strong! Your style data is getting really rich.'
                        : streak >= 7
                            ? 'A whole week! Your suggestions are getting smarter.'
                            : `${streak} days and counting. Keep it up!`,
            data: { streak },
            generatedAt: now,
        });
    }

    return insights;
}

// ============================================
// NUDGE LOGIC
// ============================================

/**
 * Determine if the user should receive a re-engagement nudge.
 * Returns true if user has been inactive for too long.
 */
export function shouldNudge(lastActiveAt: string | null, threshold: number = 3): boolean {
    if (!lastActiveAt) return false;

    const daysSinceActive = (Date.now() - new Date(lastActiveAt).getTime()) / 86400000;
    return daysSinceActive >= threshold;
}

/**
 * Get the most impactful nudge type based on user state.
 */
export function getNudgeType(
    items: ClothingItem[],
    wearLogs: WearLog[],
    streak: number
): 'streak_at_risk' | 'unworn_items' | 'low_utilization' | 'none' {
    // Streak at risk (logged yesterday but not today)
    const today = new Date().toISOString().split('T')[0];
    const hasLoggedToday = wearLogs.some((log) => log.date === today);

    if (streak > 0 && !hasLoggedToday) {
        return 'streak_at_risk';
    }

    // Low utilization
    const utilization = getClosetUtilization(items, wearLogs, 30);
    if (utilization < 30 && items.length > 5) {
        return 'low_utilization';
    }

    // Unworn items
    const unworn = getUnwornItems(items, wearLogs, 30);
    if (unworn.length > items.length * 0.5) {
        return 'unworn_items';
    }

    return 'none';
}

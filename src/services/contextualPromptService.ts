/**
 * Contextual Prompt Service — Smart feature discovery
 *
 * Shows relevant prompts at the right moment:
 * - "You haven't worn 40% of your wardrobe" → analytics
 * - "Trip coming up? Let us pack." → trip planner
 * - "Try Surprise Me!" → when suggestions are stale
 * - "Your streak is 🔥 7 days!" → celebration
 *
 * Rate-limited: max 1 prompt per session, respects dismissals.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import type { ClothingItem, WearLog } from '../../src/types/domain';

const DISMISSED_KEY = 'contextual_prompts_dismissed';
const LAST_SHOWN_KEY = 'contextual_prompt_last_shown';
const SESSION_COOLDOWN_MS = 4 * 60 * 60 * 1000; // 4 hours between prompts

export interface ContextualPrompt {
    id: string;
    title: string;
    message: string;
    icon: string;
    action: {
        label: string;
        route: keyof import('../../navigation/types').RootStackParamList;
        params?: Record<string, unknown>;
    };
    priority: number; // higher = more important
    color: string;
}

// ============================================
// PROMPT RULES
// ============================================

function evaluatePrompts(
    items: ClothingItem[],
    wearLogs: WearLog[],
    streak: number,
    utilization: number,
): ContextualPrompt[] {
    const prompts: ContextualPrompt[] = [];
    const now = Date.now();

    // Rule 1: High unworn percentage
    if (items.length >= 5) {
        const cutoff = new Date(now - 30 * 86400000).toISOString().split('T')[0];
        const recentLogs = wearLogs.filter(log => log.date >= cutoff);
        const wornIds = new Set(recentLogs.flatMap(log => log.itemIds));
        const unwornPct = Math.round(((items.length - wornIds.size) / items.length) * 100);

        if (unwornPct >= 40) {
            prompts.push({
                id: 'unworn_alert',
                title: 'Hidden Gems 💎',
                message: `${unwornPct}% of your wardrobe hasn't been worn in 30 days. Discover what you're missing!`,
                icon: 'analytics-outline',
                action: {
                    label: 'View Analytics',
                    route: 'WardrobeAnalytics',
                },
                priority: 8,
                color: '#F59E0B',
            });
        }
    }

    // Rule 2: Stale suggestions (no wear log in 3+ days)
    if (items.length >= 3) {
        const lastLog = wearLogs.length > 0 ? wearLogs[0] : null;
        const daysSinceLog = lastLog
            ? Math.floor((now - new Date(lastLog.createdAt).getTime()) / 86400000)
            : 999;

        if (daysSinceLog >= 3) {
            prompts.push({
                id: 'surprise_me',
                title: 'Shake Things Up 🎲',
                message: "Haven't logged a fit in a while? Let AI surprise you with a fresh combo!",
                icon: 'shuffle-outline',
                action: {
                    label: 'Surprise Me',
                    route: 'Main',
                },
                priority: 6,
                color: '#8B5CF6',
            });
        }
    }

    // Rule 3: Streak celebration
    if (streak >= 7 && streak % 7 === 0) {
        prompts.push({
            id: `streak_${streak}`,
            title: `🔥 ${streak}-Day Streak!`,
            message: `You've logged your outfits ${streak} days in a row. Keep it going!`,
            icon: 'flame-outline',
            action: {
                label: 'View Insights',
                route: 'WardrobeAnalytics',
            },
            priority: 9,
            color: '#EF4444',
        });
    }

    // Rule 4: Low utilization
    if (utilization <= 20 && items.length >= 10) {
        prompts.push({
            id: 'low_utilization',
            title: 'Closet Potential 📊',
            message: `Only ${utilization}% of your wardrobe is being used. See what's being neglected.`,
            icon: 'pie-chart-outline',
            action: {
                label: 'See Stats',
                route: 'WardrobeAnalytics',
            },
            priority: 7,
            color: '#3B82F6',
        });
    }

    // Rule 5: New user nudge (< 5 items)
    if (items.length > 0 && items.length < 5) {
        prompts.push({
            id: 'grow_wardrobe',
            title: 'Build Your Closet 👕',
            message: `You have ${items.length} items. Scan more clothes to unlock AI outfit suggestions!`,
            icon: 'camera-outline',
            action: {
                label: 'Scan Now',
                route: 'WardrobeVideo',
                params: {},
            },
            priority: 5,
            color: '#10B981',
        });
    }

    // Sort by priority descending
    return prompts.sort((a, b) => b.priority - a.priority);
}

// ============================================
// PUBLIC API
// ============================================

/**
 * Get the highest-priority contextual prompt for this session.
 * Returns null if rate-limited or all prompts dismissed.
 */
export async function getContextualPrompt(
    items: ClothingItem[],
    wearLogs: WearLog[],
    streak: number,
    utilization: number,
): Promise<ContextualPrompt | null> {
    try {
        // Rate limit: check last shown time
        const lastShown = await AsyncStorage.getItem(LAST_SHOWN_KEY);
        if (lastShown) {
            const elapsed = Date.now() - parseInt(lastShown, 10);
            if (elapsed < SESSION_COOLDOWN_MS) return null;
        }

        // Get dismissed prompt IDs
        const dismissedRaw = await AsyncStorage.getItem(DISMISSED_KEY);
        const dismissed = new Set<string>(dismissedRaw ? JSON.parse(dismissedRaw) : []);

        // Evaluate all rules
        const prompts = evaluatePrompts(items, wearLogs, streak, utilization);

        // Find first non-dismissed
        const prompt = prompts.find(p => !dismissed.has(p.id)) || null;

        return prompt;
    } catch {
        return null;
    }
}

/** Mark a prompt as shown (updates rate limit timestamp) */
export async function markPromptShown(): Promise<void> {
    try {
        await AsyncStorage.setItem(LAST_SHOWN_KEY, Date.now().toString());
    } catch {
        // Ignore storage errors
    }
}

/** Dismiss a prompt permanently (won't show again) */
export async function dismissPrompt(promptId: string): Promise<void> {
    try {
        const raw = await AsyncStorage.getItem(DISMISSED_KEY);
        const dismissed: string[] = raw ? JSON.parse(raw) : [];
        if (!dismissed.includes(promptId)) {
            dismissed.push(promptId);
            await AsyncStorage.setItem(DISMISSED_KEY, JSON.stringify(dismissed));
        }
    } catch {
        // Ignore storage errors
    }
}

/** Reset all dismissed prompts (for testing or after major wardrobe changes) */
export async function resetPrompts(): Promise<void> {
    try {
        await AsyncStorage.multiRemove([DISMISSED_KEY, LAST_SHOWN_KEY]);
    } catch {
        // Ignore
    }
}

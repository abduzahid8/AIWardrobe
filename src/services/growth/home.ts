/**
 * Phase 6F–6J — Home action center, empty-state intelligence,
 * retention loops, weekly value summary, non-manipulative streaks.
 *
 * Home answers: "What is the most useful thing for me right now?"
 * Never a promotional carousel. Streaks are optional, never shaming.
 */

import type { HomeActionCandidate, HomeActionKind, StreakState, VerifiedSaving } from '../../types/growth';

export const HOME_ACTION_COOLDOWN_HOURS: Record<HomeActionKind, number> = {
    todays_outfit: 12,
    upcoming_event: 24,
    wardrobe_gap: 72,
    underused_item: 168,
    price_change: 24,
    fit_feedback_request: 168,
    weekly_plan: 168,
    no_buy_reminder: 168,
    continue_closet_scan: 24,
    seasonal_review: 720,
};

function priorityScore(candidate: HomeActionCandidate): number {
    // Urgency × confidence × value, with goal alignment bonus.
    // Expired candidates are ineligible (score -Infinity).
    if (candidate.expiresInHours !== null && candidate.expiresInHours <= 0) return Number.NEGATIVE_INFINITY;
    if (candidate.hoursSinceLastShown < (HOME_ACTION_COOLDOWN_HOURS[candidate.kind] ?? 24)) return Number.NEGATIVE_INFINITY;
    const base = candidate.urgency * candidate.confidence * candidate.expectedValue;
    return candidate.goalAligned ? base * 1.2 : base;
}

/**
 * Returns prioritized home actions, highest value first. Promotional
 * content must never be passed in as a candidate.
 */
export function prioritizeHomeActions(candidates: HomeActionCandidate[], limit = 3): HomeActionCandidate[] {
    return candidates
        .map((c) => ({ c, score: priorityScore(c) }))
        .filter((s) => s.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit)
        .map((s) => s.c);
}

// ── Empty-state intelligence (6G) ────────────────────────────────

export type EmptySurface = 'wardrobe' | 'price_history' | 'style_dna' | 'fit_evidence' | 'events' | 'plans';

const EMPTY_STATE_COPY: Record<EmptySurface, { title: string; action: string }> = {
    wardrobe: { title: 'Your wardrobe is empty', action: 'Add 3–5 items to start building outfits.' },
    price_history: { title: 'No price history yet', action: 'We don’t have enough history yet.' },
    style_dna: { title: 'Style profile is still learning', action: 'Save or wear a few outfits so recommendations can learn.' },
    fit_evidence: { title: 'No fit evidence yet', action: 'Mark one garment that fits you well.' },
    events: { title: 'No upcoming events', action: 'Plan a casual week instead.' },
    plans: { title: 'No saved plans', action: 'Create a weekly plan to get started.' },
};

export function emptyStateFor(surface: EmptySurface): { title: string; action: string } {
    return EMPTY_STATE_COPY[surface];
}

// ── Retention loops (6H) ─────────────────────────────────────────

export type RetentionLoopKind = 'daily_outfit' | 'weekly_plan' | 'shopping_triggered' | 'event_triggered' | 'lifecycle';

export interface RetentionLoop {
    kind: RetentionLoopKind;
    /** Natural cadence — weekly use cases must not be forced daily. */
    cadence: 'daily' | 'weekly' | 'triggered';
    enabled: boolean;
}

export function defaultRetentionLoops(): RetentionLoop[] {
    return [
        { kind: 'daily_outfit', cadence: 'daily', enabled: true },
        { kind: 'weekly_plan', cadence: 'weekly', enabled: true },
        { kind: 'shopping_triggered', cadence: 'triggered', enabled: true },
        { kind: 'event_triggered', cadence: 'triggered', enabled: true },
        { kind: 'lifecycle', cadence: 'triggered', enabled: true },
    ];
}

// ── Weekly value summary (6I) ────────────────────────────────────

export interface WeeklyValueSummary {
    outfitsWorn: number;
    itemsReused: number;
    itemsUnused: number;
    verifiedSavingsCents: number;
    bestOutfitId: string | null;
    upcomingNeed: string | null;
    goalProgress: string | null;
}

export function buildWeeklyValueSummary(input: {
    outfitsWorn: number;
    itemsReused: number;
    itemsUnused: number;
    savings: VerifiedSaving[];
    bestOutfitId: string | null;
    upcomingNeed: string | null;
    goalProgress: string | null;
}): WeeklyValueSummary {
    // Conservative: only verified savings counted; never fabricated.
    const verifiedSavingsCents = input.savings.reduce((sum, s) => sum + Math.max(0, s.amountCents), 0);
    return {
        outfitsWorn: input.outfitsWorn,
        itemsReused: input.itemsReused,
        itemsUnused: input.itemsUnused,
        verifiedSavingsCents,
        bestOutfitId: input.bestOutfitId,
        upcomingNeed: input.upcomingNeed,
        goalProgress: input.goalProgress,
    };
}

// ── Streaks (6J) ─────────────────────────────────────────────────

export function startStreak(id: string): StreakState {
    return { id, count: 0, lastActiveWeek: null, optedIn: true };
}

/** Increments for the active week; idempotent within the same week. */
export function markStreakWeek(streak: StreakState, weekKey: string): StreakState {
    if (!streak.optedIn) return streak;
    if (streak.lastActiveWeek === weekKey) return streak;
    return { ...streak, count: streak.count + 1, lastActiveWeek: weekKey };
}

/** Broken streaks simply pause — no punishment, no loss messaging. */
export function streakStatusCopy(streak: StreakState): string {
    if (streak.count <= 0) return 'Start your first week of wardrobe planning when you like.';
    return `${streak.count} week${streak.count === 1 ? '' : 's'} of wardrobe planning`;
}

/**
 * Phase 6K–6L, 6M, 6BL — Notification policy engine, preferences,
 * re-engagement domain, value-based message templates.
 *
 * LOCAL ONLY: candidates are computed on-device. No remote sending,
 * no campaigns in this task. Generic "come back!" spam is rejected
 * by construction (expectedValue floor + actionable requirement).
 */

import type {
    NotificationCandidate,
    NotificationChannel,
    NotificationPreference,
    NotificationType,
    NotificationUrgency,
} from '../../types/growth';

export const CHANNELS: NotificationChannel[] = [
    'outfit_reminders',
    'event_reminders',
    'price_alerts',
    'fit_feedback',
    'weekly_review',
    'travel_packing',
    'shopping_reminders',
];

const TYPE_TO_CHANNEL: Record<NotificationType, NotificationChannel> = {
    price_target_reached: 'price_alerts',
    outfit_weather_conflict: 'outfit_reminders',
    event_missing_piece: 'event_reminders',
    planned_item_unavailable: 'outfit_reminders',
    weekly_plan_ready: 'weekly_review',
    fit_feedback_request: 'fit_feedback',
    seasonal_review: 'weekly_review',
    trial_ending: 'shopping_reminders',
    renewal_reminder: 'shopping_reminders',
    billing_issue: 'shopping_reminders',
};

/** Minimum expected value for a candidate to be sendable. Kills generic spam. */
export const MIN_SENDABLE_VALUE = 0.3;

const URGENCY_RANK: Record<NotificationUrgency, number> = { low: 0, normal: 1, high: 2, critical: 3 };

export type PreferenceMap = Record<NotificationChannel, NotificationPreference>;

export function defaultPreferences(): PreferenceMap {
    return {
        outfit_reminders: 'normal',
        event_reminders: 'normal',
        price_alerts: 'normal',
        fit_feedback: 'important_only',
        weekly_review: 'normal',
        travel_packing: 'normal',
        shopping_reminders: 'important_only',
    };
}

export interface PolicyDecision {
    sendable: boolean;
    reason: string;
}

/**
 * Decides whether a candidate may surface. Quiet hours respected unless
 * the candidate is explicitly quiet-hours compatible or critical.
 */
export function evaluateCandidate(
    candidate: NotificationCandidate,
    prefs: PreferenceMap,
    inQuietHours: boolean,
): PolicyDecision {
    if (candidate.expiresInHours !== null && candidate.expiresInHours <= 0) {
        return { sendable: false, reason: 'expired' };
    }
    if (!candidate.actionable) return { sendable: false, reason: 'not_actionable' };
    if (candidate.expectedValue < MIN_SENDABLE_VALUE) return { sendable: false, reason: 'low_value' };
    if (candidate.hoursSinceLastSameType < candidate.cooldownHours) return { sendable: false, reason: 'cooldown' };
    const channel = TYPE_TO_CHANNEL[candidate.type];
    const pref = prefs[channel];
    if (pref === 'off') return { sendable: false, reason: 'channel_off' };
    if (pref === 'important_only' && URGENCY_RANK[candidate.urgency] < URGENCY_RANK.high) {
        return { sendable: false, reason: 'below_important_threshold' };
    }
    if (inQuietHours && !candidate.quietHoursCompatible && candidate.urgency !== 'critical') {
        return { sendable: false, reason: 'quiet_hours' };
    }
    return { sendable: true, reason: 'ok' };
}

/** Picks the single best sendable candidate (highest urgency, then value). */
export function selectTopCandidate(
    candidates: NotificationCandidate[],
    prefs: PreferenceMap,
    inQuietHours: boolean,
): NotificationCandidate | null {
    const sendable = candidates.filter((c) => evaluateCandidate(c, prefs, inQuietHours).sendable);
    if (sendable.length === 0) return null;
    sendable.sort((a, b) => URGENCY_RANK[b.urgency] - URGENCY_RANK[a.urgency] || b.expectedValue - a.expectedValue);
    return sendable[0];
}

// ── Re-engagement (6M) — value-based re-entry only ───────────────

export type ReengagementTrigger =
    | 'resume_closet_scan'
    | 'upcoming_trip'
    | 'saved_price_watch'
    | 'seasonal_rotation'
    | 'wardrobe_review';

export function reengagementCopy(trigger: ReengagementTrigger): { title: string; body: string } {
    const copy: Record<ReengagementTrigger, { title: string; body: string }> = {
        resume_closet_scan: { title: 'Your closet scan is waiting', body: 'Pick up where you left off — 2 minutes finishes it.' },
        upcoming_trip: { title: 'Packing help is ready', body: 'Build a travel capsule from clothes you own.' },
        saved_price_watch: { title: 'Price update on a saved item', body: 'One of your watched items changed price.' },
        seasonal_rotation: { title: 'Season change check-in', body: 'Review what to store, repair, or bring back.' },
        wardrobe_review: { title: 'Your weekly wardrobe review', body: 'See what worked this week and what needs attention.' },
    };
    return copy[trigger];
}

// ── Message templates (6BL) — value-based, never promotional spam ─

export type MessageTemplateId = 'weekly_plan_ready' | 'price_watch_hit' | 'event_prep' | 'seasonal_review' | 'trial_ending' | 'renewal' | 'billing_issue' | 'grace_period';

export function renderTemplate(id: MessageTemplateId, vars: Record<string, string>): { title: string; body: string } {
    switch (id) {
        case 'weekly_plan_ready':
            return { title: 'Your weekly plan is ready', body: `Outfits planned for ${vars.week ?? 'this week'} from your wardrobe.` };
        case 'price_watch_hit':
            return { title: 'Price target reached', body: `${vars.item ?? 'A saved item'} is now ${vars.price ?? 'at your target price'}.` };
        case 'event_prep':
            return { title: `Ready for ${vars.event ?? 'your event'}?`, body: 'Your outfit is planned and checked against the weather.' };
        case 'seasonal_review':
            return { title: 'Seasonal wardrobe review', body: 'A calm review of what to keep, store, or repair.' };
        case 'trial_ending':
            return { title: `Trial ends ${vars.date ?? 'soon'}`, body: 'Your wardrobe and history stay available either way.' };
        case 'renewal':
            return { title: 'Subscription renews soon', body: `Renews ${vars.date ?? 'soon'}. Manage anytime in Settings.` };
        case 'billing_issue':
            return { title: 'Payment needs attention', body: 'Your subscription hit a billing issue. Premium features pause; your wardrobe stays fully available.' };
        case 'grace_period':
            return { title: 'Still in grace period', body: 'Premium continues briefly while billing resolves.' };
    }
}

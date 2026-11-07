/**
 * Phase 6BO–6BQ, 6BR — Growth analytics: dashboard metric
 * definitions (vendor-neutral), versioned KPIs, anti-vanity
 * guardrails, low-friction feedback with cooldown.
 */

import type { FeedbackEvent, FeedbackReason, KpiDefinition } from '../../types/growth';

export const GROWTH_DASHBOARD_DOMAINS = [
    'activation',
    'retention',
    'engagement',
    'wardrobe_value',
    'purchase_intelligence',
    'premium_conversion',
    'premium_retention',
    'provider_cost',
    'recommendation_quality',
] as const;

// ── KPI definitions (6BP) — versioned, never silently changed ────

export const KPI_DEFINITIONS: KpiDefinition[] = [
    { kpiId: 'activated_user', version: 1, definition: 'Completed at least one high-value wardrobe action (saved outfit, confirmed wear, BuyCheck, or aha moment).' },
    { kpiId: 'successful_outfit_recommendation', version: 1, definition: 'Saved, worn, or positively rated — not merely viewed.' },
    { kpiId: 'successful_buycheck', version: 1, definition: 'Result viewed plus later outcome evidence or explicit helpful rating.' },
    { kpiId: 'no_buy_acceptance_rate', version: 1, definition: 'Share of no-buy recommendations followed by no purchase of a similar item within 30 days.' },
];

export function kpiDefinition(kpiId: string): KpiDefinition | null {
    return KPI_DEFINITIONS.find((k) => k.kpiId === kpiId) ?? null;
}

// ── Anti-vanity guardrail (6BQ) ──────────────────────────────────

export const VANITY_METRICS = ['raw_screen_views', 'notification_opens', 'catalog_scroll_depth'] as const;

export function isVanityMetric(metric: string): boolean {
    return (VANITY_METRICS as readonly string[]).includes(metric);
}

// ── Feedback collection (6BR) ────────────────────────────────────

export const FEEDBACK_COOLDOWN_ACTIONS = 5;

export function shouldAskFeedback(actionsSinceLastAsk: number, lastHelpful: boolean | null): boolean {
    if (actionsSinceLastAsk < FEEDBACK_COOLDOWN_ACTIONS) return false;
    if (lastHelpful === false && actionsSinceLastAsk < FEEDBACK_COOLDOWN_ACTIONS * 2) return false; // back off after negative
    return true;
}

export function recordFeedback(feature: string, helpful: boolean, reason: FeedbackReason | null): FeedbackEvent {
    return { helpful, reason, feature, at: new Date().toISOString() };
}

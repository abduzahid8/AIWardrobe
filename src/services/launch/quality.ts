/**
 * Phase 7BO–7BQ, 7BT–7BZ, 7CA–7CH — Quality: bug triage, debt register,
 * known limitations, QA plans, scale/abuse models.
 */

import type { BlockerSeverity } from '../../types/launch';

export type BugCategory =
    | 'security_data' | 'crash' | 'core_flow' | 'billing' | 'ai_quality'
    | 'ui' | 'performance' | 'low_priority';

export interface BugReport {
    category: BugCategory;
    reproducible: boolean;
    affectedVersion: string;
    workaround: string | null;
    severity: BlockerSeverity;
    owner: string;
}

export function triagePriority(bug: BugReport): number {
    const sev = { BLOCKER: 0, HIGH: 1, MEDIUM: 2, LOW: 3 } as const;
    let score = sev[bug.severity] * 10;
    if (bug.category === 'security_data' || bug.category === 'billing') score -= 5;
    if (!bug.reproducible) score += 3;
    if (bug.workaround) score += 2;
    return score;
}

export type DebtHorizon = 'before_launch' | 'within_7d' | 'within_30d' | 'acceptable';

export interface DebtItem {
    id: string;
    horizon: DebtHorizon;
    description: string;
    owner: string;
}

export const QUALITY_DEBT_SEED: DebtItem[] = [
    { id: 'QD-full-suite-flakes', horizon: 'within_30d', description: '5 pre-existing failing suites (calendar/occasion/pipeline/lifecycle/stylist) predate Phase 5, unrelated to launch domains', owner: 'qa' },
];

export const KNOWN_LIMITATIONS = [
    'Limited merchant coverage; unsupported URLs fall back to manual add.',
    'Price history unavailable for some products; shown as unavailable, never fabricated.',
    'Fit confidence depends on evidence; low evidence never produces a guarantee.',
    'VTON is visualization, not fit proof.',
    'Visual parsing may need user correction; correction always wins.',
] as const;

export const MANUAL_QA_MATRIX = [
    'ios_current', 'ios_old_supported', 'android_current', 'android_low_memory',
    'small_screen', 'large_screen', 'slow_network', 'offline',
    'fresh_install', 'upgrade_install', 'free_tier', 'premium_tier',
    'dark_mode', 'light_mode',
] as const;

export const DEVICE_MATRIX = [
    'older_low_memory_android', 'current_android', 'older_supported_iphone', 'current_iphone',
] as const;

// ── Scale model (7CA–7CE) — scenario ranges, no invented traffic ──

export type LoadScenario = 'users_100' | 'users_1k' | 'users_10k' | 'users_100k';

export interface LoadEstimate {
    scenario: LoadScenario;
    outfitRequestsPerDay: number | null;
    buycheckRequestsPerDay: number | null;
    imageAnalysesPerDay: number | null;
    note: string;
}

/** Nulls where no data exists — ranges filled from real metrics post-launch. */
export function loadModel(inputs: { dau: number; outfitsPerUserPerDay: number | null }): LoadEstimate[] {
    const per = (rate: number | null) => (rate === null ? null : Math.round(inputs.dau * rate));
    const scenario: LoadScenario = inputs.dau <= 100 ? 'users_100' : inputs.dau <= 1000 ? 'users_1k' : inputs.dau <= 10000 ? 'users_10k' : 'users_100k';
    return [{
        scenario,
        outfitRequestsPerDay: per(inputs.outfitsPerUserPerDay),
        buycheckRequestsPerDay: null,
        imageAnalysesPerDay: null,
        note: 'Nulls = unknown baseline; measure first 7 days before sizing jobs.',
    }];
}

export const DB_HOT_PATHS = [
    'shop_catalog_discovery', 'wardrobe_items', 'outfit_history', 'style_evidence',
    'fit_evidence', 'price_history', 'recommendation_feedback',
] as const;

export const ABUSE_SIGNALS = [
    'automated_vton_abuse', 'expensive_ai_spam', 'catalog_scraping', 'referral_abuse',
    'signup_abuse', 'url_fetch_abuse', 'large_image_abuse',
] as const;

export function webhookIdempotencyKey(eventId: string, eventType: string): string {
    return `${eventType}:${eventId}`;
}

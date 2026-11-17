/**
 * Phase 6BI–6BJ, 6BN, 6BM, 6BU, 6BX–6BY — Monetization guardrails:
 * affiliate/sponsorship boundaries, purchase-intent separation,
 * subscription communication safety, server authority boundary,
 * store compliance checklist, performance safety, safe defaults.
 */

import type { RankedRecommendation } from '../../types/growth';

// ── Affiliate boundary (6BI) ─────────────────────────────────────
// Ranking stays user-value-first; commission never alters scores.

export function applyAffiliateAttribution<T extends RankedRecommendation>(
    ranked: T[],
    affiliateFlags: Record<string, boolean>,
): (T & { affiliateAttributed: boolean })[] {
    // Attribution logged separately; order and scores untouched.
    return ranked.map((r) => ({ ...r, affiliateAttributed: affiliateFlags[r.productId] === true }));
}

// ── Sponsorship safety (6BJ) ─────────────────────────────────────

export interface SponsoredSlot {
    slotId: string;
    productId: string;
    label: 'Sponsored';
}

export function mergeSponsoredSlot<T extends RankedRecommendation>(
    organic: T[],
    slot: SponsoredSlot | null,
    hardConstraintPass: (productId: string) => boolean,
): (T | (T & { sponsoredLabel: 'Sponsored' }))[] {
    if (!slot) return organic;
    // Sponsored never overrides hard fit/wardrobe constraints.
    if (!hardConstraintPass(slot.productId)) return organic;
    // Separate slot at the end, labeled — never interleaved silently.
    return [...organic, { ...(organic[0] as T | undefined ?? { productId: slot.productId, score: 0 } as T), productId: slot.productId, sponsoredLabel: 'Sponsored' as const }];
}

// ── Purchase intent vs recommendation (6BK) ──────────────────────

export type ShoppingIntent = 'user_asked_to_shop' | 'system_noticed_gap' | 'system_recommends_no_buy' | 'unrelated_task';

export function shoppingCtaAllowed(intent: ShoppingIntent): boolean {
    // No shopping CTAs during unrelated wardrobe tasks.
    return intent === 'user_asked_to_shop' || intent === 'system_noticed_gap';
}

// ── Communication safety (6BM) ───────────────────────────────────

export type SubscriptionMessageKind = 'trial_ending' | 'renewal' | 'billing_issue' | 'cancellation' | 'grace_period';

/** Every lifecycle state links restore/manage; never implies failure falsely. */
export function subscriptionMessage(kind: SubscriptionMessageKind): { text: string; includesManageLink: boolean } {
    const copy: Record<SubscriptionMessageKind, string> = {
        trial_ending: 'Your trial ends soon. Manage or cancel anytime in Settings — your wardrobe stays available.',
        renewal: 'Your subscription renews soon. Manage anytime in Settings.',
        billing_issue: 'A billing issue paused premium features. Your wardrobe and history are intact.',
        cancellation: 'Cancellation confirmed — premium runs until the period ends, then free features continue.',
        grace_period: 'You are in a grace period: premium continues briefly while billing resolves.',
    };
    return { text: copy[kind], includesManageLink: true };
}

// ── Server authority boundary (6BU) ──────────────────────────────

export interface ServerEntitlementVerifier {
    /** Trusted backend verifies entitlement for expensive server actions. */
    verify(userId: string, featureId: string): Promise<{ allowed: boolean; reason: string }>;
}

export class AllowlistServerVerifier implements ServerEntitlementVerifier {
    constructor(private allowed: Set<string>) {}
    async verify(userId: string, featureId: string) {
        const key = `${userId}:${featureId}`;
        return this.allowed.has(key) ? { allowed: true, reason: 'ok' } : { allowed: false, reason: 'not_entitled' };
    }
}

// ── Store compliance checklist (6BN) ─────────────────────────────

export const STORE_COMPLIANCE_CHECKLIST = [
    'subscription_disclosure',
    'restore_purchase_button',
    'manage_subscription_link',
    'terms_link',
    'privacy_link',
    'localized_price_display',
    'trial_copy_accuracy',
    'cancellation_handling',
    'account_deletion_access',
] as const;

export function complianceChecklist(): { item: string; done: boolean }[] {
    return STORE_COMPLIANCE_CHECKLIST.map((item) => ({ item, done: false }));
}

// ── Performance safety (6BX) ─────────────────────────────────────

export const GROWTH_PERF_RULES = {
    paywallSdkLazy: true,
    noEntitlementNetworkBeforeCoreUi: true,
    entitlementCached: true,
    analyticsNonBlocking: true,
    maxGrowthConfigKbAtStartup: 50,
} as const;

// ── Safe defaults (6BY) ──────────────────────────────────────────

export type GrowthDegradation = 'store_unavailable' | 'analytics_unavailable' | 'referral_unavailable';

export function safeDefaultFor(degradation: GrowthDegradation): string {
    switch (degradation) {
        case 'store_unavailable':
            return 'Launch with free features; premium state unknown/degraded, no endless paywall loading.';
        case 'analytics_unavailable':
            return 'Product works fully; events dropped locally.';
        case 'referral_unavailable':
            return 'Sharing works without rewards.';
    }
}

/**
 * Phase 6O–6Q, 6R, 6AU, 6X–6Z — Entitlement matrix, subscription
 * lifecycle, usage metering, graceful limits, quota resets,
 * RevenueCat infrastructure boundary, restore/recovery.
 *
 * Wraps the existing tier system (`free | lite | premium | vip` from
 * `store/subscriptionStore`) and IAP service. Domain code asks this
 * module — never the RevenueCat/vendor SDK directly. No boolean-only
 * isPremium: lifecycle states degrade safely.
 */

import type {
    CostClass,
    FeatureDefinition,
    QuotaState,
    SubscriptionLifecycleState,
    SubscriptionTierName,
} from '../../types/growth';

export const ENTITLEMENT_MATRIX_VERSION = 1;

/**
 * Central entitlement matrix (6P). Tiers mirror the existing store tiers.
 * Free stays genuinely useful: wardrobe, basic outfits, manual planning.
 */
export const ENTITLEMENT_MATRIX: FeatureDefinition[] = [
    {
        featureId: 'wardrobe_management',
        freeAvailability: true, freeLimit: null,
        premiumAvailability: true, premiumLimit: null,
        costClass: 'free_compute', fallback: 'always_available',
        upgradeReason: '', requiredEntitlement: 'free',
    },
    {
        featureId: 'basic_outfit_generation',
        freeAvailability: true, freeLimit: 5,
        premiumAvailability: true, premiumLimit: null,
        costClass: 'cheap_api', fallback: 'manual_planning',
        upgradeReason: 'Unlimited outfit generation with deeper analysis.',
        requiredEntitlement: 'free',
    },
    {
        featureId: 'basic_buycheck',
        freeAvailability: true, freeLimit: 10,
        premiumAvailability: true, premiumLimit: null,
        costClass: 'cheap_api', fallback: 'basic_fit_guidance',
        upgradeReason: 'Unlimited deep purchase checks with price history.',
        requiredEntitlement: 'free',
    },
    {
        featureId: 'advanced_visual_analysis',
        freeAvailability: true, freeLimit: 5,
        premiumAvailability: true, premiumLimit: null,
        costClass: 'expensive_vision', fallback: 'basic_outfit_generation',
        upgradeReason: 'Advanced visual analysis costs more to run per use.',
        requiredEntitlement: 'premium',
    },
    {
        featureId: 'recreate_look',
        freeAvailability: true, freeLimit: 3,
        premiumAvailability: true, premiumLimit: null,
        costClass: 'expensive_vision', fallback: 'manual_reference_browse',
        upgradeReason: 'Unlimited look recreation with wardrobe matching.',
        requiredEntitlement: 'premium',
    },
    {
        featureId: 'deep_buycheck',
        freeAvailability: false, freeLimit: 0,
        premiumAvailability: true, premiumLimit: null,
        costClass: 'cheap_api', fallback: 'basic_buycheck',
        upgradeReason: 'Deep checks add price history and multi-merchant comparison.',
        requiredEntitlement: 'premium',
    },
    {
        featureId: 'price_watch',
        freeAvailability: true, freeLimit: 1,
        premiumAvailability: true, premiumLimit: 20,
        costClass: 'monitored_service', fallback: 'manual_price_check',
        upgradeReason: 'Track price drops on up to 20 saved items.',
        requiredEntitlement: 'premium',
    },
    {
        featureId: 'weekly_autonomous_planner',
        freeAvailability: false, freeLimit: 0,
        premiumAvailability: true, premiumLimit: null,
        costClass: 'cheap_api', fallback: 'manual_planning',
        upgradeReason: 'Automatic weekly outfit plans built from your wardrobe.',
        requiredEntitlement: 'premium',
    },
    {
        featureId: 'premium_vton',
        freeAvailability: false, freeLimit: 0,
        premiumAvailability: true, premiumLimit: 30,
        costClass: 'expensive_generation', fallback: 'reference_preview',
        upgradeReason: 'High-cost try-on renders are a premium convenience.',
        requiredEntitlement: 'premium',
    },
    {
        featureId: 'travel_capsule_automation',
        freeAvailability: false, freeLimit: 0,
        premiumAvailability: true, premiumLimit: null,
        costClass: 'cheap_api', fallback: 'manual_capsule',
        upgradeReason: 'Automatic travel capsules from your wardrobe.',
        requiredEntitlement: 'premium',
    },
    {
        featureId: 'advanced_wardrobe_analytics',
        freeAvailability: true, freeLimit: 1,
        premiumAvailability: true, premiumLimit: null,
        costClass: 'free_compute', fallback: 'basic_insights',
        upgradeReason: 'Deeper wardrobe analytics and history depth.',
        requiredEntitlement: 'premium',
    },
];

export function featureDefinition(featureId: string): FeatureDefinition | null {
    return ENTITLEMENT_MATRIX.find((f) => f.featureId === featureId) ?? null;
}

function isPremiumTier(tier: SubscriptionTierName): boolean {
    return tier === 'premium' || tier === 'vip' || tier === 'lite';
}

/** Central access check — the only place tier logic lives (6P). */
export function canAccessFeature(featureId: string, tier: SubscriptionTierName): boolean {
    const def = featureDefinition(featureId);
    if (!def) return false;
    return isPremiumTier(tier) ? def.premiumAvailability : def.freeAvailability;
}

/** Effective limit for a tier; null = unlimited. */
export function limitFor(featureId: string, tier: SubscriptionTierName): number | null {
    const def = featureDefinition(featureId);
    if (!def) return 0;
    return isPremiumTier(tier) ? def.premiumLimit : def.freeLimit;
}

// ── Subscription lifecycle (6Z) ──────────────────────────────────

export interface SubscriptionSnapshot {
    tier: SubscriptionTierName;
    lifecycle: SubscriptionLifecycleState;
    trialEndsAt: string | null;
    renewsAt: string | null;
    lastVerifiedAt: string | null;
}

/** Maps store/IAP state onto the explicit lifecycle (safe defaults, 6BY). */
export function deriveLifecycle(input: {
    tier: SubscriptionTierName;
    trialActive: boolean;
    trialPending: boolean;
    billingIssue: boolean;
    gracePeriod: boolean;
    storeReachable: boolean;
}): SubscriptionLifecycleState {
    if (input.trialPending || !input.storeReachable) return 'unknown';
    if (input.billingIssue) return input.gracePeriod ? 'grace_period' : 'billing_issue';
    if (input.trialActive) return 'trial';
    if (input.tier === 'free') return 'free';
    return 'premium_active';
}

/**
 * Premium-gated but safe: unknown/offline store never locks the user out
 * of core wardrobe (fail-open for free features, degrade premium, 6Y/6BY).
 */
export function effectiveAccess(
    snapshot: SubscriptionSnapshot,
    featureId: string,
): { allowed: boolean; degraded: boolean; reason: string } {
    if (snapshot.lifecycle === 'unknown' || snapshot.lifecycle === 'unavailable') {
        const def = featureDefinition(featureId);
        const freeOk = def !== null && def.freeAvailability;
        return { allowed: freeOk, degraded: true, reason: 'entitlement_unknown_use_free' };
    }
    if (snapshot.lifecycle === 'billing_issue' || snapshot.lifecycle === 'grace_period') {
        // Wardrobe and history always available; premium regeneration pauses.
        const def = featureDefinition(featureId);
        const freeOk = def !== null && def.freeAvailability;
        return { allowed: freeOk, degraded: true, reason: 'billing_issue_free_only' };
    }
    const tier: SubscriptionTierName =
        snapshot.lifecycle === 'free' || snapshot.lifecycle === 'expired' ? 'free' : snapshot.tier;
    return { allowed: canAccessFeature(featureId, tier), degraded: false, reason: 'ok' };
}

// ── Usage metering (6Q) + quota reset (6AU) ───────────────────────

export function quotaRemaining(quota: QuotaState): number | null {
    if (quota.limit === null) return null; // unlimited
    return Math.max(0, quota.limit - quota.used);
}

export function isQuotaExhausted(quota: QuotaState): boolean {
    if (quota.limit === null) return false;
    return quota.used >= quota.limit;
}

/** Graceful limit UX (6R): what remains, why, when it resets. */
export function limitMessage(featureId: string, quota: QuotaState): { title: string; body: string } {
    const def = featureDefinition(featureId);
    const reset = quota.resetsAt ? ` Resets ${quota.resetsAt.slice(0, 10)}.` : '';
    return {
        title: `You've used this month's ${quota.limit ?? ''} ${featureId.replace(/_/g, ' ')} allowance.`.replace('  ', ' ').trim(),
        body: `${def?.fallback ? `Basic features still work (${def.fallback.replace(/_/g, ' ')}). ` : ''}${def?.upgradeReason ?? ''}${reset}`,
    };
}

// ── RevenueCat infrastructure boundary (6X) ──────────────────────
// Domain code depends on this interface; vendor SDK stays in infra.

export interface EntitlementProvider {
    readonly providerId: 'revenuecat' | 'mock' | 'storekit' | 'play_billing';
    currentEntitlement(): Promise<{ tier: SubscriptionTierName; verifiedAt: string }>;
    availablePlans(): Promise<{ planId: string; localizedPrice: string | null }[]>;
    restore(): Promise<{ tier: SubscriptionTierName; restored: boolean }>;
}

/** Local mock for tests/dev — no SDK, no network. */
export class MockEntitlementProvider implements EntitlementProvider {
    readonly providerId = 'mock' as const;
    constructor(private tier: SubscriptionTierName = 'free') {}
    async currentEntitlement() {
        return { tier: this.tier, verifiedAt: new Date().toISOString() };
    }
    async availablePlans() {
        return [
            { planId: 'monthly', localizedPrice: null },
            { planId: 'annual', localizedPrice: null },
        ];
    }
    async restore() {
        return { tier: this.tier, restored: true };
    }
}

export function costClassOf(featureId: string): CostClass {
    return featureDefinition(featureId)?.costClass ?? 'free_compute';
}

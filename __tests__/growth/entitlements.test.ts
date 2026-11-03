import {
    canAccessFeature,
    limitFor,
    deriveLifecycle,
    effectiveAccess,
    quotaRemaining,
    isQuotaExhausted,
    limitMessage,
    MockEntitlementProvider,
    ENTITLEMENT_MATRIX,
} from '../../src/services/growth/entitlements';
import { isNeverPaywalled } from '../../src/services/growth/monetization';
import { buildPaywallContext, paywallEligible, trialDisclosureCopy, trialDaysRemaining, whyPremiumCopy } from '../../src/services/growth/paywall';

describe('entitlement matrix', () => {
    test('free stays useful; premium-only features gated centrally', () => {
        expect(canAccessFeature('wardrobe_management', 'free')).toBe(true);
        expect(canAccessFeature('basic_buycheck', 'free')).toBe(true);
        expect(canAccessFeature('weekly_autonomous_planner', 'free')).toBe(false);
        expect(canAccessFeature('weekly_autonomous_planner', 'premium')).toBe(true);
        expect(limitFor('price_watch', 'free')).toBe(1);
        expect(limitFor('price_watch', 'premium')).toBe(20);
        expect(limitFor('wardrobe_management', 'free')).toBeNull();
        expect(canAccessFeature('nonexistent', 'premium')).toBe(false);
    });

    test('lifecycle unknown/offline degrades to free, never locks wardrobe', () => {
        const snap = { tier: 'premium' as const, lifecycle: 'unknown' as const, trialEndsAt: null, renewsAt: null, lastVerifiedAt: null };
        expect(effectiveAccess(snap, 'wardrobe_management')).toMatchObject({ allowed: true, degraded: true });
        expect(effectiveAccess(snap, 'weekly_autonomous_planner').allowed).toBe(false);
        const billing = { ...snap, lifecycle: 'billing_issue' as const };
        expect(effectiveAccess(billing, 'wardrobe_management').allowed).toBe(true);
        expect(deriveLifecycle({ tier: 'free', trialActive: false, trialPending: true, billingIssue: false, gracePeriod: false, storeReachable: true })).toBe('unknown');
        expect(deriveLifecycle({ tier: 'premium', trialActive: false, trialPending: false, billingIssue: true, gracePeriod: true, storeReachable: true })).toBe('grace_period');
    });

    test('quotas meter in meaningful units; limit UX shows fallback + reset', () => {
        expect(quotaRemaining({ featureId: 'x', used: 3, limit: 5, resetsAt: null, period: 'monthly' })).toBe(2);
        expect(isQuotaExhausted({ featureId: 'x', used: 5, limit: 5, resetsAt: null, period: 'monthly' })).toBe(true);
        expect(isQuotaExhausted({ featureId: 'x', used: 999, limit: null, resetsAt: null, period: null })).toBe(false);
        const msg = limitMessage('advanced_visual_analysis', { featureId: 'advanced_visual_analysis', used: 5, limit: 5, resetsAt: '2026-12-01T00:00:00Z', period: 'monthly' });
        expect(msg.body).toMatch(/still work/);
    });

    test('mock provider: no SDK leakage; essentials never paywalled', async () => {
        const p = new MockEntitlementProvider('premium');
        expect(p.providerId).toBe('mock');
        expect((await p.currentEntitlement()).tier).toBe('premium');
        for (const f of ['account_access', 'restore_purchases', 'basic_wardrobe_access', 'offline_wardrobe_visibility']) {
            expect(isNeverPaywalled(f)).toBe(true);
        }
        expect(ENTITLEMENT_MATRIX.length).toBeGreaterThanOrEqual(10);
    });
});

describe('paywall system', () => {
    test('context always carries free fallback; no paywall on first launch', () => {
        const ctx = buildPaywallContext({ triggerFeatureId: 'recreate_look', triggerDescription: 'Unlimited recreations', resetsAt: null, benefits: ['a'], plans: [], trialDisclosure: null });
        expect(ctx.freeFallback).toBeTruthy();
        expect(paywallEligible({ appLaunches: 1, coreActionsCompleted: 5 })).toBe(false);
        expect(paywallEligible({ appLaunches: 3, coreActionsCompleted: 0 })).toBe(false);
        expect(paywallEligible({ appLaunches: 3, coreActionsCompleted: 1 })).toBe(true);
    });

    test('trial disclosure clear; why-premium from safe local context', () => {
        const copy = trialDisclosureCopy({ trialDays: 7, startsAt: '2026-11-01', endsAt: '2026-11-08', convertsToPlanId: 'annual' });
        expect(copy).toMatch(/unless cancelled/);
        expect(trialDaysRemaining('2026-11-08T00:00:00Z', '2026-11-06T00:00:00Z')).toBe(2);
        expect(whyPremiumCopy('basic_buycheck')).toMatch(/BuyCheck often/);
    });
});

import { effectiveAccess, MockEntitlementProvider } from '../../src/services/growth/entitlements';
import { AllowlistServerVerifier } from '../../src/services/growth/compliance';
import { applyDowngrade } from '../../src/services/growth/value';
import { buildPaywallContext, paywallEligible, defaultPlans } from '../../src/services/growth/paywall';
import { formatRegionalPrice, unknownStudentEligibility, promoEligible } from '../../src/services/growth/pricing';
import type { SubscriptionSnapshot } from '../../src/services/growth/entitlements';

function snap(lifecycle: SubscriptionSnapshot['lifecycle'], tier: SubscriptionSnapshot['tier'] = 'free'): SubscriptionSnapshot {
    return { tier, lifecycle, trialEndsAt: null, renewsAt: null, lastVerifiedAt: new Date().toISOString() };
}

describe('subscription downgrade safety (6BS)', () => {
    test('premium → free preserves data; trial → free preserves history', () => {
        for (const used of [['premium_vton'], ['weekly_autonomous_planner', 'deep_buycheck']]) {
            const out = applyDowngrade(used);
            expect(out.userDataPreserved).toBe(true);
            expect(out.historyReadable).toBe(true);
            expect(out.expensiveRegenerationDisabled).toEqual(used);
        }
        // After downgrade the matrix still allows free features.
        expect(effectiveAccess(snap('expired'), 'wardrobe_management').allowed).toBe(true);
        expect(effectiveAccess(snap('expired'), 'weekly_autonomous_planner').allowed).toBe(false);
    });

    test('offline unknown entitlement never locks core wardrobe', () => {
        const unknown = snap('unknown', 'premium');
        expect(effectiveAccess(unknown, 'wardrobe_management')).toMatchObject({ allowed: true, degraded: true });
        expect(effectiveAccess(unknown, 'premium_vton').allowed).toBe(false);
    });

    test('quota exhausted blocks regeneration but keeps free fallback', () => {
        const billing = snap('billing_issue', 'premium');
        expect(effectiveAccess(billing, 'basic_buycheck').allowed).toBe(true);
        expect(effectiveAccess(billing, 'deep_buycheck').allowed).toBe(false);
    });
});

describe('monetization adversarial tests (6BT)', () => {
    test('client entitlement is display-only: server verifier is authoritative', async () => {
        // Spoofed client claims "premium" — server check still denies.
        const v = new AllowlistServerVerifier(new Set());
        expect((await v.verify('spoofed-user', 'deep_buycheck')).allowed).toBe(false);
    });

    test('stale entitlement cache degrades to unknown, not premium', () => {
        const stale = snap('unknown');
        const access = effectiveAccess(stale, 'deep_buycheck');
        expect(access.allowed).toBe(false);
        expect(access.degraded).toBe(true);
    });

    test('restore race: mock restore resolves consistently', async () => {
        const p = new MockEntitlementProvider('premium');
        const [a, b] = await Promise.all([p.restore(), p.restore()]);
        expect(a.tier).toBe(b.tier);
        expect(a.restored).toBe(true);
    });

    test('paywall loop guard: no paywall before understood value', () => {
        expect(paywallEligible({ appLaunches: 1, coreActionsCompleted: 0 })).toBe(false);
        // Context always offers a free fallback + close path (close access).
        const ctx = buildPaywallContext({ triggerFeatureId: 'price_watch', triggerDescription: 'd', resetsAt: null, benefits: [], plans: defaultPlans({ trialDays: null }), trialDisclosure: null });
        expect(ctx.freeFallback).toBeTruthy();
        expect(ctx.plans.length).toBe(2);
    });

    test('unavailable price never fabricates a number', () => {
        expect(formatRegionalPrice({ planId: 'm', region: 'UZ', currency: 'UZS', displayPrice: null, taxIncluded: false })).toMatch(/unavailable/);
        expect(formatRegionalPrice({ planId: 'm', region: 'DE', currency: 'EUR', displayPrice: '4,99 €', taxIncluded: true })).toMatch(/incl\. tax/);
        expect(unknownStudentEligibility().eligible).toBe(false);
        expect(promoEligible({ isNewUser: true, isReturningUser: false, region: 'US', campaign: 'x', hadPriorTrial: false, hadPriorSubscription: false, storeOfferAvailable: false }).eligible).toBe(false);
        expect(promoEligible({ isNewUser: true, isReturningUser: false, region: 'US', campaign: 'x', hadPriorTrial: false, hadPriorSubscription: false, storeOfferAvailable: true }).eligible).toBe(true);
    });
});


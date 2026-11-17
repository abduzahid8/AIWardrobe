import { evaluateCandidate, selectTopCandidate, defaultPreferences, reengagementCopy, renderTemplate } from '../../src/services/growth/notifications';
import { kpiDefinition, isVanityMetric, shouldAskFeedback, recordFeedback } from '../../src/services/growth/analytics';
import {
    applyAffiliateAttribution,
    mergeSponsoredSlot,
    shoppingCtaAllowed,
    subscriptionMessage,
    AllowlistServerVerifier,
    safeDefaultFor,
} from '../../src/services/growth/compliance';
import {
    scorePremiumCandidate,
    valueRevenueGuardrail,
    grossContributionPerUser,
    registerReferral,
    __resetReferralIdempotencyForTests,
    winBackMessage,
    recordCancellationReason,
} from '../../src/services/growth/monetization';
import { buildShareCard, socialProofCopy, challengeFor } from '../../src/services/growth/sharing';
import { totalVerifiedSavingsShim } from './helpers';
import { verifySaving, utilizationSummary, cohortForIntent, contextualUpsell, applyDowngrade } from '../../src/services/growth/value';
import { appendTimelineEvent, summarizeGrowthDebug } from '../../src/services/growth/debug';

const base = {
    type: 'weekly_plan_ready' as const,
    urgency: 'normal' as const,
    expectedValue: 0.8,
    confidence: 0.9,
    cooldownHours: 168,
    hoursSinceLastSameType: 200,
    expiresInHours: 48,
    actionable: true,
    quietHoursCompatible: true,
};

describe('notification policy', () => {
    test('spam rejected: low value, non-actionable, cooldown, channel off, quiet hours', () => {
        const prefs = defaultPreferences();
        expect(evaluateCandidate({ ...base, expectedValue: 0.1 }, prefs, false).sendable).toBe(false);
        expect(evaluateCandidate({ ...base, actionable: false }, prefs, false).reason).toBe('not_actionable');
        expect(evaluateCandidate({ ...base, hoursSinceLastSameType: 1 }, prefs, false).reason).toBe('cooldown');
        expect(evaluateCandidate({ ...base, expiresInHours: 0 }, prefs, false).reason).toBe('expired');
        expect(evaluateCandidate({ ...base, quietHoursCompatible: false }, prefs, true).reason).toBe('quiet_hours');
        const off = { ...prefs, weekly_review: 'off' as const };
        expect(evaluateCandidate(base, off, false).reason).toBe('channel_off');
        // important_only blocks normal urgency
        const imp = { ...prefs, weekly_review: 'important_only' as const };
        expect(evaluateCandidate(base, imp, false).reason).toBe('below_important_threshold');
        // top candidate selection prefers urgency then value
        const top = selectTopCandidate([
            base,
            { ...base, type: 'price_target_reached' as const, urgency: 'high' as const, expectedValue: 0.5, cooldownHours: 24, hoursSinceLastSameType: 100 },
        ], prefs, false);
        expect(top?.type).toBe('price_target_reached');
        expect(reengagementCopy('upcoming_trip').title).toBeTruthy();
        expect(renderTemplate('billing_issue', {}).body).toMatch(/wardrobe stays fully available/);
    });
});

describe('monetization guardrails', () => {
    test('premium scorecard keeps cheap substitutable features free', () => {
        const keep = scorePremiumCandidate({ featureId: 'x', userValue: 0.4, recurringValue: 0.3, providerCost: 0.1, differentiation: 0.2, retentionImpact: 0.2, freeSubstituteExists: true, abusePotential: 0.1 });
        expect(keep.belongsInPremium).toBe(false);
        const prem = scorePremiumCandidate({ featureId: 'vton', userValue: 0.8, recurringValue: 0.7, providerCost: 0.9, differentiation: 0.8, retentionImpact: 0.6, freeSubstituteExists: false, abusePotential: 0.3 });
        expect(prem.belongsInPremium).toBe(true);
        expect(valueRevenueGuardrail(true, true).verdict).toBe('hold');
        expect(valueRevenueGuardrail(false, true).verdict).toBe('rollback');
        expect(valueRevenueGuardrail(true, false).verdict).toBe('ship');
        expect(grossContributionPerUser({ cohort: 'c', freeRetentionRate: 0.5, paidConversionRate: 0.1, paidRetentionRate: 0.8, avgRevenuePerPaidUser: 50, avgCostToServe: 2, refundRate: 0.05 })).toBeCloseTo(1.8, 5);
    });

    test('cancellation never blocked; win-back respects cooling-off', () => {
        expect(recordCancellationReason('too_expensive').reason).toBe('too_expensive');
        expect(winBackMessage({ daysSinceCancel: 5, improvementsSince: [], offerConfigured: true })).toBeNull();
        expect(winBackMessage({ daysSinceCancel: 45, improvementsSince: ['Faster plans'], offerConfigured: false })?.headline).toBeTruthy();
    });

    test('referral: self-referral + duplicates rejected, idempotent', () => {
        __resetReferralIdempotencyForTests();
        const reg = { inviterUserId: 'a', inviteeUserId: 'b', inviteToken: 'inv_12345678', idempotencyKey: 'k1' };
        expect(registerReferral(reg, { inviterUserId: null, inviteToken: null, status: 'none', rewardIdempotencyKey: null }).accepted).toBe(true);
        expect(registerReferral(reg, { inviterUserId: null, inviteToken: null, status: 'none', rewardIdempotencyKey: null }).reason).toBe('duplicate');
        expect(registerReferral({ ...reg, inviterUserId: 'b', inviteeUserId: 'b', idempotencyKey: 'k2' }, { inviterUserId: null, inviteToken: null, status: 'none', rewardIdempotencyKey: null }).reason).toBe('self_referral');
    });
});

describe('sharing, savings, analytics, compliance', () => {
    test('share cards block private fields; social proof needs cohort', () => {
        expect(buildShareCard({ kind: 'outfit', title: 't', summary: 's', payloadKeys: ['colors', 'price_history'], expiresAt: null })).toMatchObject({ error: expect.stringMatching(/share_blocked/) });
        const ok = buildShareCard({ kind: 'capsule', title: 't', summary: 's', payloadKeys: ['colors'], expiresAt: null });
        expect((ok as { shareToken: null }).shareToken).toBeNull();
        expect(socialProofCopy({ label: 'Popular', cohortSize: 10, valid: true })).toBeNull();
        expect(socialProofCopy({ label: 'Popular', cohortSize: 60, valid: true })).toMatch(/60/);
        expect(challengeFor('capsule_week').optedIn).toBe(false);
    });

    test('savings verified-only; utilization is insight not shame', () => {
        expect(verifySaving({ kind: 'skipped_duplicate', amountCents: 5000, currency: 'USD', provenance: 'buycheck:r1', actedUpon: true })?.amountCents).toBe(5000);
        expect(verifySaving({ kind: 'skipped_duplicate', amountCents: 5000, currency: 'USD', provenance: 'buycheck:r1', actedUpon: false })).toBeNull();
        expect(totalVerifiedSavingsShim()).toBe(0);
        expect(utilizationSummary(30, 18)).toMatch(/18 of 30/);
        expect(cohortForIntent('check_before_buy')).toBe('shopping_first');
        expect(contextualUpsell('price_watch', 'manual_price_check', 'Track more.')).toMatch(/manual price check/);
        const down = applyDowngrade(['premium_vton']);
        expect(down.userDataPreserved).toBe(true);
        expect(down.expensiveRegenerationDisabled).toEqual(['premium_vton']);
    });

    test('kpis versioned; vanity metrics flagged; feedback cooldown', () => {
        expect(kpiDefinition('activated_user')?.version).toBe(1);
        expect(kpiDefinition('nope')).toBeNull();
        expect(isVanityMetric('notification_opens')).toBe(true);
        expect(isVanityMetric('no_buy_acceptance_rate')).toBe(false);
        expect(shouldAskFeedback(2, null)).toBe(false);
        expect(shouldAskFeedback(6, null)).toBe(true);
        expect(shouldAskFeedback(6, false)).toBe(false);
        expect(recordFeedback('buycheck', true, null).helpful).toBe(true);
    });

    test('affiliate never reorders ranking; sponsored labeled + constraint-checked; shopping CTA gated', () => {
        const ranked = [{ productId: 'a', score: 0.9, sponsored: false }, { productId: 'b', score: 0.8, sponsored: false }];
        const attributed = applyAffiliateAttribution(ranked, { a: true });
        expect(attributed.map((r) => r.productId)).toEqual(['a', 'b']);
        expect(attributed[0].affiliateAttributed).toBe(true);
        expect(mergeSponsoredSlot(ranked, { slotId: 's', productId: 'ad', label: 'Sponsored' }, () => false)).toHaveLength(2);
        const merged = mergeSponsoredSlot(ranked, { slotId: 's', productId: 'ad', label: 'Sponsored' }, () => true);
        expect(merged).toHaveLength(3);
        expect(shoppingCtaAllowed('unrelated_task')).toBe(false);
        expect(shoppingCtaAllowed('user_asked_to_shop')).toBe(true);
        expect(subscriptionMessage('cancellation').includesManageLink).toBe(true);
        expect(safeDefaultFor('store_unavailable')).toMatch(/free features/);
    });

    test('server verifier authoritative; timeline first-occurrence; debug snapshot', async () => {
        const v = new AllowlistServerVerifier(new Set(['u:deep_buycheck']));
        expect((await v.verify('u', 'deep_buycheck')).allowed).toBe(true);
        expect((await v.verify('u', 'premium_vton')).reason).toBe('not_entitled');
        let tl = appendTimelineEvent([], 'first_outfit', 'Saved first outfit');
        tl = appendTimelineEvent(tl, 'first_outfit', 'duplicate ignored');
        expect(tl).toHaveLength(1);
        const snap = summarizeGrowthDebug({
            activation: { wardrobeStarted: true, enoughItemsForOutfit: true, firstOutfitGenerated: true, firstOutfitSaved: false, firstWearConfirmed: false, firstPurchaseCheck: false, firstPersonalizationSignal: false, firstAhaMoment: false, lastMilestoneAt: null },
            activated: false, lifecycle: 'free', quotas: [], paywallEligible: false, notificationCandidates: 0,
            referral: { inviterUserId: null, inviteToken: null, status: 'none', rewardIdempotencyKey: null }, verifiedSavingsCents: 0,
        });
        expect(snap.activationMilestones).toBe(3);
    });
});

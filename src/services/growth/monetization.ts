/**
 * Phase 6AB–6AH, 6AY–6AZ, 6BA–6BD — Monetization guardrails:
 * cost-to-serve, premium value scorecard, conversion/subscription
 * metrics, cancellation feedback, win-back, LTV modeling, value-vs-
 * revenue guardrails, paywall/onboarding/retention experiment hooks,
 * referral abuse prevention.
 *
 * Non-paywallable essentials (6AA): account/data access, restore,
 * basic wardrobe, saved data, safety notices, user corrections,
 * offline wardrobe visibility — never held hostage.
 */

import type { CancellationReason, ReferralState, VerifiedSaving } from '../../types/growth';

// ── Non-paywallable essentials (6AA) ─────────────────────────────

export const NEVER_PAYWALLED = [
    'account_access',
    'data_export',
    'data_delete',
    'restore_purchases',
    'basic_wardrobe_access',
    'saved_wardrobe_data',
    'safety_notices',
    'user_corrections',
    'offline_wardrobe_visibility',
] as const;

export function isNeverPaywalled(featureId: string): boolean {
    return (NEVER_PAYWALLED as readonly string[]).includes(featureId);
}

// ── Cost-to-serve (6AC) — configured estimates, never invented ───

export interface CostToServe {
    feature: string;
    provider: string;
    /** Null = unconfigured; never invent provider prices. */
    estimatedVariableCost: number | null;
    cacheBenefit: string;
    freeCapRationale: string;
    premiumMarginNote: string;
}

export function costToServe(input: {
    feature: string;
    provider: string;
    configuredUnitCost: number | null;
    cacheBenefit: string;
    freeCapRationale: string;
}): CostToServe {
    return {
        feature: input.feature,
        provider: input.provider,
        estimatedVariableCost: input.configuredUnitCost,
        cacheBenefit: input.cacheBenefit,
        freeCapRationale: input.freeCapRationale,
        premiumMarginNote: input.configuredUnitCost === null ? 'unpriced — do not set caps from cost' : 'review margin quarterly',
    };
}

// ── Premium value scorecard (6AD) ────────────────────────────────

export interface PremiumScorecard {
    featureId: string;
    userValue: number; // 0..1
    recurringValue: number;
    providerCost: number; // 0..1 (higher = more expensive)
    differentiation: number;
    retentionImpact: number;
    freeSubstituteExists: boolean;
    abusePotential: number;
    belongsInPremium: boolean;
    rationale: string;
}

export function scorePremiumCandidate(input: Omit<PremiumScorecard, 'belongsInPremium' | 'rationale'>): PremiumScorecard {
    const value = input.userValue * 0.35 + input.recurringValue * 0.25 + input.retentionImpact * 0.2 + input.differentiation * 0.2;
    const cost = input.providerCost * 0.6 + input.abusePotential * 0.4;
    const belongsInPremium = value >= 0.5 && (cost >= 0.3 || !input.freeSubstituteExists);
    return {
        ...input,
        belongsInPremium,
        rationale: belongsInPremium
            ? 'High recurring value with real serving cost or no free substitute.'
            : 'Keep free: low cost or adequate free substitute exists.',
    };
}

// ── Conversion funnel + subscription quality (6AE–6AF) ──────────

export const CONVERSION_FUNNEL_STEPS = [
    'paywall_viewed',
    'benefit_expanded',
    'plan_selected',
    'checkout_started',
    'purchase_completed',
    'restore_attempted',
    'restore_success',
    'paywall_closed',
] as const;

export const SUBSCRIPTION_QUALITY_EVENTS = [
    'trial_start',
    'trial_conversion',
    'first_renewal',
    'cancellation',
    'refund',
    'restore',
    'billing_issue',
    'feature_usage_after_upgrade',
    'premium_retention',
] as const;

// ── Cancellation + win-back (6AG–6AH) ────────────────────────────

export function recordCancellationReason(reason: CancellationReason): { reason: CancellationReason; at: string } {
    return { reason, at: new Date().toISOString() };
}

export type WinBackOffer = { headline: string; body: string } | null;

/** Respectful, infrequent; never aggressive repeated prompts. */
export function winBackMessage(input: {
    daysSinceCancel: number;
    improvementsSince: string[];
    offerConfigured: boolean;
}): WinBackOffer {
    if (input.daysSinceCancel < 30) return null; // cooling-off period
    const body = [
        'Everything free still works, and your wardrobe is intact.',
        ...input.improvementsSince.slice(0, 2).map((i) => `New since you left: ${i}.`),
        ...(input.offerConfigured ? ['An optional welcome-back offer is available.'] : []),
    ].join(' ');
    return { headline: 'Your wardrobe kept working without you', body };
}

// ── LTV modeling (6AY) — offline analytics abstraction ──────────

export interface CohortLtvInputs {
    cohort: string;
    freeRetentionRate: number;
    paidConversionRate: number;
    paidRetentionRate: number;
    avgRevenuePerPaidUser: number;
    avgCostToServe: number;
    refundRate: number;
}

export function grossContributionPerUser(input: CohortLtvInputs): number {
    const paidShare = input.paidConversionRate * input.paidRetentionRate;
    const revenue = paidShare * input.avgRevenuePerPaidUser * (1 - input.refundRate);
    return revenue - input.avgCostToServe;
}

// ── Value-vs-revenue guardrail (6AZ) ─────────────────────────────

export interface GuardrailCheck {
    revenueUp: boolean;
    coreValueDown: boolean;
    verdict: 'ship' | 'hold' | 'rollback';
}

export function valueRevenueGuardrail(revenueUp: boolean, coreValueDown: boolean): GuardrailCheck {
    if (coreValueDown) return { revenueUp, coreValueDown, verdict: revenueUp ? 'hold' : 'rollback' };
    return { revenueUp, coreValueDown, verdict: 'ship' };
}

// ── Experiment hooks (6BA–6BC; execution via Phase 5 framework) ──

export const PAYWALL_EXPERIMENT_VARIABLES = ['timing', 'messaging', 'feature_framing', 'plan_emphasis', 'free_limit', 'trial_visibility'] as const;

export const ONBOARDING_EXPERIMENT_VARIANTS = ['closet_first', 'first_outfit_first', 'buycheck_first', 'reference_look_first'] as const;

export const RETENTION_EXPERIMENT_GUARDRAILS = ['dismiss_rate', 'notification_disable_rate', 'session_quality', 'complaint_rate'] as const;

// ── Referral abuse prevention (6BD) ──────────────────────────────

export interface ReferralRegistration {
    inviterUserId: string;
    inviteeUserId: string;
    inviteToken: string;
    idempotencyKey: string;
}

const seenIdempotencyKeys = new Set<string>();

export function registerReferral(
    reg: ReferralRegistration,
    existing: ReferralState,
): { accepted: boolean; reason: string } {
    // Self-referral guard.
    if (reg.inviterUserId === reg.inviteeUserId) return { accepted: false, reason: 'self_referral' };
    if (!reg.inviteToken || reg.inviteToken.length < 8) return { accepted: false, reason: 'invalid_token' };
    // Reward idempotency.
    if (seenIdempotencyKeys.has(reg.idempotencyKey) || existing.rewardIdempotencyKey === reg.idempotencyKey) {
        return { accepted: false, reason: 'duplicate' };
    }
    seenIdempotencyKeys.add(reg.idempotencyKey);
    return { accepted: true, reason: 'ok' };
}

export function __resetReferralIdempotencyForTests(): void {
    seenIdempotencyKeys.clear();
}

// ── Savings ledger helpers (6AN detail) ──────────────────────────

export function totalVerifiedSavings(savings: VerifiedSaving[]): number {
    return savings.reduce((sum, s) => sum + Math.max(0, s.amountCents), 0);
}

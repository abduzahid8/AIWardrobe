/**
 * Phase 6S–6W, 6BG, 6BV–6BW — Paywall component system, contextual
 * timing, plan architecture, trial safety, localized copy.
 *
 * Principles: sell outcomes (not "powerful AI"), paywall at a moment of
 * understood value, always show the free fallback, never paywall right
 * after launch. Prices localized via store data; i18n keys, no hardcoded
 * English billing strings in components (keys defined in 6BW section).
 */

import type { PaywallContext, PlanDefinition } from '../../types/growth';
import { featureDefinition } from './entitlements';

export const PAYWALL_I18N_KEYS = {
    title: 'growth.paywall.title',
    benefits: 'growth.paywall.benefits',
    monthly: 'growth.paywall.plan_monthly',
    annual: 'growth.paywall.plan_annual',
    trialDisclosure: 'growth.paywall.trial_disclosure',
    restore: 'growth.paywall.restore',
    manage: 'growth.paywall.manage_subscription',
    close: 'growth.paywall.close',
    terms: 'growth.paywall.terms',
    privacy: 'growth.paywall.privacy',
    currentPlan: 'growth.paywall.current_plan',
} as const;

/** Builds paywall context at the moment of blocked value (6S). */
export function buildPaywallContext(input: {
    triggerFeatureId: string;
    triggerDescription: string;
    resetsAt: string | null;
    benefits: string[];
    plans: PlanDefinition[];
    trialDisclosure: string | null;
}): PaywallContext {
    const def = featureDefinition(input.triggerFeatureId);
    return {
        triggerFeatureId: input.triggerFeatureId,
        triggerDescription: input.triggerDescription,
        freeFallback: def?.fallback ?? 'basic features',
        resetsAt: input.resetsAt,
        benefits: input.benefits,
        plans: input.plans,
        trialDisclosure: input.trialDisclosure,
    };
}

/**
 * Contextual eligibility (6S): paywall only after understood value —
 * user has completed at least one core action. Never on first launch.
 */
export function paywallEligible(input: { appLaunches: number; coreActionsCompleted: number }): boolean {
    if (input.appLaunches < 2) return false;
    return input.coreActionsCompleted >= 1;
}

// ── Plans (6V) ───────────────────────────────────────────────────

export function defaultPlans(input: { trialDays: number | null }): PlanDefinition[] {
    return [
        { planId: 'monthly', billingPeriod: 'monthly', localizedPrice: null, trialDays: null, eligible: true },
        { planId: 'annual', billingPeriod: 'annual', localizedPrice: null, trialDays: input.trialDays, eligible: true },
    ];
}

// ── Trial safety (6W) ────────────────────────────────────────────

export interface TrialTerms {
    trialDays: number;
    startsAt: string;
    endsAt: string;
    convertsToPlanId: string;
}

export function trialDisclosureCopy(terms: TrialTerms): string {
    return (
        `Free for ${terms.trialDays} days, then converts to ${terms.convertsToPlanId} ` +
        `on ${terms.endsAt.slice(0, 10)} unless cancelled. Manage or cancel anytime in Settings.`
    );
}

export function trialDaysRemaining(endsAt: string, nowIso?: string): number {
    const ms = new Date(endsAt).getTime() - new Date(nowIso ?? new Date().toISOString()).getTime();
    return Math.max(0, Math.ceil(ms / 86_400_000));
}

// ── "Why Premium?" personalization (6BG) ─────────────────────────
// Personalized only from safe local context (feature usage counts).

export function whyPremiumCopy(topUsedFreeFeature: string | null): string {
    switch (topUsedFreeFeature) {
        case 'basic_buycheck':
            return 'You use BuyCheck often. Premium adds unlimited deep purchase checks and price watches.';
        case 'basic_outfit_generation':
            return 'You generate outfits often. Premium removes limits and adds weekly auto-planning.';
        case 'recreate_look':
            return 'You recreate looks often. Premium adds unlimited recreations from your wardrobe.';
        default:
            return 'Premium adds automation, deeper intelligence, and higher limits — your wardrobe stays useful either way.';
    }
}

// ── Accessibility contract (6BV) ─────────────────────────────────

export const PAYWALL_A11Y = {
    minCloseTargetPt: 44,
    requiresPlanComparisonLabels: true,
    requiresSelectedPlanAnnouncement: true,
    requiresDynamicTypeSupport: true,
} as const;

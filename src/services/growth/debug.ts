/**
 * Phase 6BE–6BF — Growth debug panel model + user value timeline.
 * Development-only view; no secrets, no private data.
 */

import type { ActivationState, ReferralState, SubscriptionLifecycleState } from '../../types/growth';

export interface GrowthDebugSnapshot {
    activationMilestones: number;
    activated: boolean;
    lifecycle: SubscriptionLifecycleState;
    quotaSummary: { featureId: string; used: number; limit: number | null }[];
    paywallEligible: boolean;
    notificationCandidates: number;
    referralStatus: ReferralState['status'];
    verifiedSavingsCents: number;
}

export type ValueTimelineKind =
    | 'first_outfit'
    | 'first_wear'
    | 'first_buycheck'
    | 'first_saving'
    | 'first_fit_correction'
    | 'first_capsule'
    | 'first_price_alert'
    | 'first_avoided_duplicate';

export interface ValueTimelineEvent {
    kind: ValueTimelineKind;
    at: string;
    /** Sanitized label only — no private details. */
    label: string;
}

export function appendTimelineEvent(
    timeline: ValueTimelineEvent[],
    kind: ValueTimelineKind,
    label: string,
): ValueTimelineEvent[] {
    if (timeline.some((e) => e.kind === kind)) return timeline; // first-occurrence timeline
    return [...timeline, { kind, at: new Date().toISOString(), label }];
}

export function summarizeGrowthDebug(input: {
    activation: ActivationState;
    activated: boolean;
    lifecycle: SubscriptionLifecycleState;
    quotas: { featureId: string; used: number; limit: number | null }[];
    paywallEligible: boolean;
    notificationCandidates: number;
    referral: ReferralState;
    verifiedSavingsCents: number;
}): GrowthDebugSnapshot {
    const milestones = (Object.values(input.activation) as unknown[]).filter((v) => v === true).length;
    return {
        activationMilestones: milestones,
        activated: input.activated,
        lifecycle: input.lifecycle,
        quotaSummary: input.quotas,
        paywallEligible: input.paywallEligible,
        notificationCandidates: input.notificationCandidates,
        referralStatus: input.referral.status,
        verifiedSavingsCents: input.verifiedSavingsCents,
    };
}

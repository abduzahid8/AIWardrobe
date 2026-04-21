/**
 * useTrialStatus — convenience hook for reading the 7-day free trial state.
 *
 * Automatically refreshes every minute so countdown display stays accurate
 * without requiring a full app restart.
 */
import { useEffect, useState } from 'react';
import useSubscriptionStore from '../store/subscriptionStore';

export interface TrialStatus {
    /** User is on free tier and within the 7-day trial window. */
    isTrialActive: boolean;
    /** Free tier trial has ended (or never started for legacy users). */
    isTrialExpired: boolean;
    /** Full days remaining in trial (0-7). */
    daysRemaining: number;
    /** Hours remaining — useful for "last 24h" display. */
    hoursRemaining: number;
    /**
     * True when the app should show the TrialExpiredScreen gate:
     * user is authenticated, trial is over, and no paid subscription is active.
     */
    shouldShowTrialGate: boolean;
    /** 0.0 → 1.0 progress through the trial (0 = day 1, 1 = last moment). */
    trialProgress: number;
}

export function useTrialStatus(): TrialStatus {
    // Prime with the current store snapshot so the first render is correct.
    const storeSnapshot = useSubscriptionStore.getState();

    const [status, setStatus] = useState<TrialStatus>(() =>
        buildStatus(storeSnapshot),
    );

    useEffect(() => {
        // Re-compute whenever the store changes…
        const unsub = useSubscriptionStore.subscribe(snap => {
            setStatus(buildStatus(snap));
        });

        // …and also refresh every 60 s so the countdown is live.
        const timer = setInterval(() => {
            setStatus(buildStatus(useSubscriptionStore.getState()));
        }, 60_000);

        return () => {
            unsub();
            clearInterval(timer);
        };
    }, []);

    return status;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

type StoreSnap = ReturnType<typeof useSubscriptionStore.getState>;

function buildStatus(snap: StoreSnap): TrialStatus {
    const {
        isTrialActive,
        isTrialExpired,
        trialDaysRemaining,
        trialStartedAt,
        hasActiveSubscription,
    } = snap;

    // Hours remaining for last-day granularity
    let hoursRemaining = 0;
    if (isTrialActive && trialStartedAt) {
        const { FREE_TRIAL_DAYS } = require('../store/subscriptionStore');
        const trialEnd = new Date(trialStartedAt);
        trialEnd.setDate(trialEnd.getDate() + FREE_TRIAL_DAYS);
        const ms = trialEnd.getTime() - Date.now();
        hoursRemaining = Math.max(0, Math.ceil(ms / (1000 * 60 * 60)));
    }

    const trialProgress = isTrialActive
        ? 1 - trialDaysRemaining / 7
        : isTrialExpired
        ? 1
        : 0;

    return {
        isTrialActive,
        isTrialExpired,
        daysRemaining: trialDaysRemaining,
        hoursRemaining,
        shouldShowTrialGate: isTrialExpired && !hasActiveSubscription,
        trialProgress: Math.min(1, Math.max(0, trialProgress)),
    };
}

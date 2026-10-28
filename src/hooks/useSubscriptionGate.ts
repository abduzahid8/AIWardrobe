/**
 * useSubscriptionGate — the single place every screen/feature
 * goes through to check "can this user do X right now?".
 *
 * It unifies:
 *   - Tier-based boolean gates (analytics, tripPlanner, tryOns, …)
 *   - Per-day quota gates (homeSwipes 3/day for free, scans 5/day, …).
 *     NOTE: free.aiOutfits is 0 post-trial → Create Outfit always paywalls.
 *   - Automatic navigation to the Paywall when access is denied
 *   - A "consume" helper that both records usage and returns whether
 *     the action was allowed — so screens don't have to duplicate logic.
 *
 * Pending fail-open: while trial status is still resolving from
 * storage/server (isTrialPending), all gates allow through so trial-active
 * users are never wrongly paywalled on cold start. The window is ~1s.
 *
 * Typical patterns:
 *
 *   const { requireFeature } = useSubscriptionGate();
 *   if (!requireFeature('tripPlanner')) return;   // auto-navigates to paywall
 *
 *   const { consumeOrPaywall } = useSubscriptionGate();
 *   const ok = await consumeOrPaywall('aiOutfits');
 *   if (!ok) return;   // user saw paywall — stop
 *   runExpensiveGenerate();
 */

import { useCallback } from 'react';
import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import useSubscriptionStore, {
    TIER_FEATURES,
    DAILY_QUOTA_FEATURES,
    type FeatureKey,
    type SubscriptionTier,
} from '../../store/subscriptionStore';
import useDailyUsageStore, { type UsageConsumeOptions } from '../../store/dailyUsageStore';
import { authEvents } from '../../store/authEvents';
import type { RootStackParamList } from '../../navigation/types';

type Nav = NativeStackNavigationProp<RootStackParamList>;

interface SubscriptionGate {
    tier: SubscriptionTier;
    effectiveTier: SubscriptionTier;
    hasActiveSubscription: boolean;
    isPremium: boolean;

    /** Does the tier grant ANY access to this feature (ignoring daily usage)? */
    canAccess: (feature: FeatureKey) => boolean;

    /**
     * Like canAccess, but also accounts for today's daily quota.
     *   true  = tier has access AND user still has budget today
     *   false = blocked by tier OR blocked by daily limit
     */
    canUseNow: (feature: FeatureKey) => boolean;

    /**
     * Tier-gate: if denied, auto-navigates to Paywall. Returns true if allowed.
     * This does NOT decrement daily usage — pair with consume() after the
     * feature actually executes.
     */
    requireFeature: (feature: FeatureKey) => boolean;

    /**
     * Full gate + consume in one call:
     *   1. If tier has no access at all → navigates to Paywall, returns false.
     *   2. If daily quota is exhausted → navigates to Paywall, returns false.
     *   3. Otherwise decrements today's counter and returns true.
     *
     * Use this immediately before executing the paid feature.
     */
    consumeOrPaywall: (feature: FeatureKey, options?: UsageConsumeOptions) => Promise<boolean>;

    /**
     * Record a successful use (server-authoritative).
     * Returns { allowed, used, remaining } — `allowed=false` means the user
     * hit the quota on this call and the work should NOT proceed.
     */
    consume: (feature: FeatureKey, amount?: number) => Promise<{ allowed: boolean; used: number; remaining: number }>;

    /** Remaining uses today (-1 = unlimited, 0 = blocked). */
    getRemaining: (feature: FeatureKey) => number;

    /** How many the user has used today. */
    getUsed: (feature: FeatureKey) => number;

    /** Total daily limit for the current tier. -1 = unlimited, 0 = no access. */
    getDailyLimit: (feature: FeatureKey) => number;
}

export function useSubscriptionGate(): SubscriptionGate {
    const navigation = useNavigation<Nav>();
    // Selective subscriptions — previously `useSubscriptionStore()` with no
    // selector re-rendered every consumer on ANY store change (loading flags,
    // last-verified timestamps). This was a top cause of list/screen jank.
    const tier = useSubscriptionStore((s) => s.tier);
    const effectiveTier = useSubscriptionStore((s) => s.effectiveTier);
    const hasActiveSubscription = useSubscriptionStore((s) => s.hasActiveSubscription);
    const isPremium = useSubscriptionStore((s) => s.isPremium);
    const checkFeatureAccess = useSubscriptionStore((s) => s.checkFeatureAccess);
    const isTrialPending = useSubscriptionStore((s) => s.isTrialPending);

    const dailyRemaining = useDailyUsageStore((s) => s.getRemaining);
    const dailyUsed = useDailyUsageStore((s) => s.getUsed);
    const dailyCanUse = useDailyUsageStore((s) => s.canUse);
    const dailyConsume = useDailyUsageStore((s) => s.consume);
    // Getters are stable functions; subscribe to their data so quota labels update.
    useDailyUsageStore((s) => s.snapshot);

    const openPaywall = useCallback(() => {
        navigation.navigate('Paywall');
    }, [navigation]);

    const canAccess = useCallback(
        (feature: FeatureKey): boolean => {
            // Fail-open while trial status resolves (cold start) — see header.
            if (isTrialPending) return true;
            return checkFeatureAccess(feature);
        },
        [checkFeatureAccess, isTrialPending]
    );

    const canUseNow = useCallback(
        (feature: FeatureKey): boolean => {
            if (isTrialPending) return true;
            if (!checkFeatureAccess(feature)) return false;
            if (DAILY_QUOTA_FEATURES.includes(feature)) {
                return dailyCanUse(feature);
            }
            return true;
        },
        [checkFeatureAccess, dailyCanUse, isTrialPending]
    );

    const requireFeature = useCallback(
        (feature: FeatureKey): boolean => {
            if (isTrialPending) return true;
            if (!checkFeatureAccess(feature)) {
                openPaywall();
                return false;
            }
            if (DAILY_QUOTA_FEATURES.includes(feature) && !dailyCanUse(feature)) {
                openPaywall();
                return false;
            }
            return true;
        },
        [checkFeatureAccess, dailyCanUse, openPaywall, isTrialPending]
    );

    const consume = useCallback(
        async (feature: FeatureKey, amount = 1) => {
            return dailyConsume(feature, amount);
        },
        [dailyConsume]
    );

    const consumeOrPaywall = useCallback(
        async (feature: FeatureKey, options: UsageConsumeOptions = {}): Promise<boolean> => {
            const revision = authEvents.getRevision();
            const day = new Date().toISOString().slice(0, 10);
            if (options.cancelSignal?.aborted) return false;
            if (isTrialPending && !options.requireServer) return true;
            if (!isTrialPending && !checkFeatureAccess(feature)) {
                openPaywall();
                return false;
            }
            if (DAILY_QUOTA_FEATURES.includes(feature)) {
                // Server-authoritative check — the RPC either allows and
                // increments atomically, or denies (quota already hit).
                const result = await dailyConsume(feature, 1, options);
                if (options.cancelSignal?.aborted || revision !== authEvents.getRevision() || day !== new Date().toISOString().slice(0, 10)) return false;
                if (!result.allowed) {
                    openPaywall();
                    return false;
                }
            }
            return true;
        },
        [checkFeatureAccess, dailyConsume, openPaywall, isTrialPending]
    );

    const getRemaining = useCallback(
        (feature: FeatureKey): number => {
            // Fail-open while pending so the UI never flashes a lock/countdown
            // for trial-active users on cold start.
            if (isTrialPending) return -1;
            if (!DAILY_QUOTA_FEATURES.includes(feature)) {
                // Non-daily boolean features: -1 if granted, 0 if locked
                return checkFeatureAccess(feature) ? -1 : 0;
            }
            return dailyRemaining(feature);
        },
        [dailyRemaining, checkFeatureAccess, isTrialPending]
    );

    const getUsed = useCallback(
        (feature: FeatureKey): number => dailyUsed(feature),
        [dailyUsed]
    );

    const getDailyLimit = useCallback(
        (feature: FeatureKey): number => {
            const limit = TIER_FEATURES[effectiveTier][feature];
            if (typeof limit === 'boolean') return limit ? -1 : 0;
            return limit as number;
        },
        [effectiveTier]
    );

    return {
        tier,
        effectiveTier,
        hasActiveSubscription,
        isPremium,
        canAccess,
        canUseNow,
        requireFeature,
        consumeOrPaywall,
        consume,
        getRemaining,
        getUsed,
        getDailyLimit,
    };
}

export default useSubscriptionGate;

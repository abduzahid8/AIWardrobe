/**
 * useSubscriptionGate — the single place every screen/feature
 * goes through to check "can this user do X right now?".
 *
 * It unifies:
 *   - Tier-based boolean gates (analytics, tripPlanner, tryOns, …)
 *   - Per-day quota gates (aiOutfits 10/day, scans 5/day, …)
 *   - Automatic navigation to the Paywall when access is denied
 *   - A "consume" helper that both records usage and returns whether
 *     the action was allowed — so screens don't have to duplicate logic.
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
import useDailyUsageStore from '../../store/dailyUsageStore';
import type { RootStackParamList } from '../../navigation/types';

type Nav = NativeStackNavigationProp<RootStackParamList>;

interface SubscriptionGate {
    tier: SubscriptionTier;
    effectiveTier: SubscriptionTier;
    hasActiveSubscription: boolean;
    isPremium: boolean;
    isVIP: boolean;

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
    consumeOrPaywall: (feature: FeatureKey) => Promise<boolean>;

    /** Record a successful use (for non-gated accounting). */
    consume: (feature: FeatureKey, amount?: number) => Promise<number>;

    /** Remaining uses today (-1 = unlimited, 0 = blocked). */
    getRemaining: (feature: FeatureKey) => number;

    /** How many the user has used today. */
    getUsed: (feature: FeatureKey) => number;

    /** Total daily limit for the current tier. -1 = unlimited, 0 = no access. */
    getDailyLimit: (feature: FeatureKey) => number;
}

export function useSubscriptionGate(): SubscriptionGate {
    const navigation = useNavigation<Nav>();
    const {
        tier,
        effectiveTier,
        hasActiveSubscription,
        isPremium,
        isVIP,
        checkFeatureAccess,
    } = useSubscriptionStore();

    const {
        getRemaining: dailyRemaining,
        getUsed: dailyUsed,
        canUse: dailyCanUse,
        consume: dailyConsume,
    } = useDailyUsageStore();

    const openPaywall = useCallback(() => {
        navigation.navigate('Paywall');
    }, [navigation]);

    const canAccess = useCallback(
        (feature: FeatureKey): boolean => checkFeatureAccess(feature),
        [checkFeatureAccess]
    );

    const canUseNow = useCallback(
        (feature: FeatureKey): boolean => {
            if (!checkFeatureAccess(feature)) return false;
            if (DAILY_QUOTA_FEATURES.includes(feature)) {
                return dailyCanUse(feature);
            }
            return true;
        },
        [checkFeatureAccess, dailyCanUse]
    );

    const requireFeature = useCallback(
        (feature: FeatureKey): boolean => {
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
        [checkFeatureAccess, dailyCanUse, openPaywall]
    );

    const consume = useCallback(
        async (feature: FeatureKey, amount = 1) => {
            return dailyConsume(feature, amount);
        },
        [dailyConsume]
    );

    const consumeOrPaywall = useCallback(
        async (feature: FeatureKey): Promise<boolean> => {
            if (!checkFeatureAccess(feature)) {
                openPaywall();
                return false;
            }
            if (DAILY_QUOTA_FEATURES.includes(feature)) {
                if (!dailyCanUse(feature)) {
                    openPaywall();
                    return false;
                }
                await dailyConsume(feature);
            }
            return true;
        },
        [checkFeatureAccess, dailyCanUse, dailyConsume, openPaywall]
    );

    const getRemaining = useCallback(
        (feature: FeatureKey): number => {
            if (!DAILY_QUOTA_FEATURES.includes(feature)) {
                // Non-daily boolean features: -1 if granted, 0 if locked
                return checkFeatureAccess(feature) ? -1 : 0;
            }
            return dailyRemaining(feature);
        },
        [dailyRemaining, checkFeatureAccess]
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
        isVIP,
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

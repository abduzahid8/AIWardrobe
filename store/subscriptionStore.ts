import { authEvents } from './authEvents';
import readinessLogger from '../src/utils/logger';
import { create } from "zustand";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { supabase } from '../lib/supabase';

// ─────────────────────────────────────────────────────────────
// Storage keys (local cache — Supabase is source of truth for tier)
// ─────────────────────────────────────────────────────────────
const SUBSCRIPTION_KEY = "subscription_tier";
const SUBSCRIPTION_EXPIRY_KEY = "subscription_expiry";
const DAILY_USAGE_KEY = "daily_usage_v1";
const TRIAL_START_KEY = "trial_started_at_v1";

// ─────────────────────────────────────────────────────────────
// Trial config
// ─────────────────────────────────────────────────────────────
export const FREE_TRIAL_DAYS = 7;

// ─────────────────────────────────────────────────────────────
// Tiers
// ─────────────────────────────────────────────────────────────
export type SubscriptionTier = 'free' | 'lite' | 'premium' | 'vip';

/** Friendly display labels used in the UI. */
export const TIER_DISPLAY_NAMES: Record<SubscriptionTier, string> = {
    free: 'Free',
    lite: 'Lite',
    premium: 'Pro',
    vip: 'Max',
};

// ─────────────────────────────────────────────────────────────
// Feature matrix
// ─────────────────────────────────────────────────────────────
export const FREE_HOME_SWIPES_PER_DAY = 3;

export const TIER_FEATURES = {
    free: {
        /**
         * AI outfit generations per day (resets at local midnight).
         * 0 after the 7-day trial ends → "Create Outfit" is fully locked
         * and redirects to Paywall via useSubscriptionGate.
         * During trial effectiveTier is 'premium' so trial users get unlimited.
         */
        aiOutfits: 0,
        /** AI Try-On renders per day */
        tryOns: 0,
        /**
         * Home swipe-deck swipes per day. Expired-trial users get 3 swipes,
         * then the deck locks and redirects to Paywall.
         */
        homeSwipes: FREE_HOME_SWIPES_PER_DAY,
        /** Max clothing items the user can store in their closet */
        wardrobeItems: 20,
        /** Wardrobe scans per day (camera → AI detection) */
        wardrobeScans: 5,
        /** Wardrobe insights / analytics dashboard */
        analytics: false,
        /** AI-powered trip / travel outfit planner */
        tripPlanner: false,
        /** Full outfit calendar (past + future planning) */
        fullCalendar: false,
        /** AI stylist chat (AIAssistant + OutfitAI) */
        aiStylistChat: false,
        /** Priority high-end AI model */
        priorityModel: false,
        /** No ads anywhere in the app */
        adFree: false,
        /** Early access to beta features */
        earlyAccess: false,
        /** Priority customer support */
        prioritySupport: false,
    },
    lite: {
        aiOutfits: -1,
        tryOns: 0,
        homeSwipes: -1,
        wardrobeItems: 100,
        wardrobeScans: 30,
        analytics: true,
        tripPlanner: false,
        fullCalendar: true,
        aiStylistChat: true,
        priorityModel: false,
        adFree: false,
        earlyAccess: false,
        prioritySupport: false,
    },
    premium: {
        aiOutfits: -1,
        tryOns: -1,
        homeSwipes: -1,
        wardrobeItems: -1,
        wardrobeScans: -1,
        analytics: true,
        tripPlanner: true,
        fullCalendar: true,
        aiStylistChat: true,
        priorityModel: true,
        adFree: true,
        earlyAccess: true,
        prioritySupport: true,
    },
    vip: {
        aiOutfits: -1,
        tryOns: -1,
        homeSwipes: -1,
        wardrobeItems: -1,
        wardrobeScans: -1,
        analytics: true,
        tripPlanner: true,
        fullCalendar: true,
        aiStylistChat: true,
        priorityModel: true,
        adFree: true,
        earlyAccess: true,
        prioritySupport: true,
        /** VIP-exclusive: seasonal style collections curated by stylists */
        exclusiveCollections: true,
    },
} as const;

export type FeatureKey = keyof typeof TIER_FEATURES.free;

/** Keys that consume a daily quota (numeric limit per day). */
export const DAILY_QUOTA_FEATURES: FeatureKey[] = [
    'aiOutfits',
    'tryOns',
    'wardrobeScans',
    'homeSwipes',
];

// ─────────────────────────────────────────────────────────────
// Pricing
// ─────────────────────────────────────────────────────────────
export const SUBSCRIPTION_PRICING = {
    lite: {
        price: 2.99,
        currency: 'USD',
        period: 'month',
        productId: '2.99',
    },
    premium: {
        price: 9.99,
        currency: 'USD',
        period: 'month',
        productId: 'com.aiwardrobe.premium.monthly',
    },
    vip: {
        price: 99.99,
        currency: 'USD',
        period: 'year',
        productId: 'com.aiwardrobe.premium.yearly',
    },
} as const;

interface SubscriptionState {
    tier: SubscriptionTier;
    /**
     * The tier used for feature-access checks.
     * During an active free trial this is bumped to 'premium' (Pro access).
     */
    effectiveTier: SubscriptionTier;
    expiryDate: string | null;
    /** ISO timestamp when the user's 7-day trial started. null = no trial recorded. */
    trialStartedAt: string | null;
    isLoading: boolean;
    lastVerifiedAt: string | null;

    // Derived
    isPremium: boolean;
    hasActiveSubscription: boolean;
    isSubscriptionExpired: boolean;
    /** True when free tier user is within the 7-day trial window. */
    isTrialActive: boolean;
    /**
     * True when the user is on the free tier AND the 7-day trial has ended
     * (or was never started for legacy users). This does NOT block the app —
     * the user keeps free-tier access and individual premium features
     * redirect to Paywall via useSubscriptionGate.
     */
    isTrialExpired: boolean;
    /**
     * True when trial status is not yet determined (e.g. user just logged in
     * and initializeTrial / verifySubscriptionFromServer hasn't completed).
     * The gate must NOT be shown while this is true.
     */
    isTrialPending: boolean;
    /** Days left in the trial (0-7). */
    trialDaysRemaining: number;
    /** True when free-tier user has no trial date and no promo code redeemed.
     *  They should see the PromoCodeScreen instead of TrialExpiredScreen. */
    needsPromoCode: boolean;

    // Actions
    /** Reconcile time-derived access without a network request or persistence write. */
    refreshTimeState: () => void;
    initializeSubscription: () => Promise<void>;
    verifySubscriptionFromServer: () => Promise<void>;
    /**
     * Called once after a new user registers. Writes trial_started_at to
     * AsyncStorage + Supabase (gracefully — the column may not exist yet).
     */
    initializeTrial: (userId: string) => Promise<void>;
    setSubscription: (tier: SubscriptionTier, expiryDate?: string | null, productId?: string) => Promise<void>;
    clearSubscription: () => Promise<void>;
    /** Feature gate check (uses effectiveTier for trial-aware access). */
    checkFeatureAccess: (feature: FeatureKey) => boolean;
    /** @deprecated Use dailyUsageStore.getRemaining() for daily quotas. */
    getTriesRemaining: (feature: FeatureKey, usedCount: number) => number;
}

// ─────────────────────────────────────────────────────────────
// Trial helpers
// ─────────────────────────────────────────────────────────────
function computeTrialState(
    tier: SubscriptionTier,
    trialStartedAt: string | null,
    /** Pass true while we haven't yet fetched the trial date from storage/server */
    isPending = false,
) {
    // Trial only applies to the free tier
    if (tier !== 'free') {
        return { isTrialActive: false, isTrialExpired: false, isTrialPending: false, trialDaysRemaining: 0 };
    }

    if (!trialStartedAt) {
        // If we're still loading, mark as pending so the gate doesn't fire yet.
        // Only treat as truly expired once we've confirmed there is no trial date.
        if (isPending) {
            return { isTrialActive: false, isTrialExpired: false, isTrialPending: true, trialDaysRemaining: 0 };
        }
        // No trial date after full resolution → user hasn't redeemed a promo code yet.
        // Surface this as "needs promo code" (not pending, not expired) so the
        // PromoCodeScreen gate can fire in RootNavigator.
        return { isTrialActive: false, isTrialExpired: false, isTrialPending: false, trialDaysRemaining: 0 };
    }

    // Match the server's elapsed seven-day window across timezone changes.
    const trialEnd = new Date(Date.parse(trialStartedAt) + FREE_TRIAL_DAYS * 24 * 60 * 60 * 1000);
    const now = new Date();

    if (now < trialEnd) {
        const msRemaining = trialEnd.getTime() - now.getTime();
        const daysRemaining = Math.ceil(msRemaining / (1000 * 60 * 60 * 24));
        return { isTrialActive: true, isTrialExpired: false, isTrialPending: false, trialDaysRemaining: daysRemaining };
    }

    return { isTrialActive: false, isTrialExpired: true, isTrialPending: false, trialDaysRemaining: 0 };
}

function deriveState(
    tier: SubscriptionTier,
    expiryDate: string | null,
    trialStartedAt: string | null,
    /** Set to true during app boot before trial date is resolved from storage/server */
    isPending = false,
) {
    const isExpired = expiryDate ? new Date(expiryDate) <= new Date() : false;
    // Keep the last verified tier, but stop granting its access at expiry.
    const accessTier: SubscriptionTier = isExpired ? 'free' : tier;
    const trial = computeTrialState(accessTier, trialStartedAt, isPending);

    // During an active trial, feature checks use Pro (premium) limits
    const effectiveTier: SubscriptionTier = trial.isTrialActive ? 'premium' : accessTier;

    // Free-tier user with no trial date yet → must redeem a promo code.
    const needsPromoCode =
        tier === 'free' &&
        !trial.isTrialActive &&
        !trial.isTrialExpired &&
        !trial.isTrialPending &&
        !trialStartedAt;

    return {
        isPremium: accessTier === 'premium' || trial.isTrialActive,
        hasActiveSubscription: accessTier !== 'free' || trial.isTrialActive,
        isSubscriptionExpired: isExpired,
        effectiveTier,
        needsPromoCode,
        ...trial,
    };
}

const useSubscriptionStore = create<SubscriptionState>((set, get) => ({
    tier: 'free',
    expiryDate: null,
    trialStartedAt: null,
    isLoading: false,
    lastVerifiedAt: null,
    // Start as pending so the gate is never shown before initialization completes
    ...deriveState('free', null, null, true),

    refreshTimeState: () => {
        const current = get();
        const next = deriveState(current.tier, current.expiryDate, current.trialStartedAt, current.isTrialPending);
        if (Object.entries(next).some(([key, value]) => current[key as keyof SubscriptionState] !== value)) {
            set(next);
        }
    },

    // ─────────────────────────────────────────────────────────
    // verifySubscriptionFromServer
    // ─────────────────────────────────────────────────────────
    verifySubscriptionFromServer: async () => {
        const revision = authEvents.getRevision();
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session?.user?.id) return;

            const userId = session.user.id;
            if (!authEvents.canApplyToUser(userId)) return;
            const { data, error } = await supabase.rpc('get_subscription_status');
            if (error || !Array.isArray(data) || data.length !== 1) throw error || new Error('Invalid subscription status response');
            const { data: latest } = await supabase.auth.getSession();
            if (revision !== authEvents.getRevision() || latest.session?.user?.id !== userId) return;
            const status = data[0];
            if (!['free', 'lite', 'premium', 'vip'].includes(status.tier)) throw new Error('Invalid subscription tier');
            const resolvedTier = status.tier as SubscriptionTier;
            const resolvedExpiry: string | null = status.expiry_date;
            if (resolvedExpiry !== null && (typeof resolvedExpiry !== 'string' || !Number.isFinite(Date.parse(resolvedExpiry)))) {
                throw new Error('Invalid subscription expiry');
            }
            if (status.trial_started_at) {
                await AsyncStorage.setItem(TRIAL_START_KEY, status.trial_started_at);
                if (revision !== authEvents.getRevision()) return;
                set({ trialStartedAt: status.trial_started_at });
            } else if (resolvedTier === 'free') {
                await get().initializeTrial(userId);
            }
            if (revision !== authEvents.getRevision()) return;

            // A successful authoritative read wins over cached entitlements,
            // including refunds and account transfers. Read errors preserve cache.
            await AsyncStorage.setItem(SUBSCRIPTION_KEY, resolvedTier);
            if (revision !== authEvents.getRevision()) return;
            if (resolvedExpiry) {
                await AsyncStorage.setItem(SUBSCRIPTION_EXPIRY_KEY, resolvedExpiry);
                if (revision !== authEvents.getRevision()) return;
            } else {
                await AsyncStorage.removeItem(SUBSCRIPTION_EXPIRY_KEY);
                if (revision !== authEvents.getRevision()) return;
            }

            // Use the trial date that was already set (either from local cache or
            // Supabase profile sync above). isPending is now resolved = false.
            const trialStartedAt = get().trialStartedAt;
            set({
                tier: resolvedTier,
                expiryDate: resolvedExpiry,
                ...deriveState(resolvedTier, resolvedExpiry, trialStartedAt, false),
                lastVerifiedAt: new Date().toISOString(),
            });
        } catch (error) {
            if (revision !== authEvents.getRevision()) return;
            console.error('Failed to verify subscription from server:', error);
            // On error, clear pending so the UI isn't stuck waiting forever
            const { tier, expiryDate, trialStartedAt } = get();
            set(deriveState(tier, expiryDate, trialStartedAt, false));
        }
    },

    // ─────────────────────────────────────────────────────────
    // initializeSubscription (app boot, reads local cache first)
    // ─────────────────────────────────────────────────────────
    initializeSubscription: async () => {
        try {
            set({ isLoading: true });

            const storedTier = await AsyncStorage.getItem(SUBSCRIPTION_KEY);
            let storedExpiry = await AsyncStorage.getItem(SUBSCRIPTION_EXPIRY_KEY);
            const storedTrial = await AsyncStorage.getItem(TRIAL_START_KEY);

            let tier = 'free' as SubscriptionTier;

            if (storedTier && storedExpiry) {
                const expiryDate = new Date(storedExpiry);
                if (expiryDate > new Date()) {
                    tier = storedTier as SubscriptionTier;
                } else {
                    tier = 'free';
                    storedExpiry = null;
                    await AsyncStorage.setItem(SUBSCRIPTION_KEY, tier);
                    await AsyncStorage.removeItem(SUBSCRIPTION_EXPIRY_KEY);
                }
            } else if (storedTier && storedTier !== 'free' && !storedExpiry) {
                // Explicit lifetime entitlements have no expiry. Cache them until
                // an authoritative server read confirms or revokes access.
                tier = storedTier as SubscriptionTier;
                console.warn('[subscriptionStore] initializeSubscription: paid tier cached without expiry — trusting tier, will verify from server', { storedTier });
            }

            // If no trial date is cached locally yet, check the server before
            // marking the state as resolved. This prevents new/reinstalled users
            // from briefly seeing the TrialExpiredScreen on first launch.
            const { data: { session } } = await supabase.auth.getSession();
            const userId = session?.user?.id;

            if (!storedTrial && userId) {
                // Keep isPending=true while we resolve from server
                set({
                    tier,
                    expiryDate: storedExpiry,
                    trialStartedAt: null,
                    ...deriveState(tier, storedExpiry, null, true),
                    isLoading: false,
                });
                // verifySubscriptionFromServer will set isPending=false once done
                await get().verifySubscriptionFromServer();

                // Post-verification check: if we user has no trial date after syncing
                // from server, they need to enter a promo code first.
                // (No longer auto-start trial — promo code is required.)
            } else {
                // We have a cached trial date (or user is not logged in) — resolve immediately
                set({
                    tier,
                    expiryDate: storedExpiry,
                    trialStartedAt: storedTrial,
                    ...deriveState(tier, storedExpiry, storedTrial, false),
                    isLoading: false,
                });
                if (userId) {
                    get().verifySubscriptionFromServer().catch(() => {});
                }
            }
        } catch (error) {
            console.error('Failed to initialize subscription:', error);
            // On error, fall back to non-pending free state to avoid infinite spinner
            set({ isLoading: false, ...deriveState('free', null, null, false) });
        }
    },

    // ─────────────────────────────────────────────────────────
    // initializeTrial — called once after new user registration
    // ─────────────────────────────────────────────────────────
    initializeTrial: async (userId: string) => {
        if (!authEvents.canApplyToUser(userId)) return;
        const revision = authEvents.getRevision();
        try {
            // Confirm the server record even when a device cache exists.
            const { data: profile, error: readError } = await supabase
                .from('profiles')
                .select('trial_started_at')
                .eq('id', userId)
                .single();
            if (readError) throw readError;
            if (revision !== authEvents.getRevision()) return;
            let trialDate: string | null = (profile as any)?.trial_started_at ?? null;

            if (!trialDate) {
                // The database clamps the requested date to server time. Use
                // the returned record, never the device clock, for local access.
                const { data: updated, error: writeError } = await supabase
                    .from('profiles')
                    .update({ trial_started_at: new Date().toISOString() } as any)
                    .eq('id', userId)
                    .select('trial_started_at')
                    .single();
                if (writeError) throw writeError;
                trialDate = (updated as any)?.trial_started_at ?? null;
            }
            if (!trialDate || !Number.isFinite(Date.parse(trialDate))) {
                throw new Error('Server did not confirm a valid trial date');
            }

            if (revision !== authEvents.getRevision()) return;
            await AsyncStorage.setItem(TRIAL_START_KEY, trialDate);
            if (revision !== authEvents.getRevision()) return;
            const currentTier = get().tier;
            set({
                trialStartedAt: trialDate,
                ...deriveState(currentTier, get().expiryDate, trialDate),
            });
        } catch (error) {
            if (revision !== authEvents.getRevision()) return;
            console.error('Failed to initialize trial:', error);
        }
    },

    // ─────────────────────────────────────────────────────────
    // setSubscription
    // ─────────────────────────────────────────────────────────
    setSubscription: async (tier: SubscriptionTier, expiryDate?: string | null, productId?: string) => {
        if (tier !== 'free' && !authEvents.canApplyToUser()) return;
        const revision = authEvents.getRevision();
        try {
            const expiry = expiryDate === undefined ? getDefaultExpiry(tier, productId) : expiryDate;
            await AsyncStorage.setItem(SUBSCRIPTION_KEY, tier);
            if (revision !== authEvents.getRevision()) return;
            if (expiry === null) {
                await AsyncStorage.removeItem(SUBSCRIPTION_EXPIRY_KEY);
            } else {
                await AsyncStorage.setItem(SUBSCRIPTION_EXPIRY_KEY, expiry);
            }
            if (revision !== authEvents.getRevision()) return;

            const trialStartedAt = get().trialStartedAt;
            set({
                tier,
                expiryDate: expiry,
                ...deriveState(tier, expiry, trialStartedAt),
            });

            // RevenueCat-confirmed state is cached locally for immediate UI feedback.
            // Only verified server reconciliation may write authoritative entitlements.

        } catch (error) {
            if (revision !== authEvents.getRevision()) return;
            // Guarantee local state is set even if Supabase calls fail.
            console.error('[subscriptionStore] setSubscription: unexpected error:', error);
            const expiry2 = expiryDate === undefined ? getDefaultExpiry(tier) : expiryDate;
            const trialStartedAt2 = get().trialStartedAt;
            set({ tier, expiryDate: expiry2, ...deriveState(tier, expiry2, trialStartedAt2) });
        }
    },

    // ─────────────────────────────────────────────────────────
    // clearSubscription
    // ─────────────────────────────────────────────────────────
    clearSubscription: async () => {
        try {
            await AsyncStorage.removeItem(SUBSCRIPTION_KEY);
            await AsyncStorage.removeItem(SUBSCRIPTION_EXPIRY_KEY);

            const trialStartedAt = get().trialStartedAt;
            set({
                tier: 'free',
                expiryDate: null,
                ...deriveState('free', null, trialStartedAt),
            });


        } catch (error) {
            console.error('Failed to clear subscription:', error);
        }
    },

    // ─────────────────────────────────────────────────────────
    // checkFeatureAccess — uses effectiveTier (trial-aware)
    // ─────────────────────────────────────────────────────────
    checkFeatureAccess: (feature: FeatureKey) => {
        get().refreshTimeState();
        const { effectiveTier } = get();

        const value = TIER_FEATURES[effectiveTier][feature];
        if (typeof value === 'boolean') return value;
        return value === -1 || value > 0;
    },

    getTriesRemaining: (feature: FeatureKey, usedCount: number) => {
        get().refreshTimeState();
        const { effectiveTier } = get();

        const limit = TIER_FEATURES[effectiveTier][feature];
        if (typeof limit === 'boolean') return limit ? -1 : 0;
        if (limit === -1) return -1;
        return Math.max(0, limit - usedCount);
    },
}));

function getDefaultExpiry(tier: SubscriptionTier, productId?: string): string {
    const now = new Date();
    const isYearly = productId?.includes('yearly');
    if (isYearly) {
        now.setFullYear(now.getFullYear() + 1);
    } else if (tier === 'lite' || tier === 'premium' || tier === 'vip') {
        now.setMonth(now.getMonth() + 1);
    }
    return now.toISOString();
}

export { DAILY_USAGE_KEY, TRIAL_START_KEY };
export default useSubscriptionStore;

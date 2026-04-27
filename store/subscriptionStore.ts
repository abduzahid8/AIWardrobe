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
export type SubscriptionTier = 'free' | 'premium' | 'vip';

/** Friendly display labels used in the UI. */
export const TIER_DISPLAY_NAMES: Record<SubscriptionTier, string> = {
    free: 'Free',
    premium: 'Pro',
    vip: 'Max',
};

// ─────────────────────────────────────────────────────────────
// Feature matrix
// ─────────────────────────────────────────────────────────────
export const TIER_FEATURES = {
    free: {
        /** AI outfit generations per day (resets at local midnight) */
        aiOutfits: 10,
        /** AI Try-On renders per day */
        tryOns: 0,
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
        /** Priority high-end AI model */
        priorityModel: false,
        /** No ads anywhere in the app */
        adFree: false,
        /** Early access to beta features */
        earlyAccess: false,
        /** Priority customer support */
        prioritySupport: false,
    },
    premium: {
        aiOutfits: -1,
        tryOns: -1,
        wardrobeItems: -1,
        wardrobeScans: -1,
        analytics: true,
        tripPlanner: true,
        fullCalendar: true,
        priorityModel: true,
        adFree: true,
        earlyAccess: true,
        prioritySupport: true,
    },
    vip: {
        aiOutfits: -1,
        tryOns: -1,
        wardrobeItems: -1,
        wardrobeScans: -1,
        analytics: true,
        tripPlanner: true,
        fullCalendar: true,
        priorityModel: true,
        adFree: true,
        earlyAccess: true,
        prioritySupport: true,
        /** VIP-exclusive: seasonal style collections curated by stylists */
        exclusiveCollections: true,
        /** VIP-exclusive: AI stylist chat with personalized recommendations */
        aiStylistChat: true,
    },
} as const;

export type FeatureKey = keyof typeof TIER_FEATURES.free;

/** Keys that consume a daily quota (numeric limit per day). */
export const DAILY_QUOTA_FEATURES: FeatureKey[] = [
    'aiOutfits',
    'tryOns',
    'wardrobeScans',
];

// ─────────────────────────────────────────────────────────────
// Pricing
// ─────────────────────────────────────────────────────────────
export const SUBSCRIPTION_PRICING = {
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
     * (or was never started for legacy users). This gates the TrialExpiredScreen.
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
    initializeSubscription: () => Promise<void>;
    verifySubscriptionFromServer: () => Promise<void>;
    /**
     * Called once after a new user registers. Writes trial_started_at to
     * AsyncStorage + Supabase (gracefully — the column may not exist yet).
     */
    initializeTrial: (userId: string) => Promise<void>;
    setSubscription: (tier: SubscriptionTier, expiryDate?: string, productId?: string) => Promise<void>;
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
        // No trial date after full resolution → NOT expired, but needs initialization.
        // We set isTrialPending: true here as well until initializeTrial is called.
        return { isTrialActive: false, isTrialExpired: false, isTrialPending: true, trialDaysRemaining: 0 };
    }

    const trialEnd = new Date(trialStartedAt);
    trialEnd.setDate(trialEnd.getDate() + FREE_TRIAL_DAYS);
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
    const trial = computeTrialState(tier, trialStartedAt, isPending);

    // During an active trial, feature checks use Pro (premium) limits
    const effectiveTier: SubscriptionTier = trial.isTrialActive ? 'premium' : tier;

    // NOTE: Promo code gate is DISABLED for App Store submission.
    // Re-enable after approval: tier === 'free' && !trial.isTrialActive && !trial.isTrialExpired && !trial.isTrialPending && !trialStartedAt;
    const needsPromoCode = false;

    return {
        isPremium: tier === 'premium' || trial.isTrialActive,
        hasActiveSubscription: (tier !== 'free' && !isExpired) || trial.isTrialActive,
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

    // ─────────────────────────────────────────────────────────
    // verifySubscriptionFromServer
    // ─────────────────────────────────────────────────────────
    verifySubscriptionFromServer: async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session?.user?.id) return;

            const userId = session.user.id;
            let resolvedTier: SubscriptionTier = 'free';
            let resolvedExpiry: string | null = null;

            const { data: activeSub } = await supabase
                .from('subscriptions')
                .select('tier, end_date, status')
                .eq('user_id', userId)
                .in('status', ['active', 'trial'])
                .gte('end_date', new Date().toISOString())
                .order('end_date', { ascending: false })
                .limit(1)
                .maybeSingle();

            if (activeSub) {
                resolvedTier = activeSub.tier as SubscriptionTier;
                resolvedExpiry = activeSub.end_date;
            } else {
                const { data: profile } = await supabase
                    .from('profiles')
                    .select('subscription_tier, subscription_expires_at, trial_started_at')
                    .eq('id', userId)
                    .single();

                if (profile) {
                    const profileTier = (profile.subscription_tier || 'free') as SubscriptionTier;
                    const profileExpiry = profile.subscription_expires_at;

                    if (profileTier !== 'free' && profileExpiry && new Date(profileExpiry) > new Date()) {
                        // Profile shows a valid, non-expired paid tier — trust it.
                        resolvedTier = profileTier;
                        resolvedExpiry = profileExpiry;
                    } else if (profileTier !== 'free' && !profileExpiry) {
                        // Profile shows paid tier but no expiry yet.
                        // This happens in the window between a client-side purchase writing
                        // the profiles row and the RevenueCat webhook arriving with the real
                        // expiry date. Do NOT actively downgrade — keep whatever the client set.
                        // The next verifySubscriptionFromServer call (after the webhook fires)
                        // will find the subscriptions row and resolve correctly.
                        const currentState = get();
                        resolvedTier = currentState.tier !== 'free' ? currentState.tier : profileTier;
                        resolvedExpiry = currentState.expiryDate;
                        console.log('[subscriptionStore] Profile shows paid tier with no expiry — holding current state to avoid race condition', { profileTier, currentTier: currentState.tier });
                    } else if (profileTier !== 'free' && profileExpiry && new Date(profileExpiry) <= new Date()) {
                        // Paid tier but genuinely expired — safe to downgrade.
                        await supabase
                            .from('profiles')
                            .update({ subscription_tier: 'free', subscription_expires_at: null })
                            .eq('id', userId);
                    }

                    // Sync trial start date from Supabase (handles reinstall scenario)
                    if (profile.trial_started_at) {
                        await AsyncStorage.setItem(TRIAL_START_KEY, profile.trial_started_at);
                        const derived = deriveState(resolvedTier, resolvedExpiry, profile.trial_started_at, false);
                        set({
                            trialStartedAt: profile.trial_started_at,
                            ...derived,
                        });
                    }
                }
            }

            // ─── RACE-CONDITION GUARD ──────────────────────────────────────────────
            // If the server returned 'free' but the local store already has a paid
            // tier with a FUTURE expiry, do NOT downgrade. This covers the window
            // immediately after a purchase where:
            //   • setSubscription() wrote premium to AsyncStorage + Zustand + Supabase
            //   • verifySubscriptionFromServer() fired (fire-and-forget) before the
            //     Supabase row/profile was replicated to the read replica, so the
            //     server query sees the old 'free' state and incorrectly overwrites it.
            // We only skip when resolvedTier is 'free' — if the server found an
            // active subscription we always respect that.
            if (resolvedTier === 'free') {
                const currentState = get();
                if (
                    currentState.tier !== 'free' &&
                    currentState.expiryDate &&
                    new Date(currentState.expiryDate) > new Date()
                ) {
                    console.warn(
                        '[subscriptionStore] verifySubscriptionFromServer: server returned free but ' +
                        'local state has a valid paid subscription — keeping local state to prevent ' +
                        'race-condition downgrade.',
                        { localTier: currentState.tier, localExpiry: currentState.expiryDate }
                    );
                    // Mark as verified so callers don't re-enter, but preserve tier.
                    set({ lastVerifiedAt: new Date().toISOString() });
                    return;
                }
            }
            // ──────────────────────────────────────────────────────────────────────

            await AsyncStorage.setItem(SUBSCRIPTION_KEY, resolvedTier);
            if (resolvedExpiry) {
                await AsyncStorage.setItem(SUBSCRIPTION_EXPIRY_KEY, resolvedExpiry);
            } else {
                await AsyncStorage.removeItem(SUBSCRIPTION_EXPIRY_KEY);
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
                // Stored tier exists (paid) but no expiry cached yet — this happens
                // in the brief window after setSubscription writes the tier before
                // the server returns the real expiry. Preserve the paid tier and let
                // verifySubscriptionFromServer fill in the expiry.
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
                    get().verifySubscriptionFromServer();
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
        try {
            // Idempotent — if trial already recorded locally, just sync state
            const existing = await AsyncStorage.getItem(TRIAL_START_KEY);
            if (existing) {
                const currentTier = get().tier;
                set({
                    trialStartedAt: existing,
                    ...deriveState(currentTier, get().expiryDate, existing),
                });
                return;
            }

            // Check Supabase first (handles reinstall scenario)
            let trialDate: string | null = null;
            try {
                const { data: profile } = await supabase
                    .from('profiles')
                    .select('trial_started_at')
                    .eq('id', userId)
                    .single();
                trialDate = (profile as any)?.trial_started_at ?? null;
            } catch {
                // Column may not exist yet — silently fall back
            }

            if (!trialDate) {
                trialDate = new Date().toISOString();
                // Write to Supabase (gracefully — column may not be migrated yet)
                try {
                    await supabase
                        .from('profiles')
                        .update({ trial_started_at: trialDate } as any)
                        .eq('id', userId);
                } catch {
                    // Silently ignore — will be written on next verifySubscriptionFromServer
                }
            }

            await AsyncStorage.setItem(TRIAL_START_KEY, trialDate);
            const currentTier = get().tier;
            set({
                trialStartedAt: trialDate,
                ...deriveState(currentTier, get().expiryDate, trialDate),
            });
        } catch (error) {
            console.error('Failed to initialize trial:', error);
        }
    },

    // ─────────────────────────────────────────────────────────
    // setSubscription
    // ─────────────────────────────────────────────────────────
    setSubscription: async (tier: SubscriptionTier, expiryDate?: string, productId?: string) => {
        try {
            const expiry = expiryDate || getDefaultExpiry(tier, productId);
            await AsyncStorage.setItem(SUBSCRIPTION_KEY, tier);
            await AsyncStorage.setItem(SUBSCRIPTION_EXPIRY_KEY, expiry);

            const trialStartedAt = get().trialStartedAt;
            set({
                tier,
                expiryDate: expiry,
                ...deriveState(tier, expiry, trialStartedAt),
            });

            const { data: { session } } = await supabase.auth.getSession();
            if (session?.user?.id) {
                const userId = session.user.id;

                // Update profiles denormalized fields
                const { error: profileError } = await supabase
                    .from('profiles')
                    .update({ subscription_tier: tier, subscription_expires_at: expiry })
                    .eq('id', userId);
                    
                if (profileError) {
                    // Non-fatal: log but continue to try writing the subscriptions row.
                    console.error('[subscriptionStore] setSubscription: profile update failed (non-fatal):', profileError);
                }

                // Also upsert a row into subscriptions so verifySubscriptionFromServer
                // finds the active subscription even if the webhook hasn't fired yet.
                // The webhook will overwrite this with the real Apple expiry later.
                if (tier !== 'free') {
                    try {
                        const { data: existing, error: queryError } = await supabase
                            .from('subscriptions')
                            .select('id')
                            .eq('user_id', userId)
                            .in('status', ['active', 'trial'])
                            .limit(1);

                        if (queryError) {
                            console.error('[subscriptionStore] setSubscription: query subs failed:', queryError);
                        } else if (existing && existing.length > 0) {
                            const { error: updateError } = await supabase
                                .from('subscriptions')
                                .update({
                                    tier,
                                    status: 'active',
                                    end_date: expiry,
                                    auto_renew: true,
                                })
                                .eq('id', existing[0].id);
                            if (updateError) {
                                console.error('[subscriptionStore] setSubscription: update sub failed:', updateError);
                            }
                        } else {
                            const { error: insertError } = await supabase
                                .from('subscriptions')
                                .insert({
                                    user_id: userId,
                                    tier,
                                    status: 'active',
                                    start_date: new Date().toISOString(),
                                    end_date: expiry,
                                    auto_renew: true,
                                    platform: 'ios',
                                    product_id: productId || (tier === 'vip' ? 'com.aiwardrobe.premium.yearly' : 'com.aiwardrobe.premium.monthly'),
                                });
                            if (insertError) {
                                console.error('[subscriptionStore] setSubscription: insert sub failed:', insertError);
                            }
                        }
                    } catch (subError) {
                        console.error('[subscriptionStore] setSubscription: subscriptions write failed (non-fatal):', subError);
                    }
                }
            }
        } catch (error) {
            // Guarantee local state is set even if Supabase calls fail.
            console.error('[subscriptionStore] setSubscription: unexpected error:', error);
            const expiry2 = expiryDate || getDefaultExpiry(tier);
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

            const { data: { session } } = await supabase.auth.getSession();
            if (session?.user?.id) {
                await supabase
                    .from('profiles')
                    .update({ subscription_tier: 'free', subscription_expires_at: null })
                    .eq('id', session.user.id);
            }
        } catch (error) {
            console.error('Failed to clear subscription:', error);
        }
    },

    // ─────────────────────────────────────────────────────────
    // checkFeatureAccess — uses effectiveTier (trial-aware)
    // ─────────────────────────────────────────────────────────
    checkFeatureAccess: (feature: FeatureKey) => {
        const { effectiveTier } = get();

        const value = TIER_FEATURES[effectiveTier][feature];
        if (typeof value === 'boolean') return value;
        return value === -1 || value > 0;
    },

    getTriesRemaining: (feature: FeatureKey, usedCount: number) => {
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
    } else if (tier === 'premium' || tier === 'vip') {
        now.setMonth(now.getMonth() + 1);
    }
    return now.toISOString();
}

export { DAILY_USAGE_KEY, TRIAL_START_KEY };
export default useSubscriptionStore;

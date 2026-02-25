import { create } from "zustand";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { supabase } from '../lib/supabase';
import Config from '../src/config/env';

// Storage Keys — used as a local cache only, not source of truth
const SUBSCRIPTION_KEY = "subscription_tier";
const SUBSCRIPTION_EXPIRY_KEY = "subscription_expiry";

// Subscription Tiers
export type SubscriptionTier = 'free' | 'premium' | 'vip';

// Feature Access by Tier
export const TIER_FEATURES = {
    free: {
        maxUses: 5,
        aiOutfits: 5,
        wardrobeScans: 5,
        tryOns: 3,
        analytics: false,
        prioritySupport: false,
        unlimitedStorage: false,
    },
    premium: {
        maxUses: -1, // Unlimited
        aiOutfits: -1,
        wardrobeScans: -1,
        tryOns: 50,
        analytics: true,
        prioritySupport: false,
        unlimitedStorage: false,
    },
    vip: {
        maxUses: -1,
        aiOutfits: -1,
        wardrobeScans: -1,
        tryOns: -1,
        analytics: true,
        prioritySupport: true,
        unlimitedStorage: true,
    },
};

// Pricing
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
        productId: 'com.aiwardrobe.vip.yearly',
    },
};

interface SubscriptionState {
    tier: SubscriptionTier;
    expiryDate: string | null;
    isLoading: boolean;
    lastVerifiedAt: string | null;

    // Computed
    isPremium: boolean;
    isVIP: boolean;
    hasActiveSubscription: boolean;

    // Actions
    initializeSubscription: () => Promise<void>;
    verifySubscriptionFromServer: () => Promise<void>;
    setSubscription: (tier: SubscriptionTier, expiryDate?: string) => Promise<void>;
    clearSubscription: () => Promise<void>;
    checkFeatureAccess: (feature: keyof typeof TIER_FEATURES.free) => boolean;
    getTriesRemaining: (usedCount: number) => number;
}

const useSubscriptionStore = create<SubscriptionState>((set, get) => ({
    tier: 'free',
    expiryDate: null,
    isLoading: false,
    lastVerifiedAt: null,
    isPremium: false,
    isVIP: false,
    hasActiveSubscription: false,

    /**
     * Verify subscription from the server (authoritative source of truth).
     * Falls back to cached local state if the request fails.
     * Call this on app launch and when returning from paywall.
     */
    verifySubscriptionFromServer: async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session?.access_token) return;

            const response = await fetch(`${Config.api.url}/subscription-status`, {
                headers: { Authorization: `Bearer ${session.access_token}` },
            });

            if (!response.ok) return;

            const data = await response.json();
            const tier: SubscriptionTier = data.tier || 'free';
            const expiryDate: string | null = data.expiresAt || null;
            const hasActiveSubscription: boolean = tier !== 'free';

            // Update local cache
            await AsyncStorage.setItem(SUBSCRIPTION_KEY, tier);
            if (expiryDate) await AsyncStorage.setItem(SUBSCRIPTION_EXPIRY_KEY, expiryDate);

            set({
                tier,
                expiryDate,
                isPremium: tier === 'premium' || tier === 'vip',
                isVIP: tier === 'vip',
                hasActiveSubscription,
                lastVerifiedAt: new Date().toISOString(),
            });
        } catch (error) {
            console.error('Failed to verify subscription from server:', error);
        }
    },

    initializeSubscription: async () => {
        try {
            set({ isLoading: true });

            const storedTier = await AsyncStorage.getItem(SUBSCRIPTION_KEY);
            const storedExpiry = await AsyncStorage.getItem(SUBSCRIPTION_EXPIRY_KEY);

            let tier = 'free' as SubscriptionTier;
            let hasActiveSubscription = false;

            if (storedTier && storedExpiry) {
                const expiryDate = new Date(storedExpiry);
                if (expiryDate > new Date()) {
                    tier = storedTier as SubscriptionTier;
                    hasActiveSubscription = tier !== 'free';
                } else {
                    tier = 'free';
                    hasActiveSubscription = false;
                    await AsyncStorage.setItem(SUBSCRIPTION_KEY, tier);
                    await AsyncStorage.removeItem(SUBSCRIPTION_EXPIRY_KEY);
                }
            }

            set({
                tier,
                expiryDate: storedExpiry,
                isPremium: tier === 'premium' || tier === 'vip',
                isVIP: tier === 'vip',
                hasActiveSubscription,
                isLoading: false,
            });
        } catch (error) {
            console.error('Failed to initialize subscription:', error);
            set({ isLoading: false });
        }
    },

    setSubscription: async (tier: SubscriptionTier, expiryDate?: string) => {
        try {
            const expiry = expiryDate || getDefaultExpiry(tier);

            await AsyncStorage.setItem(SUBSCRIPTION_KEY, tier);
            await AsyncStorage.setItem(SUBSCRIPTION_EXPIRY_KEY, expiry);

            set({
                tier,
                expiryDate: expiry,
                isPremium: tier === 'premium' || tier === 'vip',
                isVIP: tier === 'vip',
                hasActiveSubscription: tier !== 'free',
            });

            console.log(`Subscription set to ${tier} until ${expiry}`);
        } catch (error) {
            console.error('Failed to set subscription:', error);
        }
    },

    clearSubscription: async () => {
        try {
            await AsyncStorage.removeItem(SUBSCRIPTION_KEY);
            await AsyncStorage.removeItem(SUBSCRIPTION_EXPIRY_KEY);

            set({
                tier: 'free',
                expiryDate: null,
                isPremium: false,
                isVIP: false,
                hasActiveSubscription: false,
            });
        } catch (error) {
            console.error('Failed to clear subscription:', error);
        }
    },

    checkFeatureAccess: (feature: keyof typeof TIER_FEATURES.free) => {
        const { tier } = get();
        const tierFeatures = TIER_FEATURES[tier];
        const value = tierFeatures[feature];

        if (typeof value === 'boolean') {
            return value;
        }
        return value === -1 || value > 0;
    },

    getTriesRemaining: (usedCount: number) => {
        const { tier } = get();
        const maxUses = TIER_FEATURES[tier].maxUses;

        if (maxUses === -1) return -1; // Unlimited
        return Math.max(0, maxUses - usedCount);
    },
}));

// Helper to get default expiry date
function getDefaultExpiry(tier: SubscriptionTier): string {
    const now = new Date();
    if (tier === 'premium') {
        now.setMonth(now.getMonth() + 1); // 1 month
    } else if (tier === 'vip') {
        now.setFullYear(now.getFullYear() + 1); // 1 year
    }
    return now.toISOString();
}

export default useSubscriptionStore;

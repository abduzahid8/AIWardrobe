import { create } from "zustand";
import AsyncStorage from "@react-native-async-storage/async-storage";

// Storage Keys
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

    // Computed
    isPremium: boolean;
    isVIP: boolean;
    hasActiveSubscription: boolean;

    // Actions
    initializeSubscription: () => Promise<void>;
    setSubscription: (tier: SubscriptionTier, expiryDate?: string) => Promise<void>;
    clearSubscription: () => Promise<void>;
    checkFeatureAccess: (feature: keyof typeof TIER_FEATURES.free) => boolean;
    getTriesRemaining: (usedCount: number) => number;
}

const useSubscriptionStore = create<SubscriptionState>((set, get) => ({
    tier: 'free',
    expiryDate: null,
    isLoading: false,
    isPremium: false,
    isVIP: false,
    hasActiveSubscription: false,

    initializeSubscription: async () => {
        try {
            set({ isLoading: true });

            const storedTier = await AsyncStorage.getItem(SUBSCRIPTION_KEY);
            const storedExpiry = await AsyncStorage.getItem(SUBSCRIPTION_EXPIRY_KEY);

            let tier: SubscriptionTier = 'free';
            let hasActiveSubscription = false;

            if (storedTier && storedExpiry) {
                const expiryDate = new Date(storedExpiry);
                if (expiryDate > new Date()) {
                    tier = storedTier as SubscriptionTier;
                    hasActiveSubscription = tier !== 'free';
                } else {
                    // Subscription expired, reset to free
                    await AsyncStorage.removeItem(SUBSCRIPTION_KEY);
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

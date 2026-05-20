import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../lib/supabase';
import useSubscriptionStore from './subscriptionStore';
import { createLogger } from '../src/utils/logger';

const log = createLogger('PromoCodeStore');

const PROMO_REDEEMED_KEY = 'promo_redeemed_v1';
const PROMO_SKIPPED_KEY = 'promo_skipped_v1';

interface PromoCodeState {
    /** Whether the user has already redeemed a promo code. */
    hasRedeemedPromo: boolean;
    /** Whether the user skipped the promo code screen (chose to pay later). */
    hasSkippedPromo: boolean;
    /** Whether we've checked the user's promo status on this boot. */
    isHydrated: boolean;

    // Actions
    hydrate: () => Promise<void>;
    redeemPromoCode: (code: string) => Promise<{ success: boolean; trialDays?: number; error?: string }>;
    skipPromo: () => Promise<void>;
    /** Check if the user should see the promo code screen. */
    shouldShowPromoScreen: () => boolean;
}

const usePromoCodeStore = create<PromoCodeState>((set, get) => ({
    hasRedeemedPromo: false,
    hasSkippedPromo: false,
    isHydrated: false,

    hydrate: async () => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            const userId = session?.user?.id;

            // Check local cache first
            const [localRedeemed, localSkipped] = await Promise.all([
                AsyncStorage.getItem(PROMO_REDEEMED_KEY),
                AsyncStorage.getItem(PROMO_SKIPPED_KEY),
            ]);

            if (localRedeemed === 'true') {
                set({ hasRedeemedPromo: true, isHydrated: true });
                return;
            }

            if (localSkipped === 'true') {
                set({ hasSkippedPromo: true, isHydrated: true });
                // Still check server — maybe they redeemed on another device
            }

            // Check server for existing redemption
            if (userId) {
                const { data: redemption } = await supabase
                    .from('promo_redemptions')
                    .select('id, trial_days, redeemed_at')
                    .eq('user_id', userId)
                    .maybeSingle();

                if (redemption) {
                    await AsyncStorage.setItem(PROMO_REDEEMED_KEY, 'true');
                    set({ hasRedeemedPromo: true, isHydrated: true });
                    return;
                }
            }

            set({ isHydrated: true });
        } catch (error) {
            log.error('Failed to hydrate promo code store', error);
            set({ isHydrated: true });
        }
    },

    redeemPromoCode: async (code: string) => {
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session?.access_token) {
                return { success: false, error: 'Not authenticated' };
            }

            const { data, error } = await supabase.functions.invoke('redeem-promo', {
                method: 'POST',
                body: { code: code.trim() },
            });

            if (error) {
                let message = 'Redemption failed';
                
                // Try to parse error context if it is a FunctionsHttpError
                if (error.context) {
                    try {
                        const body = typeof error.context.json === 'function' 
                            ? await error.context.json() 
                            : JSON.parse(await error.context.text());
                        message = body?.error || body?.message || message;
                    } catch (_) {
                        try {
                            if (typeof error.context.text === 'function') {
                                const text = await error.context.text();
                                if (text) message = text;
                            }
                        } catch (__) {}
                    }
                }
                
                if (message === 'Redemption failed' || message === 'Edge Function returned a non-2xx status code') {
                    message = error.message || message;
                }
                
                log.error('Promo code redemption failed', message);
                return { success: false, error: message };
            }

            if (!data?.success) {
                return { success: false, error: data?.error || 'Redemption failed' };
            }

            // Success — update local state
            await AsyncStorage.setItem(PROMO_REDEEMED_KEY, 'true');
            set({ hasRedeemedPromo: true });

            // Refresh subscription store so trial state is updated
            await useSubscriptionStore.getState().verifySubscriptionFromServer();

            log.info('Promo code redeemed successfully', { trialDays: data.trial_days });
            return { success: true, trialDays: data.trial_days };
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Redemption failed';
            log.error('Promo code redemption error', error);
            return { success: false, error: message };
        }
    },

    skipPromo: async () => {
        await AsyncStorage.setItem(PROMO_SKIPPED_KEY, 'true');
        set({ hasSkippedPromo: true });
    },

    shouldShowPromoScreen: () => {
        // PRODUCTION: Always return false — Apple Guideline 3.1.1 prohibits
        // custom promo code mechanisms from gating access to the app.
        // The PromoCode screen remains accessible voluntarily (e.g. from Profile
        // via the "Redeem Offer Code" button) but is NEVER a navigation gate.
        if (!__DEV__) return false;

        // DEV ONLY: allow testing the promo gate flow locally.
        const { hasRedeemedPromo, isHydrated } = get();
        if (!isHydrated) return false;
        if (hasRedeemedPromo) return false;

        const subStore = useSubscriptionStore.getState();
        if (subStore.tier !== 'free') return false;
        if (subStore.isTrialActive) return false;
        if (subStore.isTrialExpired) return false;

        return true;
    },
}));

export default usePromoCodeStore;

/**
 * Store Bootstrap — wires cross-store event subscriptions.
 *
 * Import this file once from App.tsx or RootNavigator to activate
 * the event bridge between auth and wardrobe stores.
 */

import { authEvents } from './authEvents';
import useWardrobeStore from './wardrobeStore';
import useSubscriptionStore from './subscriptionStore';

let initialized = false;

export function bootstrapStores() {
    if (initialized) return;
    initialized = true;

    authEvents.onLogin(async (userId) => {
        try {
            const state = useWardrobeStore.getState();

            const hasOrphanedItems = state.items.some((i) => !i.userId || i.userId === '');
            if (hasOrphanedItems) {
                state.items.forEach((item) => {
                    if (!item.userId || item.userId === '') {
                        state.updateItem(item.id, { userId });
                    }
                });
            }

            await state.rehydrateFromCloud();
            state.subscribeToRealtime();
        } catch (e) {
            console.error('[bootstrap] Wardrobe rehydration failed:', e);
        }
    });

    authEvents.onLogout(() => {
        try {
            useWardrobeStore.getState().unsubscribeRealtime();
        } catch {
            // Store may not be initialized
        }

        // Reset subscription/trial in-memory state so that stale trial data
        // from the previous user is never shown to the next one who logs in.
        try {
            useSubscriptionStore.setState({
                tier: 'free',
                effectiveTier: 'free',
                expiryDate: null,
                trialStartedAt: null,
                isTrialActive: false,
                isTrialExpired: false,
                isTrialPending: true, // pending until initializeSubscription runs for new user
                trialDaysRemaining: 0,
                isPremium: false,
                isVIP: false,
                hasActiveSubscription: false,
                isSubscriptionExpired: false,
                lastVerifiedAt: null,
            });
        } catch {
            // Non-critical
        }
    });
}

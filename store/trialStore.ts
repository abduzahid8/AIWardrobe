/**
 * trialStore.ts — Free trial usage tracking
 *
 * Uses Zustand persist (AsyncStorage) — no manual AsyncStorage calls.
 *
 * Paywall timing:
 *   - When trialCount reaches MAX_TRIAL_COUNT the flag `pendingPaywall` is
 *     set to true.  The UI reads this and shows the bottom sheet after a
 *     SHORT_DELAY_MS delay so it never interrupts the current gesture.
 *   - Call dismissPaywall() to clear the flag once the sheet has been shown.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

const MAX_TRIAL_COUNT   = 5;   // free AI calls before paywall
const SHORT_DELAY_MS    = 800; // delay before bottom sheet appears

interface TrialState {
    trialCount:      number;
    firstLaunchDate: string | null;
    isTrialExpired:  boolean;
    /** True after the trial expires and before the paywall sheet is shown. */
    pendingPaywall:  boolean;

    // Actions
    initializeTrial:    () => void;
    incrementTrialCount: () => void;
    checkTrialExpired:  () => boolean;
    getTrialsRemaining: () => number;
    /** Called by the paywall sheet once it is visible — clears the pending flag. */
    dismissPaywall:     () => void;
    resetTrial:         () => void;
}

const useTrialStore = create<TrialState>()(
    persist(
        (set, get) => ({
            trialCount:      0,
            firstLaunchDate: null,
            isTrialExpired:  false,
            pendingPaywall:  false,

            initializeTrial: () => {
                const { firstLaunchDate } = get();
                if (!firstLaunchDate) {
                    set({ firstLaunchDate: new Date().toISOString() });
                }
            },

            incrementTrialCount: () => {
                const newCount       = get().trialCount + 1;
                const isTrialExpired = newCount >= MAX_TRIAL_COUNT;

                set({
                    trialCount: newCount,
                    isTrialExpired,
                    // Schedule the paywall flag after a short delay so UI can
                    // finish the current animation before showing the sheet.
                    pendingPaywall: false,
                });

                if (isTrialExpired) {
                    setTimeout(() => {
                        set({ pendingPaywall: true });
                    }, SHORT_DELAY_MS);
                }
            },

            checkTrialExpired: () => get().trialCount >= MAX_TRIAL_COUNT,

            getTrialsRemaining: () => Math.max(0, MAX_TRIAL_COUNT - get().trialCount),

            dismissPaywall: () => set({ pendingPaywall: false }),

            resetTrial: () => set({
                trialCount:      0,
                firstLaunchDate: null,
                isTrialExpired:  false,
                pendingPaywall:  false,
            }),
        }),
        {
            name: 'trial-storage',
            storage: createJSONStorage(() => AsyncStorage),
            partialize: (state) => ({
                trialCount:      state.trialCount,
                firstLaunchDate: state.firstLaunchDate,
                isTrialExpired:  state.isTrialExpired,
            }),
        }
    )
);

export default useTrialStore;

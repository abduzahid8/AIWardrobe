import { create } from "zustand";
import AsyncStorage from "@react-native-async-storage/async-storage";

const TRIAL_COUNT_KEY = "trial_count";
const TRIAL_FIRST_LAUNCH_KEY = "trial_first_launch";
const MAX_TRIAL_COUNT = 3;

interface TrialState {
    trialCount: number;
    firstLaunchDate: string | null;
    isTrialExpired: boolean;
    loading: boolean;

    // Actions
    initializeTrial: () => Promise<void>;
    incrementTrialCount: () => Promise<void>;
    checkTrialExpired: () => boolean;
    resetTrial: () => Promise<void>; // For testing purposes
    getTrialsRemaining: () => number;
}

const useTrialStore = create<TrialState>((set, get) => ({
    trialCount: 0,
    firstLaunchDate: null,
    isTrialExpired: false,
    loading: false,

    initializeTrial: async () => {
        try {
            set({ loading: true });

            // Get stored trial count
            const storedCount = await AsyncStorage.getItem(TRIAL_COUNT_KEY);
            const storedDate = await AsyncStorage.getItem(TRIAL_FIRST_LAUNCH_KEY);

            const trialCount = storedCount ? parseInt(storedCount, 10) : 0;
            const firstLaunchDate = storedDate || new Date().toISOString();

            // If first time, save the launch date
            if (!storedDate) {
                await AsyncStorage.setItem(TRIAL_FIRST_LAUNCH_KEY, firstLaunchDate);
            }

            const isTrialExpired = trialCount >= MAX_TRIAL_COUNT;

            set({
                trialCount,
                firstLaunchDate,
                isTrialExpired,
                loading: false,
            });
        } catch (error) {
            console.error("Failed to initialize trial:", error);
            set({ loading: false });
        }
    },

    incrementTrialCount: async () => {
        try {
            const currentCount = get().trialCount;
            const newCount = currentCount + 1;

            await AsyncStorage.setItem(TRIAL_COUNT_KEY, newCount.toString());

            const isTrialExpired = newCount >= MAX_TRIAL_COUNT;

            set({
                trialCount: newCount,
                isTrialExpired,
            });

            console.log(`Trial count incremented to ${newCount}/${MAX_TRIAL_COUNT}`);
        } catch (error) {
            console.error("Failed to increment trial count:", error);
        }
    },

    checkTrialExpired: () => {
        return get().trialCount >= MAX_TRIAL_COUNT;
    },

    getTrialsRemaining: () => {
        const remaining = MAX_TRIAL_COUNT - get().trialCount;
        return Math.max(0, remaining);
    },

    resetTrial: async () => {
        try {
            await AsyncStorage.removeItem(TRIAL_COUNT_KEY);
            await AsyncStorage.removeItem(TRIAL_FIRST_LAUNCH_KEY);
            set({
                trialCount: 0,
                firstLaunchDate: null,
                isTrialExpired: false,
            });
            console.log("Trial reset successfully");
        } catch (error) {
            console.error("Failed to reset trial:", error);
        }
    },
}));

export default useTrialStore;

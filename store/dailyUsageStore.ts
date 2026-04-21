/**
 * dailyUsageStore — tracks per-day usage counters (AI outfits, try-ons, scans).
 *
 * Counters are keyed by local calendar date (YYYY-MM-DD) so they reset
 * automatically at the user's local midnight. Persisted to AsyncStorage,
 * so usage survives app restarts but a device that travels across time
 * zones still sees a fresh daily bucket when a new local day begins.
 *
 * Usage:
 *   const { consume, getUsed, getRemaining } = useDailyUsageStore();
 *   consume('aiOutfits');
 *   const remaining = getRemaining('aiOutfits');
 */

import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import useSubscriptionStore, {
    DAILY_USAGE_KEY,
    TIER_FEATURES,
    type FeatureKey,
} from './subscriptionStore';

interface DailyUsageSnapshot {
    /** YYYY-MM-DD in local time */
    date: string;
    counts: Partial<Record<FeatureKey, number>>;
}

function todayKey(): string {
    const d = new Date();
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
}

interface DailyUsageState {
    snapshot: DailyUsageSnapshot;
    hydrated: boolean;
    hydrate: () => Promise<void>;
    /** Record one use of a feature. Returns the new used count. */
    consume: (feature: FeatureKey, amount?: number) => Promise<number>;
    /** How many times the user has used the feature today. */
    getUsed: (feature: FeatureKey) => number;
    /**
     * How many uses remain today for the current tier.
     *   -1 = unlimited
     *    0 = blocked (either tier has no access or daily cap hit)
     *   N = uses left
     */
    getRemaining: (feature: FeatureKey) => number;
    /** True if the user can still use the feature today. */
    canUse: (feature: FeatureKey) => boolean;
    /** Force-reset today's counters (e.g. after successful upgrade). */
    resetToday: () => Promise<void>;
}

function emptySnapshot(): DailyUsageSnapshot {
    return { date: todayKey(), counts: {} };
}

async function persist(snapshot: DailyUsageSnapshot) {
    try {
        await AsyncStorage.setItem(DAILY_USAGE_KEY, JSON.stringify(snapshot));
    } catch (e) {
        console.warn('[dailyUsageStore] persist failed', e);
    }
}

/** Normalize the snapshot: if the stored date isn't today, roll it over. */
function rollover(snapshot: DailyUsageSnapshot): DailyUsageSnapshot {
    const today = todayKey();
    if (snapshot.date !== today) {
        return { date: today, counts: {} };
    }
    return snapshot;
}

const useDailyUsageStore = create<DailyUsageState>((set, get) => ({
    snapshot: emptySnapshot(),
    hydrated: false,

    hydrate: async () => {
        try {
            const raw = await AsyncStorage.getItem(DAILY_USAGE_KEY);
            let snapshot = emptySnapshot();
            if (raw) {
                try {
                    snapshot = rollover(JSON.parse(raw));
                } catch {
                    snapshot = emptySnapshot();
                }
            }
            set({ snapshot, hydrated: true });
            if (raw && snapshot.date !== (JSON.parse(raw).date ?? '')) {
                await persist(snapshot);
            }
        } catch (e) {
            console.warn('[dailyUsageStore] hydrate failed', e);
            set({ hydrated: true });
        }
    },

    consume: async (feature: FeatureKey, amount = 1) => {
        const current = rollover(get().snapshot);
        const used = (current.counts[feature] ?? 0) + amount;
        const next: DailyUsageSnapshot = {
            date: current.date,
            counts: { ...current.counts, [feature]: used },
        };
        set({ snapshot: next });
        await persist(next);
        return used;
    },

    getUsed: (feature: FeatureKey) => {
        const current = rollover(get().snapshot);
        return current.counts[feature] ?? 0;
    },

    getRemaining: (feature: FeatureKey) => {
        const current = rollover(get().snapshot);
        const tier = useSubscriptionStore.getState().effectiveTier;
        const limit = TIER_FEATURES[tier][feature];

        if (typeof limit === 'boolean') {
            return limit ? -1 : 0;
        }
        if (limit === -1) return -1;
        const used = current.counts[feature] ?? 0;
        return Math.max(0, (limit as number) - used);
    },

    canUse: (feature: FeatureKey) => {
        const remaining = get().getRemaining(feature);
        return remaining === -1 || remaining > 0;
    },

    resetToday: async () => {
        const next = emptySnapshot();
        set({ snapshot: next });
        await persist(next);
    },
}));

export default useDailyUsageStore;

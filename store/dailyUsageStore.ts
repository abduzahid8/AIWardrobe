import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import useSubscriptionStore, {
  TIER_FEATURES,
  type FeatureKey,
} from './subscriptionStore';

type DailyUsageSnapshot = {
  date: string;
  counts: Partial<Record<FeatureKey, number>>;
};

type DailyUsageState = {
  snapshot: DailyUsageSnapshot;
  hydrated: boolean;
  hydrate: () => Promise<void>;
  consume: (feature: FeatureKey, amount?: number) => Promise<number>;
  getUsed: (feature: FeatureKey) => number;
  getRemaining: (feature: FeatureKey) => number;
  canUse: (feature: FeatureKey) => boolean;
  resetToday: () => Promise<void>;
};

const DAILY_USAGE_KEY = 'daily_usage_v1';

function todayKey(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function emptySnapshot(): DailyUsageSnapshot {
  return { date: todayKey(), counts: {} };
}

function rollover(snapshot: DailyUsageSnapshot): DailyUsageSnapshot {
  const today = todayKey();
  if (snapshot.date !== today) {
    return { date: today, counts: {} };
  }
  return snapshot;
}

async function persist(snapshot: DailyUsageSnapshot) {
  try {
    await AsyncStorage.setItem(DAILY_USAGE_KEY, JSON.stringify(snapshot));
  } catch (error) {
    console.warn('[dailyUsageStore] persist failed', error);
  }
}

const useDailyUsageStore = create<DailyUsageState>((set, get) => ({
  snapshot: emptySnapshot(),
  hydrated: false,

  hydrate: async () => {
    try {
      const raw = await AsyncStorage.getItem(DAILY_USAGE_KEY);
      if (!raw) {
        set({ snapshot: emptySnapshot(), hydrated: true });
        return;
      }

      let parsed = emptySnapshot();
      try {
        parsed = JSON.parse(raw) as DailyUsageSnapshot;
      } catch {
        parsed = emptySnapshot();
      }

      const snapshot = rollover(parsed);
      set({ snapshot, hydrated: true });
      if (snapshot.date !== parsed.date) {
        await persist(snapshot);
      }
    } catch (error) {
      console.warn('[dailyUsageStore] hydrate failed', error);
      set({ hydrated: true });
    }
  },

  consume: async (feature, amount = 1) => {
    const current = rollover(get().snapshot);
    const nextCount = (current.counts[feature] ?? 0) + amount;
    const next = {
      date: current.date,
      counts: { ...current.counts, [feature]: nextCount },
    };
    set({ snapshot: next });
    await persist(next);
    return nextCount;
  },

  getUsed: (feature) => {
    const current = rollover(get().snapshot);
    return current.counts[feature] ?? 0;
  },

  getRemaining: (feature) => {
    const current = rollover(get().snapshot);
    const tier = useSubscriptionStore.getState().effectiveTier;
    const limit = TIER_FEATURES[tier][feature];

    if (typeof limit === 'boolean') {
      return limit ? -1 : 0;
    }

    if (limit === -1) {
      return -1;
    }

    const used = current.counts[feature] ?? 0;
    return Math.max(0, limit - used);
  },

  canUse: (feature) => {
    const remaining = get().getRemaining(feature);
    return remaining === -1 || remaining > 0;
  },

  resetToday: async () => {
    const snapshot = emptySnapshot();
    set({ snapshot });
    await persist(snapshot);
  },
}));

export default useDailyUsageStore;

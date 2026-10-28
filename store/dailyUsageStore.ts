import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import useSubscriptionStore, {
  TIER_FEATURES,
  type FeatureKey,
} from './subscriptionStore';
import { supabase } from '../lib/supabase';
import { authEvents } from './authEvents';
import { withRequestDeadline } from '../src/lib/requestDeadline';

// Time may have advanced while no screen was mounted or while an RPC was pending.
function currentFeatureLimit(feature: FeatureKey): number {
  useSubscriptionStore.getState().refreshTimeState();
  const tier = useSubscriptionStore.getState().effectiveTier;
  const raw = TIER_FEATURES[tier][feature];
  return typeof raw === 'number' ? raw : raw ? -1 : 0;
}

// ─────────────────────────────────────────────────────────────
// Source-of-truth: Supabase RPC `check_and_increment_usage`.
// AsyncStorage is a read-through cache only (used when offline
// and on cold start so the UI is not blank).
// ─────────────────────────────────────────────────────────────

type DailyUsageSnapshot = {
  date: string;
  counts: Partial<Record<FeatureKey, number>>;
};

type ConsumeResult = {
  allowed: boolean;
  used: number;
  remaining: number;
};

export type UsageConsumeOptions = { requireServer?: boolean; cancelSignal?: AbortSignal; homeCardKey?: string };
export class UsageUnavailableError extends Error {
  constructor() { super('Usage verification unavailable'); this.name = 'UsageUnavailableError'; }
}

type DailyUsageState = {
  snapshot: DailyUsageSnapshot;
  hydrated: boolean;
  hydrate: () => Promise<void>;
  /**
   * Atomically check quota on the server and increment on success.
   * requireServer rejects unavailable verification; other callers retain the offline cache behavior.
   */
  consume: (feature: FeatureKey, amount?: number, options?: UsageConsumeOptions) => Promise<ConsumeResult>;
  getUsed: (feature: FeatureKey) => number;
  getRemaining: (feature: FeatureKey) => number;
  canUse: (feature: FeatureKey) => boolean;
  resetToday: () => Promise<void>;
};

export const DAILY_USAGE_STORAGE_PREFIX = 'daily_usage_v2::';
const storageKey = (owner: string) => `${DAILY_USAGE_STORAGE_PREFIX}${encodeURIComponent(owner)}`;
let writeChain: Promise<void> = Promise.resolve();
export function drainDailyUsageWrites(): Promise<void> { return writeChain; }

function todayKey(): string {
  // Match the UTC day used by Postgres quota counters in every timezone.
  return new Date().toISOString().slice(0, 10);
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

async function persist(snapshot: DailyUsageSnapshot, owner: string | null, revision: number) {
  if (!owner) return;
  writeChain = writeChain.then(async () => {
    if (revision !== authEvents.getRevision() || owner !== authEvents.getUserId()) return;
    try {
      await AsyncStorage.setItem(storageKey(owner), JSON.stringify(snapshot));
    } catch (error) {
      console.warn('[dailyUsageStore] persist failed', error);
    }
  });
  await writeChain;
}

const useDailyUsageStore = create<DailyUsageState>((set, get) => ({
  snapshot: emptySnapshot(),
  hydrated: false,

  hydrate: async () => {
    const revision = authEvents.getRevision();
    const owner = authEvents.getUserId();
    try {
      // Legacy unowned counters cannot safely be assigned to an account.
      const raw = owner ? await AsyncStorage.getItem(storageKey(owner)) : null;
      if (revision !== authEvents.getRevision()) return;
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

      if (!parsed || typeof parsed !== 'object' || !parsed.counts || typeof parsed.counts !== 'object' || Array.isArray(parsed.counts)) {
        parsed = emptySnapshot();
      }
      parsed.counts = Object.fromEntries(Object.entries(parsed.counts).filter(([, count]) => Number.isSafeInteger(count) && (count as number) >= 0));
      const snapshot = rollover(parsed);
      set({ snapshot, hydrated: true });
      if (snapshot.date !== parsed.date) {
        await persist(snapshot, owner, revision);
      }
    } catch (error) {
      console.warn('[dailyUsageStore] hydrate failed', error);
      if (revision !== authEvents.getRevision()) return;
      set({ hydrated: true });
    }
  },

  consume: async (feature, amount = 1, options = {}) => {
    const revision = authEvents.getRevision();
    const owner = authEvents.getUserId();
    const requestDay = todayKey();
    const stillCurrent = () => revision === authEvents.getRevision() && requestDay === todayKey();
    const cancelled = () => ({ allowed: false, used: get().getUsed(feature), remaining: get().getRemaining(feature) });
    if (options.cancelSignal?.aborted) return cancelled();
    if (!Number.isSafeInteger(amount) || amount < 1 || amount > 100) {
      return { allowed: false, used: get().getUsed(feature), remaining: get().getRemaining(feature) };
    }
    // A stable displayed-card identity lets a later attempt recover a committed
    // authorization after timeout, cancellation, remount or app restart.
    const cardKey = options.homeCardKey;
    if (cardKey !== undefined && (feature !== 'homeSwipes' || amount !== 1 ||
        !options.requireServer || typeof cardKey !== 'string' || !cardKey.length || [...cardKey].length > 512)) {
      throw new UsageUnavailableError();
    }
    let limit = currentFeatureLimit(feature);

    // Try the server RPC first — it is atomic and tamper-proof.
    // Bound it with a timeout: on slow networks (15s+ Supabase stalls)
    // awaiting the RPC hangs the Generate button with zero feedback.
    // A caller requiring verification never grants a cached allowance on timeout.
    const RPC_TIMEOUT_MS = 6000;
    try {
      const { data, error } = await withRequestDeadline(signal => {
        const query = cardKey !== undefined ? supabase.rpc('authorize_home_swipe', {
          p_card_key: cardKey, p_day: requestDay,
        }) : supabase.rpc('check_and_increment_usage', {
          p_feature: feature, p_limit: limit, p_amount: amount,
        });
        return typeof query.abortSignal === 'function' ? query.abortSignal(signal) : query;
      }, RPC_TIMEOUT_MS, options.cancelSignal);
      if (!stillCurrent() || options.cancelSignal?.aborted) return cancelled();

      if (!error && Array.isArray(data) && data.length === 1 &&
          typeof data[0]?.allowed === 'boolean' && Number.isSafeInteger(data[0]?.used) && data[0].used >= 0 &&
          Number.isSafeInteger(data[0]?.remaining) && data[0].remaining >= -1) {
        const row = data[0] as { allowed: boolean; used: number; remaining: number };

        // Mirror to local cache so the UI updates immediately even offline.
        const previous = get().snapshot;
        const current = rollover(previous);
        const nextUsed = Math.max(current.counts[feature] ?? 0, row.used);
        const next = {
          date: current.date,
          counts: { ...current.counts, [feature]: nextUsed },
        };
        if (previous.date !== next.date || (current.counts[feature] ?? 0) !== nextUsed) {
          set({ snapshot: next });
          const cacheWrite = persist(next, owner, revision);
          // Server approval is authoritative; cache storage must not stall the gesture.
          if (options.requireServer) void cacheWrite.catch(() => {});
          else await cacheWrite;
        }
        if (!stillCurrent() || options.cancelSignal?.aborted) return cancelled();

        return { allowed: row.allowed, used: row.used, remaining: row.remaining };
      }
    } catch (err) {
      if (options.cancelSignal?.aborted) return cancelled();
      console.warn('[dailyUsageStore] RPC unavailable', err);
    }

    if (!stillCurrent() || options.cancelSignal?.aborted) return cancelled();
    if (options.requireServer) throw new UsageUnavailableError();

    // Re-evaluate after the await: a plan or trial may have expired meanwhile.
    limit = currentFeatureLimit(feature);
    // Offline fallback — best-effort enforcement against the cached counter.
    const current = rollover(get().snapshot);
    const used = (current.counts[feature] ?? 0);
    const allowed = limit === -1 ? true : used + amount <= limit;
    if (!allowed) {
      return { allowed: false, used, remaining: Math.max(0, limit - used) };
    }
    const nextCount = used + amount;
    const next = {
      date: current.date,
      counts: { ...current.counts, [feature]: nextCount },
    };
    set({ snapshot: next });
    await persist(next, owner, revision);
    if (!stillCurrent() || options.cancelSignal?.aborted) return cancelled();
    return {
      allowed: true,
      used: nextCount,
      remaining: limit === -1 ? -1 : Math.max(0, limit - nextCount),
    };
  },

  getUsed: (feature) => {
    const current = rollover(get().snapshot);
    return current.counts[feature] ?? 0;
  },

  getRemaining: (feature) => {
    const current = rollover(get().snapshot);
    const limit = currentFeatureLimit(feature);

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
    const owner = authEvents.getUserId();
    const revision = authEvents.getRevision();
    const snapshot = emptySnapshot();
    set({ snapshot });
    await persist(snapshot, owner, revision);
  },
}));

export default useDailyUsageStore;

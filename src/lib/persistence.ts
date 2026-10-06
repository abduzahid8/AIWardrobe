/**
 * Persistence Key Registry
 *
 * Every Zustand persist() slice or raw AsyncStorage consumer that
 * holds per-user data MUST register its storage key here so that
 * account deletion and logout fully wipe local state on the device.
 *
 * Adding a key here is a one-line change. Forgetting to add one is
 * a data-leak bug on shared devices.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

import { deleteSecureItem } from '../utils/secureStorage';
import { useStylePreferenceStore } from '../../store/stylePreferenceStore';
import useAvatarStore from '../../store/avatarStore';

/**
 * AsyncStorage keys used by persist(...) middleware or direct reads.
 * Keep this list in sync with every `persist({ name: '...' })` block.
 */
export const PERSISTED_ASYNC_STORAGE_KEYS: readonly string[] = [
    // Zustand persist stores
    'wardrobe-storage',
    'avatar-storage',
    'body-profile-storage-v1', // bodyProfileStore (Month 1 of body-fit plan)
    'subscription-storage',
    'style-preferences', // stylePreferenceStore (was incorrectly 'style-preference-storage')

    // Direct AsyncStorage keys (non-persist stores)
    'daily_usage_v1', // dailyUsageStore (was incorrectly 'daily-usage-storage')
    // NOTE: '@app_language' is intentionally NOT listed here. Language is a
    // device-level preference chosen on the first-run gate — it must survive
    // logout / account deletion. Otherwise the user would be forced through
    // the language screen again on every re-login.
    'offline_queue', // offlineQueue.ts (was incorrectly 'offline_request_queue')
    'upload_queue_v1', // uploadQueue.ts (was incorrectly 'upload_queue')
    '@image_cache_index', // imageCache.ts
    'aiwardrobe_hints_seen', // useFeatureHints.ts

    // Legacy / direct storage keys
    'userToken',
    'analytics_event_queue',
    'analytics_enabled', // Added - separate from queue
    'crash_reports',

    // Subscription / trial keys written directly (not via Zustand persist)
    'subscription_tier',
    'subscription_expiry',
    'trial_started_at_v1',

    // Promo code keys (promoCodeStore) — must be cleared to prevent cross-user leakage
    'promo_redeemed_v1',
    'promo_skipped_v1',

    // Per-user keys that previously leaked across accounts on shared devices
    // (delete -> create without app restart served user A's data to user B).
    'myWardrobeItems', // legacy useWardrobeItems hook
    'wardrobeFavorites', // legacy useWardrobeItems hook
    'recentSearches', // shoppingService
    'shoppingWishlist', // shoppingService
    'pending_corrections', // feedbackService
    'notificationSettings', // notificationService
    'lastWardrobeVideo', // HomeScreen saved video — per-user content
] as const;

/**
 * AsyncStorage key prefixes whose entries are per-user but keyed dynamically
 * (e.g. `daily_ai_outfit_v18::<userId>::<style>`). These cannot be listed
 * statically above — they are wiped by prefix scan in
 * `clearAllPersistedUserData()`.
 */
const PERSISTED_KEY_PREFIXES: readonly string[] = [
    'daily_ai_outfit_',
] as const;

/**
 * SecureStore keys. Listed separately because SecureStore has a
 * different (and smaller) API surface and must be cleared with
 * deleteSecureItem rather than AsyncStorage.multiRemove.
 */
export const PERSISTED_SECURE_KEYS: readonly string[] = [
    'supabase.auth.token',
    'supabase.auth.refreshToken',
] as const;

/**
 * Reset every in-memory Zustand singleton to its logged-out initial state.
 *
 * `AsyncStorage.multiRemove` only clears the on-disk cache; each Zustand
 * store also holds the previous user's data in memory. Without this reset,
 * a newly created account in the same session renders user A's wardrobe,
 * subscription tier, promo flags, and daily-usage counters until the next
 * cold start — and Home thrashes between the essentials grid and the AI
 * decks as the real (empty) wardrobe arrives, freezing the JS thread.
 *
 * Uses lazy `require()` (like bodyProfileStore below) so this module never
 * creates an import cycle: wardrobe/subscription stores import auth, auth
 * imports this file.
 */
export function resetInMemoryUserState(): void {
    // Wardrobe — the biggest leak: items/outfits/wearLogs/lastSyncedAt.
    // Also drops any in-flight rehydrate so user A's fetch can neither block
    // nor resolve into user B's store, and unsubscribes the realtime channel
    // (currently dead code, but the filter captures userId at subscribe time).
    try {
        const useWardrobeStore = require('../../store/wardrobeStore').default;
        try {
            useWardrobeStore.getState().unsubscribeRealtime?.();
        } catch {}
        try {
            const { invalidateRehydrateState } =
                require('../../store/slices/wardrobeSyncSlice');
            invalidateRehydrateState?.();
        } catch {}
        useWardrobeStore.setState({
            items: [],
            outfits: [],
            wearLogs: [],
            dailySuggestion: null,
            streak: 0,
            lastWearDate: null,
            lastSyncedAt: null,
            pendingActions: [],
            dislikedOutfitKeys: [],
            isLoading: false,
            isSyncing: false,
        });
    } catch {}

    // Subscription — tier/expiry/trial must not leak; derived flags are reset
    // to the free-tier baseline (pending=false so gates don't spin).
    try {
        const useSubscriptionStore = require('../../store/subscriptionStore').default;
        useSubscriptionStore.setState({
            tier: 'free',
            expiryDate: null,
            trialStartedAt: null,
            isLoading: false,
            lastVerifiedAt: null,
            isPremium: false,
            hasActiveSubscription: false,
            isSubscriptionExpired: false,
            isTrialActive: false,
            isTrialExpired: false,
            isTrialPending: false,
            trialDaysRemaining: 0,
            needsPromoCode: false,
            effectiveTier: 'free',
        });
    } catch {}

    // Daily usage counters — per-user quota cache.
    try {
        const useDailyUsageStore = require('../../store/dailyUsageStore').default;
        const today = new Date();
        const dateKey = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
        useDailyUsageStore.setState({
            snapshot: { date: dateKey, counts: {} },
            hydrated: false,
        });
    } catch {}

    // Promo flags — must not leak redemption/skip across accounts.
    try {
        const usePromoCodeStore = require('../../store/promoCodeStore').default;
        usePromoCodeStore.setState({
            hasRedeemedPromo: false,
            hasSkippedPromo: false,
            isHydrated: false,
        });
    } catch {}

    // Shop catalog is global (not per-user) but holds stale pagination state;
    // clearing avoids showing user A's paginated list while user B's fetch runs.
    try {
        const useShopCatalogStore = require('../../store/shopCatalogStore').default;
        useShopCatalogStore.setState({ items: [], loading: true });
    } catch {}

    // Style / avatar / body-profile (same resets as before, centralized).
    try { useStylePreferenceStore.getState().clearAll(); } catch {}
    try { useAvatarStore.setState({ heightCm: '175', weightKg: '70', bodyType: 'average', gender: 'male' }); } catch {}
    try {
        const useBodyProfileStore = require('../../store/bodyProfileStore').default;
        useBodyProfileStore.getState().reset();
    } catch {}

    // Upload queue — destroy the AppState listener AND drop queued payloads
    // (imageBase64 tied to the old user) so they are never uploaded as the
    // new user via the success handler.
    try {
        const { uploadQueue } = require('../services/uploadQueue');
        try { uploadQueue.destroy(); } catch {}
        try {
            const maybePromise = uploadQueue.clearQueue?.();
            if (maybePromise && typeof (maybePromise as Promise<void>).catch === 'function') {
                (maybePromise as Promise<void>).catch(() => undefined);
            }
        } catch {}
    } catch {}

    // API token cache + React Query cache — never serve user A's authed
    // responses to user B.
    try {
        const { clearCachedToken } = require('./api');
        clearCachedToken?.();
    } catch {}
    try {
        const { clearQueryCache } = require('./queryClient');
        clearQueryCache?.();
    } catch {}
}

/**
 * Wipe every piece of per-user state on this device.
 * Called from auth.logout() and auth.deleteAccount().
 *
 * Swallows individual errors — we still want to clear as much as
 * possible even if one key fails.
 */
export async function clearAllPersistedUserData(): Promise<void> {
    try {
        const keys: string[] = [...PERSISTED_ASYNC_STORAGE_KEYS];
        try {
            const allKeys = await AsyncStorage.getAllKeys();
            for (const prefix of PERSISTED_KEY_PREFIXES) {
                for (const key of allKeys) {
                    if (key.startsWith(prefix) && !keys.includes(key)) {
                        keys.push(key);
                    }
                }
            }
        } catch {
            // Prefix scan is best-effort; static keys above still clear.
        }
        await AsyncStorage.multiRemove(keys);
    } catch {
        // Best-effort: keep going to secure store + memory reset
    }

    // Reset in-memory state of every persisted Zustand store so the
    // next login does not inherit the previous user's data from memory.
    // AsyncStorage.multiRemove only clears the on-disk cache; the
    // Zustand singleton holds state in memory until explicitly reset.
    resetInMemoryUserState();

    await Promise.all(
        PERSISTED_SECURE_KEYS.map((key) =>
            deleteSecureItem(key).catch(() => undefined),
        ),
    );
}

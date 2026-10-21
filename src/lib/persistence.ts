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
import { authEvents } from '../../store/authEvents';
import { drainCalendarWrites, CALENDAR_STORAGE_PREFIX } from '../services/calendarStorage';

import { deleteSecureItem } from '../utils/secureStorage';
import { useStylePreferenceStore } from '../../store/stylePreferenceStore';
import useAvatarStore from '../../store/avatarStore';

/**
 * AsyncStorage keys used by persist(...) middleware or direct reads.
 * Keep this list in sync with every `persist({ name: '...' })` block.
 */
export const PERSISTED_ASYNC_STORAGE_KEYS: readonly string[] = [
    'private_cache_owner_v1', // verified owner of the private persisted caches
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
    'wardrobe-upload-queue', // photo/video analysis queue (separate Zustand store)
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
    'outfitLogs', // legacy unowned calendar plans; owner keys cleared by prefix
    '@outfit_logs', // legacy calendar hook
    '@ai_chat_history',
    '@selected_style',
    'savedInspo', // savedShopStore
    'aiRecentShownItemIds',
    'aiExcludeShownItemIds',
    'notificationLastScheduled',
] as const;

/**
 * AsyncStorage key prefixes whose entries are per-user but keyed dynamically
 * (e.g. `daily_ai_outfit_v18::<userId>::<style>`). These cannot be listed
 * statically above — they are wiped by prefix scan in
 * `clearAllPersistedUserData()`.
 */
const PERSISTED_KEY_PREFIXES: readonly string[] = [
    CALENDAR_STORAGE_PREFIX,
    'daily_ai_outfit_',
    'aiExcludeShownItemIds::',
    'daily_usage_v2::', // dailyUsageStore owner-scoped counters
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
 * Saved shop state additionally queues ordered cleanup of its shared disk key;
 * clearAllPersistedUserData awaits that queue before returning.
 */
export function resetInMemoryUserState(): Promise<void> {
    const pendingPersistence: Promise<unknown>[] = [];
    let persistenceFailed = false;
    const retain = (work: Promise<unknown>) => {
        pendingPersistence.push(work.catch(() => { persistenceFailed = true; }));
    };
    try {
        const { resetUploadQueueState } = require('../store/uploadQueueStore');
        resetUploadQueueState();
    } catch {}

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
            wardrobeCapacityIssue: null,
            pendingReviewAcknowledgements: [],
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

    // Saved shop likes are held outside persist middleware as well as on disk.
    // Invalidate pending hydration before another account can see its result.
    try {
        const savedShop = require('../../store/savedShopStore');
        savedShop.resetSavedShopState();
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
                retain(maybePromise as Promise<void>);
            }
        } catch { persistenceFailed = true; }
    } catch { persistenceFailed = true; }

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

    // Store resets above can schedule new debounced writes after the earlier
    // registry removal. Drain them as part of cleanup, rather than allowing
    // the authentication queue to release the next account prematurely.
    for (const flush of [
        () => require('../../store/wardrobeStore').flushWardrobePersistence?.(),
        () => require('../store/uploadQueueStore').flushUploadQueuePersistence?.(),
        () => require('../../store/savedShopStore').flushSavedShopPersistence?.(),
    ]) {
        try {
            const work = flush();
            if (work) retain(Promise.resolve(work));
        } catch { persistenceFailed = true; }
    }
    return Promise.all(pendingPersistence).then(() => {
        if (persistenceFailed) throw new Error('Local account cleanup failed');
    });
}

/** Marker is intentionally outside the registry: only confirmed cleanup removes it. */
export const ACCOUNT_CLEANUP_PENDING_KEY = 'account_cleanup_pending_v1';
let cleanupRequired = false;
let cleanupTail: Promise<void> = Promise.resolve();

/** Retry incomplete cleanup, including a marker restored after process restart. */
export async function ensurePreviousAccountCleanup(): Promise<void> {
    // A failed read must not be treated as an absent marker.
    const marker = await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY);
    if (cleanupRequired || marker !== null) await clearAllPersistedUserData();
}

/** Serialized cleanup drains each available stage and reports incomplete removal. */
export function clearAllPersistedUserData(expectedRevision = authEvents.getRevision()): Promise<void> {
    const work = cleanupTail.then(() => performAccountCleanup(expectedRevision));
    cleanupTail = work.catch(() => undefined);
    return work;
}

async function performAccountCleanup(expectedRevision: number): Promise<void> {
    const stillCurrent = () => expectedRevision === authEvents.getRevision();
    if (!stillCurrent()) return;
    cleanupRequired = true;
    let failed = false;
    const attempt = async (operation: () => Promise<unknown>) => {
        try { await operation(); } catch { failed = true; }
    };
    await attempt(() => AsyncStorage.setItem(ACCOUNT_CLEANUP_PENDING_KEY, '1'));
    await attempt(drainCalendarWrites);
    await attempt(async () => {
        const { drainDailyUsageWrites } = require('../../store/dailyUsageStore');
        await drainDailyUsageWrites?.();
    });
    if (!stillCurrent()) return;
    await attempt(async () => {
        const { clearUploadQueuePersistence } = require('../store/uploadQueueStore');
        await clearUploadQueuePersistence();
    });
    if (!stillCurrent()) return;
    await attempt(async () => {
        const wardrobe = require('../../store/wardrobeStore');
        await wardrobe.clearWardrobePersistence?.();
    });
    if (!stillCurrent()) return;
    const keys: string[] = [...PERSISTED_ASYNC_STORAGE_KEYS];
    await attempt(async () => {
        const allKeys = await AsyncStorage.getAllKeys();
        for (const prefix of PERSISTED_KEY_PREFIXES) {
            for (const key of allKeys) {
                if (key.startsWith(prefix) && !keys.includes(key)) keys.push(key);
            }
        }
    });
    if (!stillCurrent()) return;
    await attempt(() => AsyncStorage.multiRemove(keys));
    if (!stillCurrent()) return;
    await attempt(resetInMemoryUserState);
    if (!stillCurrent()) return;
    await attempt(async () => {
        // Native-only and lazy to avoid initializing media dependencies at startup.
        const { Platform } = require('react-native');
        if (Platform.OS === 'web') return;
        const { clearAllPendingWardrobeImages } = require('../services/pendingWardrobeMedia');
        await clearAllPendingWardrobeImages(expectedRevision);
    });
    if (!stillCurrent()) return;
    await attempt(async () => {
        const { Platform } = require('react-native');
        if (Platform.OS === 'web') return;
        const { clearAllPendingScanMedia } = require('../services/pendingScanMedia');
        await clearAllPendingScanMedia(expectedRevision);
    });
    if (!stillCurrent()) return;
    await attempt(async () => {
        const { Platform } = require('react-native');
        if (Platform.OS === 'web') return;
        const { clearAllScanReviews } = require('../services/scanReviews');
        await clearAllScanReviews(expectedRevision);
    });
    if (!stillCurrent()) return;
    await Promise.all(PERSISTED_SECURE_KEYS.map(key => attempt(() => deleteSecureItem(key))));
    if (!stillCurrent()) return;
    if (!failed) await attempt(() => AsyncStorage.removeItem(ACCOUNT_CLEANUP_PENDING_KEY));
    if (failed) throw new Error('Local account cleanup failed');
    cleanupRequired = false;
}

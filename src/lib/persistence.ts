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

/**
 * AsyncStorage keys used by persist(...) middleware or direct reads.
 * Keep this list in sync with every `persist({ name: '...' })` block.
 */
export const PERSISTED_ASYNC_STORAGE_KEYS: readonly string[] = [
    // Zustand persist stores
    'wardrobe-storage',
    'try-on-looks-storage',
    'avatar-storage',
    'subscription-storage',
    'price-tracking-storage',
    'style-preferences', // stylePreferenceStore (was incorrectly 'style-preference-storage')

    // Direct AsyncStorage keys (non-persist stores)
    'daily_usage_v1', // dailyUsageStore (was incorrectly 'daily-usage-storage')
    '@app_language', // languageStore
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
 * Wipe every piece of per-user state on this device.
 * Called from auth.logout() and auth.deleteAccount().
 *
 * Swallows individual errors — we still want to clear as much as
 * possible even if one key fails.
 */
export async function clearAllPersistedUserData(): Promise<void> {
    try {
        await AsyncStorage.multiRemove([...PERSISTED_ASYNC_STORAGE_KEYS]);
    } catch {
        // Best-effort: keep going to secure store
    }

    await Promise.all(
        PERSISTED_SECURE_KEYS.map((key) =>
            deleteSecureItem(key).catch(() => undefined),
        ),
    );
}

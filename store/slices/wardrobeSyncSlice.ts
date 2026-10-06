/**
 * Wardrobe Sync Slice — cloud sync, realtime subscriptions, rehydration
 */

import type { StateCreator } from 'zustand';
import { supabase } from '../../lib/supabase';
import { wardrobeApi, wearLogApi, type ApiClothingItem } from '../../src/lib/api';
import { fetchItemsFromServer, processPendingActions } from '../wardrobeSyncService';
import type { PendingAction } from '../wardrobeSyncService';
import type { RealtimeChannel } from '@supabase/supabase-js';
import type { ClothingItem, ClothingCategory, WearLog, Season, Occasion } from '../../src/types/domain';
import type { WardrobeState } from '../wardrobeStore';
import { calculateStreak, mapApiItem } from './helpers';
import useAuthStore from '../auth';

export interface SyncSlice {
    isSyncing: boolean;
    lastSyncedAt: string | null;
    pendingActions: PendingAction[];

    fetchItems: () => Promise<void>;
    syncToServer: () => Promise<void>;
    rehydrateFromCloud: (force?: boolean) => Promise<void>;
    subscribeToRealtime: () => void;
    unsubscribeRealtime: () => void;
}

// 60-second TTL: skip a full Supabase round-trip when items are already
// loaded and the last sync was recent. Upload/delete pass force=true to
// bypass this guard and immediately reflect the mutation.
const REHYDRATE_TTL_MS = 60_000;

let _realtimeChannel: RealtimeChannel | null = null;
let activeRehydratePromise: Promise<void> | null = null;
// Tracks which user the last successful rehydrate belonged to. The TTL guard
// below must be user-scoped: after delete -> create, the new (empty) account
// must never skip its initial fetch because the previous user synced recently.
let _lastRehydrateUserId: string | null = null;

/**
 * Invalidate any in-flight rehydrate and forget which user last synced.
 * Called on logout / account deletion / new-session reset so a stale
 * `activeRehydratePromise` from user A can never block — or resolve into —
 * user B's store, and the TTL guard never skips user B's first fetch.
 */
export function invalidateRehydrateState(): void {
    activeRehydratePromise = null;
    _lastRehydrateUserId = null;
}

export const createSyncSlice: StateCreator<WardrobeState, [], [], SyncSlice> = (set, get, store) => ({
    isSyncing: false,
    lastSyncedAt: null,
    pendingActions: [],

    fetchItems: async () => {
        try {
            set({ isLoading: true });
            const apiItems = await wardrobeApi.list();
            const items: ClothingItem[] = apiItems.map(mapApiItem);
            set({ items, isLoading: false, lastSyncedAt: new Date().toISOString() });
        } catch {
            try {
                const items = await fetchItemsFromServer();
                if (items) set({ items, isLoading: false, lastSyncedAt: new Date().toISOString() });
                else set({ isLoading: false });
            } catch (err) {
                console.error('[WardrobeStore] Fetch failed:', err);
                set({ isLoading: false });
            }
        }
    },

    syncToServer: async () => {
        const { pendingActions, items } = get();
        if (pendingActions.length === 0) return;

        set({ isSyncing: true });
        try {
            const { processedIds, updatedItems } = await processPendingActions(pendingActions, items);
            set((state) => ({
                items: updatedItems,
                pendingActions: state.pendingActions.filter((a) => !processedIds.includes(a.id)),
                isSyncing: false,
                lastSyncedAt: new Date().toISOString(),
            }));
        } catch (err) {
            console.error('[WardrobeStore] Sync failed:', err);
            set({ isSyncing: false });
        }
    },

    rehydrateFromCloud: async (force = false) => {
        const rehydrateStart = Date.now();
        // Capture the user this fetch belongs to. If the account changes
        // mid-flight (delete -> create without restart), the result must be
        // discarded instead of writing user A's items into user B's store.
        const startUserId = useAuthStore.getState().user?.id ?? null;

        // Wait for Zustand storage hydration to finish if it hasn't already
        const persistStore = store as any;
        if (persistStore && persistStore.persist && typeof persistStore.persist.hasHydrated === 'function' && !persistStore.persist.hasHydrated()) {
            const hydrationStart = Date.now();
            await new Promise<void>((resolve) => {
                const unsub = persistStore.persist.onFinishHydration(() => {
                    unsub();
                    resolve();
                });
            });
            if (__DEV__) console.log(`[PERF] 🟡 wardrobe hydration wait: ${Date.now() - hydrationStart}ms`);
        }

        // TTL guard: skip network round-trip when items are already loaded
        // and the last sync was recent. Upload/delete operations pass force=true
        // to bypass this and immediately reflect the mutation.
        // User-scoped: a different user must always fetch, even if the
        // previous user synced seconds ago (delete -> create freeze fix).
        const { items: currentItems, lastSyncedAt } = get();
        const isSameUserAsLastSync = startUserId !== null && startUserId === _lastRehydrateUserId;
        if (
            !force &&
            isSameUserAsLastSync &&
            currentItems.length > 0 &&
            lastSyncedAt &&
            Date.now() - new Date(lastSyncedAt).getTime() < REHYDRATE_TTL_MS
        ) {
            if (__DEV__) console.log(`[PERF] 🟢 wardrobe rehydrate skipped (TTL ${Math.round((Date.now() - new Date(lastSyncedAt).getTime()) / 1000)}s ago)`);
            return;
        }

        if (activeRehydratePromise) {
            return activeRehydratePromise;
        }

        activeRehydratePromise = (async () => {
            set({ isLoading: true });
            try {
                const { pendingActions, wearLogs: localWearLogs } = get();
                if (pendingActions.length > 0) {
                    const syncStart = Date.now();
                    await get().syncToServer();
                    if (__DEV__) console.log(`[PERF] 🟡 wardrobe syncToServer: ${Date.now() - syncStart}ms`);
                }

                // Skip the wear-log fetch when the local store already has persisted
                // data — up to 1000 logs are saved to AsyncStorage by the persist
                // middleware, so a full server round-trip on every cold start is
                // unnecessary (fixes Defect 1.7).
                const hasLocalWearLogs = localWearLogs.length > 0;

                const fetchStart = Date.now();
                // Bound the custom-backend call: on Render cold start it can hang
                // for seconds before the catch-path falls back to Supabase
                // directly (the 2263ms rehydrate in startup logs). 8s is plenty
                // for a healthy backend; slower means fall back now.
                const backendListWithTimeout = Promise.race([
                    wardrobeApi.list(),
                    new Promise<never>((_, reject) =>
                        setTimeout(() => reject(new Error('wardrobe backend timeout')), 8000),
                    ),
                ]);
                const [apiItems, apiLogs] = await Promise.all([
                    backendListWithTimeout,
                    hasLocalWearLogs ? Promise.resolve(null) : wearLogApi.list(),
                ]);
                if (__DEV__) console.log(`[PERF] 🟡 wardrobe fetch (${hasLocalWearLogs ? 'items only' : 'items + wearLogs'}): ${Date.now() - fetchStart}ms`);

                // Account-switch guard: if the user changed while the network
                // was in flight, drop this result instead of writing user A's
                // wardrobe into user B's store (delete -> create freeze fix).
                const endUserId = useAuthStore.getState().user?.id ?? null;
                if (endUserId !== startUserId) {
                    if (__DEV__) console.log('[WardrobeStore] Rehydrate discarded — account changed mid-fetch');
                    set({ isLoading: false });
                    return;
                }

                const items: ClothingItem[] = apiItems.map(mapApiItem);

                // Use server wear logs only when we had none locally; otherwise
                // keep the persisted logs to avoid the extra network round-trip.
                const wearLogs: WearLog[] = hasLocalWearLogs
                    ? localWearLogs
                    : (apiLogs ?? []).map((l) => ({
                        id: l.id,
                        userId: l.userId,
                        itemIds: l.itemIds,
                        outfitId: l.outfitId ?? undefined,
                        date: l.date,
                        occasion: l.occasion ?? undefined,
                        weatherTemp: l.weatherTemp ?? undefined,
                        weatherCondition: l.weatherCondition ?? undefined,
                        notes: l.notes ?? undefined,
                        createdAt: l.createdAt,
                    }));

                set({
                    items,
                    wearLogs,
                    streak: calculateStreak(wearLogs),
                    isLoading: false,
                    lastSyncedAt: new Date().toISOString(),
                });
                _lastRehydrateUserId = endUserId;
            } catch {
                try {
                    // Single fallback query — items only. Wear logs are less critical
                    // for the fallback path, so we keep the current in-memory state
                    // rather than issuing a second Supabase round-trip (Defect 4.4 fix).
                    const serverItems = await fetchItemsFromServer();
                    if (serverItems) {
                        const fallbackUserId = useAuthStore.getState().user?.id ?? null;
                        if (fallbackUserId !== startUserId) {
                            if (__DEV__) console.log('[WardrobeStore] Rehydrate fallback discarded — account changed mid-fetch');
                            set({ isLoading: false });
                        } else {
                            set({
                                items: serverItems,
                                wearLogs: get().wearLogs,
                                isLoading: false,
                                lastSyncedAt: new Date().toISOString(),
                            });
                            _lastRehydrateUserId = fallbackUserId;
                        }
                    } else {
                        set({ isLoading: false });
                    }
                } catch (err) {
                    console.error('[WardrobeStore] Rehydrate failed:', err);
                    set({ isLoading: false });
                }
            }
        })().finally(() => {
            activeRehydratePromise = null;
            if (__DEV__) console.log(`[PERF] 🟢 wardrobe rehydrateFromCloud total: ${Date.now() - rehydrateStart}ms`);
        });

        return activeRehydratePromise;
    },

    subscribeToRealtime: () => {
        if (_realtimeChannel) {
            const status = _realtimeChannel.state;
            if (status !== 'closed' && status !== 'errored') return;
            // Channel is dead — clean it up and re-subscribe
            supabase.removeChannel(_realtimeChannel);
            _realtimeChannel = null;
        }

        const userId = useAuthStore.getState().user?.id;

        const channel: RealtimeChannel = supabase
            .channel('wardrobe-realtime')
            .on(
                'postgres_changes',
                {
                    event: '*',
                    schema: 'public',
                    table: 'clothing_items',
                    ...(userId ? { filter: `user_id=eq.${userId}` } : {}),
                },
                (payload) => {
                    const { eventType } = payload;
                    if (eventType === 'INSERT' || eventType === 'UPDATE') {
                        const row = payload.new as Record<string, unknown>;
                        const { items } = get();
                        const mappedItem: ClothingItem = {
                            id: row.id as string,
                            userId: row.user_id as string,
                            imageUrl: row.image_url as string,
                            thumbnailUrl: row.thumbnail_url as string | undefined,
                            category: (row.category as ClothingCategory) || 'top',
                            subCategory: (row.sub_category as string) || '',
                            primaryColor: (row.primary_color as string) || '',
                            colorHex: (row.color_hex as string) || '#000000',
                            pattern: (row.pattern as string) || 'solid',
                            material: (row.material as string) || '',
                            brand: row.brand as string | undefined,
                            name: row.name as string | undefined,
                            seasons: (row.seasons as Season[]) || [],
                            occasions: (row.occasions as Occasion[]) || [],
                            wearCount: (row.wear_count as number) || 0,
                            lastWornAt: row.last_worn_at as string | null,
                            isFavorite: (row.is_favorite as boolean) || false,
                            createdAt: row.created_at as string,
                            updatedAt: row.updated_at as string,
                            detectionConfidence: row.detection_confidence as number | undefined,
                        };

                        const exists = items.find(i => i.id === mappedItem.id);
                        if (exists) {
                            set({ items: items.map(i => i.id === mappedItem.id ? mappedItem : i) });
                        } else {
                            set({ items: [mappedItem, ...items] });
                        }
                    } else if (eventType === 'DELETE') {
                        const oldRow = payload.old as Record<string, unknown>;
                        set({ items: get().items.filter(i => i.id !== oldRow.id) });
                    }
                }
            )
            .subscribe();

        _realtimeChannel = channel;
    },

    unsubscribeRealtime: () => {
        if (_realtimeChannel) {
            supabase.removeChannel(_realtimeChannel);
            _realtimeChannel = null;
        }
    },
});

import readinessLogger from '../../src/utils/logger';
/**
 * Wardrobe Sync Slice — cloud sync, realtime subscriptions, rehydration
 */

import type { StateCreator } from 'zustand';
import type { PersistOptions, PersistStorage } from 'zustand/middleware';
import { supabase } from '../../lib/supabase';
import { fetchItemsFromServer, fetchWearLogsFromServer, processPendingActions, reconcileCloudItems } from '../wardrobeSyncService';
import type { PendingAction, WardrobeCapacityIssue } from '../wardrobeSyncService';
import type { RealtimeChannel } from '@supabase/supabase-js';
import type { ClothingItem, ClothingCategory, Season, Occasion } from '../../src/types/domain';
import type { WardrobeState } from '../wardrobeStore';
import { calculateStreak } from './helpers';
import useAuthStore from '../auth';
import { authEvents } from '../authEvents';
import { replaceSavedOutfitImages } from '../../src/utils/confirmedOutfitMedia';

export interface SyncSlice {
    isSyncing: boolean;
    lastSyncedAt: string | null;
    pendingActions: PendingAction[];
    wardrobeCapacityIssue: WardrobeCapacityIssue | null;
    /** Durable cleanup journal: garment confirmation must precede review removal. */
    pendingReviewAcknowledgements: string[];

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
let activeSync: { revision: number; promise: Promise<void> } | null = null;

/**
 * Invalidate any in-flight rehydrate and forget which user last synced.
 * Called on logout / account deletion / new-session reset so a stale
 * `activeRehydratePromise` from user A can never block — or resolve into —
 * user B's store, and the TTL guard never skips user B's first fetch.
 */
export function invalidateRehydrateState(): void {
    activeRehydratePromise = null;
    activeSync = null;
    _lastRehydrateUserId = null;
}

export const createSyncSlice: StateCreator<WardrobeState, [], [], SyncSlice> = (set, get, store) => ({
    isSyncing: false,
    lastSyncedAt: null,
    pendingActions: [],
    wardrobeCapacityIssue: null,
    pendingReviewAcknowledgements: [],

    fetchItems: async () => {
        const revision = authEvents.getRevision();
        const userId = useAuthStore.getState().user?.id;
        if (!userId) return;
        const sameAccount = () => revision === authEvents.getRevision()
            && useAuthStore.getState().user?.id === userId;
        const before = get().items;
        set({ isLoading: true });
        try {
            const items = await fetchItemsFromServer();
            if (!sameAccount()) return;
            if (items) set(state => ({ items: reconcileCloudItems(items, before, state.items, state.pendingActions), lastSyncedAt: new Date().toISOString() }));
        } catch (err) {
            console.error('[WardrobeStore] Fetch failed:', err);
        } finally { if (sameAccount()) set({ isLoading: false }); }
    },

    syncToServer: async () => {
        const revision = authEvents.getRevision();
        const userId = useAuthStore.getState().user?.id;
        if (!userId) return;
        if (activeSync?.revision === revision) return activeSync.promise;
        const { pendingActions, items } = get();
        if (pendingActions.length === 0 && get().pendingReviewAcknowledgements.length === 0) return;
        const sameAccount = () => revision === authEvents.getRevision()
            && useAuthStore.getState().user?.id === userId;
        const before = new Map(items.map(item => [item.id, item]));
        const startingActionIds = new Set(pendingActions.map(action => action.id));
        set({ isSyncing: true });
        const operation = (async () => {
            let acknowledgedActions: PendingAction[] = [];
            try {
                // Force a fresh snapshot: an earlier debounced write may have failed.
                const persistence = (store as typeof store & { persist?: {
                    getOptions: () => Partial<PersistOptions<WardrobeState, unknown>>;
                } }).persist?.getOptions();
                const storage = persistence?.storage as (PersistStorage<unknown> & { flush?: () => Promise<void> }) | undefined;
                if (!storage || !persistence?.name) throw new Error('Wardrobe persistence unavailable');
                const persistSnapshot = async () => {
                    const snapshot = persistence.partialize ? persistence.partialize(get()) : get();
                    const write = Promise.resolve(storage.setItem(persistence.name!, {
                        state: snapshot, version: persistence.version ?? 0,
                    }));
                    await Promise.all([write, Promise.resolve().then(() => storage.flush?.())]);
                };
                await persistSnapshot();
                if (!sameAccount()) return;
                const checkpointFavorite = async (actions: PendingAction[]) => {
                    if (!sameAccount()) throw new Error('Wardrobe checkpoint account changed');
                    const updates = new Map(actions.map(action => [action.id, action]));
                    if (actions.some(action => !get().pendingActions.some(current => current.id === action.id))) {
                        throw new Error('Favorite action no longer pending');
                    }
                    set(state => ({ pendingActions: state.pendingActions.map(action => updates.get(action.id) ?? action) }));
                    await persistSnapshot();
                    if (!sameAccount()) throw new Error('Wardrobe checkpoint account changed');
                };
                const { processedIds, updatedItems, capacityIssue } = (pendingActions.length ? await processPendingActions(pendingActions, items, userId, checkpointFavorite)
                    : { processedIds: [] as string[], updatedItems: items, capacityIssue: undefined });
                if (!sameAccount()) return;
                // Keep persisted versions if acknowledgement persistence fails.
                acknowledgedActions = get().pendingActions.filter(action => startingActionIds.has(action.id) && processedIds.includes(action.id));
                const confirmedReviewIds = acknowledgedActions.filter(action => action.type === 'add_item')
                    .map(action => action.payload.id).filter((id): id is string => typeof id === 'string'
                        && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id));
                const previousIssue = get().wardrobeCapacityIssue;
                const reportedIssue = capacityIssue ?? previousIssue;
                const remainingIssueIds = reportedIssue?.actionIds.filter(id => !processedIds.includes(id) && get().pendingActions.some(action => action.id === id)) ?? [];
                const nextIssue = reportedIssue && remainingIssueIds.length ? { ...reportedIssue, actionIds: remainingIssueIds } : null;
                const issueChanged = JSON.stringify(previousIssue) !== JSON.stringify(nextIssue);
                const synced = new Map(updatedItems.map(item => [item.id, item]));
                const imageReplacements = new Map<string, string>();
                for (const item of updatedItems) {
                    const previous = before.get(item.id);
                    for (const field of ['imageUrl', 'thumbnailUrl'] as const) {
                        const old = previous?.[field], next = item[field];
                        if (old && next && old !== next) imageReplacements.set(old, next);
                    }
                }
                set((state) => {
                    const unconfirmedFavorites = new Set(state.pendingActions.filter(action =>
                        action.type === 'update_item' && !processedIds.includes(action.id)
                        && typeof action.payload.itemId === 'string').map(action => String(action.payload.itemId)));
                    return {
                        // Preserve additions, removals and edits made during the request.
                        items: state.items.map(item => {
                            const previous = before.get(item.id), next = synced.get(item.id);
                            if (!next) return item;
                            if (previous === item && !unconfirmedFavorites.has(item.id)) return next;
                            // An intervening favorite/name edit must not retain a
                            // temporary image, nor overwrite a newly selected photo.
                            return { ...item,
                                ...(previous?.imageUrl === item.imageUrl && next.imageUrl ? { imageUrl: next.imageUrl } : {}),
                                ...(previous?.thumbnailUrl === item.thumbnailUrl && next.thumbnailUrl ? { thumbnailUrl: next.thumbnailUrl } : {}),
                                ...(previous && previous.isFavorite === item.isFavorite && typeof next.isFavorite === 'boolean'
                                    && !unconfirmedFavorites.has(item.id) ? { isFavorite: next.isFavorite } : {}),
                            };
                        }),
                        outfits: state.outfits?.map(outfit => replaceSavedOutfitImages(outfit, imageReplacements)) ?? [],
                        dailySuggestion: state.dailySuggestion ? { ...state.dailySuggestion,
                            outfit: replaceSavedOutfitImages(state.dailySuggestion.outfit, imageReplacements),
                        } : state.dailySuggestion,
                        pendingActions: state.pendingActions.filter(action => !processedIds.includes(action.id)).map(action => {
                            if (action.type !== 'add_item') return action;
                            const id = String(action.payload.id), previous = before.get(id), next = synced.get(id);
                            if (!previous || !next || typeof next.isFavorite !== 'boolean' || unconfirmedFavorites.has(id)
                                || action.payload.isFavorite !== previous.isFavorite || action.payload.isFavorite === next.isFavorite) return action;
                            // A later metadata upload must not resurrect the
                            // superseded favorite value from its old snapshot.
                            return { ...action, payload: { ...action.payload, isFavorite: next.isFavorite } };
                        }),
                        pendingReviewAcknowledgements: [...new Set([...state.pendingReviewAcknowledgements, ...confirmedReviewIds])],
                        wardrobeCapacityIssue: nextIssue,
                        ...(processedIds.length ? { lastSyncedAt: new Date().toISOString() } : {}),
                    };
                });
                // Persist acknowledgement before any later file cleanup can run.
                if (acknowledgedActions.length || issueChanged) await persistSnapshot();
                acknowledgedActions = [];
                const reviewIds = [...get().pendingReviewAcknowledgements];
                if (reviewIds.length) {
                    const { acknowledgeSavedScanReviewItems } = require('../../src/services/scanReviews') as typeof import('../../src/services/scanReviews');
                    await acknowledgeSavedScanReviewItems(userId, reviewIds);
                    if (!sameAccount()) return;
                    set(state => ({ pendingReviewAcknowledgements: state.pendingReviewAcknowledgements.filter(id => !reviewIds.includes(id)) }));
                    try { await persistSnapshot(); }
                    catch (error) {
                        if (sameAccount()) set(state => ({ pendingReviewAcknowledgements: [...new Set([...reviewIds, ...state.pendingReviewAcknowledgements])] }));
                        throw error;
                    }
                }
            } catch (err) {
                if (sameAccount() && acknowledgedActions.length) {
                    // Unknown disk state must remain retryable. Stable IDs make
                    // replay safe, and old actions must stay ahead of newer edits.
                    set(state => ({ pendingActions: [
                        ...acknowledgedActions.filter(action => !state.pendingActions.some(current => current.id === action.id)),
                        ...state.pendingActions,
                    ] }));
                }
                console.error('[WardrobeStore] Sync failed:', err);
            } finally {
                if (sameAccount()) set({ isSyncing: false });
            }
        })();
        const entry = { revision, promise: operation };
        activeSync = entry;
        try { await operation; }
        finally {
            if (activeSync === entry) activeSync = null;
            // Drain newly queued user actions; failed old actions alone do not spin.
            if (sameAccount() && get().pendingActions.some(action => !startingActionIds.has(action.id))) {
                await get().syncToServer();
            }
        }
    },

    rehydrateFromCloud: async (force = false) => {
        const rehydrateStart = Date.now();
        // Capture the user this fetch belongs to. If the account changes
        // mid-flight (delete -> create without restart), the result must be
        // discarded instead of writing user A's items into user B's store.
        const startUserId = useAuthStore.getState().user?.id ?? null;
        const revision = authEvents.getRevision();
        if (!startUserId) return;

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
            if (__DEV__) readinessLogger.debug(String(`[PERF] 🟡 wardrobe hydration wait: ${Date.now() - hydrationStart}ms`), []);
        }

        if (revision !== authEvents.getRevision() || useAuthStore.getState().user?.id !== startUserId) return;

        // TTL guard: skip network round-trip when items are already loaded
        // and the last sync was recent. Upload/delete operations pass force=true
        // to bypass this and immediately reflect the mutation.
        // User-scoped: a different user must always fetch, even if the
        // previous user synced seconds ago (delete -> create freeze fix).
        const { items: currentItems, lastSyncedAt, pendingActions } = get();
        const isSameUserAsLastSync = startUserId !== null && startUserId === _lastRehydrateUserId;
        if (
            !force &&
            pendingActions.length === 0 &&
            get().pendingReviewAcknowledgements.length === 0 &&
            isSameUserAsLastSync &&
            currentItems.length > 0 &&
            lastSyncedAt &&
            Date.now() - new Date(lastSyncedAt).getTime() < REHYDRATE_TTL_MS
        ) {
            if (__DEV__) readinessLogger.debug(String(`[PERF] 🟢 wardrobe rehydrate skipped (TTL ${Math.round((Date.now() - new Date(lastSyncedAt).getTime()) / 1000)}s ago)`), []);
            return;
        }

        if (activeRehydratePromise) {
            return activeRehydratePromise;
        }

        const sameAccount = () => revision === authEvents.getRevision() && useAuthStore.getState().user?.id === startUserId;
        const operation = (async () => {
            set({ isLoading: true });
            try {
                if (get().pendingActions.length || get().pendingReviewAcknowledgements.length) await get().syncToServer();
                if (!sameAccount()) return;
                const beforeItems = get().items;
                const beforeLogs = new Set(get().wearLogs.map(log => log.id));
                const [items, cloudLogs] = await Promise.all([fetchItemsFromServer(), fetchWearLogsFromServer()]);
                if (!sameAccount()) return;
                set(state => {
                    const logs = cloudLogs === null ? [...state.wearLogs] : [...cloudLogs];
                    const pendingLogIds = new Set(state.pendingActions.filter(action => action.type === 'add_wear_log').map(action => action.payload.id));
                    const byId = new Map(logs.map(log => [log.id, log]));
                    for (const log of state.wearLogs) if (!beforeLogs.has(log.id) || pendingLogIds.has(log.id)) byId.set(log.id, log);
                    const wearLogs = [...byId.values()].sort((a, b) => b.date.localeCompare(a.date));
                    return {
                        ...(items ? { items: reconcileCloudItems(items, beforeItems, state.items, state.pendingActions) } : {}),
                        wearLogs, streak: calculateStreak(wearLogs),
                        ...(items !== null && cloudLogs !== null ? { lastSyncedAt: new Date().toISOString() } : {}),
                    };
                });
                if (items !== null && cloudLogs !== null) _lastRehydrateUserId = startUserId;
                if (items !== null) {
                    // Recover pre-journal reviews only from authoritative rows,
                    // never cached or optimistic garments after a failed fetch.
                    const { findConfirmedScanReviewItems } = require('../../src/services/scanReviews') as typeof import('../../src/services/scanReviews');
                    const confirmed = await findConfirmedScanReviewItems(startUserId, items.map(item => item.id));
                    if (!sameAccount()) return;
                    if (confirmed.length) {
                        set(state => ({ pendingReviewAcknowledgements: [...new Set([...state.pendingReviewAcknowledgements, ...confirmed])] }));
                        const wasSyncing = get().isSyncing;
                        await get().syncToServer();
                        // A concurrent sync may have already captured its journal.
                        if (wasSyncing && sameAccount() && get().pendingReviewAcknowledgements.some(id => confirmed.includes(id))) {
                            await get().syncToServer();
                        }
                    }
                }
            } catch (err) { console.error('[WardrobeStore] Rehydrate failed:', err); }
            finally { if (sameAccount()) set({ isLoading: false }); }
        })();
        activeRehydratePromise = operation;
        try { await operation; }
        finally {
            if (activeRehydratePromise === operation) activeRehydratePromise = null;
            if (__DEV__) readinessLogger.debug(`[PERF] wardrobe rehydrate: ${Date.now() - rehydrateStart}ms`, []);
        }
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

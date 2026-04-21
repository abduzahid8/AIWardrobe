/**
 * Wardrobe Sync Slice — cloud sync, realtime subscriptions, rehydration
 */

import type { StateCreator } from 'zustand';
import { supabase } from '../../lib/supabase';
import { wardrobeApi, wearLogApi, type ApiClothingItem } from '../../src/lib/api';
import { fetchItemsFromServer, fetchWearLogsFromServer, processPendingActions } from '../wardrobeSyncService';
import type { PendingAction } from '../wardrobeSyncService';
import type { RealtimeChannel } from '@supabase/supabase-js';
import type { ClothingItem, ClothingCategory, WearLog, Season, Occasion } from '../../src/types/domain';
import type { WardrobeState } from '../wardrobeStore';
import { calculateStreak, mapApiItem } from './helpers';

export interface SyncSlice {
    isSyncing: boolean;
    lastSyncedAt: string | null;
    pendingActions: PendingAction[];

    fetchItems: () => Promise<void>;
    syncToServer: () => Promise<void>;
    rehydrateFromCloud: () => Promise<void>;
    subscribeToRealtime: () => void;
    unsubscribeRealtime: () => void;
}

let _realtimeChannel: RealtimeChannel | null = null;

export const createSyncSlice: StateCreator<WardrobeState, [], [], SyncSlice> = (set, get) => ({
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

    rehydrateFromCloud: async () => {
        set({ isLoading: true });
        try {
            const { pendingActions } = get();
            if (pendingActions.length > 0) {
                await get().syncToServer();
            }

            const [apiItems, apiLogs] = await Promise.all([
                wardrobeApi.list(),
                wearLogApi.list(),
            ]);

            const items: ClothingItem[] = apiItems.map(mapApiItem);
            const wearLogs: WearLog[] = apiLogs.map((l) => ({
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
        } catch {
            try {
                const serverItems = await fetchItemsFromServer();
                const serverLogs = await fetchWearLogsFromServer();
                if (serverItems) {
                    set({
                        items: serverItems,
                        wearLogs: serverLogs || get().wearLogs,
                        isLoading: false,
                        lastSyncedAt: new Date().toISOString(),
                    });
                } else {
                    set({ isLoading: false });
                }
            } catch (err) {
                console.error('[WardrobeStore] Rehydrate failed:', err);
                set({ isLoading: false });
            }
        }
    },

    subscribeToRealtime: () => {
        if (_realtimeChannel) return;

        const useAuthStore = require('../auth').default;
        const userId = useAuthStore.getState()?.user?.id;

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

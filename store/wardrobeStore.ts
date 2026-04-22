/**
 * Wardrobe Store — Composed from domain slices
 *
 * Single source of truth for: clothing items, outfits, wear logs,
 * cloud sync, and realtime subscriptions.
 *
 * Architecture: each slice is a separate file under store/slices/
 * to keep the codebase modular and testable.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

import type { ClothingItem } from '../src/types/domain';
import type { PendingAction } from './wardrobeSyncService';

import { createItemsSlice, type ItemsSlice } from './slices/wardrobeItemsSlice';
import { createOutfitsSlice, type OutfitsSlice } from './slices/wardrobeOutfitsSlice';
import { createWearLogSlice, type WearLogSlice } from './slices/wardrobeWearLogSlice';
import { createSyncSlice, type SyncSlice } from './slices/wardrobeSyncSlice';

// ============================================
// COMPOSED STATE TYPE
// ============================================

export type WardrobeState = ItemsSlice & OutfitsSlice & WearLogSlice & SyncSlice;

// ============================================
// STORE
// ============================================

const useWardrobeStore = create<WardrobeState>()(
    persist(
        (...a) => ({
            ...createItemsSlice(...a),
            ...createOutfitsSlice(...a),
            ...createWearLogSlice(...a),
            ...createSyncSlice(...a),
        }),
        {
            name: 'wardrobe-storage',
            version: 1,
            storage: createJSONStorage(() => AsyncStorage),
            migrate: (persistedState: unknown, version: number) => {
                const state = persistedState as Record<string, unknown>;
                if (version < 1) {
                    const isUUID = (id: unknown) =>
                        typeof id === 'string' &&
                        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);
                    if (Array.isArray(state.pendingActions)) {
                        state.pendingActions = (state.pendingActions as PendingAction[]).filter((a) => {
                            if (a.type === 'add_item') {
                                return isUUID((a.payload as { id?: unknown }).id);
                            }
                            return true;
                        });
                    }
                    if (Array.isArray(state.items)) {
                        state.items = (state.items as ClothingItem[]).filter((i) => isUUID(i.id));
                    }
                }
                return state;
            },
            partialize: (state) => ({
                items: state.items,
                outfits: state.outfits,
                wearLogs: state.wearLogs.slice(0, 1000),
                dailySuggestion: state.dailySuggestion,
                streak: state.streak,
                lastWearDate: state.lastWearDate,
                lastSyncedAt: state.lastSyncedAt,
                pendingActions: state.pendingActions,
                dislikedOutfitKeys: state.dislikedOutfitKeys,
            }),
        }
    )
);

export default useWardrobeStore;

/**
 * Wardrobe Store — Unified data layer for closet management
 *
 * Single source of truth for:
 * - Clothing items (with offline-first caching)
 * - Outfits (AI-generated and user-created)
 * - Wear logs (behavioral tracking)
 * - Daily suggestions
 * - Streak tracking
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../lib/supabase';
import { fetchItemsFromServer, processPendingActions } from './wardrobeSyncService';
import type { PendingAction } from './wardrobeSyncService';

import type {
    ClothingItem,
    ClothingCategory,
    Outfit,
    WearLog,
    DailySuggestion,
    Occasion,
    Season,
} from '../src/types/domain';

// ============================================
// TYPES
// ============================================

interface WardrobeState {
    // Data
    items: ClothingItem[];
    outfits: Outfit[];
    wearLogs: WearLog[];
    dailySuggestion: DailySuggestion | null;

    // Engagement
    streak: number;
    lastWearDate: string | null;      // YYYY-MM-DD

    // Sync status
    isLoading: boolean;
    isSyncing: boolean;
    lastSyncedAt: string | null;
    pendingActions: PendingAction[];   // Offline queue

    // Item CRUD
    addItem: (item: Omit<ClothingItem, 'id' | 'createdAt' | 'updatedAt' | 'wearCount' | 'lastWornAt' | 'isFavorite'>) => Promise<void>;
    removeItem: (id: string) => Promise<void>;
    updateItem: (id: string, updates: Partial<ClothingItem>) => void;
    toggleFavorite: (id: string) => void;

    // Outfit management
    addOutfit: (outfit: Omit<Outfit, 'id' | 'createdAt' | 'wornCount' | 'lastWornAt' | 'saved'>) => void;
    saveOutfit: (id: string) => void;
    rateOutfit: (id: string, rating: 1 | 2 | 3 | 4 | 5) => void;

    // Wear logging (behavioral loop)
    logWear: (itemIds: string[], occasion?: Occasion | string, weather?: { temp: number; condition: string }) => void;
    getStreak: () => number;
    getClosetUtilization: (days?: number) => number;
    getUnwornItems: (days?: number) => ClothingItem[];

    // Daily suggestion
    setDailySuggestion: (suggestion: DailySuggestion) => void;
    clearDailySuggestion: () => void;

    // Data fetching
    fetchItems: () => Promise<void>;
    syncToServer: () => Promise<void>;

    // Filters / queries
    getItemsByCategory: (category: ClothingCategory) => ClothingItem[];
    getItemsBySeason: (season: Season) => ClothingItem[];
    getRecentWearLogs: (count?: number) => WearLog[];
}

// PendingAction type is imported from wardrobeSyncService.ts

// ============================================
// HELPERS
// ============================================

const generateId = () => `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

const getTodayDate = () => {
    const now = new Date();
    return now.toISOString().split('T')[0]; // YYYY-MM-DD
};

const calculateStreak = (wearLogs: WearLog[]): number => {
    if (wearLogs.length === 0) return 0;

    // Get unique dates, sorted descending
    const dates = [...new Set(wearLogs.map(log => log.date))].sort().reverse();

    const today = getTodayDate();
    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];

    // Streak must include today or yesterday
    if (dates[0] !== today && dates[0] !== yesterday) return 0;

    let streak = 1;
    for (let i = 1; i < dates.length; i++) {
        const prev = new Date(dates[i - 1]);
        const curr = new Date(dates[i]);
        const diffDays = (prev.getTime() - curr.getTime()) / 86400000;

        if (Math.round(diffDays) === 1) {
            streak++;
        } else {
            break;
        }
    }

    return streak;
};

// ============================================
// STORE
// ============================================

const useWardrobeStore = create<WardrobeState>()(
    persist(
        (set, get) => ({
            // Initial state
            items: [],
            outfits: [],
            wearLogs: [],
            dailySuggestion: null,
            streak: 0,
            lastWearDate: null,
            isLoading: false,
            isSyncing: false,
            lastSyncedAt: null,
            pendingActions: [],

            // ── Item CRUD ──────────────────────────────

            addItem: async (itemInput) => {
                const now = new Date().toISOString();
                const newItem: ClothingItem = {
                    ...itemInput,
                    id: generateId(),
                    wearCount: 0,
                    lastWornAt: null,
                    isFavorite: false,
                    createdAt: now,
                    updatedAt: now,
                };

                set((state) => ({
                    items: [newItem, ...state.items],
                    pendingActions: [
                        ...state.pendingActions,
                        {
                            id: generateId(),
                            type: 'add_item',
                            payload: newItem as unknown as Record<string, unknown>,
                            createdAt: now,
                        },
                    ],
                }));

                // Try to sync immediately
                try {
                    await get().syncToServer();
                } catch {
                    // Will be synced later when online
                    // Item saved offline, will sync later
                }
            },

            removeItem: async (id) => {
                const now = new Date().toISOString();

                set((state) => ({
                    items: state.items.filter((item) => item.id !== id),
                    outfits: state.outfits.map((outfit) => ({
                        ...outfit,
                        itemIds: outfit.itemIds.filter((itemId) => itemId !== id),
                    })),
                    pendingActions: [
                        ...state.pendingActions,
                        {
                            id: generateId(),
                            type: 'remove_item',
                            payload: { itemId: id },
                            createdAt: now,
                        },
                    ],
                }));

                try {
                    await get().syncToServer();
                } catch {
                    // Remove queued offline
                }
            },

            updateItem: (id, updates) => {
                set((state) => ({
                    items: state.items.map((item) =>
                        item.id === id
                            ? { ...item, ...updates, updatedAt: new Date().toISOString() }
                            : item
                    ),
                }));
            },

            toggleFavorite: (id) => {
                set((state) => ({
                    items: state.items.map((item) =>
                        item.id === id
                            ? { ...item, isFavorite: !item.isFavorite }
                            : item
                    ),
                }));
            },

            // ── Outfit Management ──────────────────────

            addOutfit: (outfitInput) => {
                const newOutfit: Outfit = {
                    ...outfitInput,
                    id: generateId(),
                    saved: false,
                    wornCount: 0,
                    lastWornAt: null,
                    createdAt: new Date().toISOString(),
                };

                set((state) => ({
                    outfits: [newOutfit, ...state.outfits],
                }));
            },

            saveOutfit: (id) => {
                set((state) => ({
                    outfits: state.outfits.map((outfit) =>
                        outfit.id === id ? { ...outfit, saved: true } : outfit
                    ),
                }));
            },

            rateOutfit: (id, rating) => {
                set((state) => ({
                    outfits: state.outfits.map((outfit) =>
                        outfit.id === id ? { ...outfit, rating } : outfit
                    ),
                }));
            },

            // ── Wear Logging (Behavioral Loop) ────────

            logWear: (itemIds, occasion, weather) => {
                const today = getTodayDate();
                const now = new Date().toISOString();

                const newLog: WearLog = {
                    id: generateId(),
                    userId: '', // Will be set from auth store
                    itemIds,
                    date: today,
                    occasion,
                    weatherTemp: weather?.temp,
                    weatherCondition: weather?.condition,
                    createdAt: now,
                };

                set((state) => {
                    // Update wear counts on items
                    const updatedItems = state.items.map((item) => {
                        if (itemIds.includes(item.id)) {
                            return {
                                ...item,
                                wearCount: item.wearCount + 1,
                                lastWornAt: now,
                            };
                        }
                        return item;
                    });

                    // Update outfit wear count if applicable
                    const matchingOutfit = state.outfits.find(
                        (o) => o.itemIds.length === itemIds.length &&
                            o.itemIds.every((id) => itemIds.includes(id))
                    );

                    const updatedOutfits = matchingOutfit
                        ? state.outfits.map((o) =>
                            o.id === matchingOutfit.id
                                ? { ...o, wornCount: o.wornCount + 1, lastWornAt: now }
                                : o
                        )
                        : state.outfits;

                    const newLogs = [newLog, ...state.wearLogs].slice(0, 500); // Keep last 500
                    const newStreak = calculateStreak(newLogs);

                    return {
                        items: updatedItems,
                        outfits: updatedOutfits,
                        wearLogs: newLogs,
                        streak: newStreak,
                        lastWearDate: today,
                        pendingActions: [
                            ...state.pendingActions,
                            {
                                id: generateId(),
                                type: 'add_wear_log',
                                payload: newLog as unknown as Record<string, unknown>,
                                createdAt: now,
                            },
                        ],
                    };
                });
            },

            getStreak: () => {
                return calculateStreak(get().wearLogs);
            },

            getClosetUtilization: (days = 30) => {
                const { items, wearLogs } = get();
                if (items.length === 0) return 0;

                const cutoff = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
                const recentLogs = wearLogs.filter((log) => log.date >= cutoff);
                const wornItemIds = new Set(recentLogs.flatMap((log) => log.itemIds));

                return Math.round((wornItemIds.size / items.length) * 100);
            },

            getUnwornItems: (days = 30) => {
                const { items, wearLogs } = get();
                const cutoff = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
                const recentLogs = wearLogs.filter((log) => log.date >= cutoff);
                const wornItemIds = new Set(recentLogs.flatMap((log) => log.itemIds));

                return items.filter((item) => !wornItemIds.has(item.id));
            },

            // ── Daily Suggestion ───────────────────────

            setDailySuggestion: (suggestion) => {
                set({ dailySuggestion: suggestion });
            },

            clearDailySuggestion: () => {
                set({ dailySuggestion: null });
            },

            // ── Data Fetching ──────────────────────────

            fetchItems: async () => {
                try {
                    set({ isLoading: true });
                    const items = await fetchItemsFromServer();
                    if (items) {
                        set({ items, isLoading: false, lastSyncedAt: new Date().toISOString() });
                    } else {
                        set({ isLoading: false });
                    }
                } catch (err) {
                    console.error('[WardrobeStore] Fetch failed:', err);
                    set({ isLoading: false });
                }
            },

            syncToServer: async () => {
                const { pendingActions, items } = get();
                if (pendingActions.length === 0) return;

                set({ isSyncing: true });

                try {
                    const { processedIds, updatedItems } = await processPendingActions(
                        pendingActions,
                        items
                    );

                    set((state) => ({
                        items: updatedItems,
                        pendingActions: state.pendingActions.filter(
                            (a) => !processedIds.includes(a.id)
                        ),
                        isSyncing: false,
                        lastSyncedAt: new Date().toISOString(),
                    }));
                } catch (err) {
                    console.error('[WardrobeStore] Sync failed:', err);
                    set({ isSyncing: false });
                }
            },

            // ── Filters / Queries ──────────────────────

            getItemsByCategory: (category) => {
                return get().items.filter((item) => item.category === category);
            },

            getItemsBySeason: (season) => {
                return get().items.filter((item) => item.seasons.includes(season));
            },

            getRecentWearLogs: (count = 10) => {
                return get().wearLogs.slice(0, count);
            },
        }),
        {
            name: 'wardrobe-storage',
            storage: createJSONStorage(() => AsyncStorage),
            // Only persist essential data, not loading states
            partialize: (state) => ({
                items: state.items,
                outfits: state.outfits,
                wearLogs: state.wearLogs.slice(0, 200), // Cap persisted logs
                dailySuggestion: state.dailySuggestion,
                streak: state.streak,
                lastWearDate: state.lastWearDate,
                lastSyncedAt: state.lastSyncedAt,
                pendingActions: state.pendingActions,
            }),
        }
    )
);

export default useWardrobeStore;

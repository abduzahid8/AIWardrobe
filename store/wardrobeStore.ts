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
import { wardrobeApi, wearLogApi, ApiClothingItem } from '../src/lib/api';
import { fetchItemsFromServer, fetchWearLogsFromServer, processPendingActions } from './wardrobeSyncService';
import type { PendingAction } from './wardrobeSyncService';
import type { RealtimeChannel } from '@supabase/supabase-js';

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

    // Data fetching & cloud sync
    fetchItems: () => Promise<void>;
    syncToServer: () => Promise<void>;
    rehydrateFromCloud: () => Promise<void>;
    subscribeToRealtime: () => void;
    unsubscribeRealtime: () => void;

    // Filters / queries
    getItemsByCategory: (category: ClothingCategory) => ClothingItem[];
    getItemsBySeason: (season: Season) => ClothingItem[];
    getRecentWearLogs: (count?: number) => WearLog[];

    // Disliked outfits (used to avoid suggesting the same combos)
    /** Sorted comma-joined itemId keys of outfits the user has disliked. */
    dislikedOutfitKeys: string[];
    /** Record a dislike for an outfit by its item IDs. */
    dislikeOutfit: (itemIds: string[]) => void;
    /** Remove a dislike (undo). */
    undislikeOutfit: (itemIds: string[]) => void;
    /** Return human-readable summaries for the AI context prompt. */
    getDislikedSummaries: () => string[];
}

// PendingAction type is imported from wardrobeSyncService.ts

// ============================================
// HELPERS
// ============================================

const generateId = (): string => {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
};

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
            dislikedOutfitKeys: [],

            // ── Item CRUD ──────────────────────────────

            addItem: async (itemInput) => {
                const now = new Date().toISOString();
                const tempId = generateId();

                // Normalize category to match DB CHECK constraint values
                const categoryMap: Record<string, string> = {
                    top: 'Tops', tops: 'Tops',
                    bottom: 'Bottoms', bottoms: 'Bottoms',
                    shoe: 'Shoes', shoes: 'Shoes',
                    outerwear: 'Outerwear',
                    accessory: 'Accessories', accessories: 'Accessories',
                    dress: 'Dresses', dresses: 'Dresses',
                };
                const normalizedCategory = categoryMap[(itemInput.category || '').toLowerCase()] ?? 'Other';
                const optimisticItem: ClothingItem = {
                    ...itemInput,
                    id: tempId,
                    wearCount: 0,
                    lastWornAt: null,
                    isFavorite: false,
                    createdAt: now,
                    updatedAt: now,
                };

                // Optimistic add
                set((state) => ({ items: [optimisticItem, ...state.items] }));

                const { data: sessionData } = await supabase.auth.getSession();
                const userId = sessionData?.session?.user?.id;

                if (!userId) {
                    // No session — queue for later sync
                    set((state) => ({
                        pendingActions: [
                            ...state.pendingActions,
                            {
                                id: generateId(),
                                type: 'add_item',
                                payload: optimisticItem as unknown as Record<string, unknown>,
                                createdAt: now,
                            },
                        ],
                    }));
                    return;
                }

                try {
                    // Save directly to Supabase
                    const { data, error } = await supabase
                        .from('clothing_items')
                        .insert({
                            user_id: userId,
                            image_url: itemInput.imageUrl,
                            thumbnail_url: itemInput.thumbnailUrl,
                            category: normalizedCategory,
                            sub_category: itemInput.subCategory,
                            type: itemInput.subCategory,
                            primary_color: itemInput.primaryColor,
                            color_hex: itemInput.colorHex,
                            pattern: itemInput.pattern,
                            material: itemInput.material,
                            brand: itemInput.brand,
                            name: itemInput.name,
                            seasons: itemInput.seasons ?? [],
                            occasions: itemInput.occasions ?? [],
                            wear_count: 0,
                            is_favorite: false,
                            detection_confidence: (itemInput as any).detectionConfidence,
                            created_at: now,
                            updated_at: now,
                        })
                        .select()
                        .single();

                    if (error) throw error;

                    const serverItem: ClothingItem = {
                        ...optimisticItem,
                        id: data.id,
                        createdAt: data.created_at,
                        updatedAt: data.updated_at,
                    };
                    set((state) => ({
                        items: state.items.map((i) => i.id === tempId ? serverItem : i),
                        lastSyncedAt: new Date().toISOString(),
                    }));
                } catch (err) {
                    console.warn('[WardrobeStore] addItem Supabase insert failed, queuing for later sync:', err);
                    // Offline — queue for later sync
                    set((state) => ({
                        pendingActions: [
                            ...state.pendingActions,
                            {
                                id: generateId(),
                                type: 'add_item',
                                payload: optimisticItem as unknown as Record<string, unknown>,
                                createdAt: now,
                            },
                        ],
                    }));
                }
            },

            removeItem: async (id) => {
                const now = new Date().toISOString();

                // Optimistic removal
                set((state) => ({
                    items: state.items.filter((item) => item.id !== id),
                    outfits: state.outfits.map((outfit) => ({
                        ...outfit,
                        itemIds: outfit.itemIds.filter((itemId) => itemId !== id),
                    })),
                }));

                try {
                    await wardrobeApi.remove(id);
                } catch {
                    // Offline — queue for later sync
                    set((state) => ({
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
                // Optimistic toggle
                set((state) => ({
                    items: state.items.map((item) =>
                        item.id === id
                            ? { ...item, isFavorite: !item.isFavorite }
                            : item
                    ),
                }));
                // Persist to backend (fire and forget — local state already updated)
                wardrobeApi.toggleFavorite(id).catch(() => {
                    // Revert on failure
                    set((state) => ({
                        items: state.items.map((item) =>
                            item.id === id
                                ? { ...item, isFavorite: !item.isFavorite }
                                : item
                        ),
                    }));
                });
            },

            // ── Outfit Management ──────────────────────

            addOutfit: (outfitInput) => {
                set((state) => {
                    // De-duplicate: keep at most 1 item per clothing category
                    const seenCategories = new Set<string>();
                    const validItemIds = outfitInput.itemIds.filter((id) => {
                        const item = state.items.find((i) => i.id === id);
                        if (!item) return true; // keep IDs not in wardrobe (shop items etc.)
                        if (seenCategories.has(item.category)) return false;
                        seenCategories.add(item.category);
                        return true;
                    });

                    const newOutfit: Outfit = {
                        ...outfitInput,
                        itemIds: validItemIds,
                        id: generateId(),
                        saved: false,
                        wornCount: 0,
                        lastWornAt: null,
                        createdAt: new Date().toISOString(),
                    };

                    return { outfits: [newOutfit, ...state.outfits] };
                });
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

                    const newLogs = [newLog, ...state.wearLogs].slice(0, 1000); // Keep last 1000
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

            // ── Data Fetching & Cloud Sync ─────────────

            fetchItems: async () => {
                try {
                    set({ isLoading: true });
                    const apiItems = await wardrobeApi.list();
                    const items: ClothingItem[] = apiItems.map(mapApiItem);
                    set({ items, isLoading: false, lastSyncedAt: new Date().toISOString() });
                } catch {
                    // Fallback to legacy Supabase sync if backend is unreachable
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

            /**
             * Rehydrate from cloud — called after login/session restore.
             * Merges server items with any un-synced local items to prevent data loss.
             */
            rehydrateFromCloud: async () => {
                set({ isLoading: true });
                try {
                    // 1. Flush pending offline actions first
                    const { pendingActions } = get();
                    if (pendingActions.length > 0) {
                        await get().syncToServer();
                    }

                    // 2. Fetch from new backend API
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
                    // Fallback to legacy Supabase direct fetch
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

            /**
             * Subscribe to Supabase realtime changes on clothing_items table.
             * Keeps local store in sync with server-side changes.
             */
            subscribeToRealtime: () => {
                // Avoid duplicate subscriptions
                if ((useWardrobeStore as any)._realtimeChannel) return;

                const channel: RealtimeChannel = supabase
                    .channel('wardrobe-realtime')
                    .on(
                        'postgres_changes',
                        { event: '*', schema: 'public', table: 'clothing_items' },
                        (payload) => {
                            const { eventType } = payload;
                            if (eventType === 'INSERT' || eventType === 'UPDATE') {
                                const row = payload.new as Record<string, unknown>;
                                // Only process if it's for current user
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

                (useWardrobeStore as any)._realtimeChannel = channel;
            },

            /** Unsubscribe from realtime — called on logout */
            unsubscribeRealtime: () => {
                const channel = (useWardrobeStore as any)._realtimeChannel;
                if (channel) {
                    supabase.removeChannel(channel);
                    (useWardrobeStore as any)._realtimeChannel = null;
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

            // ── Disliked Outfits ───────────────────────

            dislikeOutfit: (itemIds) => {
                const key = [...itemIds].sort().join(',');
                set((state) => ({
                    dislikedOutfitKeys: state.dislikedOutfitKeys.includes(key)
                        ? state.dislikedOutfitKeys
                        : [...state.dislikedOutfitKeys, key],
                }));
            },

            undislikeOutfit: (itemIds) => {
                const key = [...itemIds].sort().join(',');
                set((state) => ({
                    dislikedOutfitKeys: state.dislikedOutfitKeys.filter((k) => k !== key),
                }));
            },

            getDislikedSummaries: () => {
                const { dislikedOutfitKeys, items } = get();
                return dislikedOutfitKeys.slice(0, 10).map((key) => {
                    const ids = key.split(',');
                    const names = ids
                        .map((id) => {
                            const item = items.find((i) => i.id === id);
                            return item ? (item.name || item.subCategory || item.category) : id;
                        })
                        .join(' + ');
                    return `Disliked combo: ${names}`;
                });
            },
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
                    // Drop pending actions whose item payload has a non-UUID id
                    if (Array.isArray(state.pendingActions)) {
                        state.pendingActions = (state.pendingActions as PendingAction[]).filter((a) => {
                            if (a.type === 'add_item') {
                                return isUUID((a.payload as { id?: unknown }).id);
                            }
                            return true;
                        });
                    }
                    // Drop local-only items with non-UUID ids (they can never be synced)
                    if (Array.isArray(state.items)) {
                        state.items = (state.items as ClothingItem[]).filter((i) => isUUID(i.id));
                    }
                }
                return state;
            },
            // Only persist essential data, not loading states
            partialize: (state) => ({
                items: state.items,
                outfits: state.outfits,
                wearLogs: state.wearLogs.slice(0, 1000), // Cap persisted logs
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

// ── API → domain mapper ────────────────────────────────────────────────────

function mapApiItem(a: ApiClothingItem): ClothingItem {
    return {
        id: a.id,
        userId: a.userId,
        imageUrl: a.imageUrl,
        thumbnailUrl: a.thumbnailUrl ?? undefined,
        category: a.category as ClothingCategory,
        subCategory: a.subCategory,
        primaryColor: a.primaryColor,
        colorHex: a.colorHex,
        pattern: a.pattern,
        material: a.material,
        brand: a.brand ?? undefined,
        name: a.name ?? undefined,
        seasons: a.seasons as Season[],
        occasions: a.occasions as Occasion[],
        wearCount: a.wearCount,
        lastWornAt: a.lastWornAt ?? null,
        isFavorite: a.isFavorite,
        detectionConfidence: a.detectionConfidence ?? undefined,
        createdAt: a.createdAt,
        updatedAt: a.updatedAt,
    };
}

export default useWardrobeStore;

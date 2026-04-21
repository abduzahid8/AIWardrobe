/**
 * Wardrobe Items Slice — CRUD operations for clothing items
 */

import type { StateCreator } from 'zustand';
import { supabase } from '../../lib/supabase';
import { wardrobeApi } from '../../src/lib/api';
import { normalizeCategory } from '../../src/utils/categoryMapper';
import type { ClothingItem, ClothingCategory } from '../../src/types/domain';
import type { WardrobeState } from '../wardrobeStore';
import { generateId } from './helpers';
import useSubscriptionStore, { TIER_FEATURES } from '../subscriptionStore';

/**
 * Thrown by addItem when the user's tier has a wardrobe cap and
 * adding one more item would exceed it. UI surfaces that call addItem
 * should catch this error by checking `err.code === 'WARDROBE_LIMIT_REACHED'`
 * and then navigate to the Paywall.
 */
export class WardrobeLimitError extends Error {
    code = 'WARDROBE_LIMIT_REACHED' as const;
    limit: number;
    constructor(limit: number) {
        super(`Wardrobe limit reached (${limit} items).`);
        this.limit = limit;
    }
}

export interface ItemsSlice {
    items: ClothingItem[];
    isLoading: boolean;

    addItem: (item: Omit<ClothingItem, 'id' | 'createdAt' | 'updatedAt' | 'wearCount' | 'lastWornAt' | 'isFavorite'>) => Promise<void>;
    removeItem: (id: string) => Promise<void>;
    updateItem: (id: string, updates: Partial<ClothingItem>) => void;
    toggleFavorite: (id: string) => void;
    getItemsByCategory: (category: ClothingCategory) => ClothingItem[];
    getItemsBySeason: (season: string) => ClothingItem[];
}

export const createItemsSlice: StateCreator<WardrobeState, [], [], ItemsSlice> = (set, get) => ({
    items: [],
    isLoading: false,

    addItem: async (itemInput) => {
        // Enforce free-tier wardrobe cap before touching state or Supabase.
        const tier = useSubscriptionStore.getState().tier;
        const cap = TIER_FEATURES[tier].wardrobeItems;
        if (cap !== -1 && get().items.length >= cap) {
            throw new WardrobeLimitError(cap);
        }

        const now = new Date().toISOString();
        const tempId = generateId();
        const normalizedCategory = normalizeCategory(itemInput.category || '');

        const optimisticItem: ClothingItem = {
            ...itemInput,
            id: tempId,
            category: normalizedCategory,
            wearCount: 0,
            lastWornAt: null,
            isFavorite: false,
            createdAt: now,
            updatedAt: now,
        };

        set((state) => ({ items: [optimisticItem, ...state.items] }));

        const { data: sessionData } = await supabase.auth.getSession();
        const userId = sessionData?.session?.user?.id;

        if (!userId) {
            set((state) => ({
                pendingActions: [
                    ...state.pendingActions,
                    { id: generateId(), type: 'add_item', payload: optimisticItem as unknown as Record<string, unknown>, createdAt: now },
                ],
            }));
            return;
        }

        try {
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
            console.warn('[WardrobeStore] addItem failed, queuing:', err);
            set((state) => ({
                pendingActions: [
                    ...state.pendingActions,
                    { id: generateId(), type: 'add_item', payload: optimisticItem as unknown as Record<string, unknown>, createdAt: now },
                ],
            }));
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
        }));

        try {
            await wardrobeApi.remove(id);
        } catch {
            set((state) => ({
                pendingActions: [
                    ...state.pendingActions,
                    { id: generateId(), type: 'remove_item', payload: { itemId: id }, createdAt: now },
                ],
            }));
        }
    },

    updateItem: (id, updates) => {
        set((state) => ({
            items: state.items.map((item) =>
                item.id === id ? { ...item, ...updates, updatedAt: new Date().toISOString() } : item
            ),
        }));
    },

    toggleFavorite: (id) => {
        set((state) => ({
            items: state.items.map((item) =>
                item.id === id ? { ...item, isFavorite: !item.isFavorite } : item
            ),
        }));
        wardrobeApi.toggleFavorite(id).catch(() => {
            set((state) => ({
                items: state.items.map((item) =>
                    item.id === id ? { ...item, isFavorite: !item.isFavorite } : item
                ),
            }));
        });
    },

    getItemsByCategory: (category) => get().items.filter((item) => item.category === category),
    getItemsBySeason: (season) => get().items.filter((item) => item.seasons.includes(season as any)),
});

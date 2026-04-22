/**
 * Wardrobe Outfits Slice — outfit management + disliked tracking
 */

import type { StateCreator } from 'zustand';
import type { Outfit, Occasion } from '../../src/types/domain';
import type { WardrobeState } from '../wardrobeStore';
import { generateId } from './helpers';

export interface OutfitsSlice {
    outfits: Outfit[];
    dislikedOutfitKeys: string[];

    addOutfit: (outfit: Omit<Outfit, 'id' | 'createdAt' | 'wornCount' | 'lastWornAt' | 'saved'>) => void;
    saveOutfit: (id: string) => void;
    rateOutfit: (id: string, rating: 1 | 2 | 3 | 4 | 5) => void;
    dislikeOutfit: (itemIds: string[]) => void;
    undislikeOutfit: (itemIds: string[]) => void;
    getDislikedSummaries: () => string[];
}

export const createOutfitsSlice: StateCreator<WardrobeState, [], [], OutfitsSlice> = (set, get) => ({
    outfits: [],
    dislikedOutfitKeys: [],

    addOutfit: (outfitInput) => {
        set((state) => {
            const seenCategories = new Set<string>();
            const validItemIds = outfitInput.itemIds.filter((id) => {
                const item = state.items.find((i) => i.id === id);
                if (!item) return true;
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
});

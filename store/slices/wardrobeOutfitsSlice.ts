/**
 * Wardrobe Outfits Slice — outfit management + disliked tracking
 */

import type { StateCreator } from 'zustand';
import type { Outfit, Occasion } from '../../src/types/domain';
import type { WardrobeState } from '../wardrobeStore';
import { generateId } from './helpers';
import { getMacroCategory } from '../../src/utils/categoryMapper';

export interface OutfitsSlice {
    outfits: Outfit[];
    dislikedOutfitKeys: string[];

    addOutfit: (outfit: Omit<Outfit, 'id' | 'createdAt' | 'wornCount' | 'lastWornAt' | 'saved'>) => string;
    saveOutfit: (id: string, collectionCategory?: string) => void;
    unsaveOutfit: (id: string) => void;
    rateOutfit: (id: string, rating: 1 | 2 | 3 | 4 | 5) => void;
    dislikeOutfit: (itemIds: string[]) => void;
    undislikeOutfit: (itemIds: string[]) => void;
    getDislikedSummaries: () => string[];
}

export const createOutfitsSlice: StateCreator<WardrobeState, [], [], OutfitsSlice> = (set, get) => ({
    outfits: [],
    dislikedOutfitKeys: [],

    addOutfit: (outfitInput) => {
        let newId = '';
        set((state) => {
            // For AI-generated outfits, allow up to 2 items per macro-category
            // (layered outfits have base top + outerwear which may both
            // share macro-category 'top'). For user-created outfits, keep the
            // stricter 1-per-category rule.
            const maxPerCategory = outfitInput.generatedBy === 'ai' ? 2 : 1;
            const macroCounts = new Map<string, number>();
            const seenIds = new Set<string>();
            const validItemIds = outfitInput.itemIds.filter((id) => {
                if (seenIds.has(id)) return false;
                seenIds.add(id);
                const item = state.items.find((i) => i.id === id);
                if (!item) {
                    // Unknown item — allow it but prevent exact duplicates
                    return true;
                }
                const macro = getMacroCategory(item.category);
                const count = macroCounts.get(macro) ?? 0;
                if (count >= maxPerCategory) return false;
                macroCounts.set(macro, count + 1);
                return true;
            });

            const id = generateId();
            newId = id;
            const newOutfit: Outfit = {
                ...outfitInput,
                itemIds: validItemIds,
                id,
                saved: false,
                wornCount: 0,
                lastWornAt: null,
                createdAt: new Date().toISOString(),
            };

            return { outfits: [newOutfit, ...state.outfits] };
        });
        return newId;
    },

    saveOutfit: (id, collectionCategory?: string) => {
        set((state) => ({
            outfits: state.outfits.map((outfit) =>
                outfit.id === id ? { ...outfit, saved: true, ...(collectionCategory ? { collectionCategory } : {}) } : outfit
            ),
        }));
    },

    unsaveOutfit: (id) => {
        set((state) => ({
            outfits: state.outfits.map((outfit) =>
                outfit.id === id ? { ...outfit, saved: false } : outfit
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

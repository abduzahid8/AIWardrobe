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
    saveOutfit: (id: string, collectionCategory?: string, savedFrom?: Outfit['savedFrom']) => void;
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

            // Dedup: the daily-deck fallback is deterministic and can serve
            // the exact same 15 combinations day after day — re-liking one
            // must NOT append a duplicate outfit. Without this, `outfits`
            // (persisted + re-scanned by Profile memos and every persist
            // write) grew by one full object per swipe, forever: the
            // slow-burn freeze behind "every page gets slower over time".
            const comboKey = [...validItemIds].sort().join('|');
            const existing = state.outfits.find(
                (o) =>
                    [...o.itemIds].sort().join('|') === comboKey &&
                    (o.occasion ?? '') === (outfitInput.occasion ?? ''),
            );
            if (existing) {
                newId = existing.id;
                // Refresh the frozen snapshot when the old save predates
                // snapshots — keeps Profile collages rendering without
                // growing the array.
                const existingAny = existing as Outfit & { snapshotItems?: unknown };
                const inputAny = outfitInput as { snapshotItems?: unknown };
                if (
                    (!Array.isArray(existingAny.snapshotItems) ||
                        existingAny.snapshotItems.length === 0) &&
                    Array.isArray(inputAny.snapshotItems) &&
                    inputAny.snapshotItems.length > 0
                ) {
                    return {
                        outfits: state.outfits.map((o) =>
                            o.id === existing.id
                                ? { ...o, snapshotItems: inputAny.snapshotItems }
                                : o,
                        ),
                    } as Partial<WardrobeState>;
                }
                return {};
            }

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

    saveOutfit: (id, collectionCategory?: string, savedFrom?: Outfit['savedFrom']) => {
        set((state) => ({
            outfits: state.outfits.map((outfit) =>
                outfit.id === id ? { ...outfit, saved: true, ...(collectionCategory ? { collectionCategory } : {}), ...(savedFrom ? { savedFrom } : {}) } : outfit
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
        set((state) => {
            if (state.dislikedOutfitKeys.includes(key)) return {};
            // Bounded: dislikes are only surfaced via getDislikedSummaries
            // (first 10) and persisted on every write — an unbounded list
            // made each persist payload (and each swipe) heavier forever.
            const next = [...state.dislikedOutfitKeys, key];
            return {
                dislikedOutfitKeys:
                    next.length > 500 ? next.slice(next.length - 500) : next,
            };
        });
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

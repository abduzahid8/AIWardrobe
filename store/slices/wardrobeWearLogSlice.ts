/**
 * Wardrobe Wear Log Slice — wear logging, streaks, utilization
 */

import type { StateCreator } from 'zustand';
import type { ClothingItem, WearLog, Occasion } from '../../src/types/domain';
import type { WardrobeState } from '../wardrobeStore';
import { generateId, getTodayDate, calculateStreak } from './helpers';

export interface WearLogSlice {
    wearLogs: WearLog[];
    streak: number;
    lastWearDate: string | null;
    dailySuggestion: any | null;

    logWear: (itemIds: string[], occasion?: Occasion | string, weather?: { temp: number; condition: string }) => void;
    getStreak: () => number;
    getClosetUtilization: (days?: number) => number;
    getUnwornItems: (days?: number) => ClothingItem[];
    getRecentWearLogs: (count?: number) => WearLog[];
    setDailySuggestion: (suggestion: any) => void;
    clearDailySuggestion: () => void;
}

export const createWearLogSlice: StateCreator<WardrobeState, [], [], WearLogSlice> = (set, get) => ({
    wearLogs: [],
    streak: 0,
    lastWearDate: null,
    dailySuggestion: null,

    logWear: (itemIds, occasion, weather) => {
        const today = getTodayDate();
        const now = new Date().toISOString();

        const newLog: WearLog = {
            id: generateId(),
            userId: '',
            itemIds,
            date: today,
            occasion,
            weatherTemp: weather?.temp,
            weatherCondition: weather?.condition,
            createdAt: now,
        };

        set((state) => {
            const updatedItems = state.items.map((item) => {
                if (itemIds.includes(item.id)) {
                    return { ...item, wearCount: item.wearCount + 1, lastWornAt: now };
                }
                return item;
            });

            const matchingOutfit = state.outfits.find(
                (o) => o.itemIds.length === itemIds.length && o.itemIds.every((id) => itemIds.includes(id))
            );

            const updatedOutfits = matchingOutfit
                ? state.outfits.map((o) =>
                    o.id === matchingOutfit.id ? { ...o, wornCount: o.wornCount + 1, lastWornAt: now } : o
                )
                : state.outfits;

            const newLogs = [newLog, ...state.wearLogs].slice(0, 1000);
            const newStreak = calculateStreak(newLogs);

            return {
                items: updatedItems,
                outfits: updatedOutfits,
                wearLogs: newLogs,
                streak: newStreak,
                lastWearDate: today,
                pendingActions: [
                    ...state.pendingActions,
                    { id: generateId(), type: 'add_wear_log', payload: newLog as unknown as Record<string, unknown>, createdAt: now },
                ],
            };
        });
    },

    getStreak: () => calculateStreak(get().wearLogs),

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

    getRecentWearLogs: (count = 10) => get().wearLogs.slice(0, count),

    setDailySuggestion: (suggestion) => set({ dailySuggestion: suggestion }),
    clearDailySuggestion: () => set({ dailySuggestion: null }),
});

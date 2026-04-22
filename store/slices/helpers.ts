/**
 * Shared helpers for wardrobe store slices
 */

import type { ClothingItem, ClothingCategory, Season, Occasion, WearLog } from '../../src/types/domain';
import type { ApiClothingItem } from '../../src/lib/api';

export const generateId = (): string => {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
};

export const getTodayDate = () => {
    return new Date().toISOString().split('T')[0];
};

export const calculateStreak = (wearLogs: WearLog[]): number => {
    if (wearLogs.length === 0) return 0;

    const dates = [...new Set(wearLogs.map(log => log.date))].sort().reverse();
    const today = getTodayDate();
    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];

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

export function mapApiItem(a: ApiClothingItem): ClothingItem {
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

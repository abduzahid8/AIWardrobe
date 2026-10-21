/**
 * Shared helpers for wardrobe store slices
 */

import { localDateKey, parseCalendarDate, shiftCalendarDay } from '../../src/utils/calendarDate';
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

export const getTodayDate = () => localDateKey(new Date());

export const calculateStreak = (wearLogs: Pick<WearLog, 'date'>[]): number => {
    const today = new Date();
    const todayKey = localDateKey(today);
    const dates = new Set(wearLogs.map(log => log.date).filter(date => date <= todayKey && parseCalendarDate(date)));
    let cursor = dates.has(todayKey) ? today : shiftCalendarDay(today, -1);
    let streak = 0;
    while (dates.has(localDateKey(cursor))) {
        streak++;
        cursor = shiftCalendarDay(cursor, -1);
    }
    return streak;
};

/** Inclusive window: today and the preceding days - 1 local calendar days. */
export function recentWornItemIds(wearLogs: WearLog[], days: number): Set<string> {
    if (!Number.isFinite(days) || days < 1) return new Set();
    const today = new Date();
    const todayKey = localDateKey(today);
    const cutoff = localDateKey(shiftCalendarDay(today, -(Math.floor(days) - 1)));
    return new Set(wearLogs.filter(log => log.date >= cutoff && log.date <= todayKey && parseCalendarDate(log.date))
        .flatMap(log => log.itemIds));
}

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

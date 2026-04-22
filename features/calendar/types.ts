/**
 * Calendar Feature — Single Source of Truth for Types & Constants
 *
 * All calendar-related types, constants, and factory functions live here.
 * Re-exports canonical domain types where possible to avoid duplication.
 */

import type { ClothingCategory as DomainClothingCategory } from '../../src/types/domain';
import { getMacroCategory } from '../../src/utils/categoryMapper';

export const CLOTHING_CATEGORIES = {
    top: ['top', 'tops', 'shirt', 't-shirt', 'tshirt', 'tee', 'sweater', 'hoodie', 'blouse', 'jacket', 'blazer', 'coat', 'outerwear', 'vest'],
    pants: ['bottom', 'bottoms', 'pant', 'pants', 'jean', 'jeans', 'short', 'shorts', 'skirt', 'trouser', 'trousers'],
    shoes: ['shoe', 'shoes', 'sneaker', 'sneakers', 'boot', 'boots', 'heel', 'heels', 'flat', 'flats', 'sandal', 'sandals', 'loafer', 'loafers'],
} as const;

export type ClothingCategory = keyof typeof CLOTHING_CATEGORIES;

/**
 * Matches an item type string against a clothing category.
 * E.g. matchesCategory('t-shirt', 'top') → true
 */
export const matchesCategory = (itemType: string, category: ClothingCategory): boolean => {
    const t = (itemType || '').toLowerCase();
    return CLOTHING_CATEGORIES[category].some(keyword => t.includes(keyword));
};

// ── Occasions ───────────────────────────────────────────────────

export const OCCASION_IDS = ['work', 'casual', 'date', 'party', 'sport', 'formal'] as const;

export type OccasionId = (typeof OCCASION_IDS)[number];

export interface Occasion {
    readonly id: OccasionId;
    readonly label: string;
    readonly icon: string;
    readonly color: string;
}

export const OCCASIONS: readonly Occasion[] = [
    { id: 'work',   label: 'Work',   icon: '💼', color: '#3B82F6' },
    { id: 'casual', label: 'Casual', icon: '☕', color: '#22C55E' },
    { id: 'date',   label: 'Date',   icon: '💕', color: '#EC4899' },
    { id: 'party',  label: 'Party',  icon: '🎉', color: '#F59E0B' },
    { id: 'sport',  label: 'Sport',  icon: '🏃', color: '#8B5CF6' },
    { id: 'formal', label: 'Formal', icon: '🎩', color: '#1A1A1A' },
] as const;

export const getOccasionColor = (id: string): string =>
    OCCASIONS.find(o => o.id === id)?.color ?? '#6B7280';

export const isValidOccasion = (id: string): id is OccasionId =>
    OCCASION_IDS.includes(id as OccasionId);

// ── Calendar Helpers ────────────────────────────────────────────

export type Month = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11;

export const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] as const;
export const WEEKDAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'] as const;

export const getDaysInMonth = (year: number, month: number): number =>
    new Date(year, month + 1, 0).getDate();

export const getFirstDayOfMonth = (year: number, month: number): number =>
    new Date(year, month, 1).getDay();

/**
 * Formats a date as YYYY-MM-DD. This is the canonical date key format
 * used for outfit log lookups.
 */
export const formatDate = (year: number, month: number, day: number): string =>
    `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

// ── Outfit Item ─────────────────────────────────────────────────

export interface OutfitItem {
    readonly id: string;
    readonly type: string;
    readonly image: string;
    readonly color?: string;
    readonly name?: string;
}

/**
 * Validates the minimum shape for an outfit item.
 * Returns null if invalid, the item if valid.
 */
export const validateOutfitItem = (raw: unknown): OutfitItem | null => {
    if (
        typeof raw === 'object' && raw !== null &&
        typeof (raw as any).id === 'string' &&
        typeof (raw as any).type === 'string' &&
        typeof (raw as any).image === 'string'
    ) {
        const r = raw as any;
        return { id: r.id, type: r.type, image: r.image, color: r.color };
    }
    return null;
};

// ── Outfit Log ──────────────────────────────────────────────────

export type OutfitRating = 1 | 2 | 3 | 4 | 5;

export interface OutfitLog {
    readonly date: string;
    readonly items: readonly OutfitItem[];
    readonly occasion: OccasionId;
    readonly note?: string;
    readonly rating?: OutfitRating;
}

/**
 * Factory function to create a validated OutfitLog.
 * Throws if items array is empty or occasion is invalid.
 */
export const createOutfitLog = (
    date: string,
    items: OutfitItem[],
    occasion: string,
    note?: string,
    rating?: number,
): OutfitLog => {
    if (items.length === 0) {
        throw new Error('OutfitLog requires at least one item');
    }
    if (items.length > 6) {
        throw new Error('OutfitLog allows at most 6 items');
    }
    if (!isValidOccasion(occasion)) {
        throw new Error(`Invalid occasion: "${occasion}". Must be one of: ${OCCASION_IDS.join(', ')}`);
    }
    const validRating = rating != null
        ? ([1, 2, 3, 4, 5].includes(rating) ? (rating as OutfitRating) : undefined)
        : undefined;

    return {
        date,
        items: Object.freeze([...items]),
        occasion,
        note,
        rating: validRating,
    };
};

// ── Wardrobe Item (superset used in wardrobe feature) ───────────

export interface WardrobeItem {
    readonly id: string;
    readonly type: string;
    readonly image: string;
    readonly imageUrl?: string;
    readonly localImage?: string;
    readonly color?: string;
    readonly name?: string;
    readonly category?: string;
}

/**
 * Converts a WardrobeItem to an OutfitItem for logging.
 */
export const wardrobeToOutfitItem = (w: WardrobeItem): OutfitItem => ({
    id: w.id,
    type: w.type || w.category || '',
    image: w.image || w.imageUrl || '',
    color: w.color,
    name: w.name,
});

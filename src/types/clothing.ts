/**
 * Clothing Types — Re-exports from domain.ts + legacy display/input types
 *
 * Canonical domain types live in ./domain.ts (single source of truth).
 * This file provides backward-compatible legacy types for UI layers
 * that haven't migrated yet.
 */

// Re-export canonical types
export type {
    ClothingItem,
    Outfit,
    WearLog,
    ClothingCategory,
    Season,
    Occasion,
} from './domain';

import type { ClothingStyle } from './api';

/**
 * Clothing item for display in UI (extends domain ClothingItem)
 */
export interface ClothingItemDisplay {
    id: string;
    imageUrl: string;
    category: string;
    primaryColor: string;
    isFavorite?: boolean;
    isSelected?: boolean;
}

/**
 * New clothing item input (before saving to DB)
 */
export interface ClothingItemInput {
    type: string;
    color: string;
    style?: string;
    description?: string;
    season?: string;
    imageUrl?: string;
}

/**
 * Batch add request
 */
export interface BatchAddRequest {
    items: ClothingItemInput[];
}

/**
 * Batch add response
 */
export interface BatchAddResponse {
    success: boolean;
    count: number;
}

/**
 * Clothing filter options
 */
export interface ClothingFilter {
    category?: string;
    season?: string;
    style?: ClothingStyle;
    color?: string;
    searchQuery?: string;
}

/**
 * Wardrobe statistics
 */
export interface WardrobeStats {
    totalItems: number;
    byCategory: Record<string, number>;
    bySeason: Record<string, number>;
    byColor: Record<string, number>;
}

/**
 * Clothing Item Types
 * Types for wardrobe and clothing management
 */

import { ClothingStyle } from './api';

/**
 * Clothing item stored in database
 */
export interface ClothingItem {
    _id: string;
    userId: string;
    type: string;
    color: string;
    style: string;
    description: string;
    season: Season;
    imageUrl: string;
    createdAt: Date;
    updatedAt?: Date;
}

/**
 * Season type for clothing items
 */
export type Season =
    | 'Spring'
    | 'Summer'
    | 'Fall'
    | 'Winter'
    | 'All Seasons';

/**
 * Clothing category type
 */
export type ClothingCategory =
    | 'Tops'
    | 'Bottoms'
    | 'Dresses'
    | 'Outerwear'
    | 'Shoes'
    | 'Accessories'
    | 'All';

/**
 * Clothing item for display in UI
 */
export interface ClothingItemDisplay extends ClothingItem {
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
    season?: Season;
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
    category?: ClothingCategory;
    season?: Season;
    style?: ClothingStyle;
    color?: string;
    searchQuery?: string;
}

/**
 * Wardrobe statistics
 */
export interface WardrobeStats {
    totalItems: number;
    byCategory: Record<ClothingCategory, number>;
    bySeason: Record<Season, number>;
    byColor: Record<string, number>;
}

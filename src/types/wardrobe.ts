/**
 * wardrobe.ts — DEPRECATED: Re-exports from domain.ts
 *
 * All canonical types now live in ./domain.ts.
 * This file exists solely for backward compatibility.
 * New code should import directly from './domain'.
 */

export type {
    ClothingCategory,
    ClothingItem,
    Outfit,
    WearLog,
    DailySuggestion,
    StyleInsight,
    UserProfile,
    Season,
    Occasion,
    SubscriptionTier,
} from './domain';

// Legacy aliases for files that used the old shape
export type ClothingStyle = 'casual' | 'formal' | 'sport' | 'semi_classic' | 'business' | 'evening';

export interface APIResponse<T> {
    success: boolean;
    data?: T;
    error?: string;
    message?: string;
}

export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    page: number;
    pageSize: number;
    hasMore: boolean;
}

export interface WeatherData {
    date: string;
    temperature: number;
    condition: 'sunny' | 'cloudy' | 'rainy' | 'snowy' | 'windy';
    humidity?: number;
}

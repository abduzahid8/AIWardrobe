/**
 * Domain Models — Canonical data types for AIWardrobe
 *
 * These are the single source of truth for all data shapes.
 * Every store, service, and component should reference these types.
 */

// ============================================
// CLOTHING ITEM
// ============================================

/** Macro categories — keeps taxonomy simple and consistent.
 *  Must stay in sync with the CHECK constraint in supabase/schema.sql */
export type ClothingCategory = 'top' | 'bottom' | 'dress' | 'shoes' | 'outerwear' | 'accessory' | 'other';

/** Seasons for clothing items */
export type Season = 'spring' | 'summer' | 'fall' | 'winter';

/** Occasions for wearing items */
export type Occasion = 'casual' | 'work' | 'formal' | 'sport' | 'date' | 'travel';

/** Core clothing item — stored in Supabase and cached locally */
export interface ClothingItem {
    id: string;
    userId: string;
    imageUrl: string;
    thumbnailUrl?: string;

    // AI-detected (user can edit)
    category: ClothingCategory;
    subCategory: string;          // "t-shirt", "jeans", "sneakers" etc.
    primaryColor: string;
    colorHex: string;
    pattern: string;              // "solid", "striped", "plaid" etc.
    material: string;             // "cotton", "denim", "leather" etc.

    // User-provided
    brand?: string;
    name?: string;
    seasons: Season[];
    occasions: Occasion[];

    // Engagement / tracking
    wearCount: number;
    lastWornAt: string | null;    // ISO date
    isFavorite: boolean;

    // Metadata
    createdAt: string;
    updatedAt: string;

    // AI confidence (0-1)
    detectionConfidence?: number;
}

// ============================================
// OUTFIT
// ============================================

export interface Outfit {
    id: string;
    userId: string;
    itemIds: string[];            // ClothingItem IDs
    occasion: Occasion | string;
    generatedBy: 'ai' | 'user';
    previewImageUrl?: string;

    // Engagement
    saved: boolean;
    wornCount: number;
    lastWornAt: string | null;
    rating?: 1 | 2 | 3 | 4 | 5;

    // AI metadata
    reasoning?: string;           // Why AI suggested this
    colorHarmony?: string;
    style?: string;

    createdAt: string;
}

// ============================================
// WEAR LOG
// ============================================

export interface WearLog {
    id: string;
    userId: string;
    outfitId?: string;
    itemIds: string[];
    date: string;                 // YYYY-MM-DD
    occasion?: Occasion | string;

    // Context
    weatherTemp?: number;
    weatherCondition?: string;

    createdAt: string;
}

// ============================================
// USER PROFILE
// ============================================

export type SubscriptionTier = 'free' | 'premium';

export interface UserProfile {
    id: string;
    email: string;
    username: string;
    gender?: 'male' | 'female' | 'other' | 'prefer_not_to_say';
    profileImage?: string;

    // Style preferences (from quiz)
    preferredStyles: string[];
    preferredColors: string[];
    bodyType?: string;

    // Subscription
    tier: SubscriptionTier;
    tierExpiresAt?: string;

    // Engagement
    onboardingComplete: boolean;
    lastActiveAt: string;
    streakDays: number;

    createdAt: string;
    updatedAt: string;
}

// ============================================
// DAILY SUGGESTION
// ============================================

export interface DailySuggestion {
    outfit: Outfit;
    reason: string;               // "Based on today's weather (22°C) and your Tuesday style"
    weatherContext?: {
        temp: number;
        condition: string;
        city?: string;
    };
    generatedAt: string;
}

// ============================================
// STYLE INSIGHT
// ============================================

export interface StyleInsight {
    type: 'color_pattern' | 'utilization' | 'unworn_nudge' | 'streak' | 'variety';
    title: string;
    description: string;
    data?: Record<string, unknown>;
    generatedAt: string;
}

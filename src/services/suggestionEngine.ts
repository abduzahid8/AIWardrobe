import {validatedQuartets} from '../domain/validatedQuartets';
import {stylingContext} from '../domain/stylingContext';
import {fourSlotOutfitReason} from '../domain/outfitStructure';
import {passesStyleJudge,assessOutfitQuality} from '../domain/occasionRules';
import {assembleDiverseDeck,productIdentityKeys,diversitySlotOf} from '../domain/outfitDiversity';
import {stylingCopy} from './stylingCopy';
/**
 * Suggestion Engine — Core outfit recommendation logic
 *
 * Pure rule-based + weighted scoring. No LLM calls.
 * This is the brain of the daily suggestion system.
 *
 * Architecture:
 *   1. TIER    — Assign formality tier (1–5) to every item
 *   2. FILTER  — Remove items incompatible with context (weather, occasion tier)
 *   3. COMBINE — Generate candidate outfits (TOP + LAYER + BOTTOM + SHOES through the shared domain kernel)
 *   4. SCORE   — Rank by formality(30%) + novelty(25%) + color_harmony(25%) + weather(20%)
 *   5. RETURN  — Top N candidates with reasoning
 *
 * Formality Tiers:
 *   1 = Formal        (suits, dress shirts, Oxford shoes, ties)
 *   2 = Business Casual (blazers, chinos, loafers, turtlenecks)
 *   3 = Smart Casual  (clean jeans, polos, Chelsea boots, minimal sneakers)
 *   4 = Casual        (t-shirts, hoodies, joggers, casual sneakers)
 *   5 = Sport/Active  (athletic wear, trainers, technical fabrics)
 */

import type {
    ClothingItem,
    ClothingCategory,
    Outfit,
    Occasion,
    Season,
    WearLog,
} from '../types/domain';

// ============================================
// FORMALITY TIER TYPES
// ============================================

/** 1=Formal, 2=Business Casual, 3=Smart Casual, 4=Casual, 5=Sport/Active */
export type FormalityTier = 1 | 2 | 3 | 4 | 5;

export const FORMALITY_TIER_LABELS: Record<FormalityTier, string> = {
    1: 'Formal',
    2: 'Business Casual',
    3: 'Smart Casual',
    4: 'Casual',
    5: 'Sport / Active',
};

/** Dress code label shown in UI carousel badges */
export const DRESS_CODE_BADGE: Record<FormalityTier, string> = {
    1: 'Black Tie / Formal',
    2: 'Business Casual',
    3: 'Smart Casual',
    4: 'Casual',
    5: 'Activewear',
};

/** Occasion label shown in UI carousel cards */
export const OCCASION_LABEL: Record<string, string> = {
    work: 'Team Collaboration',
    date: 'Date Night',
    casual: 'Weekend',
    formal: 'Formal Event',
    sport: 'Active Day',
    travel: 'Travel Ready',
    'night-out': 'Night Out',
    wedding: 'Wedding Guest',
    interview: 'Interview',
};

// ============================================
// TYPES
// ============================================

/** A clothing item that may be a shopping suggestion (user doesn't own it). */
export interface ShoppingSuggestion {
    category: ClothingCategory;
    subCategory: string;
    primaryColor: string;
    reason: string;
    /** Always true — used by UI to show the black dot badge */
    isSuggestion: true;
}

export interface WeatherContext {
    temp: number;          // Celsius
    condition: string;     // 'clear' | 'cloudy' | 'rain' | 'snow' | 'wind'
    humidity?: number;
}

export interface UserPreferences {
    preferredColors: string[];
    avoidColors: string[];
    preferredStyles: string[];
    adventurousness: number; // 0-1: 0 = safe picks, 1 = push boundaries
}

export interface SuggestionRequest {
    items: ClothingItem[];
    wearLogs: WearLog[];
    occasion: Occasion;
    weather?: WeatherContext;
    preferences?: UserPreferences;
    excludeOutfitIds?: string[];
    /** When set, only generate outfits containing this item ID. */
    anchorItemId?: string;
}

export interface ScoredOutfit {
    outfit: Omit<Outfit, 'id' | 'userId' | 'createdAt' | 'saved' | 'wornCount' | 'lastWornAt'>;
    score: number;
    reasoning: string;
    /** Average formality tier of outfit items */
    formalityTier: FormalityTier;
    /** Dress code label for carousel badge */
    dressCode: string;
    /** Occasion label for carousel card */
    occasionLabel: string;
    /** Items that are shopping suggestions (not owned) */
    shoppingSuggestions: ShoppingSuggestion[];
    breakdown: {
        preferenceScore: number;
        weatherScore: number;
        noveltyScore: number;
        harmonyScore: number;
        formalityScore: number;
    };
}

/** Result of generateDailyOutfits — 4 occasion-targeted outfits */
export interface DailyOutfits {
    work: ScoredOutfit | null;
    smartCasual: ScoredOutfit | null;
    weekendCasual: ScoredOutfit | null;
    wildcard: ScoredOutfit | null;
}

// ============================================
// CONSTANTS
// ============================================

/** Scoring weights per spec: formality 30%, novelty 25%, harmony 25%, weather 20%. */
const WEIGHTS = {
    formality: 0.30,
    novelty: 0.25,
    harmony: 0.25,
    weather: 0.20,
    preference: 0.0,  // Folded into formality tier selection
};

/** Materials suitable for temperature ranges — used for SCORING only, not hard exclusion */
const WARM_MATERIALS = ['wool', 'fleece', 'cashmere', 'down', 'sherpa', 'corduroy', 'tweed'];
const COOL_MATERIALS = ['linen', 'cotton', 'silk', 'rayon', 'chiffon', 'mesh', 'seersucker'];
const RAIN_MATERIALS = ['nylon', 'polyester', 'gore-tex', 'leather', 'waxed'];

/** Occasion → allowed formality tier range [min, max, preferred] */
const OCCASION_TIER_MAP: Record<string, [number, number, FormalityTier]> = {
    formal:    [1, 1, 1],
    work:      [1, 3, 2],
    interview: [1, 2, 1],
    wedding:   [1, 2, 1],
    date:      [2, 3, 3],
    'night-out': [2, 3, 2],
    casual:    [3, 4, 3],
    travel:    [3, 4, 3],
    beach:     [4, 4, 4],
    sport:     [5, 5, 5],
    workout:   [5, 5, 5],
    'alpine-skiing': [5, 5, 5],
};

/** Color harmony rules (complementary pairs)
 *  Expanded with the "Golden 8" formulas from expert style guides:
 *  white, blue, light blue, brown, green, cream/milk, burgundy, grey.
 *  Key pairings: blue+brown, blue+green, green+brown always work.
 *  Black is NOT universal — it only pairs with white for classic looks.
 *  Navy, charcoal, dark brown are far more versatile replacements for black.
 */
const COLOR_HARMONIES: Record<string, string[]> = {
    navy: ['white', 'cream', 'beige', 'khaki', 'light blue', 'camel', 'brown', 'burgundy', 'tan', 'charcoal', 'grey'],
    black: ['white', 'gray', 'grey', 'cream'],
    white: ['navy', 'blue', 'light blue', 'gray', 'grey', 'beige', 'olive', 'brown', 'burgundy', 'camel', 'charcoal'],
    gray: ['white', 'navy', 'blue', 'light blue', 'pink', 'burgundy', 'charcoal', 'beige'],
    grey: ['white', 'navy', 'blue', 'light blue', 'pink', 'burgundy', 'charcoal', 'beige'],
    charcoal: ['white', 'light blue', 'cream', 'beige', 'navy', 'grey', 'burgundy', 'tan'],
    blue: ['white', 'cream', 'beige', 'brown', 'khaki', 'light blue', 'navy', 'green', 'grey', 'tan'],
    'light blue': ['navy', 'white', 'cream', 'brown', 'charcoal', 'grey', 'beige', 'tan', 'camel'],
    brown: ['white', 'cream', 'blue', 'light blue', 'green', 'beige', 'navy', 'olive', 'tan', 'camel', 'burgundy'],
    beige: ['navy', 'white', 'brown', 'olive', 'blue', 'light blue', 'black', 'camel', 'tan', 'charcoal', 'burgundy'],
    cream: ['navy', 'brown', 'beige', 'blue', 'light blue', 'camel', 'charcoal', 'burgundy', 'tan', 'green', 'olive'],
    olive: ['white', 'cream', 'beige', 'brown', 'khaki', 'navy', 'blue', 'tan', 'camel'],
    camel: ['navy', 'cream', 'white', 'brown', 'beige', 'charcoal', 'burgundy', 'olive', 'tan'],
    tan: ['navy', 'white', 'cream', 'brown', 'beige', 'olive', 'blue', 'light blue', 'charcoal', 'camel'],
    khaki: ['navy', 'white', 'cream', 'brown', 'olive', 'blue', 'beige'],
    burgundy: ['grey', 'gray', 'cream', 'white', 'navy', 'camel', 'charcoal', 'beige', 'brown'],
    red: ['white', 'navy', 'grey', 'gray', 'denim blue', 'charcoal'],
    green: ['white', 'cream', 'brown', 'beige', 'navy', 'blue', 'tan', 'olive'],
    'forest green': ['cream', 'white', 'navy', 'brown', 'beige', 'tan', 'camel'],
    pink: ['grey', 'gray', 'navy', 'white', 'cream', 'charcoal'],
    sand: ['navy', 'white', 'brown', 'beige', 'cream', 'charcoal', 'olive'],
    stone: ['navy', 'white', 'brown', 'beige', 'cream', 'charcoal', 'black'],
    chocolate: ['cream', 'white', 'navy', 'beige', 'light blue', 'camel', 'tan'],
    'midnight blue': ['cream', 'white', 'beige', 'camel', 'charcoal', 'silver'],
    ivory: ['navy', 'brown', 'beige', 'charcoal', 'camel', 'burgundy', 'forest green'],
};

// ============================================
// FORMALITY TIER ASSIGNMENT
// ============================================

/**
 * Determines the formality tier (1–5) of a clothing item based on its
 * subCategory, category, and material.
 *
 * Tier 1 = Formal | Tier 2 = Business Casual | Tier 3 = Smart Casual
 * Tier 4 = Casual | Tier 5 = Sport/Active
 *
 * Default is tier 3 (Smart Casual) when tier cannot be determined.
 */
export function getFormalityTier(item: ClothingItem): FormalityTier {
    const type = (item.subCategory || item.category || '').toLowerCase();
    const mat  = (item.material || '').toLowerCase();
    const cat  = (item.category || '').toLowerCase();

    // Tier 1 — Formal
    const tier1 = ['suit', 'tuxedo', 'oxford shoe', 'dress shoe', 'tie', 'bow tie',
        'dress shirt', 'waistcoat', 'vest', 'cummerbund', 'pocket square', 'blazer suit'];
    if (tier1.some((t) => type.includes(t))) return 1;
    if (mat.includes('satin') && cat === 'top') return 1;

    // Tier 2 — Business Casual
    const tier2 = ['blazer', 'chinos', 'loafer', 'monk strap', 'turtleneck',
        'dress pant', 'trouser', 'button-down', 'button down', 'slacks',
        'polo shirt', 'smart shoe', 'brogues', 'derby'];
    if (tier2.some((t) => type.includes(t))) return 2;
    if (mat.includes('tweed') || mat.includes('wool') && cat === 'top') return 2;

    // Tier 3 — Smart Casual
    const tier3 = ['polo', 'chino', 'chelsea boot', 'clean jeans', 'dark jeans',
        'cardigan', 'crewneck', 'merino', 'boat shoe', 'minimalist sneaker',
        'slim jeans', 'tapered trouser', 'henley', 'quarter-zip'];
    if (tier3.some((t) => type.includes(t))) return 3;
    if (type.includes('jean') && !type.includes('distressed')) return 3;
    if (type.includes('sneaker') && (mat.includes('leather') || mat.includes('suede'))) return 3;

    // Tier 5 — Sport/Active
    const tier5 = ['athletic', 'running', 'gym', 'trainer', 'tracksuit', 'jogger',
        'compression', 'legging', 'cycling', 'technical', 'windbreaker sport',
        'board short', 'swim', 'activewear'];
    if (tier5.some((t) => type.includes(t))) return 5;
    if (mat.includes('lycra') || mat.includes('spandex') || mat.includes('dri-fit')) return 5;

    // Tier 4 — Casual (widest net — catches remaining items)
    const tier4 = ['t-shirt', 'tee', 'hoodie', 'sweatshirt', 'sweater', 'jeans',
        'denim', 'shorts', 'sneaker', 'sandal', 'flip-flop', 'tank', 'cap',
        'casual jacket', 'bomber', 'puffer', 'joggers', 'sweatpant'];
    if (tier4.some((t) => type.includes(t))) return 4;
    if (mat.includes('fleece') || mat.includes('sweat') || mat.includes('terry')) return 4;

    return 3; // Default: Smart Casual
}

/**
 * Check whether two formality tiers can coexist in the same outfit.
 * Tiers 2+3 mixing is intentional (business casual). No other cross-tier mixing.
 */
function tiersCompatible(tierA: FormalityTier, tierB: FormalityTier): boolean {
    const diff = Math.abs(tierA - tierB);
    if (diff === 0) return true;
    // Tier 2 + Tier 3 = intentional business casual mixing
    if ((tierA === 2 && tierB === 3) || (tierA === 3 && tierB === 2)) return true;
    // 1 tier apart is acceptable
    if (diff === 1) return true;
    // 2+ tiers apart = clash (e.g., formal + casual = invalid)
    return false;
}

function tempToSeason(temp: number): Season[] {
    if (temp < 5) return ['winter'];
    if (temp < 15) return ['fall', 'winter', 'spring'];
    if (temp < 25) return ['spring', 'fall'];
    return ['summer', 'spring'];
}

// ============================================
// STEP 1: FILTER (soft — no hard exclusions)
// ============================================

/**
 * Filter by weather using SOFT scoring hints only.
 * Per spec: no absolute exclusions — just score items lower when weather-mismatched.
 * This function only removes items with 0% weather fit (extreme outliers).
 */
function filterByWeather(items: ClothingItem[], weather?: WeatherContext): ClothingItem[] {
    if (!weather) return items;

    // Keep all items — extreme mismatches are handled by scoreWeather() returning ~0
    return items;
}

/**
 * Filter items by occasion-compatible formality tier.
 * Items outside the allowed tier range for the occasion are excluded.
 * Untagged items (no subCategory) default to tier 3 and are always included.
 */
function filterByOccasion(items: ClothingItem[], occasion: Occasion): ClothingItem[] {
    const tierRange = OCCASION_TIER_MAP[occasion as string] ?? [1, 5, 3];
    const [minTier, maxTier] = tierRange;

    return items.filter((item) => {
        const tier = getFormalityTier(item);
        return tier >= minTier && tier <= maxTier;
    });
}

function filterBySeason(items: ClothingItem[], weather?: WeatherContext): ClothingItem[] {
    if(!weather)return items;
    const seasons = tempToSeason(weather.temp);

    return items.filter((item) => {
        if (item.seasons && item.seasons.length > 0) {
            return item.seasons.some((s) => seasons.includes(s));
        }
        // Untagged items are always included
        return true;
    });
}

// ============================================
// STEP 2: COMBINE — Generate candidate outfits
// ============================================

/**
 * Build shopping suggestions for outfit slots that have no owned items.
 * Returns placeholder ShoppingSuggestion objects so the outfit card can display
 * what the user should buy to complete the look.
 */
function buildShoppingSuggestions(
    occasion: Occasion,
    missingCategories: ClothingCategory[]
): ShoppingSuggestion[] {
    const tierRange = OCCASION_TIER_MAP[occasion as string] ?? [3, 4, 3];
    const preferredTier = tierRange[2];
    const tierLabel = FORMALITY_TIER_LABELS[preferredTier];

    const defaults: Record<ClothingCategory, { subCategory: string; primaryColor: string }> = {
        top:       { subCategory: preferredTier <= 2 ? 'dress shirt' : preferredTier === 3 ? 'polo shirt' : 't-shirt', primaryColor: 'white' },
        bottom:    { subCategory: preferredTier <= 2 ? 'chinos' : preferredTier === 3 ? 'slim jeans' : 'jeans', primaryColor: 'navy' },
        dress:     { subCategory: preferredTier <= 2 ? 'cocktail dress' : 'casual dress', primaryColor: 'black' },
        shoes:     { subCategory: preferredTier <= 2 ? 'loafers' : preferredTier === 3 ? 'chelsea boots' : 'sneakers', primaryColor: 'black' },
        outerwear: { subCategory: preferredTier <= 2 ? 'blazer' : 'casual jacket', primaryColor: 'navy' },
        accessory: { subCategory: 'watch', primaryColor: 'silver' },
        other:     { subCategory: 'item', primaryColor: 'black' },
    };

    return missingCategories.map((cat) => ({
        category: cat,
        subCategory: defaults[cat]?.subCategory ?? cat,
        primaryColor: defaults[cat]?.primaryColor ?? 'black',
        reason: `Complete this ${tierLabel} look`,
        isSuggestion: true as const,
    }));
}

function groupByCategory(items: ClothingItem[]): Record<ClothingCategory, ClothingItem[]> {
    const groups: Record<string, ClothingItem[]> = {
        top: [],
        bottom: [],
        dress: [],
        shoes: [],
        outerwear: [],
        accessory: [],
        other: [],
    };

    items.forEach((item) => {
        if (groups[item.category]) {
            groups[item.category].push(item);
        }
    });

    return groups as Record<ClothingCategory, ClothingItem[]>;
}

/**
 * Generate outfit candidate item-ID arrays.
 * Only combines items whose formality tiers are compatible.
 * Cold weather adds outerwear if available.
 * anchorItemId forces every candidate to include that specific item.
 */
function evidence(item:ClothingItem){return {...item,type:item.subCategory,macroCategory:item.category,color:item.primaryColor};}
function generateCandidates(
    groups: Record<ClothingCategory, ClothingItem[]>,weather?: WeatherContext,maxCandidates=60,anchorItemId?:string,occasion?:Occasion,preferences?:UserPreferences,
):string[][] {
    const context=stylingContext({style:'casual',occasion,weather,sourceMode:'wardrobe',anchorId:anchorItemId,preferences:preferences?{favoriteColors:preferences.preferredColors,avoidColors:preferences.avoidColors}:undefined});
    const pool=Object.values(groups).flat().map(evidence);
    return validatedQuartets(pool,context,parts=>!fourSlotOutfitReason(parts)&&passesStyleJudge(parts,'casual',occasion,weather,anchorItemId,preferences),
        {maximum:maxCandidates,searchLimit:3000,requiredIds:anchorItemId?[anchorItemId]:[]}).map(look=>look.items.map(item=>item.id));
}

// ============================================
// STEP 3: SCORE
// ============================================

/**
 * Score outfit against user color preferences.
 * Preferred colors boost score; avoided colors penalize.
 */
function scorePreference(
    itemIds: string[],
    items: ClothingItem[],
    preferences?: UserPreferences
): number {
    if (!preferences) return 0.5;

    const outfitItems = itemIds.map((id) => items.find((i) => i.id === id)).filter(Boolean) as ClothingItem[];
    let score = 0.5;

    outfitItems.forEach((item) => {
        const color = (item.primaryColor || '').toLowerCase();
        if (preferences.preferredColors.some((c) => color.includes(c.toLowerCase()))) score += 0.15;
        if (preferences.avoidColors.some((c) => color.includes(c.toLowerCase()))) score -= 0.2;
    });

    return Math.max(0, Math.min(1, score));
}

/**
 * Score outfit by weather fit.
 * Uses fabric weight vs temperature range — no absolute exclusions.
 * A heavy wool coat in 30°C scores low but is NOT removed from the pool.
 */
function scoreWeather(
    itemIds: string[],
    items: ClothingItem[],
    weather?: WeatherContext
): number {
    if (!weather) return 0.7;

    const outfitItems = itemIds.map((id) => items.find((i) => i.id === id)).filter(Boolean) as ClothingItem[];
    const temp = weather.temp;
    const hasOuterwear = outfitItems.some((i) => i.category === 'outerwear');
    let score = 0.5;

    // Outerwear presence vs temperature
    if (temp < 8)  { if (hasOuterwear) score += 0.25; else score -= 0.15; }
    if (temp > 25) { if (hasOuterwear) score -= 0.25; }
    if (temp >= 8 && temp <= 20 && hasOuterwear) score += 0.10;

    // Material fabric-weight scoring
    outfitItems.forEach((item) => {
        const material = (item.material || '').toLowerCase();
        if (temp < 12 && WARM_MATERIALS.some((m) => material.includes(m)))  score += 0.10;
        if (temp < 12 && COOL_MATERIALS.some((m) => material.includes(m)))  score -= 0.08;
        if (temp > 22 && COOL_MATERIALS.some((m) => material.includes(m)))  score += 0.10;
        if (temp > 22 && WARM_MATERIALS.some((m) => material.includes(m)))  score -= 0.10;
        if ((weather.condition === 'rain' || weather.condition === 'snow') &&
            RAIN_MATERIALS.some((m) => material.includes(m))) score += 0.08;
    });

    return Math.max(0, Math.min(1, score));
}

function scoreNovelty(
    itemIds: string[],
    items: ClothingItem[],
    wearLogs: WearLog[]
): number {
    const outfitItems = itemIds.map((id) => items.find((i) => i.id === id)).filter(Boolean) as ClothingItem[];
    if (outfitItems.length === 0) return 0;

    // Calculate novelty per item: items worn less recently get higher novelty
    const thirtyDaysAgo = new Date(Date.now() - 30 * 86400000).toISOString().split('T')[0];
    const recentLogs = wearLogs.filter((log) => log.date >= thirtyDaysAgo);

    // Count recent wears per item
    const recentWearCounts: Record<string, number> = {};
    recentLogs.forEach((log) => {
        log.itemIds.forEach((id) => {
            recentWearCounts[id] = (recentWearCounts[id] || 0) + 1;
        });
    });

    let totalNovelty = 0;
    outfitItems.forEach((item) => {
        const recentWears = recentWearCounts[item.id] || 0;
        // Novelty = 1 / (recentWears + 1), decays with wear count
        totalNovelty += 1 / (recentWears + 1);
    });

    const avgNovelty = totalNovelty / outfitItems.length;
    return Math.min(1, avgNovelty); // Already 0-1 range
}

function scoreColorHarmony(
    itemIds: string[],
    items: ClothingItem[]
): number {
    const outfitItems = itemIds.map((id) => items.find((i) => i.id === id)).filter(Boolean) as ClothingItem[];
    if (outfitItems.length < 2) return 0.5;

    const outfitColors = outfitItems
        .map((item) => (item.primaryColor || '').toLowerCase())
        .filter(Boolean);

    if (outfitColors.length < 2) return 0.5;

    let harmonyHits = 0;
    let comparisons = 0;

    for (let i = 0; i < outfitColors.length; i++) {
        for (let j = i + 1; j < outfitColors.length; j++) {
            comparisons++;
            const colorA = outfitColors[i];
            const colorB = outfitColors[j];

            // Same color family = safe
            if (colorA === colorB) {
                harmonyHits += 0.7;
                continue;
            }

            // Check harmony table
            const harmoniesA = COLOR_HARMONIES[colorA] || [];
            const harmoniesB = COLOR_HARMONIES[colorB] || [];

            if (harmoniesA.some((h) => colorB.includes(h)) || harmoniesB.some((h) => colorA.includes(h))) {
                harmonyHits += 1;
            }

            // Neutrals always harmonize
            const neutrals = ['black', 'white', 'gray', 'grey', 'beige', 'cream', 'navy'];
            if (neutrals.some((n) => colorA.includes(n)) || neutrals.some((n) => colorB.includes(n))) {
                harmonyHits += 0.5;
            }
        }
    }

    return comparisons > 0 ? Math.min(1, harmonyHits / comparisons) : 0.5;
}

/** @deprecated Use getFormalityTier() instead. Kept for backward compatibility. */
function getFormalityLevel(item: ClothingItem): number {
    return getFormalityTier(item);
}

/**
 * Score outfit formality coherence (30% of total score).
 *
 * Two factors:
 *   1. Internal coherence — items should not span more than 1 tier apart
 *      (exception: tiers 2+3 intentional business casual mixing = no penalty)
 *   2. Occasion alignment — outfit average tier should match occasion target tier
 */
function scoreFormality(
    itemIds: string[],
    items: ClothingItem[],
    occasion?: Occasion
): number {
    const outfitItems = itemIds
        .map((id) => items.find((i) => i.id === id))
        .filter(Boolean) as ClothingItem[];
    if (outfitItems.length === 0) return 0.5;

    const tiers = outfitItems.map(getFormalityTier);
    const minTier = Math.min(...tiers) as FormalityTier;
    const maxTier = Math.max(...tiers) as FormalityTier;
    const avgTier = tiers.reduce((a, b) => a + b, 0) / tiers.length;

    let score = 1.0;

    // Internal coherence — penalize tier clashes
    const spread = maxTier - minTier;
    if (spread > 2) score -= 0.7;      // Hard clash (e.g., formal suit + hoodie)
    else if (spread === 2) {
        // Allow tier 2+3 = 0 penalty; tier 1+3, 3+5 etc = moderate penalty
        const isBizCasualMix = (minTier === 2 && maxTier === 3) || (minTier === 3 && maxTier === 4);
        if (!isBizCasualMix) score -= 0.25;
    }

    // Occasion alignment
    if (occasion) {
        const tierRange = OCCASION_TIER_MAP[occasion as string] ?? [1, 5, 3];
        const targetTier = tierRange[2];
        const diff = Math.abs(avgTier - targetTier);
        score -= diff * 0.18;
    }

    return Math.max(0, Math.min(1, score));
}

// ============================================
// MAIN: generateSuggestions
// ============================================

/**
 * Generate scored outfit suggestions from the user's wardrobe.
 * Returns no outfit when four suitable distinct garments cannot be formed.
 * Shared domain rules retain activity, layering and context at every fallback.
 */
export function generateSuggestions(request: SuggestionRequest): ScoredOutfit[] {
    const {items,wearLogs,occasion,weather,preferences,anchorItemId}=request;
    if(!items.length)return [];
    const filtered=filterBySeason(filterByOccasion(filterByWeather(items,weather),occasion),weather);
    let candidates=generateCandidates(groupByCategory(filtered),weather,60,anchorItemId,occasion,preferences);
    // Old tier estimates are soft. The shared hard judge still checks the full
    // context when broader wardrobe evidence is needed; constraints never relax.
    if(!candidates.length)candidates=generateCandidates(groupByCategory(items),weather,60,anchorItemId,occasion,preferences);
    return scoreCandidates(candidates,items,wearLogs,weather,preferences,occasion,anchorItemId);
}

function hasNoDuplicateCategories(itemIds:string[],items:ClothingItem[]):boolean {
    const parts=itemIds.map(id=>items.find(item=>item.id===id)).filter(Boolean) as ClothingItem[];
    return parts.length===itemIds.length&&!fourSlotOutfitReason(parts.map(evidence));
}

function scoreCandidates(
    candidates: string[][],
    items: ClothingItem[],
    wearLogs: WearLog[],
    weather?: WeatherContext,
    preferences?: UserPreferences,
    occasion?: Occasion,
    anchorItemId?:string
): ScoredOutfit[] {
    const validCandidates = candidates.filter((ids) => hasNoDuplicateCategories(ids, items));
    const scored: ScoredOutfit[] = validCandidates.map((itemIds) => {
        const formalityScore = scoreFormality(itemIds, items, occasion);
        const noveltyScore   = scoreNovelty(itemIds, items, wearLogs);
        const harmonyScore   = scoreColorHarmony(itemIds, items);
        const weatherScore   = scoreWeather(itemIds, items, weather);
        const preferenceScore = scorePreference(itemIds, items, preferences);

        // Compute outfit items once for layering bonus + tier + missing slots
        const outfitItems = itemIds.map((id) => items.find((i) => i.id === id)).filter(Boolean) as ClothingItem[];

        const legacyScore =
            formalityScore * WEIGHTS.formality +
            noveltyScore   * WEIGHTS.novelty   +
            harmonyScore   * WEIGHTS.harmony   +
            weatherScore   * WEIGHTS.weather;

        const totalScore=assessOutfitQuality(outfitItems.map(evidence),'casual',occasion,weather,anchorItemId,preferences).score*.7+legacyScore*.3;
        const breakdown = { preferenceScore, weatherScore, noveltyScore, harmonyScore, formalityScore };
        const reasoning = stylingCopy(outfitItems.map(evidence),stylingContext({style:'casual',occasion,weather})).join(' ');
        const tiers = outfitItems.map(getFormalityTier);
        const avgTier = tiers.length > 0
            ? Math.round(tiers.reduce((a, b) => a + b, 0) / tiers.length)
            : 3;
        const formalityTier = (occasion === 'sport' ? 5 : Math.max(1, Math.min(5, avgTier))) as FormalityTier;

        const ownedCategories = new Set(outfitItems.map((i) => i.category));
        // Exactly one TOP, LAYER, BOTTOM and SHOES.
        const requiredCats: ClothingCategory[] = ['top', 'bottom', 'shoes', 'outerwear'];
        const missing = requiredCats.filter((c) => !ownedCategories.has(c));
        const shoppingSuggestions = missing.length > 0
            ? buildShoppingSuggestions(occasion ?? 'casual', missing)
            : [];

        return {
            outfit: {
                itemIds,
                occasion: occasion ?? 'casual',
                generatedBy: 'ai' as const,
                reasoning,
                style: FORMALITY_TIER_LABELS[formalityTier].toLowerCase(),
            },
            score: totalScore,
            reasoning,
            formalityTier,
            dressCode: DRESS_CODE_BADGE[formalityTier],
            occasionLabel: OCCASION_LABEL[occasion as string ?? ''] ?? (occasion as string ?? 'Outfit'),
            shoppingSuggestions,
            breakdown,
        };
    });

    scored.sort((a, b) => b.score - a.score);
    const anchor=items.find(item=>item.id===anchorItemId);
    return assembleDiverseDeck({candidates:scored,itemsOf:look=>look.outfit.itemIds.map(id=>items.find(item=>item.id===id)!).map(evidence),qualityOf:look=>look.score,target:4,usedKeys:new Set(),poolItems:items.map(evidence),anchorKeys:anchor?new Set(productIdentityKeys(anchor)):undefined,anchorSlot:anchor?diversitySlotOf(evidence(anchor)):undefined}).deck;
}

// ============================================
// GENERATE DAILY OUTFITS — 4 occasion variants
// ============================================

/**
 * Generate 4 daily outfit variants covering work, smart casual, weekend casual,
 * and a weather-appropriate wildcard. Used by the Home screen carousel.
 *
 * Each variant is scored independently with its occasion's tier targets.
 * Returns nulls when a variant cannot be generated (closet too small for that tier).
 */
export function generateDailyOutfits(
    items: ClothingItem[],
    wearLogs: WearLog[],
    weather?: WeatherContext,
    preferences?: UserPreferences
): DailyOutfits {
    const base: SuggestionRequest = { items, wearLogs, weather, preferences, occasion: 'casual' };

    const work        = generateSuggestions({ ...base, occasion: 'work' })[0]        ?? null;
    const smartCasual = generateSuggestions({ ...base, occasion: 'date' })[0]        ?? null;
    const weekendCasual = generateSuggestions({ ...base, occasion: 'casual' })[0]    ?? null;

    // Wildcard: pick occasion based on weather
    let wildcardOccasion: Occasion = 'casual';
    if (weather) {
        if (weather.temp > 25) wildcardOccasion = 'casual';
        else if (weather.temp < 8) wildcardOccasion = 'work';
        else if (weather.condition === 'rain') wildcardOccasion = 'travel';
        else wildcardOccasion = 'date';
    }
    const wildcard = generateSuggestions({ ...base, occasion: wildcardOccasion })[1] ?? null;

    return { work, smartCasual, weekendCasual, wildcard };
}

// ============================================
// ANCHOR-PIECE MODE — Build outfits around one item
// ============================================

/**
 * Generate outfit suggestions anchored to a specific item.
 * Every returned outfit MUST contain the anchor item.
 * Used by the "Build an outfit with this" CTA on unworn items.
 */
export function generateOutfitsForItem(
    anchorItemId: string,
    items: ClothingItem[],
    wearLogs: WearLog[],
    weather?: WeatherContext
): ScoredOutfit[] {
    const anchor = items.find((i) => i.id === anchorItemId);
    if (!anchor) return [];

    // Determine suitable occasion from anchor's formality tier
    const tier = getFormalityTier(anchor);
    let occasion: Occasion = 'casual';
    if (tier <= 1) occasion = 'formal';
    else if (tier === 2) occasion = 'work';
    else if (tier === 3) occasion = 'date';
    else if (tier >= 5) occasion = 'sport';

    return generateSuggestions({
        items,
        wearLogs,
        occasion,
        weather,
        anchorItemId,
    });
}

// ============================================
// QUICK SUGGEST — Single best outfit for notifications
// ============================================

/**
 * Return the single best outfit suggestion for quick use
 * (e.g. push notification preview at 8pm).
 */
export function quickSuggest(
    items: ClothingItem[],
    wearLogs: WearLog[],
    weather?: WeatherContext
): ScoredOutfit | null {
    const suggestions = generateSuggestions({ items, wearLogs, occasion: 'casual', weather });
    return suggestions[0] ?? null;
}

// ============================================
// VARIETY MODE — For Inspo tab masonry grid
// ============================================

/**
 * Generate up to 20 diverse outfit combinations for the Inspo tab.
 * Excludes any combination worn in the last 30 days.
 * Works entirely offline — no external calls.
 */
export function generateVarietyOutfits(
    items: ClothingItem[],
    wearLogs: WearLog[]
): ScoredOutfit[] {
    const occasions: Occasion[] = ['work', 'date', 'casual', 'formal', 'sport'];
    const allOutfits: ScoredOutfit[] = [];

    const thirtyDaysAgo = new Date(Date.now() - 30 * 86400000).toISOString().split('T')[0];
    const recentLogs = wearLogs.filter((l) => l.date >= thirtyDaysAgo);

    // Build set of recently-worn item combinations to exclude
    const recentCombos = new Set(
        recentLogs.map((l) => [...l.itemIds].sort().join(','))
    );

    for (const occasion of occasions) {
        const results = generateSuggestions({ items, wearLogs, occasion });
        for (const result of results) {
            const key = [...result.outfit.itemIds].sort().join(',');
            if (!recentCombos.has(key)) {
                allOutfits.push(result);
            }
        }
        if (allOutfits.length >= 20) break;
    }

    // Sort by novelty (most-novel first)
    allOutfits.sort((a, b) => b.breakdown.noveltyScore - a.breakdown.noveltyScore);
    return allOutfits.slice(0, 20);
}

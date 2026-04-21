/**
 * Suggestion Engine — Core outfit recommendation logic
 *
 * Pure rule-based + weighted scoring. No LLM calls.
 * This is the brain of the daily suggestion system.
 *
 * Architecture:
 *   1. FILTER  — Remove items incompatible with context (weather, occasion)
 *   2. COMBINE — Generate candidate outfits (top + bottom + shoes [+ outerwear])
 *   3. SCORE   — Rank by (preference × weather × novelty × color_harmony)
 *   4. RETURN  — Top 3 candidates with reasoning
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
// TYPES
// ============================================

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
    excludeOutfitIds?: string[];  // Already shown today
}

export interface ScoredOutfit {
    outfit: Omit<Outfit, 'id' | 'userId' | 'createdAt' | 'saved' | 'wornCount' | 'lastWornAt'>;
    score: number;
    reasoning: string;
    breakdown: {
        preferenceScore: number;
        weatherScore: number;
        noveltyScore: number;
        harmonyScore: number;
    };
}

// ============================================
// CONSTANTS
// ============================================

const WEIGHTS = {
    preference: 0.25,
    weather: 0.30,
    novelty: 0.25,
    harmony: 0.20,
};

/** Materials suitable for temperature ranges */
const WARM_MATERIALS = ['wool', 'fleece', 'cashmere', 'down', 'sherpa', 'corduroy'];
const COOL_MATERIALS = ['linen', 'cotton', 'silk', 'rayon', 'chiffon', 'mesh'];
const RAIN_MATERIALS = ['nylon', 'polyester', 'gore-tex', 'leather'];

/** Color harmony rules (complementary pairs) */
const COLOR_HARMONIES: Record<string, string[]> = {
    navy: ['white', 'cream', 'beige', 'khaki', 'light blue', 'camel'],
    black: ['white', 'gray', 'red', 'cream', 'beige', 'camel'],
    white: ['navy', 'black', 'blue', 'gray', 'beige', 'olive'],
    gray: ['black', 'white', 'navy', 'blue', 'pink', 'burgundy'],
    blue: ['white', 'gray', 'beige', 'brown', 'khaki', 'cream'],
    brown: ['white', 'cream', 'blue', 'green', 'beige', 'navy'],
    beige: ['navy', 'white', 'brown', 'olive', 'blue', 'black'],
    olive: ['white', 'cream', 'beige', 'brown', 'khaki', 'navy'],
    red: ['black', 'white', 'navy', 'gray', 'denim blue'],
    green: ['white', 'cream', 'brown', 'beige', 'navy', 'black'],
    pink: ['gray', 'navy', 'white', 'cream', 'black'],
    burgundy: ['gray', 'cream', 'white', 'navy', 'camel'],
};

// ============================================
// SEASON INFERENCE
// ============================================

function getCurrentSeason(): Season {
    const month = new Date().getMonth(); // 0-11
    if (month >= 2 && month <= 4) return 'spring';
    if (month >= 5 && month <= 7) return 'summer';
    if (month >= 8 && month <= 10) return 'fall';
    return 'winter';
}

function tempToSeason(temp: number): Season[] {
    if (temp < 5) return ['winter'];
    if (temp < 15) return ['fall', 'winter', 'spring'];
    if (temp < 25) return ['spring', 'fall'];
    return ['summer', 'spring'];
}

// ============================================
// STEP 1: FILTER
// ============================================

function filterByWeather(items: ClothingItem[], weather?: WeatherContext): ClothingItem[] {
    if (!weather) return items;

    return items.filter((item) => {
        const temp = weather.temp;
        const material = (item.material || '').toLowerCase();

        // Hard rules: don't suggest shorts in freezing weather
        if (temp < 5) {
            if (item.subCategory?.toLowerCase().includes('short')) return false;
            if (item.subCategory?.toLowerCase().includes('sandal')) return false;
            if (item.subCategory?.toLowerCase().includes('tank')) return false;
        }

        // Hard rules: don't suggest heavy coat in 30°C+
        if (temp > 28) {
            if (WARM_MATERIALS.some((m) => material.includes(m))) return false;
            if (item.category === 'outerwear' && !material.includes('light')) return false;
        }

        // Rain: prefer water-resistant
        if (weather.condition === 'rain' || weather.condition === 'snow') {
            if (item.category === 'shoes') {
                // Prefer closed shoes
                if (item.subCategory?.toLowerCase().includes('sandal')) return false;
            }
        }

        return true;
    });
}

function filterByOccasion(items: ClothingItem[], occasion: Occasion): ClothingItem[] {
    return items.filter((item) => {
        // If item has occasions set, use them
        if (item.occasions && item.occasions.length > 0) {
            return item.occasions.includes(occasion);
        }
        // Otherwise, include it (user hasn't tagged it yet)
        return true;
    });
}

function filterBySeason(items: ClothingItem[], weather?: WeatherContext): ClothingItem[] {
    const seasons = weather ? tempToSeason(weather.temp) : [getCurrentSeason()];

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

function groupByCategory(items: ClothingItem[]): Record<ClothingCategory, ClothingItem[]> {
    const groups: Record<string, ClothingItem[]> = {
        top: [],
        bottom: [],
        shoes: [],
        outerwear: [],
        accessory: [],
    };

    items.forEach((item) => {
        if (groups[item.category]) {
            groups[item.category].push(item);
        }
    });

    return groups as Record<ClothingCategory, ClothingItem[]>;
}

function generateCandidates(
    groups: Record<ClothingCategory, ClothingItem[]>,
    weather?: WeatherContext,
    maxCandidates: number = 50
): string[][] {
    const tops = groups.top;
    const bottoms = groups.bottom;
    const shoes = groups.shoes;
    const outerwear = groups.outerwear;

    if (tops.length === 0 || bottoms.length === 0) return [];

    const candidates: string[][] = [];
    const needsOuterwear = weather ? weather.temp < 15 : false;

    for (const top of tops) {
        for (const bottom of bottoms) {
            const baseOutfit = [top.id, bottom.id];

            // Add shoes if available
            if (shoes.length > 0) {
                for (const shoe of shoes) {
                    const outfit = [...baseOutfit, shoe.id];
                    // Add outerwear if cold
                    if (needsOuterwear && outerwear.length > 0) {
                        for (const outer of outerwear) {
                            candidates.push([...outfit, outer.id]);
                            if (candidates.length >= maxCandidates) return candidates;
                        }
                    } else {
                        candidates.push(outfit);
                        if (candidates.length >= maxCandidates) return candidates;
                    }
                }
            } else {
                if (needsOuterwear && outerwear.length > 0) {
                    for (const outer of outerwear) {
                        candidates.push([...baseOutfit, outer.id]);
                        if (candidates.length >= maxCandidates) return candidates;
                    }
                } else {
                    candidates.push(baseOutfit);
                    if (candidates.length >= maxCandidates) return candidates;
                }
            }
        }
    }

    return candidates;
}

// ============================================
// STEP 3: SCORE
// ============================================

function scorePreference(
    itemIds: string[],
    items: ClothingItem[],
    preferences?: UserPreferences
): number {
    if (!preferences) return 0.5; // Neutral

    const outfitItems = itemIds.map((id) => items.find((i) => i.id === id)).filter(Boolean) as ClothingItem[];
    let score = 0.5;
    let factors = 0;

    outfitItems.forEach((item) => {
        const color = (item.primaryColor || '').toLowerCase();

        // Preferred colors boost
        if (preferences.preferredColors.some((c) => color.includes(c.toLowerCase()))) {
            score += 0.15;
            factors++;
        }

        // Avoided colors penalty
        if (preferences.avoidColors.some((c) => color.includes(c.toLowerCase()))) {
            score -= 0.2;
            factors++;
        }
    });

    return Math.max(0, Math.min(1, score));
}

function scoreWeather(
    itemIds: string[],
    items: ClothingItem[],
    weather?: WeatherContext
): number {
    if (!weather) return 0.7; // Slightly positive when no weather data

    const outfitItems = itemIds.map((id) => items.find((i) => i.id === id)).filter(Boolean) as ClothingItem[];
    let score = 0.5;

    const temp = weather.temp;
    const hasOuterwear = outfitItems.some((i) => i.category === 'outerwear');

    // Temperature appropriateness
    if (temp < 10 && hasOuterwear) score += 0.2;
    if (temp < 10 && !hasOuterwear) score -= 0.15;
    if (temp > 25 && hasOuterwear) score -= 0.2;

    // Material appropriateness
    outfitItems.forEach((item) => {
        const material = (item.material || '').toLowerCase();
        if (temp < 15 && WARM_MATERIALS.some((m) => material.includes(m))) score += 0.1;
        if (temp > 25 && COOL_MATERIALS.some((m) => material.includes(m))) score += 0.1;
        if (weather.condition === 'rain' && RAIN_MATERIALS.some((m) => material.includes(m))) score += 0.1;
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

function generateReasoning(
    itemIds: string[],
    items: ClothingItem[],
    breakdown: ScoredOutfit['breakdown'],
    weather?: WeatherContext,
    occasion?: Occasion
): string {
    const outfitItems = itemIds.map((id) => items.find((i) => i.id === id)).filter(Boolean) as ClothingItem[];
    const parts: string[] = [];

    // Weather reasoning
    if (weather) {
        parts.push(`${Math.round(weather.temp)}°C ${weather.condition}`);
    }

    // Novelty reasoning
    if (breakdown.noveltyScore > 0.7) {
        const leastWorn = outfitItems.reduce((a, b) => (a.wearCount < b.wearCount ? a : b));
        parts.push(`features your ${leastWorn.subCategory || leastWorn.category} (worn ${leastWorn.wearCount}×)`);
    }

    // Harmony reasoning
    if (breakdown.harmonyScore > 0.7) {
        const colors = outfitItems.map((i) => i.primaryColor).filter(Boolean).slice(0, 2);
        if (colors.length >= 2) {
            parts.push(`${colors[0]} + ${colors[1]} color harmony`);
        }
    }

    // Occasion
    if (occasion) {
        parts.push(`suited for ${occasion}`);
    }

    return parts.length > 0
        ? parts.join(' · ')
        : 'A balanced outfit from your wardrobe';
}

// ============================================
// MAIN: generateSuggestions
// ============================================

export function generateSuggestions(request: SuggestionRequest): ScoredOutfit[] {
    const { items, wearLogs, occasion, weather, preferences } = request;

    if (items.length === 0) return [];

    // Step 1: Filter
    let filtered = filterByWeather(items, weather);
    filtered = filterByOccasion(filtered, occasion);
    filtered = filterBySeason(filtered, weather);

    // Step 2: Combine
    const groups = groupByCategory(filtered);
    const candidates = generateCandidates(groups, weather);

    if (candidates.length === 0) {
        // Fallback: try without occasion filter
        const fallbackFiltered = filterByWeather(items, weather);
        const fallbackGroups = groupByCategory(fallbackFiltered);
        const fallbackCandidates = generateCandidates(fallbackGroups, weather);
        if (fallbackCandidates.length === 0) return [];
        return scoreCandidates(fallbackCandidates, items, wearLogs, weather, preferences, occasion);
    }

    return scoreCandidates(candidates, items, wearLogs, weather, preferences, occasion);
}

function scoreCandidates(
    candidates: string[][],
    items: ClothingItem[],
    wearLogs: WearLog[],
    weather?: WeatherContext,
    preferences?: UserPreferences,
    occasion?: Occasion
): ScoredOutfit[] {
    // Step 3: Score all candidates
    const scored: ScoredOutfit[] = candidates.map((itemIds) => {
        const preferenceScore = scorePreference(itemIds, items, preferences);
        const weatherScore = scoreWeather(itemIds, items, weather);
        const noveltyScore = scoreNovelty(itemIds, items, wearLogs);
        const harmonyScore = scoreColorHarmony(itemIds, items);

        const totalScore =
            preferenceScore * WEIGHTS.preference +
            weatherScore * WEIGHTS.weather +
            noveltyScore * WEIGHTS.novelty +
            harmonyScore * WEIGHTS.harmony;

        const breakdown = { preferenceScore, weatherScore, noveltyScore, harmonyScore };

        return {
            outfit: {
                itemIds,
                occasion: occasion || 'casual',
                generatedBy: 'ai' as const,
                reasoning: generateReasoning(itemIds, items, breakdown, weather, occasion),
                style: 'smart-casual',
            },
            score: totalScore,
            reasoning: generateReasoning(itemIds, items, breakdown, weather, occasion),
            breakdown,
        };
    });

    // Step 4: Sort by score descending, return top 3
    scored.sort((a, b) => b.score - a.score);

    // Deduplicate: don't show two outfits that differ by only one item
    const diverse = diversifyResults(scored, 3);

    return diverse;
}

function diversifyResults(scored: ScoredOutfit[], count: number): ScoredOutfit[] {
    const result: ScoredOutfit[] = [];

    for (const candidate of scored) {
        if (result.length >= count) break;

        // Check overlap with already selected outfits
        const isDuplicate = result.some((selected) => {
            const selectedIds = new Set(selected.outfit.itemIds);
            const overlap = candidate.outfit.itemIds.filter((id) => selectedIds.has(id));
            // If more than 60% overlap, skip
            return overlap.length / candidate.outfit.itemIds.length > 0.6;
        });

        if (!isDuplicate) {
            result.push(candidate);
        }
    }

    // If we couldn't find enough diverse results, fill with top remaining
    if (result.length < count) {
        for (const candidate of scored) {
            if (result.length >= count) break;
            if (!result.includes(candidate)) {
                result.push(candidate);
            }
        }
    }

    return result;
}

// ============================================
// QUICK SUGGEST — Single best outfit for notifications
// ============================================

export function quickSuggest(
    items: ClothingItem[],
    wearLogs: WearLog[],
    weather?: WeatherContext
): ScoredOutfit | null {
    const suggestions = generateSuggestions({
        items,
        wearLogs,
        occasion: 'casual',
        weather,
    });

    return suggestions[0] || null;
}

/**
 * Outfit Service — AI outfit generation and weather-based recommendations.
 *
 * Handles:
 * - Server-backed outfit generation with caching
 * - Local fallback suggestions when offline
 * - Weather-adaptive outfit tips
 * - AliceVision outfit recommendations
 */

import axios from 'axios';
import Config from '../../config/env';
import { withRetry, getAuthHeaders } from './shared';
import type {
    AIOutfitSuggestion,
    DetectedClothingItem,
    OutfitRecommendation,
} from './types';

const API_URL = Config.api.url;
const ALICEVISION_URL = Config.api.alicevisionUrl;
const TIMEOUT_MS = 60000;

// ── Cache ──

const outfitCache = new Map<string, { data: AIOutfitSuggestion[]; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// ── Public API ──

export async function generateOutfitSuggestions(
    occasion: string,
    stylePreferences?: string,
    wardrobeItems?: Array<any>,
    weather?: { temp: number; condition: string }
): Promise<AIOutfitSuggestion[]> {
    const cacheKey = `${occasion}-${stylePreferences || ''}-${wardrobeItems?.length || 0}-${weather?.condition || ''}`;
    const cached = outfitCache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
        return cached.data;
    }

    return withRetry(async () => {
        try {
            const response = await axios.post(
                `${API_URL}/api/generate-outfits`,
                {
                    occasion,
                    stylePreferences,
                    wardrobeItems,
                    weather,
                    limit: 5,
                },
                {
                    headers: await getAuthHeaders(),
                    timeout: TIMEOUT_MS,
                }
            );

            if (response.data.success && response.data.outfits) {
                const suggestions = response.data.outfits;
                outfitCache.set(cacheKey, { data: suggestions, timestamp: Date.now() });
                return suggestions;
            }

            return generateLocalOutfitSuggestions(occasion, stylePreferences);
        } catch {
            return generateLocalOutfitSuggestions(occasion, stylePreferences);
        }
    });
}

export function generateLocalOutfitSuggestions(
    occasion: string,
    _stylePreferences?: string
): AIOutfitSuggestion[] {
    const occasions: Record<string, AIOutfitSuggestion> = {
        date: {
            id: 'date-1',
            description: 'Romantic evening outfit with elegant touches',
            occasion: 'Date Night',
            confidence: 0.85,
            items: [
                { type: 'Top', color: '#0A1931', recommendation: 'Silk blouse or fitted sweater' },
                { type: 'Bottom', color: 'Dark Blue', recommendation: 'Tailored jeans or skirt' },
                { type: 'Shoes', color: '#0A1931', recommendation: 'Heels or clean sneakers' },
            ],
            stylingTips: [
                'Add a statement necklace for elegance',
                'Choose a signature perfume',
                'Keep makeup natural but polished',
            ],
        },
        interview: {
            id: 'interview-1',
            description: 'Professional and confident interview attire',
            occasion: 'Interview',
            confidence: 0.90,
            items: [
                { type: 'Top', color: 'White', recommendation: 'Crisp button-down shirt' },
                { type: 'Bottom', color: 'Navy', recommendation: 'Tailored trousers or pencil skirt' },
                { type: 'Jacket', color: 'Navy', recommendation: 'Well-fitted blazer' },
            ],
            stylingTips: [
                'Iron clothes the night before',
                'Keep accessories minimal',
                'Choose closed-toe shoes',
            ],
        },
        party: {
            id: 'party-1',
            description: 'Fun and stylish party look',
            occasion: 'Party',
            confidence: 0.88,
            items: [
                { type: 'Top', color: 'Metallic', recommendation: 'Sequin top or bold colors' },
                { type: 'Bottom', color: '#0A1931', recommendation: 'Leather pants or mini skirt' },
                { type: 'Shoes', color: 'Gold', recommendation: 'Statement heels' },
            ],
            stylingTips: [
                'Don\'t be afraid of sparkle',
                'Balance bold pieces with simple ones',
                'Add a clutch bag',
            ],
        },
        casual: {
            id: 'casual-1',
            description: 'Comfortable yet stylish everyday look',
            occasion: 'Casual',
            confidence: 0.92,
            items: [
                { type: 'Top', color: 'White', recommendation: 'Quality t-shirt or casual shirt' },
                { type: 'Bottom', color: 'Blue', recommendation: 'Your favorite jeans' },
                { type: 'Shoes', color: 'White', recommendation: 'Clean sneakers' },
            ],
            stylingTips: [
                'Layer with a light jacket',
                'Accessorize with a watch',
                'Keep it simple but polished',
            ],
        },
    };

    const match = occasions[occasion.toLowerCase()] || occasions.casual;
    return [match, { ...match, id: `${match.id}-alt`, confidence: match.confidence - 0.1 }];
}

export async function getWeatherBasedOutfit(
    temperature: number,
    condition: string
): Promise<AIOutfitSuggestion> {
    const occasion = 'casual';
    let additionalTips: string[] = [];

    if (temperature < 10) {
        additionalTips = ['Layer up with a warm coat', 'Don\'t forget your scarf'];
    } else if (temperature < 20) {
        additionalTips = ['A light jacket will be perfect', 'Consider layers for temperature changes'];
    } else if (temperature < 30) {
        additionalTips = ['Light fabrics will keep you cool', 'Breathable materials recommended'];
    } else {
        additionalTips = ['Stay cool with minimal layers', 'Linen and cotton are your friends'];
    }

    if (condition.includes('rain')) {
        additionalTips.push('Bring an umbrella or waterproof jacket');
    }

    const suggestions = await generateOutfitSuggestions(occasion);
    const result = suggestions[0];
    result.stylingTips = [...additionalTips, ...result.stylingTips.slice(0, 2)];
    result.description = `Weather-appropriate outfit for ${temperature}°C - ${condition}`;

    return result;
}

export async function getOutfitRecommendations(
    wardrobeItems: DetectedClothingItem[],
    occasion: string,
    weather?: { temp: number; condition: string },
    preferences?: {
        preferredStyles?: string[];
        avoidColors?: string[];
        preferredColors?: string[];
    }
): Promise<{
    success: boolean;
    outfits: OutfitRecommendation[];
    processingTimeMs: number;
}> {
    return withRetry(async () => {
        const response = await axios.post(
            `${ALICEVISION_URL}/outfit/recommend`,
            {
                wardrobe_items: wardrobeItems.map(item => ({
                    id: item.cutoutImage?.slice(0, 20) || Math.random().toString(36),
                    category: item.category,
                    specificType: item.specificType,
                    primaryColor: item.primaryColor,
                    colorHex: item.colorHex,
                    material: item.material,
                    pattern: item.pattern,
                })),
                occasion,
                weather,
                preferences,
            },
            {
                headers: await getAuthHeaders(),
                timeout: TIMEOUT_MS,
            }
        );

        return response.data;
    });
}

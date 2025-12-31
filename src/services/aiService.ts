import axios, { AxiosError } from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Alert } from 'react-native';

// API Configuration - Using local API server (change back to remote for production)
const API_URL = 'http://172.20.10.5:3000';
const TIMEOUT_MS = 60000;
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;

// Types
export interface AIOutfitSuggestion {
    id: string;
    description: string;
    occasion: string;
    confidence: number;
    items: {
        type: string;
        color: string;
        recommendation: string;
    }[];
    stylingTips: string[];
}

export interface AIAnalysisResult {
    itemType: string;
    color: string;
    style: string;
    description: string;
    confidence: number;
    material?: string;
    brand?: string;
    season?: string;
    tags?: string[];
}

export interface VirtualTryOnResult {
    success: boolean;
    imageUrl: string;
    processingTime: number;
}

export interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

export interface ChatResponse {
    text: string;
    suggestions?: string[];
}

// Retry utility
async function withRetry<T>(
    fn: () => Promise<T>,
    retries: number = MAX_RETRIES,
    delay: number = RETRY_DELAY_MS
): Promise<T> {
    try {
        return await fn();
    } catch (error) {
        if (retries > 0) {
            console.log(`Retrying... (${MAX_RETRIES - retries + 1}/${MAX_RETRIES})`);
            await new Promise(resolve => setTimeout(resolve, delay));
            return withRetry(fn, retries - 1, delay * 1.5);
        }
        throw error;
    }
}

// Enhanced error handling
function handleAPIError(error: any, context: string): never {
    console.error(`[AIService] ${context} error:`, error);

    if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;

        if (axiosError.response?.status === 500) {
            throw new Error('Server is temporarily unavailable. Please try again in a moment.');
        }
        if (axiosError.response?.status === 404) {
            throw new Error('This feature is currently being updated. Please try again later.');
        }
        if (axiosError.code === 'ECONNABORTED') {
            throw new Error('Request timed out. Please check your connection and try again.');
        }
        if (!axiosError.response) {
            throw new Error('Unable to connect to server. Please check your internet connection.');
        }
    }

    throw new Error(`${context} failed. Please try again.`);
}

// Cache for outfit suggestions
const outfitCache = new Map<string, { data: AIOutfitSuggestion[]; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

class AIService {
    private userToken: string | null = null;

    async initialize(): Promise<void> {
        this.userToken = await AsyncStorage.getItem('userToken');
    }

    private getHeaders(): Record<string, string> {
        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
        };
        if (this.userToken) {
            headers['Authorization'] = `Bearer ${this.userToken}`;
        }
        return headers;
    }

    // ==================== OUTFIT GENERATION ====================

    async generateOutfitSuggestions(
        occasion: string,
        stylePreferences?: string,
        wardrobeItems?: any[]
    ): Promise<AIOutfitSuggestion[]> {
        const cacheKey = `${occasion}-${stylePreferences || ''}`;
        const cached = outfitCache.get(cacheKey);

        if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
            console.log('[AIService] Returning cached outfit suggestions');
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
                        limit: 5,
                    },
                    {
                        headers: this.getHeaders(),
                        timeout: TIMEOUT_MS
                    }
                );

                if (response.data.success && response.data.outfits) {
                    const suggestions = response.data.outfits;
                    outfitCache.set(cacheKey, { data: suggestions, timestamp: Date.now() });
                    return suggestions;
                }

                // Fallback to smart local generation
                return this.generateLocalOutfitSuggestions(occasion, stylePreferences);
            } catch (error) {
                console.log('[AIService] Backend unavailable, using local generation');
                return this.generateLocalOutfitSuggestions(occasion, stylePreferences);
            }
        });
    }

    private generateLocalOutfitSuggestions(
        occasion: string,
        stylePreferences?: string
    ): AIOutfitSuggestion[] {
        const occasions: Record<string, AIOutfitSuggestion> = {
            date: {
                id: 'date-1',
                description: 'Romantic evening outfit with elegant touches',
                occasion: 'Date Night',
                confidence: 0.85,
                items: [
                    { type: 'Top', color: 'Black', recommendation: 'Silk blouse or fitted sweater' },
                    { type: 'Bottom', color: 'Dark Blue', recommendation: 'Tailored jeans or skirt' },
                    { type: 'Shoes', color: 'Black', recommendation: 'Heels or clean sneakers' },
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
                    { type: 'Bottom', color: 'Black', recommendation: 'Leather pants or mini skirt' },
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

    // ==================== AI CHAT ====================

    async sendChatMessage(
        message: string,
        conversationHistory: ChatMessage[] = [],
        stylePreference?: string
    ): Promise<ChatResponse> {
        return withRetry(async () => {
            try {
                const response = await axios.post(
                    `${API_URL}/ai-chat`,
                    {
                        query: message,
                        conversationHistory,
                        stylePreference,
                    },
                    {
                        headers: this.getHeaders(),
                        timeout: TIMEOUT_MS,
                    }
                );

                if (response.data.text) {
                    return {
                        text: response.data.text,
                        suggestions: response.data.suggestions,
                    };
                }
                throw new Error('No response from AI');
            } catch (error) {
                // Fallback to helpful local responses
                return this.generateLocalChatResponse(message);
            }
        });
    }

    private generateLocalChatResponse(message: string): ChatResponse {
        const lowerMessage = message.toLowerCase();

        if (lowerMessage.includes('date')) {
            return {
                text: "For a date night, I'd recommend something that makes you feel confident! A nice blouse with tailored jeans works great. Add a statement accessory and you're set. What's the vibe - casual or fancy?",
                suggestions: ['Casual date outfit', 'Fancy dinner look', 'Coffee date style'],
            };
        }
        if (lowerMessage.includes('work') || lowerMessage.includes('interview')) {
            return {
                text: "For professional settings, stick to classic pieces. A well-fitted blazer, crisp shirt, and tailored pants in neutral colors always work. Would you like specific color recommendations?",
                suggestions: ['Business casual tips', 'Interview outfit help', 'Work wardrobe basics'],
            };
        }
        if (lowerMessage.includes('party') || lowerMessage.includes('event')) {
            return {
                text: "Time to shine! For parties, don't be afraid to go bold with colors or textures. A sequin top, statement earrings, or a fabulous dress can make you stand out. What's the dress code?",
                suggestions: ['Cocktail party look', 'Birthday outfit', 'Club night style'],
            };
        }

        return {
            text: "I'm here to help with your style! Tell me about the occasion, your preferences, or ask for outfit ideas. I can suggest complete looks based on what you're doing!",
            suggestions: ['Date night outfit', 'Work attire help', 'Weekend casual look'],
        };
    }

    // ==================== VIRTUAL TRY-ON ====================

    async virtualTryOn(
        humanImageBase64: string,
        garmentImageBase64: string
    ): Promise<VirtualTryOnResult> {
        const startTime = Date.now();

        return withRetry(async () => {
            try {
                const response = await axios.post(
                    `${API_URL}/try-on`,
                    {
                        human_image: humanImageBase64,
                        garment_image: garmentImageBase64,
                        description: 'clothing',
                    },
                    {
                        headers: this.getHeaders(),
                        timeout: 120000, // 2 minutes for try-on
                    }
                );

                if (response.data.image) {
                    return {
                        success: true,
                        imageUrl: response.data.image,
                        processingTime: Date.now() - startTime,
                    };
                }
                throw new Error('No image returned');
            } catch (error) {
                handleAPIError(error, 'Virtual try-on');
            }
        });
    }

    // ==================== CLOTHING ANALYSIS ====================

    async analyzeClothing(imageBase64: string): Promise<AIAnalysisResult[]> {
        return withRetry(async () => {
            try {
                const response = await axios.post(
                    `${API_URL}/api/analyze-frames`,
                    { frames: [imageBase64] },
                    {
                        headers: this.getHeaders(),
                        timeout: TIMEOUT_MS,
                    }
                );

                if (response.data.detectedItems?.length > 0) {
                    return response.data.detectedItems;
                }

                // Return smart fallback
                return this.generateLocalAnalysis();
            } catch (error) {
                console.log('[AIService] Analysis unavailable, using local');
                return this.generateLocalAnalysis();
            }
        });
    }

    private generateLocalAnalysis(): AIAnalysisResult[] {
        return [
            {
                itemType: 'Clothing Item',
                color: 'Unknown',
                style: 'Casual',
                description: 'Detected clothing item',
                confidence: 0.5,
                tags: ['needs-review'],
            },
        ];
    }

    // ==================== WEATHER-BASED RECOMMENDATIONS ====================

    async getWeatherBasedOutfit(
        temperature: number,
        condition: string
    ): Promise<AIOutfitSuggestion> {
        let occasion = 'casual';
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

        const suggestions = await this.generateOutfitSuggestions(occasion);
        const result = suggestions[0];
        result.stylingTips = [...additionalTips, ...result.stylingTips.slice(0, 2)];
        result.description = `Weather-appropriate outfit for ${temperature}°C - ${condition}`;

        return result;
    }

    // ==================== HEALTH CHECK ====================

    async checkServerHealth(): Promise<{ healthy: boolean; message: string }> {
        try {
            const response = await axios.get(`${API_URL}/health`, { timeout: 5000 });
            return { healthy: true, message: 'Server is running' };
        } catch {
            return { healthy: false, message: 'Server is currently unavailable' };
        }
    }
}

// Export singleton instance
export const aiService = new AIService();
export default aiService;

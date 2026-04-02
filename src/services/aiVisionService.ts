/**
 * AI Vision Service — Unified provider with Gemini and NVIDIA support.
 *
 * Supports multiple vision AI providers:
 * - Google Gemini 1.5 Flash/Pro (default, free tier available)
 * - NVIDIA NIM API (free tier with 1M tokens/day)
 *
 * Switch providers by setting EXPO_PUBLIC_AI_VISION_PROVIDER in .env:
 *   - 'gemini' (default) - Uses Google Gemini Vision
 *   - 'nvidia' - Uses NVIDIA NIM API with Llama 3.2 Vision
 *
 * Get API keys:
 *   - Gemini: https://aistudio.google.com/app/apikey
 *   - NVIDIA: https://build.nvidia.com/explore/discover (free, no credit card needed)
 */

import type {
    DetectedClothingItem,
    VideoAnalysisResult,
} from './ai/types';

// ============================================
// CONFIGURATION
// ============================================

type AIProvider = 'gemini' | 'nvidia';

const PROVIDER: AIProvider = (process.env.EXPO_PUBLIC_AI_VISION_PROVIDER as AIProvider) || 'gemini';

// Gemini Config
const GEMINI_API_KEY = process.env.EXPO_PUBLIC_GEMINI_API_KEY || '';
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models';
const GEMINI_MODEL = 'gemini-1.5-flash'; // Options: gemini-1.5-flash, gemini-1.5-pro

// NVIDIA Config
const NVIDIA_API_KEY = process.env.EXPO_PUBLIC_NVIDIA_API_KEY || '';
const NVIDIA_API_URL = 'https://ai.api.nvidia.com/v1/meta/llama-3.2-90b-vision-instruct/chat/completions';
// Alternative free NVIDIA models:
// - meta/llama-3.2-11b-vision-instruct (faster)
// - google/deplot (for charts)

const TIMEOUT_MS = 30000;
const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1000;

// ============================================
// TYPES
// ============================================

interface GeminiResponse {
    candidates?: Array<{
        content?: {
            parts?: Array<{ text?: string }>;
        };
    }>;
    error?: { message: string; code: string };
}

interface NvidiaResponse {
    choices?: Array<{
        message?: { content?: string };
    }>;
    error?: { message: string };
}

interface AnalysisResult {
    success: boolean;
    items: DetectedClothingItem[];
    processingTimeMs: number;
    provider: string;
    error?: string;
}

// ============================================
// PROMPTS
// ============================================

const CLOTHING_ANALYSIS_PROMPT = `Analyze this clothing image and return a JSON object with these exact fields:
{
  "category": "main category (tops, bottoms, outerwear, footwear, accessories, dresses)",
  "specificType": "specific type (t-shirt, jeans, blazer, sneakers, etc.)",
  "primaryColor": "main color name",
  "colorHex": "approximate hex color code",
  "material": "fabric material if visible (cotton, wool, leather, etc.)",
  "pattern": "pattern type (solid, striped, floral, etc.)",
  "confidence": 0.95
}

Be precise and specific. Return ONLY valid JSON, no markdown formatting.`;

const MULTI_ITEM_PROMPT = `Analyze this image and identify ALL clothing items visible. Return a JSON array where each item has:
{
  "category": "tops/bottoms/outerwear/footwear/accessories/dresses",
  "specificType": "specific clothing type",
  "primaryColor": "color name",
  "colorHex": "hex code",
  "material": "fabric (optional)",
  "pattern": "pattern (optional)",
  "confidence": 0.0-1.0
}

Return format: { "items": [...] }`;

// ============================================
// UTILITY FUNCTIONS
// ============================================

async function withRetry<T>(fn: () => Promise<T>, retries: number = MAX_RETRIES): Promise<T> {
    let lastError: Error | null = null;
    for (let i = 0; i <= retries; i++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));
            if (i < retries) {
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY_MS * Math.pow(2, i)));
            }
        }
    }
    throw lastError;
}

function cleanBase64(imageBase64: string): string {
    return imageBase64.replace(/^data:image\/\w+;base64,/, '');
}

function parseJsonResponse(text: string): any {
    try {
        const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
        if (jsonMatch) return JSON.parse(jsonMatch[1].trim());
        return JSON.parse(text.trim());
    } catch {
        const jsonLike = text.match(/\{[\s\S]*\}/);
        if (jsonLike) {
            try { return JSON.parse(jsonLike[0]); } catch { return null; }
        }
        return null;
    }
}

// ============================================
// GEMINI PROVIDER
// ============================================

async function callGeminiVision(imageBase64: string, prompt: string): Promise<string> {
    if (!GEMINI_API_KEY) throw new Error('Gemini API key not configured');

    const cleanedImage = cleanBase64(imageBase64);
    const url = `${GEMINI_API_URL}/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;

    const body = {
        contents: [{
            parts: [
                { text: prompt },
                { inlineData: { mimeType: 'image/jpeg', data: cleanedImage } },
            ],
        }],
        generationConfig: {
            temperature: 0.2,
            maxOutputTokens: 1024,
        },
    };

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal,
        });

        clearTimeout(timeout);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error?.message || `Gemini API error: HTTP ${response.status}`);
        }

        const data: GeminiResponse = await response.json();
        if (data.error) throw new Error(`Gemini API error: ${data.error.message}`);

        const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!text) throw new Error('Empty response from Gemini API');

        return text;
    } catch (error) {
        clearTimeout(timeout);
        throw error;
    }
}

async function geminiAnalyzeClothing(imageBase64: string): Promise<AnalysisResult> {
    const startTime = Date.now();

    return withRetry(async () => {
        const responseText = await callGeminiVision(imageBase64, CLOTHING_ANALYSIS_PROMPT);
        const parsed = parseJsonResponse(responseText);

        if (!parsed) throw new Error('Failed to parse Gemini response');

        const item: DetectedClothingItem = {
            category: parsed.category || 'unknown',
            specificType: parsed.specificType || parsed.category || 'clothing item',
            confidence: parsed.confidence || 0.8,
            primaryColor: parsed.primaryColor || 'unknown',
            colorHex: parsed.colorHex || '#808080',
            material: parsed.material,
            pattern: parsed.pattern,
        };

        return {
            success: true,
            items: [item],
            processingTimeMs: Date.now() - startTime,
            provider: 'gemini',
        };
    });
}

async function geminiDetectMultiple(imageBase64: string): Promise<AnalysisResult> {
    const startTime = Date.now();

    return withRetry(async () => {
        const responseText = await callGeminiVision(imageBase64, MULTI_ITEM_PROMPT);
        const parsed = parseJsonResponse(responseText);

        if (!parsed?.items) return geminiAnalyzeClothing(imageBase64);

        const items: DetectedClothingItem[] = parsed.items.map((item: any) => ({
            category: item.category || 'unknown',
            specificType: item.specificType || item.category || 'clothing item',
            confidence: item.confidence || 0.8,
            primaryColor: item.primaryColor || 'unknown',
            colorHex: item.colorHex || '#808080',
            material: item.material,
            pattern: item.pattern,
        }));

        return { success: true, items, processingTimeMs: Date.now() - startTime, provider: 'gemini' };
    });
}

// ============================================
// NVIDIA PROVIDER
// ============================================

async function callNvidiaVision(imageBase64: string, prompt: string): Promise<string> {
    if (!NVIDIA_API_KEY) throw new Error('NVIDIA API key not configured');

    const cleanedImage = cleanBase64(imageBase64);

    const body = {
        model: 'meta/llama-3.2-90b-vision-instruct',
        messages: [
            {
                role: 'user',
                content: [
                    { type: 'text', text: prompt },
                    {
                        type: 'image_url',
                        image_url: { url: `data:image/jpeg;base64,${cleanedImage}` },
                    },
                ],
            },
        ],
        max_tokens: 1024,
        temperature: 0.2,
    };

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
        const response = await fetch(NVIDIA_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${NVIDIA_API_KEY}`,
            },
            body: JSON.stringify(body),
            signal: controller.signal,
        });

        clearTimeout(timeout);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error?.message || `NVIDIA API error: HTTP ${response.status}`);
        }

        const data: NvidiaResponse = await response.json();
        const text = data.choices?.[0]?.message?.content;
        if (!text) throw new Error('Empty response from NVIDIA API');

        return text;
    } catch (error) {
        clearTimeout(timeout);
        throw error;
    }
}

async function nvidiaAnalyzeClothing(imageBase64: string): Promise<AnalysisResult> {
    const startTime = Date.now();

    return withRetry(async () => {
        const responseText = await callNvidiaVision(imageBase64, CLOTHING_ANALYSIS_PROMPT);
        const parsed = parseJsonResponse(responseText);

        if (!parsed) throw new Error('Failed to parse NVIDIA response');

        const item: DetectedClothingItem = {
            category: parsed.category || 'unknown',
            specificType: parsed.specificType || parsed.category || 'clothing item',
            confidence: parsed.confidence || 0.8,
            primaryColor: parsed.primaryColor || 'unknown',
            colorHex: parsed.colorHex || '#808080',
            material: parsed.material,
            pattern: parsed.pattern,
        };

        return {
            success: true,
            items: [item],
            processingTimeMs: Date.now() - startTime,
            provider: 'nvidia',
        };
    });
}

async function nvidiaDetectMultiple(imageBase64: string): Promise<AnalysisResult> {
    const startTime = Date.now();

    return withRetry(async () => {
        const responseText = await callNvidiaVision(imageBase64, MULTI_ITEM_PROMPT);
        const parsed = parseJsonResponse(responseText);

        if (!parsed?.items) return nvidiaAnalyzeClothing(imageBase64);

        const items: DetectedClothingItem[] = parsed.items.map((item: any) => ({
            category: item.category || 'unknown',
            specificType: item.specificType || item.category || 'clothing item',
            confidence: item.confidence || 0.8,
            primaryColor: item.primaryColor || 'unknown',
            colorHex: item.colorHex || '#808080',
            material: item.material,
            pattern: item.pattern,
        }));

        return { success: true, items, processingTimeMs: Date.now() - startTime, provider: 'nvidia' };
    });
}

// ============================================
// UNIFIED SERVICE
// ============================================

class AIVisionService {
    private provider: AIProvider;

    constructor() {
        this.provider = PROVIDER;
        console.log(`[AIVision] Using provider: ${this.provider}`);
    }

    /** Switch provider at runtime */
    setProvider(provider: AIProvider): void {
        this.provider = provider;
        console.log(`[AIVision] Switched to provider: ${provider}`);
    }

    getProvider(): AIProvider {
        return this.provider;
    }

    /** Analyze a single clothing image */
    async analyzeClothing(imageBase64: string): Promise<AnalysisResult> {
        if (this.provider === 'nvidia') {
            return nvidiaAnalyzeClothing(imageBase64);
        }
        return geminiAnalyzeClothing(imageBase64);
    }

    /** Detect multiple items in an image */
    async detectMultipleItems(imageBase64: string): Promise<AnalysisResult> {
        if (this.provider === 'nvidia') {
            return nvidiaDetectMultiple(imageBase64);
        }
        return geminiDetectMultiple(imageBase64);
    }

    /** Ensemble detection for higher accuracy */
    async detectClothingEnsemble(imageBase64: string, votes: number = 3): Promise<{
        success: boolean;
        items: DetectedClothingItem[];
        processingTimeMs: number;
        provider: string;
        modelsUsed: string[];
    }> {
        const startTime = Date.now();
        const results: DetectedClothingItem[][] = [];

        for (let i = 0; i < votes; i++) {
            try {
                const result = await this.detectMultipleItems(imageBase64);
                if (result.success) results.push(result.items);
            } catch (error) {
                console.warn(`[AIVision] Ensemble vote ${i + 1} failed:`, error);
            }
        }

        if (results.length === 0) {
            return {
                success: false,
                items: [],
                processingTimeMs: Date.now() - startTime,
                provider: this.provider,
                modelsUsed: [],
            };
        }

        // Aggregate results
        const itemMap = new Map<string, DetectedClothingItem & { count: number }>();
        for (const result of results) {
            for (const item of result) {
                const key = `${item.category}-${item.specificType}-${item.primaryColor}`;
                const existing = itemMap.get(key);
                if (existing) {
                    existing.count++;
                    existing.confidence = Math.max(existing.confidence, item.confidence);
                } else {
                    itemMap.set(key, { ...item, count: 1 });
                }
            }
        }

        const threshold = Math.ceil(votes / 2);
        const finalItems = Array.from(itemMap.values())
            .filter(item => item.count >= threshold)
            .map(({ count, ...item }) => item);

        return {
            success: true,
            items: finalItems.length > 0 ? finalItems : results[0],
            processingTimeMs: Date.now() - startTime,
            provider: this.provider,
            modelsUsed: Array(votes).fill(`${this.provider}-vote`),
        };
    }

    /** Analyze video frames with temporal consistency */
    async analyzeVideoFrames(
        frames: string[],
        options?: { detectOutfitChanges?: boolean; minAgreement?: number }
    ): Promise<VideoAnalysisResult> {
        const startTime = Date.now();
        const minAgreement = options?.minAgreement ?? 0.5;

        const frameResults: DetectedClothingItem[][] = [];
        for (const frame of frames) {
            try {
                const result = await this.detectMultipleItems(frame);
                if (result.success) frameResults.push(result.items);
            } catch (error) {
                console.warn('[AIVision] Frame analysis failed:', error);
            }
        }

        if (frameResults.length === 0) {
            return {
                success: false,
                items: [],
                outfits: [],
                processingTimeMs: Date.now() - startTime,
            };
        }

        // Aggregate across frames
        const itemFrequency = new Map<string, { item: DetectedClothingItem; frames: number }>();
        for (let i = 0; i < frameResults.length; i++) {
            for (const item of frameResults[i]) {
                const key = `${item.category}-${item.specificType}`;
                const existing = itemFrequency.get(key);
                if (existing) {
                    existing.frames++;
                    existing.item.confidence = Math.max(existing.item.confidence, item.confidence);
                } else {
                    itemFrequency.set(key, { item, frames: 1 });
                }
            }
        }

        const threshold = Math.ceil(frames.length * minAgreement);
        const consistentItems = Array.from(itemFrequency.values())
            .filter(({ frames }) => frames >= threshold)
            .map(({ item }) => item);

        return {
            success: true,
            items: consistentItems.length > 0 ? consistentItems : frameResults[0],
            outfits: [{
                outfitId: 1,
                items: consistentItems.length > 0 ? consistentItems : frameResults[0],
            }],
            processingTimeMs: Date.now() - startTime,
        };
    }

    /** Health check */
    async checkHealth(): Promise<{
        healthy: boolean;
        provider: AIProvider;
        features: string[];
        message: string;
    }> {
        const config = this.provider === 'nvidia' ? NVIDIA_API_KEY : GEMINI_API_KEY;

        if (!config) {
            return {
                healthy: false,
                provider: this.provider,
                features: [],
                message: `${this.provider} API key not configured`,
            };
        }

        return {
            healthy: true,
            provider: this.provider,
            features: ['vision-analysis', 'multi-item-detection', 'video-frames', 'ensemble-detection'],
            message: `${this.provider} AI Vision service is ready`,
        };
    }
}

// Export singleton
export const aiVisionService = new AIVisionService();
export default aiVisionService;

// Also export individual providers for direct use
export { geminiAnalyzeClothing, geminiDetectMultiple };
export { nvidiaAnalyzeClothing, nvidiaDetectMultiple };

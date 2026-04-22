/**
 * AI Vision Service — SECURE proxy to the `ai-process` Edge Function.
 *
 * SECURITY CONTRACT
 * -----------------
 * This file used to read EXPO_PUBLIC_GEMINI_API_KEY and
 * EXPO_PUBLIC_NVIDIA_API_KEY on device and call the providers
 * directly from the mobile app. That leaked tokens into every IPA.
 *
 * Those EXPO_PUBLIC_* keys have been removed. All vision calls now
 * go through Supabase Edge Functions (`ai-process`), which read the
 * provider tokens from the `app_config` table and keep them on the
 * server side.
 *
 * The public surface (functions / types) is unchanged so existing
 * callers in src/services/ai/scanService.ts continue to work.
 */
import { supabase } from '../../lib/supabase';
import { createLogger } from '../utils/logger';

import type {
    DetectedClothingItem,
    VideoAnalysisResult,
} from './ai/types';

const log = createLogger('AIVision');

type AIProvider = 'gemini' | 'nvidia';

interface AnalysisResult {
    success: boolean;
    items: DetectedClothingItem[];
    processingTimeMs: number;
    provider: string;
    error?: string;
}

interface EdgeVisionResponse {
    success?: boolean;
    error?: string;
    provider?: string;
    items?: Array<Partial<DetectedClothingItem>>;
    classification?: {
        category?: string;
        specificType?: string;
        primaryColor?: string;
        colorHex?: string;
        material?: string;
        pattern?: string;
        confidence?: number;
    };
}

type VisionOperation = 'analyze' | 'detect-multiple';

function normalizeItem(raw: Partial<DetectedClothingItem> | undefined): DetectedClothingItem {
    return {
        category: raw?.category || 'unknown',
        specificType: raw?.specificType || raw?.category || 'clothing item',
        confidence: typeof raw?.confidence === 'number' ? raw.confidence : 0.8,
        primaryColor: raw?.primaryColor || 'unknown',
        colorHex: raw?.colorHex || '#808080',
        material: raw?.material,
        pattern: raw?.pattern,
    };
}

/**
 * Single call to the `ai-process` Edge Function. The function expects
 * { image, operation, provider? } and returns either { classification }
 * for single-item analysis or { items: [...] } for multi-item detection.
 */
async function callEdgeVision(
    imageBase64: string,
    operation: VisionOperation,
    provider?: AIProvider,
): Promise<EdgeVisionResponse> {
    const { data, error } = await supabase.functions.invoke<EdgeVisionResponse>('ai-process', {
        body: {
            image: imageBase64,
            operation,
            provider,
        },
    });

    if (error) {
        log.warn('Edge Function returned error', { operation, error: error.message });
        throw new Error(error.message || 'Edge Function error');
    }
    if (!data) throw new Error('Empty response from ai-process Edge Function');
    if (data.success === false) {
        throw new Error(data.error || 'AI vision processing failed');
    }
    return data;
}

class AIVisionService {
    private provider: AIProvider;

    constructor() {
        const configured = (process.env.EXPO_PUBLIC_AI_VISION_PROVIDER as AIProvider) || 'gemini';
        this.provider = configured === 'nvidia' ? 'nvidia' : 'gemini';
        log.debug('AIVisionService initialized', { provider: this.provider });
    }

    setProvider(provider: AIProvider): void {
        this.provider = provider;
    }

    getProvider(): AIProvider {
        return this.provider;
    }

    async analyzeClothing(imageBase64: string): Promise<AnalysisResult> {
        const startTime = Date.now();
        try {
            const data = await callEdgeVision(imageBase64, 'analyze', this.provider);
            const item = normalizeItem(data.classification);
            return {
                success: true,
                items: [item],
                processingTimeMs: Date.now() - startTime,
                provider: data.provider || this.provider,
            };
        } catch (err) {
            log.error('analyzeClothing failed', err);
            return {
                success: false,
                items: [],
                processingTimeMs: Date.now() - startTime,
                provider: this.provider,
                error: err instanceof Error ? err.message : 'Unknown error',
            };
        }
    }

    async detectMultipleItems(imageBase64: string): Promise<AnalysisResult> {
        const startTime = Date.now();
        try {
            const data = await callEdgeVision(imageBase64, 'detect-multiple', this.provider);
            const items = (data.items ?? []).map(normalizeItem);
            if (items.length === 0) {
                // Server returned no multi-item result. Fall back to single.
                return this.analyzeClothing(imageBase64);
            }
            return {
                success: true,
                items,
                processingTimeMs: Date.now() - startTime,
                provider: data.provider || this.provider,
            };
        } catch (err) {
            log.error('detectMultipleItems failed', err);
            return {
                success: false,
                items: [],
                processingTimeMs: Date.now() - startTime,
                provider: this.provider,
                error: err instanceof Error ? err.message : 'Unknown error',
            };
        }
    }

    /**
     * Ensemble detection — runs N calls to the Edge Function and keeps
     * items agreed on by a majority of votes. The heavy lifting (and
     * the secrets) stay server-side.
     */
    async detectClothingEnsemble(
        imageBase64: string,
        votes: number = 3,
    ): Promise<{
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
                if (result.success && result.items.length) results.push(result.items);
            } catch (err) {
                log.warn(`Ensemble vote ${i + 1} failed`, err);
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
            .filter((item) => item.count >= threshold)
            .map(({ count: _count, ...item }) => item);

        return {
            success: true,
            items: finalItems.length > 0 ? finalItems : results[0],
            processingTimeMs: Date.now() - startTime,
            provider: this.provider,
            modelsUsed: Array(results.length).fill(`${this.provider}-vote`),
        };
    }

    async analyzeVideoFrames(
        frames: string[],
        options?: { detectOutfitChanges?: boolean; minAgreement?: number },
    ): Promise<VideoAnalysisResult> {
        const startTime = Date.now();
        const minAgreement = options?.minAgreement ?? 0.5;

        const frameResults: DetectedClothingItem[][] = [];
        for (const frame of frames) {
            try {
                const result = await this.detectMultipleItems(frame);
                if (result.success) frameResults.push(result.items);
            } catch (err) {
                log.warn('Frame analysis failed', err);
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

        const itemFrequency = new Map<string, { item: DetectedClothingItem; frames: number }>();
        for (const items of frameResults) {
            for (const item of items) {
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
            .filter(({ frames: f }) => f >= threshold)
            .map(({ item }) => item);

        return {
            success: true,
            items: consistentItems.length > 0 ? consistentItems : frameResults[0],
            outfits: [
                {
                    outfitId: 1,
                    items: consistentItems.length > 0 ? consistentItems : frameResults[0],
                },
            ],
            processingTimeMs: Date.now() - startTime,
        };
    }

    async checkHealth(): Promise<{
        healthy: boolean;
        provider: AIProvider;
        features: string[];
        message: string;
    }> {
        try {
            const { data, error } = await supabase.functions.invoke('ai-process', {
                body: { operation: 'health', provider: this.provider },
            });
            if (error) throw error;
            const healthy = Boolean(data?.success);
            return {
                healthy,
                provider: this.provider,
                features: healthy
                    ? ['vision-analysis', 'multi-item-detection', 'video-frames', 'ensemble-detection']
                    : [],
                message: healthy
                    ? `${this.provider} AI Vision service is ready`
                    : `${this.provider} AI Vision service is not reachable`,
            };
        } catch (err) {
            return {
                healthy: false,
                provider: this.provider,
                features: [],
                message: err instanceof Error ? err.message : 'Health check failed',
            };
        }
    }
}

export const aiVisionService = new AIVisionService();
export default aiVisionService;

// ============================================================
// Back-compat named exports — kept so existing callers still work.
// Internally they route through the same secure Edge Function.
// ============================================================
export const geminiAnalyzeClothing = (imageBase64: string) => {
    const svc = new AIVisionService();
    svc.setProvider('gemini');
    return svc.analyzeClothing(imageBase64);
};

export const geminiDetectMultiple = (imageBase64: string) => {
    const svc = new AIVisionService();
    svc.setProvider('gemini');
    return svc.detectMultipleItems(imageBase64);
};

export const nvidiaAnalyzeClothing = (imageBase64: string) => {
    const svc = new AIVisionService();
    svc.setProvider('nvidia');
    return svc.analyzeClothing(imageBase64);
};

export const nvidiaDetectMultiple = (imageBase64: string) => {
    const svc = new AIVisionService();
    svc.setProvider('nvidia');
    return svc.detectMultipleItems(imageBase64);
};

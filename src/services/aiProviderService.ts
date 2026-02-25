/**
 * AI Provider Service — Single-provider abstraction layer
 *
 * Consolidates AI access through a single interface.
 * Primary provider: Google Gemini (vision + text).
 * Falls back to local heuristics when offline or quota exceeded.
 *
 * This does NOT replace aiService.ts — it provides a cleaner
 * abstraction that aiService methods can delegate to.
 */

import Config from '../config/env';

// ============================================
// TYPES
// ============================================

export interface AIProviderConfig {
    providerName: string;
    maxRetries: number;
    timeoutMs: number;
}

export interface AnalyzeImageResult {
    items: DetectedItem[];
    processingTimeMs: number;
    provider: string;
}

export interface DetectedItem {
    category: string;
    subCategory: string;
    primaryColor: string;
    colorHex: string;
    pattern: string;
    material: string;
    confidence: number;
}

export interface GenerateOutfitResult {
    items: string[];       // Item IDs
    reasoning: string;
    occasion: string;
    style: string;
    colorHarmony: string;
    confidence: number;
}

export interface ChatResult {
    response: string;
    suggestions?: string[];
}

export interface AIProviderStatus {
    isAvailable: boolean;
    latencyMs: number;
    callCount: number;
    errorCount: number;
    lastError?: string;
}

// ============================================
// COST TRACKING
// ============================================

class CostTracker {
    private callCount = 0;
    private errorCount = 0;
    private totalLatencyMs = 0;
    private lastError: string | undefined;

    record(latencyMs: number, success: boolean, error?: string) {
        this.callCount++;
        this.totalLatencyMs += latencyMs;
        if (!success) {
            this.errorCount++;
            this.lastError = error;
        }
    }

    getStatus(): AIProviderStatus {
        return {
            isAvailable: this.errorCount < this.callCount * 0.5 || this.callCount < 3,
            latencyMs: this.callCount > 0 ? Math.round(this.totalLatencyMs / this.callCount) : 0,
            callCount: this.callCount,
            errorCount: this.errorCount,
            lastError: this.lastError,
        };
    }

    reset() {
        this.callCount = 0;
        this.errorCount = 0;
        this.totalLatencyMs = 0;
        this.lastError = undefined;
    }
}

// ============================================
// AI PROVIDER
// ============================================

class AIProviderService {
    private config: AIProviderConfig;
    private tracker: CostTracker;

    constructor() {
        this.config = {
            providerName: 'gemini',
            maxRetries: 2,
            timeoutMs: 30000,
        };
        this.tracker = new CostTracker();
    }

    /**
     * Analyze an image to detect clothing items.
     * Uses the Node.js API server which proxies to Gemini/AliceVision.
     * Falls back to local analysis on failure.
     */
    async analyzeImage(imageBase64: string): Promise<AnalyzeImageResult> {
        const start = Date.now();

        try {
            const response = await this.fetchWithRetry(
                `${Config.api.url}/api/analyze-clothing`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imageBase64 }),
                }
            );

            const data = await response.json();
            const latency = Date.now() - start;
            this.tracker.record(latency, true);

            return {
                items: (data.items || []).map((item: Record<string, unknown>) => ({
                    category: item.category || 'top',
                    subCategory: item.specificType || item.subCategory || '',
                    primaryColor: item.primaryColor || '',
                    colorHex: item.colorHex || '#000000',
                    pattern: item.pattern || 'solid',
                    material: item.material || '',
                    confidence: item.confidence || 0.5,
                })),
                processingTimeMs: latency,
                provider: this.config.providerName,
            };
        } catch (error: unknown) {
            const latency = Date.now() - start;
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            this.tracker.record(latency, false, errorMsg);

            // Local fallback — return empty result, don't crash
            console.warn('[AIProvider] analyzeImage failed, returning empty:', errorMsg);
            return {
                items: [],
                processingTimeMs: latency,
                provider: 'local_fallback',
            };
        }
    }

    /**
     * Generate outfit recommendation from wardrobe items.
     * Falls back to simple rule-based matching.
     */
    async generateOutfit(
        wardrobeItemIds: string[],
        occasion: string,
        weather?: { temp: number; condition: string }
    ): Promise<GenerateOutfitResult> {
        const start = Date.now();

        try {
            const response = await this.fetchWithRetry(
                `${Config.api.url}/api/outfit-recommendation`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        itemIds: wardrobeItemIds,
                        occasion,
                        weather,
                    }),
                }
            );

            const data = await response.json();
            const latency = Date.now() - start;
            this.tracker.record(latency, true);

            return {
                items: data.itemIds || [],
                reasoning: data.reasoning || '',
                occasion: data.occasion || occasion,
                style: data.style || '',
                colorHarmony: data.colorHarmony || '',
                confidence: data.confidence || 0.5,
            };
        } catch (error: unknown) {
            const latency = Date.now() - start;
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            this.tracker.record(latency, false, errorMsg);

            // Local fallback: return random subset of items
            console.warn('[AIProvider] generateOutfit failed, using local fallback:', errorMsg);
            return this.localOutfitFallback(wardrobeItemIds, occasion);
        }
    }

    /**
     * Chat with AI stylist.
     */
    async chat(
        message: string,
        context?: { wardrobeSize?: number; occasion?: string }
    ): Promise<ChatResult> {
        const start = Date.now();

        try {
            const response = await this.fetchWithRetry(
                `${Config.api.url}/ai-chat`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, context }),
                }
            );

            const data = await response.json();
            const latency = Date.now() - start;
            this.tracker.record(latency, true);

            return {
                response: data.response || data.text || "I'm not sure, could you rephrase?",
                suggestions: data.suggestions,
            };
        } catch (error: unknown) {
            const latency = Date.now() - start;
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            this.tracker.record(latency, false, errorMsg);

            return {
                response: "I'm currently offline. Please check your connection and try again.",
                suggestions: ['Try checking your internet connection', 'Retry in a moment'],
            };
        }
    }

    /**
     * Get provider health status and cost metrics.
     */
    getStatus(): AIProviderStatus {
        return this.tracker.getStatus();
    }

    /**
     * Reset cost tracking (e.g., at start of billing period).
     */
    resetTracking() {
        this.tracker.reset();
    }

    // ── Private helpers ────────────────────────

    private async fetchWithRetry(
        url: string,
        options: RequestInit,
        retries: number = this.config.maxRetries
    ): Promise<Response> {
        let lastError: Error | null = null;

        for (let i = 0; i <= retries; i++) {
            try {
                const controller = new AbortController();
                const timeout = setTimeout(() => controller.abort(), this.config.timeoutMs);

                const response = await fetch(url, {
                    ...options,
                    signal: controller.signal,
                });

                clearTimeout(timeout);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                return response;
            } catch (error) {
                lastError = error instanceof Error ? error : new Error(String(error));

                if (i < retries) {
                    // Exponential backoff: 1s, 2s, 4s
                    await new Promise((resolve) => setTimeout(resolve, 1000 * Math.pow(2, i)));
                }
            }
        }

        throw lastError || new Error('Request failed after retries');
    }

    private localOutfitFallback(
        itemIds: string[],
        occasion: string
    ): GenerateOutfitResult {
        // Simple fallback: pick up to 4 items
        const selected = itemIds.slice(0, Math.min(4, itemIds.length));

        return {
            items: selected,
            reasoning: 'Generated locally while offline. For better suggestions, connect to the internet.',
            occasion,
            style: 'mixed',
            colorHarmony: 'unknown',
            confidence: 0.3,
        };
    }
}

// Export singleton
export const aiProvider = new AIProviderService();
export default aiProvider;

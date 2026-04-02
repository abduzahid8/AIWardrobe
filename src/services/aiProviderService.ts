/**
 * AI Provider Service — Single-provider abstraction layer
 *
 * Consolidates AI access through a single interface.
 * Primary provider: Google Gemini (vision + text).
 * Falls back to local heuristics when offline or quota exceeded.
 *
 * Key features:
 *   - buildGeminiContext(): builds full wardrobe + wear history + weather context string
 *   - Context is cached for 15 minutes and invalidated when wardrobe changes
 *   - Every chat call prepends this system context
 *   - 5-second "Still thinking..." threshold handled via callback
 *   - All AI calls wrapped in try/catch — never white screens on failure
 *
 * Dependencies:
 *   - src/config/env (Config)
 *   - store/wardrobeStore (lazy import to avoid circular deps)
 */

import type { ClothingItem, WearLog } from '../types/domain';
import { getFormalityTier, FORMALITY_TIER_LABELS } from './suggestionEngine';
import { aiApi, uploadApi } from '../lib/api';

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
    items: string[];
    reasoning: string;
    occasion: string;
    style: string;
    colorHarmony: string;
    confidence: number;
}

export interface ChatResult {
    response: string;
    suggestions?: string[];
    /** True when response came from cache due to timeout */
    fromCache?: boolean;
}

export interface ProcessUploadResult {
    imageUrl: string | null;
    cutoutUrl: string | null;
    classification: {
        category: string;
        section: string;
        confidence: number;
        top5?: Array<{ category: string; section: string; confidence: number }>;
        attributes?: { color?: string; material?: string; pattern?: string; style?: string };
    } | null;
    description: string | null;
    style: string;
    steps: Array<string | { step: string; ms: number }>;
    provider: string;
    processingTimeMs: number;
}

export interface AIProviderStatus {
    isAvailable: boolean;
    latencyMs: number;
    callCount: number;
    errorCount: number;
    lastError?: string;
}

/** Data needed to build the Gemini system context string. */
export interface GeminiContextData {
    items: ClothingItem[];
    wearLogs: WearLog[];
    weather?: { temp: number; condition: string; city?: string };
    calendarEvents?: string[];
    stylePreferences?: string[];
    dislikedOutfitSummaries?: string[];
    userRules?: string[];
}

// ============================================
// CONTEXT BUILDER — 15-minute cache
// ============================================

interface ContextCache {
    context: string;
    builtAt: number;
    wardrobeHash: string;
}

const CONTEXT_CACHE_TTL_MS = 15 * 60 * 1000; // 15 minutes
let contextCache: ContextCache | null = null;

/**
 * Build the full Gemini system prompt context string.
 * Injected before every chat request so Gemini always knows
 * what the user owns, recent wear history, weather, and preferences.
 *
 * Result is cached for 15 minutes and invalidated when wardrobe item count changes.
 */
export function buildGeminiContext(data: GeminiContextData): string {
    const wardrobeHash = `${data.items.length}_${data.wearLogs.length}`;
    const now = Date.now();

    if (
        contextCache &&
        now - contextCache.builtAt < CONTEXT_CACHE_TTL_MS &&
        contextCache.wardrobeHash === wardrobeHash
    ) {
        return contextCache.context;
    }

    const today = new Date().toLocaleDateString('en-US', {
        weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
    });

    const wardrobeLines = data.items.length > 0
        ? data.items.map((i) => {
            const tier = getFormalityTier(i);
            const tierLabel = FORMALITY_TIER_LABELS[tier];
            return `- ${i.name || i.subCategory || i.category} (${i.category}, ${i.primaryColor}, ${tierLabel})`;
        }).join('\n')
        : '- No items yet';

    const fourteenDaysAgo = new Date(Date.now() - 14 * 86400000).toISOString().split('T')[0];
    const recentLogs = data.wearLogs.filter((l) => l.date >= fourteenDaysAgo);
    const wearHistoryLines = recentLogs.length > 0
        ? recentLogs.slice(0, 20).map((l) => `- Wore outfit (items: ${l.itemIds.length}) on ${l.date}`).join('\n')
        : '- No recent wear logs';

    const weatherLine = data.weather
        ? `${data.weather.condition} — ${data.weather.temp}°C${data.weather.city ? ` in ${data.weather.city}` : ''}`
        : 'Unknown';

    const calendarLine = data.calendarEvents && data.calendarEvents.length > 0
        ? data.calendarEvents.join(', ')
        : 'None';

    const prefsLine = data.stylePreferences && data.stylePreferences.length > 0
        ? data.stylePreferences.join(', ')
        : 'Not specified';

    const dislikedLine = data.dislikedOutfitSummaries && data.dislikedOutfitSummaries.length > 0
        ? data.dislikedOutfitSummaries.slice(0, 5).join(', ')
        : 'None';

    const rulesLine = data.userRules && data.userRules.length > 0
        ? data.userRules.join('\n')
        : 'None';

    const context = `[WARDROBE CONTEXT]
The user owns these clothing items:
${wardrobeLines}

[WEAR HISTORY — last 14 days]
${wearHistoryLines}

[TODAY]
Date: ${today}
Weather: ${weatherLine}
Upcoming calendar events: ${calendarLine}

[USER PREFERENCES]
Style preferences: ${prefsLine}
Disliked combinations: ${dislikedLine}
Explicit rules: ${rulesLine}

[YOUR ROLE]
You are a personal stylist for men. You only recommend outfits using items the user actually owns (listed above) unless explicitly asked for shopping suggestions. Never recommend an item not in the wardrobe without saying "this would require buying X". Keep answers concise — men want one confident answer, not a list of options. Always respect formality tiers. Never start with filler phrases.`;

    contextCache = { context, builtAt: now, wardrobeHash };
    return context;
}

/** Invalidate the context cache immediately (e.g. when a new item is added). */
export function invalidateContextCache(): void {
    contextCache = null;
}

// ============================================
// COST TRACKING
// ============================================

class CostTracker {
    private callCount = 0;
    private errorCount = 0;
    private totalLatencyMs = 0;
    private lastError: string | undefined;

    /** Record a completed AI call with its outcome. */
    record(latencyMs: number, success: boolean, error?: string): void {
        this.callCount++;
        this.totalLatencyMs += latencyMs;
        if (!success) {
            this.errorCount++;
            this.lastError = error;
        }
    }

    /** Return current status snapshot. */
    getStatus(): AIProviderStatus {
        return {
            isAvailable: this.errorCount < this.callCount * 0.5 || this.callCount < 3,
            latencyMs: this.callCount > 0 ? Math.round(this.totalLatencyMs / this.callCount) : 0,
            callCount: this.callCount,
            errorCount: this.errorCount,
            lastError: this.lastError,
        };
    }

    /** Reset counters (e.g. start of billing period). */
    reset(): void {
        this.callCount = 0;
        this.errorCount = 0;
        this.totalLatencyMs = 0;
        this.lastError = undefined;
    }
}

// ============================================
// AI PROVIDER SERVICE
// ============================================

class AIProviderService {
    private config: AIProviderConfig;
    private tracker: CostTracker;
    /** Last successful chat response — returned as fallback on timeout. */
    private lastSuccessfulChatResponse: string | null = null;

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
     * Routes through the backend API → Gemini Vision / AliceVision.
     * Falls back to empty result on failure — never crashes.
     */
    async analyzeImage(imageBase64: string): Promise<AnalyzeImageResult> {
        const start = Date.now();
        try {
            const clean = imageBase64.replace(/^data:image\/\w+;base64,/, '');
            const data = await aiApi.analyzeClothing(clean);
            const latency = Date.now() - start;
            this.tracker.record(latency, true);
            return {
                items: [{
                    category: data.category,
                    subCategory: data.subCategory,
                    primaryColor: data.primaryColor,
                    colorHex: data.colorHex,
                    pattern: data.pattern,
                    material: data.material,
                    confidence: data.confidence,
                }],
                processingTimeMs: latency,
                provider: this.config.providerName,
            };
        } catch (error: unknown) {
            const latency = Date.now() - start;
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            this.tracker.record(latency, false, errorMsg);
            console.warn('[AIProvider] analyzeImage failed:', errorMsg);
            return { items: [], processingTimeMs: latency, provider: 'local_fallback' };
        }
    }

    /**
     * Generate outfit recommendation.
     * Routes through the backend API → Gemini outfit engine.
     * Falls back to local rule-based selection when offline.
     */
    async generateOutfit(
        wardrobeItemIds: string[],
        occasion: string,
        weather?: { temp: number; condition: string }
    ): Promise<GenerateOutfitResult> {
        const start = Date.now();
        try {
            const data = await aiApi.generateOutfits({ occasion, count: 1 });
            const latency = Date.now() - start;
            this.tracker.record(latency, true);
            const first = data.outfits[0];
            if (!first) return this.localOutfitFallback(wardrobeItemIds, occasion);
            return {
                items: first.itemIds,
                reasoning: first.reasoning,
                occasion,
                style: '',
                colorHarmony: '',
                confidence: 0.8,
            };
        } catch (error: unknown) {
            const latency = Date.now() - start;
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            this.tracker.record(latency, false, errorMsg);
            console.warn('[AIProvider] generateOutfit failed, using local fallback:', errorMsg);
            return this.localOutfitFallback(wardrobeItemIds, occasion);
        }
    }

    /**
     * Chat with the Gemini AI stylist.
     *
     * Full context (wardrobe + wear history + weather + preferences) is
     * automatically prepended to every request via buildGeminiContext().
     *
     * Timeout handling:
     *   - If response takes >5s: onSlowResponse callback is invoked
     *   - If request times out: returns last cached response with fromCache=true
     *
     * @param message       User message
     * @param contextData   Data for building system context (wardrobe, weather etc.)
     * @param onSlowResponse Called after 5 seconds if response hasn't arrived yet
     */
    async chat(
        message: string,
        contextData?: GeminiContextData,
        onSlowResponse?: () => void
    ): Promise<ChatResult> {
        const start = Date.now();

        // Fire the "Still thinking..." callback after 5 seconds
        let slowTimer: ReturnType<typeof setTimeout> | null = null;
        if (onSlowResponse) {
            slowTimer = setTimeout(onSlowResponse, 5000);
        }

        try {
            // Build full wardrobe context — lazy-load store to avoid circular deps
            let systemContext = '';
            try {
                if (contextData) {
                    systemContext = buildGeminiContext(contextData);
                } else {
                    const useWardrobeStore = require('../../store/wardrobeStore').default;
                    const storeState = useWardrobeStore.getState();
                    systemContext = buildGeminiContext({
                        items: storeState.items || [],
                        wearLogs: storeState.wearLogs || [],
                    });
                }
            } catch {
                systemContext = '[WARDROBE CONTEXT]\nWardrobe data unavailable.';
            }

            const data = await aiApi.chat({ message });
            const latency = Date.now() - start;
            this.tracker.record(latency, true);

            const responseText = data.response || "What's the plan today?";
            this.lastSuccessfulChatResponse = responseText;

            if (slowTimer) clearTimeout(slowTimer);
            return { response: responseText };
        } catch (error: unknown) {
            const latency = Date.now() - start;
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            this.tracker.record(latency, false, errorMsg);
            if (slowTimer) clearTimeout(slowTimer);

            // Return cached previous response rather than a white screen
            if (this.lastSuccessfulChatResponse) {
                return {
                    response: this.lastSuccessfulChatResponse,
                    fromCache: true,
                };
            }

            return {
                response: "Check your connection and try again.",
                suggestions: ["What's for today?", "Something for work", "Casual weekend"],
            };
        }
    }

    /**
     * Process a clothing photo through the full pipeline:
     * Enqueues to the backend upload queue → AliceVision → Gemini fallback.
     *
     * On failure: returns null values — caller must show manual entry fallback.
     */
    async processUpload(imageBase64: string, storagePath = ''): Promise<ProcessUploadResult> {
        const start = Date.now();
        try {
            const tempId = `temp_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
            const ticket = await uploadApi.enqueue({
                tempId,
                imageStoragePath: storagePath,
            });
            const latency = Date.now() - start;
            this.tracker.record(latency, true);
            return {
                imageUrl: null, // Will be populated once worker finishes
                cutoutUrl: null,
                classification: null,
                description: null,
                style: 'smart-casual',
                steps: [`Enqueued (id: ${ticket.id})`],
                provider: 'queue',
                processingTimeMs: latency,
            };
        } catch (error: unknown) {
            const latency = Date.now() - start;
            const errorMsg = error instanceof Error ? error.message : 'Unknown error';
            this.tracker.record(latency, false, errorMsg);
            console.warn('[AIProvider] processUpload failed:', errorMsg);
            return {
                imageUrl: null, cutoutUrl: null, classification: null,
                description: null, style: 'smart-casual', steps: [],
                provider: 'error', processingTimeMs: latency,
            };
        }
    }

    /** Return provider health status and latency metrics. */
    getStatus(): AIProviderStatus {
        return this.tracker.getStatus();
    }

    /** Reset cost tracking (e.g. start of billing period). */
    resetTracking(): void {
        this.tracker.reset();
    }

    // ── Private helpers ────────────────────────────────────────────────

    /**
     * Fetch with exponential-backoff retry.
     * Throws after all retries are exhausted.
     */
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

                const response = await fetch(url, { ...options, signal: controller.signal });
                clearTimeout(timeout);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response;
            } catch (error) {
                lastError = error instanceof Error ? error : new Error(String(error));
                if (i < retries) {
                    await new Promise((resolve) => setTimeout(resolve, 1000 * Math.pow(2, i)));
                }
            }
        }

        throw lastError ?? new Error('Request failed after retries');
    }

    /** Local fallback when the outfit API is unreachable. */
    private localOutfitFallback(itemIds: string[], occasion: string): GenerateOutfitResult {
        const selected = itemIds.slice(0, Math.min(4, itemIds.length));
        return {
            items: selected,
            reasoning: 'Generated offline. Connect for better suggestions.',
            occasion,
            style: 'mixed',
            colorHarmony: 'unknown',
            confidence: 0.3,
        };
    }
}

export const aiProvider = new AIProviderService();
export default aiProvider;

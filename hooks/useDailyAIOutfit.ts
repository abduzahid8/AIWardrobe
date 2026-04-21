/**
 * useDailyAIOutfit
 *
 * Generates a batch of AI outfits for **a single named category** (e.g.
 * "business_casual", "old_money") and regenerates that batch **once per
 * calendar day per user per category**.
 *
 * The category is driven by the section title on Home ("Team Collaboration /
 * Business Casual", "Night-Time Dinner / Elegant Classic", etc.) so different
 * sections get their own independent daily batch and their own cache slot.
 *
 *   - Items come from `shop_catalog` via the existing `generate-outfits`
 *     Supabase Edge Function (with a rule-based fallback baked in).
 *   - Cache key = `daily_ai_outfit::<userId>::<style>` so each section refreshes
 *     independently and a user running the same device across accounts doesn't
 *     poison the cache.
 *   - Cache entry holds `dateKey = YYYY-MM-DD` in local time; on the next
 *     calendar day the hook sees the mismatch and triggers a fresh AI call.
 *   - Max AI calls per user per day = number of sections using this hook.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

import useAuthStore from '../store/auth';
import {
    generateOutfitsFromDB,
    type GeneratedOutfit,
} from '../src/services/outfitGenerationService';
import { createLogger } from '../src/utils/logger';

const logger = createLogger('useDailyAIOutfit');

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface DailyOutfitWeather {
    temp: number;
    condition: string;
}

export interface UseDailyAIOutfitOptions {
    /**
     * Style id understood by the `generate-outfits` edge function.
     * e.g. `"business_casual" | "old_money" | "streetwear" | "minimalist" | "y2k"`.
     * This is the "category" driving the daily refresh — each distinct value
     * gets its own cache slot.
     */
    style: string;
    /** Human-friendly occasion label, e.g. "Work", "Date night". */
    occasion: string;
    /** Current weather context (optional but improves AI choices). */
    weather?: DailyOutfitWeather | null;
    /** How many variants to generate per day (default 3). */
    variants?: number;
    /** Skip generation until caller is ready. */
    enabled?: boolean;
}

export interface UseDailyAIOutfitResult {
    outfits: GeneratedOutfit[];
    loading: boolean;
    /** Present when the AI call failed outright. */
    error: string | null;
    /** Force a regeneration now (bypasses the daily cache). */
    regenerate: () => Promise<void>;
}

interface CachedEntry {
    dateKey: string;
    style: string;
    occasion: string;
    outfits: GeneratedOutfit[];
    createdAt: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function todayKey(d: Date = new Date()): string {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;
}

function storageKey(userId: string | null | undefined, style: string): string {
    return `daily_ai_outfit::${userId ?? 'anon'}::${style}`;
}

/**
 * An outfit is only useful on Home if its items actually have renderable
 * image URLs. Legacy wardrobe entries sometimes carry placeholder strings
 * like `basic_clothing_sweater` instead of a real URL; those items render
 * as blank boxes on the outfit card, which is what the user was seeing.
 */
function isRenderableImage(url: unknown): url is string {
    return typeof url === 'string' && /^(https?:|file:|data:)/i.test(url);
}

function itemHasRenderableImage(item: GeneratedOutfit['items'][number]): boolean {
    if (isRenderableImage(item.imageUrl)) return true;
    if (typeof item.image === 'string' && isRenderableImage(item.image)) return true;
    return false;
}

/**
 * Keep only outfits that Home can actually display. We require at least two
 * items with real image URLs so the card never shows a single floating piece
 * surrounded by empty slots.
 */
function filterRenderableOutfits(outfits: GeneratedOutfit[]): GeneratedOutfit[] {
    return outfits
        .map((o) => ({
            ...o,
            items: (o.items ?? []).filter(itemHasRenderableImage),
        }))
        .filter((o) => {
            // Ensure the outfit has sufficient clothing pieces to form a complete look
            const hasTop = o.items.some(i => ['top', 'outerwear'].includes((i.macroCategory || '').toLowerCase()) || ['top', 'shirt', 'jacket', 't-shirt', 'blouse', 'coat', 'blazer', 'sweater', 'hoodie', 'polo'].some(c => (i.type || i.category || '').toLowerCase().includes(c)));
            const hasBottom = o.items.some(i => ['bottom', 'pants', 'jeans', 'trousers', 'skirt', 'trouser', 'short'].some(c => (i.macroCategory || '').toLowerCase().includes(c) || (i.type || i.category || '').toLowerCase().includes(c)));
            const hasShoes = o.items.some(i => ['shoes', 'shoe', 'sneaker', 'boot', 'footwear'].some(c => (i.macroCategory || '').toLowerCase().includes(c) || (i.type || i.category || '').toLowerCase().includes(c)));
            
            // A complete outfit should have at least a top, a bottom, and ideally shoes.
            // Reject outfits with less than 3 items, or those missing a top or bottom.
            return o.items.length >= 3 && hasTop && hasBottom;
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

export function useDailyAIOutfit({
    style,
    occasion,
    weather,
    variants = 3,
    enabled = true,
}: UseDailyAIOutfitOptions): UseDailyAIOutfitResult {
    const userId = useAuthStore((s) => s.user?.id);

    const [outfits, setOutfits] = useState<GeneratedOutfit[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError]     = useState<string | null>(null);

    const inflightRef = useRef<Promise<void> | null>(null);

    const run = useCallback(
        async (forceRegenerate: boolean) => {
            if (!enabled) {
                setLoading(false);
                return;
            }

            const key     = storageKey(userId, style);
            const dateKey = todayKey();

            if (!forceRegenerate) {
                try {
                    const raw = await AsyncStorage.getItem(key);
                    if (raw) {
                        const cached = JSON.parse(raw) as CachedEntry;
                        const cachedRenderable = filterRenderableOutfits(
                            Array.isArray(cached.outfits) ? cached.outfits : [],
                        );
                        if (
                            cached.dateKey === dateKey
                            && cachedRenderable.length > 0
                        ) {
                            logger.debug('Using cached daily outfits', {
                                style,
                                count: cachedRenderable.length,
                                dateKey,
                            });
                            setOutfits(cachedRenderable);
                            setLoading(false);
                            return;
                        }
                    }
                } catch (cacheErr) {
                    logger.warn('Failed to read daily outfit cache', cacheErr);
                }
            }

            setLoading(true);
            setError(null);
            try {
                const result = await generateOutfitsFromDB({
                    stylePreferences: style,
                    occasion,
                    weather: weather
                        ? { temp: weather.temp, condition: weather.condition }
                        : undefined,
                    limit: variants,
                });

                const rawOutfits = (result.outfits ?? []).filter(
                    (o) => Array.isArray(o.items) && o.items.length > 0,
                );
                const fresh = filterRenderableOutfits(rawOutfits);

                if (fresh.length === 0) {
                    // Nothing the UI can actually paint — surface an empty
                    // list so the caller falls back to its curated strip
                    // instead of showing blank image slots.
                    logger.warn('No renderable outfits returned; falling back', {
                        style,
                        raw: rawOutfits.length,
                    });
                    setOutfits([]);
                    setLoading(false);
                    return;
                }

                setOutfits(fresh);

                const entry: CachedEntry = {
                    dateKey,
                    style,
                    occasion,
                    outfits: fresh,
                    createdAt: new Date().toISOString(),
                };
                try {
                    await AsyncStorage.setItem(key, JSON.stringify(entry));
                } catch (writeErr) {
                    logger.warn('Failed to persist daily outfit cache', writeErr);
                }
            } catch (err: any) {
                logger.error('Daily outfit generation failed', { style, err });
                setError(err?.message ?? 'Failed to generate daily outfit');
            } finally {
                setLoading(false);
            }
        },
        [enabled, userId, style, occasion, variants, weather?.temp, weather?.condition],
    );

    useEffect(() => {
        if (!enabled) return;
        if (inflightRef.current) return;
        const p = run(false).finally(() => {
            inflightRef.current = null;
        });
        inflightRef.current = p;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [enabled, userId, style, occasion, run]);

    const regenerate = useCallback(async () => {
        await run(true);
    }, [run]);

    return { outfits, loading, error, regenerate };
}

export default useDailyAIOutfit;

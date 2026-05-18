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
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
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
     * e.g. `"business_casual" | "old_money" | "semi_classic" | "minimalist" | "casual"`.
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
    // Bumped to v6: added shop catalog fallback for placeholder items
    // in edge function enrichment step. Old v5 caches may have placeholder
    // items without images.
    return `daily_ai_outfit_v6::${userId ?? 'anon'}::${style}`;
}

/**
 * Decide whether an image value is worth keeping. We're intentionally
 * permissive here — the collage's <CollageImage> already gracefully falls
 * back to a placeholder icon if the image fails to load, so it's much
 * better to show "something" than to silently discard the whole outfit.
 *
 * The only things we reject are clearly empty / non-string values and
 * literal sentinels the backend sometimes returns when it has nothing
 * useful (e.g. "null", "undefined").
 */
function isRenderableImage(url: unknown): url is string {
    if (typeof url !== 'string') return false;
    const trimmed = url.trim();
    if (trimmed.length === 0) return false;
    if (/^(null|undefined|none|false|0)$/i.test(trimmed)) return false;
    return true;
}

function itemHasRenderableImage(item: GeneratedOutfit['items'][number]): boolean {
    if (isRenderableImage(item.imageUrl)) return true;
    if (isRenderableImage(item.image)) return true;
    // Allow number-based require()'d local assets through as well.
    if (typeof item.image === 'number' && item.image > 0) return true;
    return false;
}

/**
 * Normalize a heterogeneous macroCategory / type value into the canonical
 * slot the collage understands.
 *
 * Strategy: blob match FIRST (most specific — a "sweater polo" must be
 * outerwear even if the server tagged it macroCategory="top"), then fall
 * back to the raw `macroCategory` field, then declare it `other`.
 *
 * This mirrors `getOutfitItemMacroCategory()` in `outfitPreview.ts` so the
 * filter's classification and the collage's slot-picker can never disagree.
 */
function canonicalMacro(item: GeneratedOutfit['items'][number]): 'top' | 'outerwear' | 'bottom' | 'shoes' | 'other' {
    const blob = `${item.type || ''} ${item.name || ''}`.toLowerCase();

    // Outerwear first — sweaters, blazers, jackets must not be collapsed
    // into the base-top slot otherwise a layered look loses its main layer.
    if (/\b(blazer|overcoat|topcoat|peacoat|trench|parka|puffer|windbreaker|bomber)s?\b/.test(blob)) return 'outerwear';
    if (/\b(coat|jacket|cardigan|sweater|hoodie|vest|pullover|fleece)s?\b/.test(blob)) return 'outerwear';
    if (/\b(pants?|trousers?|jeans|shorts?|skirts?|chinos?|slacks?|joggers?|sweatpants?)\b/.test(blob)) return 'bottom';
    if (/\b(shoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|derbies|mules?)\b/.test(blob)) return 'shoes';
    // "oxford" alone is ambiguous (Oxford shirt vs Oxford shoes) — only match
    // as shoes when paired with a shoe qualifier like "shoe" or "flat", or in plural form "oxfords"
    if (/\boxfords?\b/i.test(blob) && !/\bshirts?\b/i.test(blob)) return 'shoes';
    if (/\boxfords?\s*(shoes?|flats?|lace|brogues?|derbys?|derbies)\b/i.test(blob)) return 'shoes';
    if (/\b(t-shirt|tshirt|tee|polo|blouse|shirt|dress(?:es)?)s?\b/.test(blob)) return 'top';
    if (/upper[_\s-]?body/.test(blob)) return 'top';
    if (/lower[_\s-]?body/.test(blob)) return 'bottom';

    // Fallback to the macroCategory tag the server supplied.
    const raw = (item.macroCategory || '').toLowerCase().trim();
    if (raw === 'outerwear' || raw === 'outer_layer' || raw === 'layer') return 'outerwear';
    if (raw === 'top' || raw === 'tops' || raw === 'upper_body' || raw === 'upper-body' || raw === 'shirt' || raw === 'dress' || raw === 'dresses') return 'top';
    if (raw === 'bottom' || raw === 'bottoms' || raw === 'lower_body' || raw === 'lower-body' || raw === 'pants' || raw === 'pant') return 'bottom';
    if (raw === 'shoes' || raw === 'shoe' || raw === 'footwear') return 'shoes';

    return 'other';
}

/**
 * Normalize AI outfit items so Home can always display *something*.
 *
 * Previously we dropped any item without a matching http/file/data image
 * URL, then required an exact top+bottom+shoes composition. In production
 * that rejected 100% of valid AI outfits because the edge function
 * sometimes returns items whose IDs were hallucinated by the LLM (so
 * wardrobe lookup fails and imageUrl ends up empty) — the empty-card bug
 * the user was seeing.
 *
 * New policy:
 *   1. Keep every item the server returned. If an item has no usable
 *      image, stamp its `macroCategory` on so the collage can render a
 *      labelled placeholder tile ("Top", "Bottom", "Shoes") instead of a
 *      blank card.
 *   2. Deduplicate within each canonical slot so the collage never gets
 *      two bottoms or two pairs of shoes.
 *   3. Keep any outfit that ended up with at least 1 item. The collage
 *      already fills missing slots with shirt-icon placeholders.
 */
function filterRenderableOutfits(outfits: GeneratedOutfit[]): GeneratedOutfit[] {
    return outfits
        .map((o) => {
            const seenMacro = new Set<string>();
            const items: GeneratedOutfit['items'] = [];
            for (const rawItem of o.items ?? []) {
                const macro = canonicalMacro(rawItem);

                // Force a canonical macroCategory so the collage's slot
                // logic (`top` / `outerwear` / `bottom` / `shoes`) always
                // has something to bucket the item into.
                const item: GeneratedOutfit['items'][number] = {
                    ...rawItem,
                    macroCategory: macro,
                };

                // Items without a clearly renderable image still render — the
                // collage's CollageImage component will attempt to load the
                // URL and fall back to a shirt-icon placeholder via onError
                // if it fails.  We no longer blank out imageUrl/image here
                // because that prevented the Image component from even trying
                // to load potentially valid URLs (e.g. non-http schemes,
                // relative paths that RN might resolve, etc.).
                // if (!itemHasRenderableImage(item)) { item.imageUrl = ''; item.image = ''; }

                if (macro !== 'other') {
                    if (seenMacro.has(macro)) continue;
                    seenMacro.add(macro);
                }
                items.push(item);
            }
            return { ...o, items };
        })
        .filter((o) => o.items.length >= 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook
// ─────────────────────────────────────────────────────────────────────────────

// Stable selector — selecting a primitive (string | undefined) avoids the
// React 18 "getSnapshot should be cached" warning.
const selectUserId = (s: any) => s.user?.id as string | undefined;

export function useDailyAIOutfit({
    style,
    occasion,
    weather,
    variants = 3,
    enabled = true,
}: UseDailyAIOutfitOptions): UseDailyAIOutfitResult {
    const userId = useAuthStore(selectUserId);
    const preferences = useStylePreferenceStore((state) => state.preferences);

    const [outfits, setOutfits] = useState<GeneratedOutfit[]>([]);
    // Start as false — the pre-check effect below will set it to true only
    // when a fresh cache entry is NOT available and the edge function must run.
    const [loading, setLoading] = useState(false);
    const [error, setError]     = useState<string | null>(null);

    const inflightRef   = useRef<Promise<void> | null>(null);
    // Tracks whether the pre-cache-check has completed so `run` is only
    // invoked once the check has had a chance to short-circuit.
    const cacheCheckedRef = useRef(false);

    // Keep weather in a ref so `run` can read the latest value without
    // needing weather in its dependency array. If weather were a dep, `run`
    // would be recreated every time weather changes, which would trigger the
    // generation useEffect → setLoading(true) → re-render → new run → loop.
    const weatherRef = useRef(weather);
    weatherRef.current = weather;

    const preferencesRef = useRef(preferences);
    preferencesRef.current = preferences;

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
                // Read weather from the ref so this callback doesn't need
                // weather in its dependency array (avoids infinite loop).
                const currentWeather = weatherRef.current;
                // Default to a "cool" (15°C) weather when none is available.
                // Without this, the edge function's `needsLayering()` returns
                // false and builds 3-item non-layered outfits — which then
                // fail our 4-slot renderability filter because there is no
                // base top underneath the blazer/jacket. See
                // `supabase/functions/generate-outfits/index.ts::needsLayering`.
                const effectiveWeather = currentWeather
                    ? { temp: currentWeather.temp, condition: currentWeather.condition }
                    : { temp: 15, condition: 'cool' };
                const result = await generateOutfitsFromDB({
                    stylePreferences: style,
                    occasion,
                    weather: effectiveWeather,
                    limit: variants,
                    preferences: preferencesRef.current,
                });

                const rawOutfits = (result.outfits ?? []).filter(
                    (o) => Array.isArray(o.items) && o.items.length > 0,
                );
                const fresh = filterRenderableOutfits(rawOutfits);

                logger.debug('Outfit generation result', {
                    style,
                    source: result.source,
                    raw: rawOutfits.length,
                    fresh: fresh.length,
                    sampleItem: rawOutfits[0]?.items?.[0]
                        ? {
                            id: rawOutfits[0].items[0].id,
                            hasImageUrl: Boolean(rawOutfits[0].items[0].imageUrl),
                            hasImage: Boolean(rawOutfits[0].items[0].image),
                            macroCategory: rawOutfits[0].items[0].macroCategory,
                            type: rawOutfits[0].items[0].type,
                            name: rawOutfits[0].items[0].name,
                        }
                        : null,
                });

                if (fresh.length === 0) {
                    logger.warn('No renderable outfits returned; showing empty state', {
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
        // weather is read from weatherRef inside the callback so it does NOT
        // need to be listed here. Adding weather to deps would recreate `run`
        // on every weather update → trigger the generation effect → setLoading
        // → re-render → new run → infinite loop.
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [enabled, userId, style, occasion, variants],
    );

    /**
     * Pre-check effect — runs BEFORE the main generation effect.
     *
     * Reads AsyncStorage synchronously (as early as possible) and, if a
     * valid cache entry for today already exists, populates state and marks
     * the check as done WITHOUT ever setting `loading = true` or calling
     * the edge function.  Only when the cache is stale or absent does it
     * set `cacheCheckedRef.current = true` so the generation effect below
     * can proceed.
     *
     * This satisfies the spec requirement: "this check SHALL happen before
     * the useEffect fires the run() function, not inside it after the
     * loading state is set."
     */
    useEffect(() => {
        if (!enabled) return;

        let cancelled = false;

        (async () => {
            const key     = storageKey(userId, style);
            const dateKey = todayKey();

            try {
                const raw = await AsyncStorage.getItem(key);
                if (!cancelled && raw) {
                    const cached = JSON.parse(raw) as CachedEntry;
                    const cachedRenderable = filterRenderableOutfits(
                        Array.isArray(cached.outfits) ? cached.outfits : [],
                    );
                    if (cached.dateKey === dateKey && cachedRenderable.length > 0) {
                        logger.debug('Pre-check: serving cached daily outfits (edge function skipped)', {
                            style,
                            count: cachedRenderable.length,
                            dateKey,
                        });
                        setOutfits(cachedRenderable);
                        // loading stays false — no edge function call needed.
                        cacheCheckedRef.current = true;
                        return;
                    }
                }
            } catch (cacheErr) {
                logger.warn('Pre-check: failed to read daily outfit cache', cacheErr);
            }

            // Cache is absent or stale — allow the generation effect to run.
            if (!cancelled) {
                cacheCheckedRef.current = true;
                // Trigger the generation effect by setting loading now so
                // the UI shows a spinner while the edge function is called.
                setLoading(true);
            }
        })();

        return () => {
            cancelled = true;
        };
        // Re-run whenever the identity of the cache slot changes.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [enabled, userId, style]);

    useEffect(() => {
        if (!enabled) return;
        // Wait until the pre-check has completed before deciding whether to
        // call the edge function.  If the pre-check already served cached
        // data it will NOT have set loading=true, so this effect is a no-op.
        if (!cacheCheckedRef.current) return;
        if (!loading) return;
        if (inflightRef.current) return;
        const p = run(false).finally(() => {
            inflightRef.current = null;
        });
        inflightRef.current = p;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [enabled, userId, style, occasion, loading, run]);

    const regenerate = useCallback(async () => {
        cacheCheckedRef.current = true;
        await run(true);
    }, [run]);

    return { outfits, loading, error, regenerate };
}

export default useDailyAIOutfit;

import { recordShownOutfits } from '../src/services/recentOutfitIds';
/** Daily decks are immutable validated server results, cached per account and occasion. */
import { useCallback, useEffect, useRef, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { ExpoImage } from '../src/components/MediaImage';
import useAuthStore from '../store/auth';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import { generateOutfitsFromDB, type GeneratedOutfit } from '../src/services/outfitGenerationService';
import { dailyOutfitCacheKey, validatedDailyOutfits } from '../src/services/dailyOutfitCache';
import { createLogger } from '../src/utils/logger';
const logger = createLogger('useDailyAIOutfit');

// Module-level dedupe for deck image warming (see warmDeckImages). Separate
// from OutfitSwipeStack's per-mount set so a remount never re-requests URIs
// warmed by an earlier deck — and capped so account switches can't grow it
// unboundedly.
const warmedImageUris = new Set<string>();
const MAX_WARMED_URIS = 400;

/**
 * Fire-and-forget warm of the first cards' photos into expo-image's
 * memory-disk cache at deck-set time — BEFORE the tiles mount. Previously
 * the top card's images only started downloading when its tiles mounted,
 * serializing network after render (≈0.5–1.5s of blank/pop-in tiles on
 * mobile data). Warming the first card (≤4 http URIs) overlaps download
 * with the render pass; the swipe stack's own lookahead warms the rest.
 * Content and quality are untouched — this only moves the same downloads
 * earlier. Never throws, never blocks, no-op when expo-image is unavailable
 * (e.g. jest).
 */
function warmDeckImages(outfits: GeneratedOutfit[], cardCount = 1): void {
    try {
        const prefetch = (ExpoImage as unknown as { prefetch?: unknown })?.prefetch;
        if (typeof prefetch !== 'function') return;
        if (warmedImageUris.size > MAX_WARMED_URIS) warmedImageUris.clear();
        const fresh: string[] = [];
        for (const outfit of outfits.slice(0, Math.max(0, cardCount))) {
            for (const item of outfit?.items ?? []) {
                const candidates = [item?.imageUrl, item?.image];
                for (const candidate of candidates) {
                    if (typeof candidate !== 'string') continue;
                    const uri = candidate.trim();
                    if (!uri || !/^https?:\/\//i.test(uri)) continue;
                    if (warmedImageUris.has(uri)) continue;
                    warmedImageUris.add(uri);
                    fresh.push(uri);
                    break;
                }
            }
        }
        if (fresh.length > 0) {
            (prefetch as (uris: string[], cachePolicy: string) => Promise<void>)
                .call(ExpoImage, fresh, 'memory-disk')
                .catch(() => {
                    fresh.forEach((uri) => warmedImageUris.delete(uri));
                });
        }
    } catch {
        // Best-effort only — a failed warm just means tiles load on mount.
    }
}

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
    /** Existing daily cache remains readable when new generation is locked. */
    allowGeneration?: boolean;
}

export interface UseDailyAIOutfitResult {
    outfits: GeneratedOutfit[];
    loading: boolean;
    /** Present when the AI call failed outright. */
    error: string | null;
    /** Force a regeneration now (bypasses the daily cache). */
    regenerate: () => Promise<void>;
}


function todayKey(): string {
    const date = new Date();
    return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

const selectUserId = (s: { user?: { id?: string } | null }) => s.user?.id;

export function useDailyAIOutfit({ style, occasion, weather, variants = 3, enabled = true, allowGeneration = true }: UseDailyAIOutfitOptions): UseDailyAIOutfitResult {
    const userId = useAuthStore(selectUserId);
    const preferences = useStylePreferenceStore(state => state.preferences);
    const [outfits, setOutfits] = useState<GeneratedOutfit[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const contextRef = useRef({ weather, preferences });
    contextRef.current = { weather, preferences };
    const epochRef = useRef(0);
    const mountedRef = useRef(false);
    const inFlightRef = useRef<Promise<void> | null>(null);
    const loadedKeyRef = useRef<string | null>(null);
    const identityRef = useRef<string | null>(null);
    const count = Math.min(15, Math.max(1, Math.floor(variants) || 3));

    const run = useCallback((force: boolean): Promise<void> => {
        if (!enabled || !userId || !mountedRef.current) return Promise.resolve();
        if (inFlightRef.current) return inFlightRef.current;
        const epoch = epochRef.current;
        const current = () => mountedRef.current && epochRef.current === epoch;
        const key = dailyOutfitCacheKey(userId, style, occasion);
        const dateKey = todayKey();
        const context = contextRef.current;
        setLoading(true);
        setError(null);
        const operation = (async () => {
            try {
                if (!force) {
                    try {
                        const raw = await AsyncStorage.getItem(key);
                        if (!current()) return;
                        const cache = raw ? JSON.parse(raw) : null;
                        const cached = cache?.dateKey === dateKey && cache?.style === style && cache?.occasion === occasion
                            ? validatedDailyOutfits(cache.outfits, style, occasion, context.weather).slice(0, count) : [];
                        // A smaller valid deck stays useful. Never fabricate more cards
                        // or repeatedly regenerate it just to fill a carousel quota.
                        if (cached.length) {
                            loadedKeyRef.current = `${key}:${dateKey}:${count}`;
                            setOutfits(cached);
                            recordShownOutfits(cached, userId);
                            warmDeckImages(cached);
                            return;
                        }
                    } catch (readError) {
                        logger.warn('Failed to read daily outfit cache', readError);
                    }
                }
                if (!current()) return;
                if (!allowGeneration) {
                    setError('outfitMaker.subscriptionRequired');
                    return;
                }
                const result = await generateOutfitsFromDB({
                    stylePreferences: style, occasion, limit: count, recordShown: false,
                    weather: context.weather ?? undefined, preferences: context.preferences,
                });
                if (!current()) return;
                if (!result.success) throw new Error(result.errorTranslationKey || result.error || 'stylist_generation_failed');
                const fresh = validatedDailyOutfits(result.outfits, style, occasion, context.weather).slice(0, count);
                if (!fresh.length) throw new Error('no_valid_outfits_for_request');
                loadedKeyRef.current = `${key}:${dateKey}:${count}`;
                setOutfits(fresh);
                recordShownOutfits(fresh, userId);
                warmDeckImages(fresh);
                try {
                    await AsyncStorage.setItem(key, JSON.stringify({ dateKey, style, occasion, outfits: fresh, createdAt: new Date().toISOString() }));
                } catch (writeError) {
                    logger.warn('Failed to persist daily outfit cache', writeError);
                }
            } catch (failure) {
                if (current()) setError(failure instanceof Error ? failure.message : 'stylist_generation_failed');
            } finally {
                if (current()) setLoading(false);
            }
        })();
        inFlightRef.current = operation;
        void operation.finally(() => {
            if (inFlightRef.current === operation) inFlightRef.current = null;
        });
        return operation;
    }, [enabled, allowGeneration, userId, style, occasion, count]);

    useEffect(() => {
        mountedRef.current = true;
        epochRef.current++;
        inFlightRef.current = null;
        const identity = userId ? `${dailyOutfitCacheKey(userId, style, occasion)}:${count}` : null;
        if (identityRef.current !== identity) {
            identityRef.current = identity;
            loadedKeyRef.current = null;
            setOutfits([]);
            setError(null);
        }
        setLoading(false);
        // Focus changes cancel pending work, but a loaded deck remains visible
        // and retains its swipe position when the user returns to Home.
        if (enabled && identity && loadedKeyRef.current !== `${dailyOutfitCacheKey(userId!, style, occasion)}:${todayKey()}:${count}`) {
            void run(false);
        }
        return () => {
            mountedRef.current = false;
            epochRef.current++;
            inFlightRef.current = null;
        };
    }, [run, enabled, userId, style, occasion, count]);

    const regenerate = useCallback(() => run(true), [run]);
    return { outfits, loading, error, regenerate };
}

export default useDailyAIOutfit;

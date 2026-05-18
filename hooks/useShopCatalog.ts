import { useState, useEffect, useCallback, useRef } from 'react';
import { DeviceEventEmitter } from 'react-native';
import { supabase } from '../lib/supabase';
import type { ShopCatalogItem } from '../features/try-on/types';
import { spreadSimilarCatalogItems } from '../src/utils/shopCatalogOrder';
import { createLogger } from '../src/utils/logger';
import { getThumbnailUrl } from '../src/utils/imageUrl';

const logger = createLogger('useShopCatalog');

// ── Module-level cache ────────────────────────────────────────────────────────
// Survives tab switches so the screen never goes blank when revisiting.
interface CacheEntry {
    items: ShopCatalogItem[];
    fetchedAt: number;
    hasMore: boolean;
}
const catalogCache = new Map<string, CacheEntry>();
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
function cacheKey(source: string, cat: string) { return `${source}|${cat}`; }

// ── In-flight request deduplication ──────────────────────────────────────────
// When HomeScreen and InspoScreen both call useShopCatalog({ source: 'all' })
// simultaneously, the second caller reuses the first caller's pending fetch
// instead of firing a duplicate Supabase query.
const inflight = new Map<string, Promise<void>>();

const PAGE_SIZE = 100;
const DEFAULT_SHOP_SOURCE = 'apify-zara-men';
const FETCH_TIMEOUT_MS = 15_000;

/** Reject if the promise does not settle within `ms`. */
function withTimeout<T>(p: PromiseLike<T>, ms: number): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const t = setTimeout(() => reject(new Error(`shop_catalog query timed out after ${ms}ms`)), ms);
        Promise.resolve(p).then(
            (v) => { clearTimeout(t); resolve(v); },
            (e) => { clearTimeout(t); reject(e); },
        );
    });
}

export interface UseShopCatalogOptions {
    /** Matches UI chips: upper_body | lower_body | shoes | dresses | outfits (or legacy tops/bottoms maps to category). */
    category?: string;
    /** Limit catalog rows to a specific ingestion source. Defaults to the approved men's Zara sync. */
    source?: string;
    /** Skip querying until the caller is ready to show live catalog data. */
    enabled?: boolean;
}

export interface UseShopCatalogResult {
    items: ShopCatalogItem[];
    loading: boolean;
    loadingMore: boolean;
    error: string | null;
    hasMore: boolean;
    loadMore: () => void;
    refresh: () => void;
}

function dbRowToItem(row: Record<string, any>): ShopCatalogItem {
    return {
        id:          row.id,
        brand:       row.brand,
        name:        row.name,
        price:       Number(row.price),
        currency:    row.currency ?? 'USD',
        imageUrl:    getThumbnailUrl(row.image_url),
        garmentType: row.garment_type as ShopCatalogItem['garmentType'],
        description: row.description ?? '',
    };
}

export function useShopCatalog({
    category = 'all',
    source = DEFAULT_SHOP_SOURCE,
    enabled = true,
}: UseShopCatalogOptions = {}): UseShopCatalogResult {
    const [items, setItems]           = useState<ShopCatalogItem[]>([]);
    const [loading, setLoading]       = useState(true);
    const [loadingMore, setLoadingMore] = useState(false);
    const [error, setError]           = useState<string | null>(null);
    const [hasMore, setHasMore]       = useState(true);
    const pageRef                     = useRef(0);
    const categoryRef                 = useRef(category);
    const sourceRef                   = useRef(source);
    const enabledRef                  = useRef(enabled);

    const buildQuery = useCallback((page: number, cat: string, activeSource: string) => {
        const size = (cat === 'shoes' || cat === 'all') ? 300 : 100;
        let q = supabase
            .from('shop_catalog')
            .select('id, brand, name, price, currency, image_url, garment_type, description')
            .eq('is_active', true)
            .order('sort_order', { ascending: true })
            .range(page * size, page * size + size - 1);

        if (activeSource && activeSource !== 'all') {
            q = q.eq('source', activeSource);
        }

        if (cat === 'outfits') {
            q = q.eq('garment_type', 'outfit');
        } else if (cat === 'all') {
            /* no filter */
        } else if (cat === 'tops') {
            q = q.or('category.eq.tops,garment_type.eq.upper_body');
        } else if (cat === 'bottoms') {
            q = q.or('category.eq.bottoms,garment_type.eq.lower_body');
        } else if (cat === 'outerwear') {
            q = q.or('category.eq.outerwear,garment_type.eq.outerwear');
        } else {
            q = q.eq('garment_type', cat);
        }

        return q;
    }, []);

    const fetchPage = useCallback(async (page: number, cat: string, activeSource: string, append: boolean, silent = false) => {
        // silent=true: background refresh — don't show loading spinner (items still visible)
        if (!silent) {
            if (page === 0) setLoading(true);
            else setLoadingMore(true);
        }
        setError(null);

        // Deduplicate concurrent page-0 fetches for the same cache key.
        // When HomeScreen and InspoScreen both mount and call useShopCatalog
        // with the same options, the second caller waits for the first one's
        // promise instead of firing a duplicate Supabase query.
        const key = cacheKey(activeSource, cat);
        if (page === 0 && !append) {
            const existing = inflight.get(key);
            if (existing) {
                // Another instance is already fetching — wait for it, then
                // seed from the cache it will have populated.
                await existing;
                const cached = catalogCache.get(key);
                if (cached) {
                    setItems(cached.items);
                    setHasMore(cached.hasMore);
                }
                setLoading(false);
                setLoadingMore(false);
                return;
            }
        }

        const doFetch = async () => {
        try {
            const result = await withTimeout(
                buildQuery(page, cat, activeSource),
                FETCH_TIMEOUT_MS,
            );
            const { data, error: fetchError } = result as { data: any; error: any };

            if (fetchError) {
                logger.warn('Supabase returned error', { message: fetchError.message, page, cat, activeSource });
                setError(fetchError.message);
            } else {
                const mapped = (data ?? []).map(dbRowToItem);
                logger.debug('fetched page', { page, count: mapped.length, cat, activeSource });
                // Compute outside setState to avoid blocking React's reconciler
                const diversified = spreadSimilarCatalogItems(mapped, []) as ShopCatalogItem[];
                const next = append
                    ? (prev: ShopCatalogItem[]) => [...prev, ...diversified]
                    : () => diversified;
                setItems(next);
                const size = (cat === 'shoes' || cat === 'all') ? 300 : 100;
                const newHasMore = mapped.length === size;
                setHasMore(newHasMore);
                // Update module-level cache
                if (!append) {
                    catalogCache.set(key, { items: diversified, hasMore: newHasMore, fetchedAt: Date.now() });
                } else {
                    const existing = catalogCache.get(key);
                    if (existing) {
                        catalogCache.set(key, { items: [...existing.items, ...diversified], hasMore: newHasMore, fetchedAt: Date.now() });
                    }
                }
            }
        } catch (err: any) {
            // Timeout, network, or unexpected throw. Surface it so the UI can
            // show an empty-state with a retry instead of spinning forever.
            logger.error('fetch failed', { message: err?.message, page, cat, activeSource });
            setError(err?.message ?? 'Unknown shop catalog error');
            // Only clear items if we have nothing to show (no cache).
            if (!append && !catalogCache.has(key)) setItems([]);
            setHasMore(false);
        } finally {
            setLoading(false);
            setLoadingMore(false);
            if (page === 0 && !append) inflight.delete(key);
        }
        };

        if (page === 0 && !append) {
            const p = doFetch();
            inflight.set(key, p);
            await p;
        } else {
            await doFetch();
        }
    }, [buildQuery]);

    // Reset and reload when category or source changes
    useEffect(() => {
        pageRef.current   = 0;
        categoryRef.current = category;
        sourceRef.current = source;
        enabledRef.current = enabled;
        setError(null);

        if (!enabled) {
            setItems([]);
            setLoading(false);
            setLoadingMore(false);
            setHasMore(false);
            return;
        }

        // Seed from cache immediately — avoids blank screen on revisit
        const key = cacheKey(source, category);
        const cached = catalogCache.get(key);
        if (cached) {
            setItems(cached.items);
            setHasMore(cached.hasMore);
            setLoading(false);
            // If cache is stale, do a silent background refresh
            if (Date.now() - cached.fetchedAt > CACHE_TTL_MS) {
                fetchPage(0, category, source, false, true);
            }
        } else {
            // No cache — show loading state and fetch
            setItems([]);
            setHasMore(true);
            fetchPage(0, category, source, false, false);
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [category, source, enabled]);

    // Listen for cross-tab cache busting events
    useEffect(() => {
        const subscription = DeviceEventEmitter.addListener('catalog_updated', () => {
            if (enabledRef.current) {
                // Manually trigger a refresh when admin adds/edits an item
                pageRef.current = 0;
                setHasMore(true);
                fetchPage(0, categoryRef.current, sourceRef.current, false);
            }
        });
        return () => subscription.remove();
    }, [fetchPage]);

    const loadMore = useCallback(() => {
        if (!enabledRef.current) return;
        if (loading || loadingMore || !hasMore) return;
        const nextPage = pageRef.current + 1;
        pageRef.current = nextPage;
        fetchPage(nextPage, categoryRef.current, sourceRef.current, true);
    }, [loading, loadingMore, hasMore, fetchPage]);

    const refresh = useCallback(() => {
        if (!enabledRef.current) return;
        pageRef.current = 0;
        setHasMore(true);
        // Silent=true: keep existing items visible while fetching fresh data
        fetchPage(0, categoryRef.current, sourceRef.current, false, true);
    }, [fetchPage]);

    return { items, loading, loadingMore, error, hasMore, loadMore, refresh };
}

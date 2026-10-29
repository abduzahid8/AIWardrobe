import { useState, useEffect, useCallback, useRef } from 'react';
import { DeviceEventEmitter } from 'react-native';
import { supabase } from '../lib/supabase';
import type { ShopCatalogItem } from '../src/types/shop';
import { spreadSimilarCatalogItems } from '../src/utils/shopCatalogOrder';
import { createLogger } from '../src/utils/logger';
import { getThumbnailUrl } from '../src/utils/imageUrl';
// Static catalogs are require()d lazily — the ~120KB data modules would
// otherwise parse/evaluate on app startup (this hook is imported by the
// eager Home tab). First actual seed use pays the cost, off the critical
// path.
// eslint-disable-next-line @typescript-eslint/no-var-requires
function getStaticCatalogData(): { items: ShopCatalogItem[]; outfits: ShopCatalogItem[]; classics: ShopCatalogItem[] } {
    const data = require('../data/shopCatalogItems') as {
        SHOP_CATALOG_ITEMS: ShopCatalogItem[];
        SHOP_OUTFITS: ShopCatalogItem[];
    };
    const inspo = require('../data/inspoMensShopItems') as {
        CLASSIC_MENS_ITEMS: ShopCatalogItem[];
    };
    return { items: data.SHOP_CATALOG_ITEMS, outfits: data.SHOP_OUTFITS, classics: inspo.CLASSIC_MENS_ITEMS };
}

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
function cacheKey(source: string, cat: string, search = '') { return JSON.stringify([source, cat, search]); }

// ── In-flight request deduplication ──────────────────────────────────────────
// When HomeScreen and InspoScreen both call useShopCatalog({ source: 'all' })
// simultaneously, the second caller reuses the first caller's pending fetch
// instead of firing a duplicate Supabase query. The stored promise is the
// shared network load itself, so hook mounts AND idle prewarming dedupe
// against each other.
const inflight = new Map<string, Promise<{ items: ShopCatalogItem[]; hasMore: boolean }>>();

const PAGE_SIZE = 100;
// Single page size for every category. Previously 'all'/'shoes' fetched 300
// rows in one query — 3x the payload on the Home critical path, plus an
// O(n²) diversification pass over 300 items on the JS thread. 100 rows is
// plenty for the first paint (essentials need 14, decks need variety across
// 4 slots); pagination loads more on demand.
// The live catalog contains several curated men's sources, including the
// neutral-background Zara and Massimo Dutti imports. An old single-source
// default returned no rows after the catalog's ingestion sources changed.
const DEFAULT_SHOP_SOURCE = 'all';
const FETCH_TIMEOUT_MS = 15_000;
const FETCH_RETRY_DELAY_MS = 1_500;

/** Reject if the promise does not settle within `ms`. Aborts `controller` on timeout. */
function withTimeout<T>(p: PromiseLike<T>, ms: number, controller?: AbortController): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const t = setTimeout(() => {
            try { controller?.abort(); } catch { /* no-op */ }
            reject(new Error(`shop_catalog query timed out after ${ms}ms`));
        }, ms);
        Promise.resolve(p).then(
            (v) => { clearTimeout(t); resolve(v); },
            (e) => { clearTimeout(t); reject(e); },
        );
    });
}

function isTimeoutOrNetworkError(err: any): boolean {
    const msg = String(err?.message ?? err ?? '').toLowerCase();
    return (
        msg.includes('timed out') ||
        msg.includes('timeout') ||
        msg.includes('network') ||
        msg.includes('fetch failed') ||
        msg.includes('failed to fetch') ||
        err?.name === 'AbortError'
    );
}

export interface UseShopCatalogOptions {
    /** Matches UI chips: upper_body | lower_body | shoes | dresses | outfits (or legacy tops/bottoms maps to category). */
    category?: string;
    /** Limit catalog rows to a specific ingestion source. Defaults to all catalog sources. */
    source?: string;
    /** Skip querying until the caller is ready to show live catalog data. */
    enabled?: boolean;
    /** Search the entire active catalog before pagination. */
    search?: string;
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
        currency:    row.currency ?? undefined,
        imageUrl:    getThumbnailUrl(row.image_url),
        garmentType: row.garment_type as ShopCatalogItem['garmentType'],
        description: row.description ?? '',
        sourceUrl:   row.source_url ?? '',
    };
}

/**
 * Instant local seed from the bundled catalog (~200 static items).
 * Lets Home/essentials/outfit decks render on the very first frame while
 * the Supabase query flies in the background — previously the screen showed
 * a skeleton until the network returned (500ms–2s on cold start).
 * Only applies when no source filter is set; a source-filtered query has no
 * static equivalent.
 */
let staticSeedCache: ShopCatalogItem[] | null = null;
function getStaticSeed(cat: string, activeSource: string): ShopCatalogItem[] {
    if (activeSource && activeSource !== 'all') return [];
    if (!staticSeedCache) {
        const seen = new Set<string>();
        const merged: ShopCatalogItem[] = [];
        // SHOP_OUTFITS included so the `outfits` seed (Inspo guide tab) is
        // non-empty and paints instantly instead of showing a spinner until
        // the network returns.
        const { items, outfits, classics } = getStaticCatalogData();
        for (const item of [...items, ...outfits, ...classics]) {
            const id = String((item as any)?.id ?? '');
            if (!id || seen.has(id)) continue;
            seen.add(id);
            merged.push(item as ShopCatalogItem);
        }
        staticSeedCache = merged;
    }
    const all = staticSeedCache;
    if (cat === 'all') return all.slice(0, PAGE_SIZE);
    if (cat === 'outfits') return all.filter((i) => (i as any).garmentType === 'outfit').slice(0, PAGE_SIZE);
    if (cat === 'tops') return all.filter((i) => (i as any).garmentType === 'upper_body').slice(0, PAGE_SIZE);
    if (cat === 'bottoms') return all.filter((i) => (i as any).garmentType === 'lower_body').slice(0, PAGE_SIZE);
    return all.filter((i) => (i as any).garmentType === cat).slice(0, PAGE_SIZE);
}

function buildQuery(page: number, cat: string, activeSource: string, signal?: AbortSignal, search = '') {
    const size = PAGE_SIZE;
    let q = supabase
        .from('shop_catalog')
        .select('id, brand, name, price, currency, image_url, garment_type, description, source_url')
        .eq('is_active', true)
        .order('sort_order', { ascending: true })
        .order('id', { ascending: true })
        .range(page * size, page * size + size - 1);

    // Cancel the hanging PostgREST request when our timeout fires — otherwise
    // the socket stays open burning radio/battery after we've given up.
    if (signal && typeof (q as any).abortSignal === 'function') {
        q = (q as any).abortSignal(signal);
    }

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

    if (search.trim()) {
        // Quote filter values so punctuation cannot alter the PostgREST grammar.
        const pattern = `%${search.trim().replace(/[%_*\\]/g, c => `\\${c}`)}%`;
        const value = JSON.stringify(pattern);
        q = q.or(`name.ilike.${value},brand.ilike.${value},description.ilike.${value}`);
    }
    return q;
}

/**
 * Shared network load: query → map → diversify → write module cache.
 * Used by the hook AND by idle prewarming, so a tab visited for the first
 * time usually finds a warm cache instead of paying a cold Supabase
 * round-trip + O(n²) diversification during the tab-switch animation.
 * Concurrent callers (hook mount + prewarm, Home + Inspo) share one flight.
 * Throws on error — callers decide whether to surface it (hook) or swallow
 * it (prewarm).
 */
async function loadCatalogPage(
    page: number,
    cat: string,
    activeSource: string,
    append: boolean,
    search = '',
): Promise<{ items: ShopCatalogItem[]; hasMore: boolean }> {
    const key = cacheKey(activeSource, cat, search);
    const flightKey = append ? `${key}|p${page}` : key;
    const running = inflight.get(flightKey);
    if (running) return running;

    const promise = (async () => {
        // One retry on timeout/network (transient Supabase cold start).
        // The error is logged ONCE here per shared flight — hook-level
        // callers sharing this flight must not log again, otherwise a single
        // outage spams N identical errors (Home + Inspo x2 + AIOutfitmaker).
        // Timeouts are expected offline conditions: warn (breadcrumb only),
        // never error (which files a Sentry crash report).
        let lastError: any = null;
        for (let attempt = 0; attempt < 2; attempt++) {
            if (attempt > 0) {
                await new Promise<void>((r) => setTimeout(r, FETCH_RETRY_DELAY_MS));
            }
            const controller = new AbortController();
            try {
                const result = await withTimeout(
                    buildQuery(page, cat, activeSource, controller.signal, search),
                    FETCH_TIMEOUT_MS,
                    controller,
                );
                const { data, error: fetchError } = result as { data: any; error: any };
                if (fetchError) {
                    throw new Error(fetchError.message ?? 'Unknown shop catalog error');
                }
                const mapped = (data ?? []).map(dbRowToItem);
                logger.debug('fetched page', { page, count: mapped.length, cat, activeSource });
                const diversified = spreadSimilarCatalogItems(mapped, []) as ShopCatalogItem[];
                const hasMore = mapped.length === PAGE_SIZE;
                if (!append) {
                    catalogCache.set(key, { items: diversified, hasMore, fetchedAt: Date.now() });
                } else {
                    const prev = catalogCache.get(key);
                    if (prev) {
                        // Dedupe by id on append — overlapping range() pages or
                        // retry races previously doubled rows (same id twice).
                        const prevIds = new Set(prev.items.map((i) => (i as any)?.id));
                        const fresh = diversified.filter((i) => !prevIds.has((i as any)?.id));
                        catalogCache.set(key, {
                            items: [...prev.items, ...fresh],
                            hasMore,
                            fetchedAt: Date.now(),
                        });
                    } else {
                        catalogCache.set(key, { items: diversified, hasMore, fetchedAt: Date.now() });
                    }
                }
                return { items: diversified, hasMore };
            } catch (err) {
                lastError = err;
                // AbortError after our own timeout is just the timeout itself.
                // Only retry timeouts/network blips; a real server error
                // (RLS, bad query) would fail identically on retry.
                if (!isTimeoutOrNetworkError(err)) break;
            }
        }
        if (isTimeoutOrNetworkError(lastError)) {
            logger.warn('shop_catalog fetch failed (timeout/network), showing fallback', {
                message: lastError?.message,
                page,
                cat,
                activeSource,
            });
        } else {
            logger.error('shop_catalog fetch failed', lastError, { page, cat, activeSource });
        }
        throw lastError instanceof Error ? lastError : new Error(String(lastError?.message ?? lastError ?? 'Unknown shop catalog error'));
    })();

    inflight.set(flightKey, promise);
    try {
        return await promise;
    } finally {
        if (inflight.get(flightKey) === promise) inflight.delete(flightKey);
    }
}

/**
 * Idle prewarm entry point — warms the module cache for a catalog key
 * without touching any component state. No-op when the cache is fresh or a
 * load is already in flight. Never throws.
 */
export async function prefetchShopCatalog({
    category = 'all',
    source = 'all',
}: { category?: string; source?: string } = {}): Promise<void> {
    try {
        const key = cacheKey(source, category);
        const cached = catalogCache.get(key);
        if (cached && Date.now() - cached.fetchedAt <= CACHE_TTL_MS) return;
        await loadCatalogPage(0, category, source, false);
    } catch (err) {
        logger.warn('prefetchShopCatalog skipped', err);
    }
}

/** Read a warm cache entry without triggering any fetch (for prewarm image lists). */
export function getCachedShopCatalog({
    category = 'all',
    source = 'all',
}: { category?: string; source?: string } = {}): ShopCatalogItem[] {
    return catalogCache.get(cacheKey(source, category))?.items ?? [];
}

export function useShopCatalog({
    category = 'all',
    source = DEFAULT_SHOP_SOURCE,
    enabled = true,
    search: rawSearch = '',
}: UseShopCatalogOptions = {}): UseShopCatalogResult {
    const search = rawSearch.trim().toLowerCase();
    const searchRef = useRef(search);
    searchRef.current = search;
    const requestVersion = useRef(0);
    const [items, setItems]           = useState<ShopCatalogItem[]>(() =>
        enabled ? catalogCache.get(cacheKey(source, category, search))?.items ?? [] : []);
    const [loading, setLoading]       = useState(() =>
        enabled && !catalogCache.has(cacheKey(source, category, search)));
    const [loadingMore, setLoadingMore] = useState(false);
    const [error, setError]           = useState<string | null>(null);
    const [hasMore, setHasMore]       = useState(() =>
        enabled && (catalogCache.get(cacheKey(source, category, search))?.hasMore ?? true));
    const pageRef                     = useRef(0);
    const categoryRef                 = useRef(category);
    const sourceRef                   = useRef(source);
    const enabledRef                  = useRef(enabled);

    const fetchPage = useCallback(async (page: number, cat: string, activeSource: string, append: boolean, silent = false) => {
        const search = searchRef.current;
        const version = ++requestVersion.current;
        const current = () => version === requestVersion.current && search === searchRef.current && cat === categoryRef.current && activeSource === sourceRef.current && enabledRef.current;
        // silent=true: background refresh — don't show loading spinner (items still visible)
        if (!silent) {
            if (page === 0) setLoading(true);
            else setLoadingMore(true);
        }
        setError(null);

        const key = cacheKey(activeSource, cat, search);
        try {
            const result = await loadCatalogPage(page, cat, activeSource, append, search);
            if (!current()) return;
            pageRef.current = page;
            setItems(append
                ? (prev: ShopCatalogItem[]) => {
                    const prevIds = new Set(prev.map((i) => (i as any)?.id));
                    const fresh = result.items.filter((i) => !prevIds.has((i as any)?.id));
                    return [...prev, ...fresh];
                }
                : result.items);
            setHasMore(result.hasMore);
        } catch (err: any) {
            if (!current()) return;
            // Already logged once inside loadCatalogPage (shared flight) —
            // don't log again here or one outage spams N identical errors.
            // Surface it so the UI can show an empty-state with a retry
            // instead of spinning forever.
            setError(err?.message ?? 'Unknown shop catalog error');
            // Only clear items if we have nothing to show (no cache).
            if (!append && !catalogCache.has(key)) setItems([]);
            setHasMore(false);
        } finally {
            if (current()) {
            setLoading(false);
            setLoadingMore(false);
            }
        }
    }, []);

    // Reset and reload when category or source changes
    useEffect(() => {
        pageRef.current   = 0;
        categoryRef.current = category;
        sourceRef.current = source;
        enabledRef.current = enabled;
        setError(null);

        requestVersion.current += 1;
        setLoadingMore(false);
        if (!enabled) {
            setItems([]);
            setLoading(false);
            setLoadingMore(false);
            setHasMore(false);
            return;
        }

        // Seed from cache immediately — avoids blank screen on revisit
        const key = cacheKey(source, category, search);
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
            // No cache — paint the bundled static seed instantly (no spinner),
            // then refresh from Supabase silently in the background.
            // Previously this showed a loading skeleton until the network
            // returned, blocking essentials + outfit decks on cold start.
            const seed = search ? [] : getStaticSeed(category, source);
            if (seed.length > 0) {
                setItems(seed);
                setHasMore(true);
                setLoading(false);
                fetchPage(0, category, source, false, true);
            } else {
                // No cache — show loading state and fetch
                setItems([]);
                setHasMore(true);
                fetchPage(0, category, source, false, false);
            }
        }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [category, source, enabled, search]);

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

import { useState, useEffect, useCallback, useRef } from 'react';
import { supabase } from '../lib/supabase';
import type { ShopCatalogItem } from '../features/try-on/types';
import { spreadSimilarCatalogItems } from '../src/utils/shopCatalogOrder';

const PAGE_SIZE = 100;
const DEFAULT_SHOP_SOURCE = 'apify-zara-men';

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
        imageUrl:    row.image_url,
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
        let q = supabase
            .from('shop_catalog')
            .select('id, brand, name, price, currency, image_url, garment_type, description')
            .eq('is_active', true)
            .order('sort_order', { ascending: true })
            .range(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE - 1);

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

    const fetchPage = useCallback(async (page: number, cat: string, activeSource: string, append: boolean) => {
        if (page === 0) setLoading(true);
        else setLoadingMore(true);
        setError(null);

        const { data, error: fetchError } = await buildQuery(page, cat, activeSource);

        if (fetchError) {
            setError(fetchError.message);
        } else {
            const mapped = (data ?? []).map(dbRowToItem);
            setItems(prev => {
                const diversified = spreadSimilarCatalogItems(
                    mapped,
                    append ? prev.slice(-2) : [],
                );

                return append ? [...prev, ...diversified] : diversified;
            });
            setHasMore(mapped.length === PAGE_SIZE);
        }

        setLoading(false);
        setLoadingMore(false);
    }, [buildQuery]);

    // Reset and reload when category or source changes
    useEffect(() => {
        pageRef.current   = 0;
        categoryRef.current = category;
        sourceRef.current = source;
        enabledRef.current = enabled;
        setItems([]);
        setError(null);

        if (!enabled) {
            setLoading(false);
            setLoadingMore(false);
            setHasMore(false);
            return;
        }

        setHasMore(true);
        fetchPage(0, category, source, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [category, source, enabled]);

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
        setItems([]);
        setHasMore(true);
        fetchPage(0, categoryRef.current, sourceRef.current, false);
    }, [fetchPage]);

    return { items, loading, loadingMore, error, hasMore, loadMore, refresh };
}
